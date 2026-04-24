"""
multi_person_manager.py
-----------------------
Koordinátor multi-person trackingu pro max 2 osoby.

Principy:
  - Person 1 (slot 0): sledována primárně VIDEO mode detektorem (full frame).
    Při ztrátě (LOST) se detekce omezí na zamrzlý crop region – pose detektor
    vidí pouze oblast kde osoba naposledy byla a nemůže se chytit jinde.
  - Person 2 (slot 1): detekována IMAGE mode detektorem ve vlastním crop.
    Potvrzení: 3 po sobě jdoucí snímky s dobrou viditelností a dostatečnou
    vzdáleností od Person 1.
  - LOST → EMPTY: po 10 sekundách (80 snímků při 8 FPS).
  - Vzdálenost > 0.30 (normalizovaně) od Person 1 = jiná osoba → Person 2.

Každý slot ma vlastni:
  PersonTracker, PoseValidator, MotionValidator – zdroje nejsou sdíleny.
  MotionValidator každého slotu vždy dostane CELÝ frame (ne crop), takže
  motion frame buffer je v reálných souřadnicích; ROI se počítají z
  full-frame landmarků po crop→fullframe konverzi.
"""

from __future__ import annotations

import logging

import numpy as np
import cv2

from person_tracker        import PersonTracker
from pose_validator        import PoseValidator
from motion_validator      import MotionValidator
from appearance_validator  import AppearanceValidator
from pose_detector         import LANDMARK_INDEX

logger = logging.getLogger(__name__)

# ── Konstanty ────────────────────────────────────────────────────────────────

_LOST_EXPIRE_FRAMES = 80     # 10 s při 8 FPS vzorkování
_CANDIDATE_CONFIRM  = 3      # po sobě jdoucí snímky pro potvrzení Person 2
_NEW_PERSON_DIST    = 0.30   # min vzdálenost (norm.) pro detekci jako nová osoba
_MIN_KEY_VIS        = 0.65   # min průměr visibility klíčových kloubů u kandidáta
_CROP_MARGIN        = 0.40   # margin kolem bounding boxu osoby (40 % šířky/výšky)
_RELAXED_MAX        = 15     # snímků s uvolněnou pose validací po valid_pose=True
_MIN_CROP_PX        = 30     # min rozměr crop oblasti v pixelech

# ── Váhy pro kombinaci skóre (kinematika + motion + appearance) ─────────────
_W_KINEMATIC    = 0.55   # presence_prob z PersonTrackeru
_W_MOTION       = 0.30   # (1 − sim_score): vysoké = pohyb, nízké = statické
_W_APPEARANCE   = 0.15   # appearance_score: 1.0 = barva OK, 0.0 = odchylka
_FINAL_THRESHOLD = 0.40  # min final_conf pro person_present=True


# ── Pomocné funkce ───────────────────────────────────────────────────────────

def _hip_center(lm: np.ndarray) -> np.ndarray:
    """Vrátí (x, y) hip_center jako numpy pole tvaru (2,)."""
    il = LANDMARK_INDEX["left_hip"]
    ir = LANDMARK_INDEX["right_hip"]
    return (lm[il, :2] + lm[ir, :2]) / 2.0


def _compute_crop(lm: np.ndarray) -> tuple | None:
    """
    Spočítá crop bounding box (cx1, cy1, cx2, cy2) v normalizovaných [0,1]
    souřadnicích z viditelných landmarků + 40% margin.
    Vrátí None pokud není dostatek viditelných landmarků.
    """
    vis = lm[lm[:, 3] > 0.2]
    if len(vis) < 3:
        return None
    x1 = float(vis[:, 0].min());  x2 = float(vis[:, 0].max())
    y1 = float(vis[:, 1].min());  y2 = float(vis[:, 1].max())
    w  = max(x2 - x1, 0.05)
    h  = max(y2 - y1, 0.05)
    return (
        max(0.0, x1 - w * _CROP_MARGIN),
        max(0.0, y1 - h * _CROP_MARGIN),
        min(1.0, x2 + w * _CROP_MARGIN),
        min(1.0, y2 + h * _CROP_MARGIN),
    )


def _crop_to_fullframe(lm: np.ndarray, crop: tuple) -> np.ndarray:
    """Převede landmarky z crop-normalizovaných souřadnic na full-frame [0,1]."""
    cx1, cy1, cx2, cy2 = crop
    out = lm.copy()
    out[:, 0] = lm[:, 0] * (cx2 - cx1) + cx1
    out[:, 1] = lm[:, 1] * (cy2 - cy1) + cy1
    return out


def _extract_crop_pixels(frame: np.ndarray, crop: tuple) -> np.ndarray | None:
    """Extrahuje crop region z framu jako pixel array. Vrátí None pro příliš malý crop."""
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = crop
    x1 = max(0, int(cx1 * w));  y1 = max(0, int(cy1 * h))
    x2 = min(w, int(cx2 * w)); y2 = min(h, int(cy2 * h))
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


# ── PersonSlot ───────────────────────────────────────────────────────────────

class PersonSlot:
    """
    Drží veškerý stav pro jednu sledovanou osobu.

    Stavy:
        EMPTY    – slot je prázdný, čeká na detekci
        TRACKING – osoba aktivně sledována
        LOST     – osoba ztracena, crop zmrazen, hledáme ji v frozen_crop
    """

    EMPTY    = "EMPTY"
    TRACKING = "TRACKING"
    LOST     = "LOST"

    def __init__(self, slot_id: int) -> None:
        self.slot_id = slot_id

        # Per-slot pipeline komponenty (každý slot má vlastní sadu)
        self.tracker             = PersonTracker()
        self.pose_validator      = PoseValidator()
        self.motion_validator    = MotionValidator()
        self.appearance_validator = AppearanceValidator()

        # Relaxed decay counter (odpojeno od motion/trackeru – viz main.py komentáře)
        self.relaxed_count: int = 0

        # Tracking state
        self.state: str                = self.EMPTY
        self.crop: tuple | None        = None   # aktivní crop (TRACKING)
        self.frozen_crop: tuple | None = None   # zmrazený crop (LOST)
        self.last_pos: np.ndarray | None = None # poslední hip_center (full-frame norm)
        self.lost_frames: int          = 0      # snímky ve stavu LOST

    def reset(self) -> None:
        """Reset při přechodu na nové video nebo po timeoutu LOST stavu."""
        self.tracker.reset()
        self.pose_validator.reset()
        self.motion_validator.reset()
        self.appearance_validator.reset()
        self.relaxed_count = 0
        self.state         = self.EMPTY
        self.crop          = None
        self.frozen_crop   = None
        self.last_pos      = None
        self.lost_frames   = 0


# ── Kandidát na Person 2 ─────────────────────────────────────────────────────

class _Candidate:
    """
    Sleduje kandidáta na novou osobu přes čas.
    Potvrzení: 3 po sobě jdoucí snímky na podobné pozici (< 0.15 vzdálenosti).
    Pokud se pozice změní → reset kandidáta na nové místo.
    """

    def __init__(self, pos: np.ndarray, lm: np.ndarray) -> None:
        self.pos   = pos.copy()
        self.lm    = lm.copy()
        self.count = 1

    def update(self, pos: np.ndarray, lm: np.ndarray) -> bool:
        """Aktualizuje kandidáta. Vrátí True pokud byl potvrzen (3 snímky)."""
        if float(np.linalg.norm(pos - self.pos)) < 0.15:
            self.pos   = pos.copy()
            self.lm    = lm.copy()
            self.count += 1
        else:
            self.pos   = pos.copy()
            self.lm    = lm.copy()
            self.count = 1
        return self.count >= _CANDIDATE_CONFIRM


# ── MultiPersonManager ───────────────────────────────────────────────────────

class MultiPersonManager:
    """
    Koordinátor trackingu pro max 2 osoby.

    Rozhraní:
        results = manager.update(frame, timestamp_ms, video_detector, image_detector)

    results je list 2 diktů:
        results[0]  – Person 1 (primární osoba)
        results[1]  – Person 2 (sekundární osoba nebo EMPTY)

    Každý dict obsahuje:
        slot_id        (int)
        person_present (bool)
        valid_pose     (bool)
        region_dynamic (bool)
        landmarks      (ndarray|None)  – full-frame, normalized; None pokud absent
        _raw_lm        (ndarray|None)  – před validací (pro crop update)
        state          (str)           – "EMPTY"/"TRACKING"/"LOST"
        crop           (tuple|None)    – aktivní crop box
        frozen_crop    (tuple|None)    – zmrazený crop box
        track_info     (dict)
        motion_info    (dict)
    """

    def __init__(self) -> None:
        self.slots: list[PersonSlot]    = [PersonSlot(0), PersonSlot(1)]
        self._candidate: _Candidate | None = None

    # ── Per-slot pipeline ─────────────────────────────────────────────────────

    def _run_pipeline(
        self,
        slot: PersonSlot,
        frame: np.ndarray,
        raw_lm: np.ndarray | None,
    ) -> dict:
        """
        Orchestrátor jednoho slotu pro jeden snímek.

        Pořadí:
          1. PoseValidator  – tvrddý filtr (None landmarks = ihned FAIL)
          2. PersonTracker  – kinematика: pres_prob + ghost tracking
          3. MotionValidator – soft skóre: sim_score (0=pohyb, 1=statický)
          4. AppearanceValidator – soft skóre: 1.0 (stub)
          5. Váhovaná kombinace → final_conf → person_present

        Vztah:
            final_conf = W_KIN * presence_prob
                       + W_MOT * (1 − sim_score)
                       + W_APP * appearance_score
            person_present = tracker_present AND (final_conf >= FINAL_THRESHOLD)
        """
        relaxed = slot.relaxed_count > 0

        # ── 1. Pose detection + validace ─────────────────────────────────
        if raw_lm is None:
            valid_pose = False
            landmarks  = None
        else:
            valid_pose, _ = slot.pose_validator.validate(raw_lm, relaxed=relaxed)
            landmarks = raw_lm if valid_pose else None

        # ── 2. PersonTracker (kinematika) ────────────────────────────────
        tracker_present, track_info = slot.tracker.update(
            valid_pose = valid_pose,
            landmarks  = landmarks,
        )
        presence_prob = track_info.get("presence_prob", 0.0)

        # ── 3. MotionValidator (soft penalizace) ─────────────────────────
        motion_lm  = landmarks  # None pokud invalid pose
        sim_score, motion_info = slot.motion_validator.update(frame, motion_lm)

        # ── 4. AppearanceValidator (stub = 1.0) ──────────────────────────
        appearance_score = slot.appearance_validator.update(frame, landmarks)

        # ── 5. Váhovaná kombinace ────────────────────────────────────────
        final_conf = (
            _W_KINEMATIC   * presence_prob
            + _W_MOTION    * (1.0 - sim_score)
            + _W_APPEARANCE * appearance_score
        )
        person_present = tracker_present and (final_conf >= _FINAL_THRESHOLD)

        # Relaxed decay counter
        if valid_pose:
            slot.relaxed_count = _RELAXED_MAX
        else:
            slot.relaxed_count = max(slot.relaxed_count - 1, 0)

        return {
            "slot_id":          slot.slot_id,
            "person_present":   person_present,
            "tracker_present":  tracker_present,
            "valid_pose":       valid_pose,
            "final_conf":       round(final_conf, 3),
            "presence_prob":    round(presence_prob, 3),
            "sim_score":        round(sim_score, 3),
            "appearance_score": round(appearance_score, 3),
            "landmarks":        landmarks if person_present else None,
            "_raw_lm":          raw_lm,
            "track_info":       track_info,
            "motion_info":      motion_info,
        }

    def _update_state(self, slot: PersonSlot, result: dict) -> bool:
        """
        Aktualizuje stav slotu (TRACKING/LOST/EMPTY) a crop region.

        Přechod TRACKING→LOST závisí na PersonTrackeru (tracker_present=False,
        žádný ghost). Crop se aktualizuje při valid pose + tracker přítomen.

        Vrátí True pokud slot právě přešel TRACKING→LOST.
        """
        tracker_present   = result["tracker_present"]
        raw_lm            = result["_raw_lm"]
        valid_pose        = result["valid_pose"]
        track_info        = result["track_info"]
        ghost_active      = track_info.get("ghost_active", False)
        tracker_state     = track_info.get("state", "NO_PERSON")
        transitioned_lost = False

        # Crop aktualizujeme při valid pose + tracker potvrzuje osobu
        if raw_lm is not None and valid_pose and tracker_present:
            new_crop = _compute_crop(raw_lm)
            if new_crop is not None:
                slot.crop = new_crop
            slot.last_pos = _hip_center(raw_lm)

        # Přechod stavu: závisí na PersonTrackeru (s grace period + ghost tracking),
        # NE na person_present (který může být False kvůli motion validátoru).
        # Přechod TRACKING→LOST jen když tracker říká NO_PERSON bez ghost trackingu.
        still_tracked = (tracker_present or ghost_active or
                         tracker_state in ("PERSON_PRESENT", "PERSON_UNCERTAIN"))

        if still_tracked:
            if slot.state in (slot.EMPTY, slot.LOST):
                slot.state       = slot.TRACKING
                slot.frozen_crop = None
                slot.lost_frames = 0
                logger.info(
                    "Slot %d: → TRACKING (tracker=%s, pos=%.2f,%.2f)",
                    slot.slot_id, tracker_state,
                    slot.last_pos[0] if slot.last_pos is not None else 0,
                    slot.last_pos[1] if slot.last_pos is not None else 0,
                )
            else:
                slot.state       = slot.TRACKING
                slot.lost_frames = 0
        else:
            if slot.state == slot.TRACKING:
                slot.state        = slot.LOST
                slot.frozen_crop  = slot.crop
                slot.lost_frames  = 1
                transitioned_lost = True
                # Reset trackeru + validátoru → crop detekce ve stavu LOST začíná
                # čistě bez reacquire_mode (tracker byl v reacquire po ghost timeoutu).
                # MotionValidator NERESETTUJEME – jeho frame buffer obsahuje historii
                # celého videa a je potřebný pro detekci pohybu v frozen crop.
                slot.tracker.reset()
                slot.pose_validator.reset()
                logger.info(
                    "Slot %d: TRACKING → LOST (last_pos=%.2f,%.2f)",
                    slot.slot_id,
                    slot.last_pos[0] if slot.last_pos is not None else 0,
                    slot.last_pos[1] if slot.last_pos is not None else 0,
                )
            elif slot.state == slot.LOST:
                slot.lost_frames += 1
                if slot.lost_frames >= _LOST_EXPIRE_FRAMES:
                    logger.info(
                        "Slot %d: LOST → EMPTY (timeout: %d s)",
                        slot.slot_id, slot.lost_frames // 8,
                    )
                    slot.reset()

        return transitioned_lost

    # ── Scan pro Person 2 ─────────────────────────────────────────────────────

    def _handle_scan(self, scan_lm_list: list[np.ndarray]) -> None:
        """
        Zpracuje výsledky full-frame image scanu pro hledání nové osoby.
        Kandidát musí být:
          - daleko od Person 1 (> _NEW_PERSON_DIST)
          - dobrá viditelnost klíčových kloubů (> _MIN_KEY_VIS)
          - potvrzen ve 3 po sobě jdoucích snímcích
        """
        slot1 = self.slots[1]
        if slot1.state != slot1.EMPTY:
            # Slot 1 je obsazen → nechceme přidávat novou osobu
            self._candidate = None
            return

        slot0 = self.slots[0]
        kj    = [
            LANDMARK_INDEX["left_shoulder"],  LANDMARK_INDEX["right_shoulder"],
            LANDMARK_INDEX["left_hip"],       LANDMARK_INDEX["right_hip"],
        ]

        best_lm  = None
        best_pos = None
        for lm in scan_lm_list:
            pos = _hip_center(lm)
            # Musí být dostatečně daleko od Person 1
            if (slot0.last_pos is not None and
                    float(np.linalg.norm(pos - slot0.last_pos)) < _NEW_PERSON_DIST):
                continue
            # Dobrá viditelnost klíčových kloubů
            if float(np.mean([lm[j, 3] for j in kj])) < _MIN_KEY_VIS:
                continue
            best_lm  = lm
            best_pos = pos
            break  # první validní kandidát

        if best_lm is None:
            # Žádný kandidát → oslabujeme count
            if self._candidate is not None:
                self._candidate.count = max(0, self._candidate.count - 1)
                if self._candidate.count == 0:
                    self._candidate = None
            return

        # Aktualizace nebo nový kandidát
        if self._candidate is None:
            self._candidate = _Candidate(best_pos, best_lm)
        else:
            confirmed = self._candidate.update(best_pos, best_lm)
            if confirmed:
                logger.info(
                    "Person 2 potvrzena (pos=%.2f,%.2f)",
                    best_pos[0], best_pos[1],
                )
                slot1.state    = slot1.TRACKING
                slot1.last_pos = self._candidate.pos
                slot1.crop     = _compute_crop(self._candidate.lm)
                slot1.lost_frames = 0
                self._candidate = None

    # ── Hlavní update ─────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        video_detector,   # PoseDetector (VIDEO mode) – pro slot 0 vždy
        image_detector,   # PoseDetectorImage (IMAGE mode) – pro crop + scan
    ) -> tuple[list[dict], bool]:
        """
        Zpracuje jeden snímek.

        Vrátí:
            (results, slot0_lost_transition)
            results[0]  – Person 1 výsledek
            results[1]  – Person 2 výsledek
            slot0_lost_transition – True pokud Person 1 právě přešla TRACKING→LOST
                                    (main.py může resetovat temporal okno)
        """
        results: list[dict] = []

        # ── Pre-compute full-frame IMAGE scan jednou (sdíleno P1 LOST fallback + P2 scan) ──
        scan_all = image_detector.detect_all(frame)

        # ── Slot 0 (Person 1) ─────────────────────────────────────────────
        slot0 = self.slots[0]

        if slot0.state in (slot0.EMPTY, slot0.TRACKING):
            # Spec: crop-first → full-frame fallback
            # TRACKING: nejdřív zkus IMAGE mode na aktuální crop (přesnější, méně šumu)
            raw_lm0 = None
            if slot0.state == slot0.TRACKING and slot0.crop is not None:
                crop_px = _extract_crop_pixels(frame, slot0.crop)
                if crop_px is not None:
                    detected = image_detector.detect_all(crop_px)
                    if detected:
                        raw_lm0 = _crop_to_fullframe(detected[0], slot0.crop)

            # Vždy zpracujeme VIDEO mode (udržuje temporální tracking stav MediaPipe)
            raw_lm0_video = video_detector.process_frame(frame, timestamp_ms)
            # Použijeme crop výsledek pokud je k dispozici, jinak full-frame VIDEO
            if raw_lm0 is None:
                raw_lm0 = raw_lm0_video
        else:
            # LOST: nejdřív zkus IMAGE mode na frozen crop
            raw_lm0 = None
            if slot0.frozen_crop is not None:
                crop_px = _extract_crop_pixels(frame, slot0.frozen_crop)
                if crop_px is not None:
                    detected = image_detector.detect_all(crop_px)
                    if detected:
                        raw_lm0 = _crop_to_fullframe(detected[0], slot0.frozen_crop)

            # Spec: full-frame fallback pokud crop selhal
            if raw_lm0 is None and slot0.last_pos is not None and scan_all:
                # Vyber detekci nejbližší k poslední známé poloze osoby
                best = min(scan_all, key=lambda lm: float(
                    np.linalg.norm(_hip_center(lm) - slot0.last_pos)
                ))
                dist = float(np.linalg.norm(_hip_center(best) - slot0.last_pos))
                if dist < _NEW_PERSON_DIST:
                    raw_lm0 = best
                    logger.debug(
                        "Slot 0 LOST: full-frame fallback nalezl osobu, dist=%.3f", dist
                    )

        r0 = self._run_pipeline(slot0, frame, raw_lm0)
        lost_transition = self._update_state(slot0, r0)
        results.append({
            **r0,
            "state":       slot0.state,
            "crop":        slot0.crop,
            "frozen_crop": slot0.frozen_crop,
        })

        # ── Slot 1 (Person 2) ─────────────────────────────────────────────
        slot1 = self.slots[1]

        if slot1.state == slot1.EMPTY:
            results.append({
                "slot_id": 1, "person_present": False, "valid_pose": False,
                "final_conf": 0.0, "presence_prob": 0.0, "sim_score": 0.0,
                "landmarks": None, "_raw_lm": None,
                "state": slot1.EMPTY, "crop": None, "frozen_crop": None,
                "track_info": {}, "motion_info": {},
            })
        else:
            active_crop = slot1.crop if slot1.state == slot1.TRACKING else slot1.frozen_crop
            raw_lm1 = None
            if active_crop is not None:
                crop_px = _extract_crop_pixels(frame, active_crop)
                if crop_px is not None:
                    detected = image_detector.detect_all(crop_px)
                    if detected:
                        raw_lm1 = _crop_to_fullframe(detected[0], active_crop)
            r1 = self._run_pipeline(slot1, frame, raw_lm1)
            self._update_state(slot1, r1)
            results.append({
                **r1,
                "state":       slot1.state,
                "crop":        slot1.crop,
                "frozen_crop": slot1.frozen_crop,
            })

        # ── Full-frame scan pro Person 2 ───────────────────────────────────
        # Repoužijeme scan_all předpočítaný na začátku (bez duplicitního detect_all)
        self._handle_scan(scan_all)

        return results, lost_transition

    # ── Utility ───────────────────────────────────────────────────────────────

    def log_stats(self) -> None:
        """Zaloguje statistiky pose validátoru pro slot 0 (Person 1)."""
        self.slots[0].pose_validator.log_stats()

    def reset(self) -> None:
        """Reset při přechodu na nové video."""
        for slot in self.slots:
            slot.reset()
        self._candidate = None
