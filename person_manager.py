"""
person_manager.py
-----------------
Čistý manager pro sledování max 2 osob, přesně dle pipeline_new_better.md.

Hlavní logika snímku (TRACKING / LOST):
  1. Detekce v crop oblasti (IMAGE mode) → validační pipeline
  2. Pokud selže: full-frame fallback → validační pipeline
  3. Pokud selže: ghost tracking (PersonTracker drží pozici)
  4. Pokud selže: LOST / EMPTY

Validační pipeline (pro libovolný vstup landmarks):
  1. Pose přítomnost (None = FAIL)
  2. Confidence (viditelnost klíčových kloubů)
  3. Pose validation (geometrie, proporce)
  4. Kinematická brána – HARD FAIL pokud pos > MAX_KIN_DIST od predikce
  5. Motion validator (soft penalty)
  6. Appearance validator (soft penalty)
  7. Váhovaná kombinace → final_conf
  8. FAIL pokud final_conf < FINAL_THR

Nová osoba (EMPTY stav):
  - Detekce ve full-frame, vzdálenost od P1 > MIN_DIST_P2.
  - Potřeba _CANDIDATE_CONFIRM po sobě jdoucích potvrzení → TRACKING.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass

import numpy as np
import cv2

from person_tracker             import PersonTracker
from pose_validator             import PoseValidator
from motion_validator           import MotionValidator
from appearance_validator       import AppearanceValidator
from pose_consistency_validator import PoseConsistencyValidator
from scale_change_detector import ScaleChangeDetector
from pose_detector              import LANDMARK_INDEX

logger = logging.getLogger(__name__)

# ── Konstanty ──────────────────────────────────────────────────────────────────
# Crop
_CROP_MARGIN       = 0.40   # margin okolo bbox při výpočtu crop
_MIN_CROP_PX       = 30     # min rozměr crop oblasti [px]

# Stavový automat
_LOST_EXPIRE       = 80     # snímků ve stavu LOST → přechod EMPTY
_GHOST_FRAMES      = 3      # snímků ve stavu GHOST bez detekce → LOST
_LOST_HYSTERESIS   = 3      # po sobě jdoucích selhání než TRACKING → GHOST
_CANDIDATE_CONFIRM = 3      # počet potvrzení pro novou osobu (P2)
_MIN_DIST_P2       = 0.30   # min vzdálenost P2 od P1 (normalizovaná)
_RELAXED_MAX       = 15     # snímků relaxed pose validace po valid detekci

# ── PRAHY PIPELINE ── upravuj tady ───────────────────────────────────────────

# pose_conf  – průměrná viditelnost klíčových kloubů (ramena, kyčle)
_MIN_KEY_VIS              = 0.30   # pod tímto prahem = FAIL hned na začátku

# pose_vis (L1) – PoseValidator: viditelnost
_POSE_VIS_AVG_THR         = 0.15   # min průměrná visibility všech klíč. kloubů
_POSE_VIS_MIN_JOINT_THR   = 0.15   # min visibility NEJHORŠÍHO kloubu

# pose_geo (L2) – PoseValidator: geometrie
_POSE_GEO_MIN_TORSO       = 0.03   # min výška torza (norm.)

# kin_score – kinematický early filter
_MAX_KIN_DIST         = 0.45   # max odchylka od predikce (nad tím = FAIL) – normální stav
_MAX_KIN_DIST_RELAXED = 1.50   # uvolněný limit při LOST / full-frame fallback (skok, pád)

# mot_sim – tvrdý práh na pohyb (1-sim_score musí být >= prahu, jinak = FAIL)
# crop: benevolentní – osoba stojí na místě je OK
# full_frame: přísnější – ve full-frame chceme vidět pohyb
_MOTION_HARD_THR_CROP     = 0.29   # min (1-sim) pro crop pipeline  [= 1 - threshold_static(0.71)]
_MOTION_HARD_THR_FULL     = 0.40   # min (1-sim) pro full-frame pipeline

# final_conf – vážená kombinace
_W_TRACKER    = 0.40   # presence_prob z PersonTracker (temporální stabilita)
_W_KIN        = 0.15   # kinematic_score = 1 - dist/MAX (blíže = lepší)
_W_MOTION     = 0.20   # (1 - sim_score): vysoká podobnost = statický = penalizace
_W_APPEARANCE = 0.25   # appearance_score: 1.0 = barva odpovídá historii
_FINAL_THR    = 0.30   # min final_conf pro pipeline SUCCESS

# scale_change_detector – detekce přeskoku na jinou osobu
_SCALE_SWITCH_THR      = 0.45   # min scale_err pro detekci přeskoku
_SCALE_SWITCH_APPEAR   = 0.80   # max appearance_score při přeskoku (nízká = jiná osoba)

# overlap check – zamítnutí P2 pokud se překrývá s P1
_OVERLAP_SAME_PERSON_THR = 0.50  # průměrné překrytí bbox nad tímto prahem → stejná osoba


# ─────────────────────────────────────────────────────────────────────────────

# Klíčové klouby pro confidence check
_KEY_IDX = [
    LANDMARK_INDEX["left_shoulder"],  LANDMARK_INDEX["right_shoulder"],
    LANDMARK_INDEX["left_hip"],       LANDMARK_INDEX["right_hip"],
]


# ── Pomocné funkce ─────────────────────────────────────────────────────────────

def _hip_center(lm: np.ndarray) -> np.ndarray:
    """Vrátí (x, y) středu boků jako numpy array (2,)."""
    il = LANDMARK_INDEX["left_hip"]
    ir = LANDMARK_INDEX["right_hip"]
    return np.array([(lm[il, 0] + lm[ir, 0]) / 2.0,
                     (lm[il, 1] + lm[ir, 1]) / 2.0])


# Definice skupin landmarků pro overlap check (shodné s motion_validator)
_OVERLAP_LIMB_DEFS: list[dict] = [
    {"name": "torso",     "keys": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"]},
    {"name": "left_arm",  "keys": ["left_shoulder", "left_elbow", "left_wrist"]},
    {"name": "right_arm", "keys": ["right_shoulder", "right_elbow", "right_wrist"]},
    {"name": "left_leg",  "keys": ["left_hip", "left_knee", "left_ankle"]},
    {"name": "right_leg", "keys": ["right_hip", "right_knee", "right_ankle"]},
]
_OVERLAP_LIMBS: list[dict] = [
    {"name": d["name"], "indices": [LANDMARK_INDEX[k] for k in d["keys"]]}
    for d in _OVERLAP_LIMB_DEFS
]


def _compute_limb_bboxes(
    lm: np.ndarray,
    frame_wh: tuple[int, int],
    padding_frac: float = 0.03,
) -> dict[str, tuple[int, int, int, int]]:
    """Spočítá bounding boxy (v px) per-limb z normalized landmarks.

    Vrátí dict {limb_name: (x1, y1, x2, y2)} pouze pro limby s dostatkem
    viditelných landmarků.
    """
    fw, fh = frame_wh
    pad_x = int(padding_frac * fw)
    pad_y = int(padding_frac * fh)
    result: dict[str, tuple[int, int, int, int]] = {}
    for limb in _OVERLAP_LIMBS:
        vis = [(lm[i, 0], lm[i, 1]) for i in limb["indices"] if lm[i, 3] > 0.2]
        if not vis:
            continue
        x1 = max(0,  int(min(p[0] for p in vis) * fw) - pad_x)
        y1 = max(0,  int(min(p[1] for p in vis) * fh) - pad_y)
        x2 = min(fw, int(max(p[0] for p in vis) * fw) + pad_x)
        y2 = min(fh, int(max(p[1] for p in vis) * fh) + pad_y)
        if (x2 - x1) >= 4 and (y2 - y1) >= 4:
            result[limb["name"]] = (x1, y1, x2, y2)
    return result


def _bbox_overlap_fraction(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    """Vrátí překrytí jako zlomek menšího z obou bbox (0.0 – 1.0)."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    # Překrytí vůči menšímu bbox (konzervativnější, detekuje i containment)
    return inter / min(area_a, area_b)


def _compute_crop(
    lm: np.ndarray,
    predicted_center: np.ndarray | None = None,
    smoothed_side: float | None = None,
    frame_wh: tuple[int, int] = (1, 1),
) -> tuple | None:
    """
    Spočítá čtvercový crop pro PŘÍŠTÍ frame.

    Postup:
      1. Tight bbox kolem viditelných landmarks → šířka W_px, výška H_px (v pixelech)
      2. Strana čtverce v pixelech = W_px + H_px
         pokud je předán smoothed_side (normalizovaný), přepočítá se na px
      3. Střed = predikovaná pozice (kin_predicted); fallback = střed bbox
      4. Crop = čtverec side_px×side_px vycentrovaný na tento střed → zpět normalizovaně
    """
    fw, fh = frame_wh if frame_wh[0] > 1 else (1, 1)
    vis = lm[lm[:, 3] > 0.2]
    if len(vis) < 3:
        return None
    x1, x2 = float(vis[:, 0].min()), float(vis[:, 0].max())
    y1, y2 = float(vis[:, 1].min()), float(vis[:, 1].max())

    W_px = max((x2 - x1) * fw, 0.05 * fw)
    H_px = max((y2 - y1) * fh, 0.05 * fh)

    if smoothed_side is not None:
        # smoothed_side je průměr (W_norm + H_norm), přepočítáme zpět na px
        # jako čtverec jehož strana = smoothed_side * sqrt(fw*fh)
        side_px = smoothed_side * (fw * fh) ** 0.5
    else:
        side_px = W_px + H_px

    if predicted_center is not None:
        cx, cy = float(predicted_center[0]), float(predicted_center[1])
    else:
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

    half_x = (side_px / 2.0) / fw
    half_y = (side_px / 2.0) / fh
    return (
        max(0.0, cx - half_x),
        max(0.0, cy - half_y),
        min(1.0, cx + half_x),
        min(1.0, cy + half_y),
    )


_CROP_EMA_ALPHA = 2.0 / (6 + 1)   # ≈ 0.286, odpovídá oknu ~6 vzorků


# ── Snapshot helpers pro hires fallback ──────────────────────────────────────

def _snapshot_slot_validators(slot) -> dict:
    """Uloží deepcopy stavu slotu, který mutuje _score_and_decide."""
    return {
        "motion_validator":      copy.deepcopy(slot.motion_validator),
        "appearance_validator":  copy.deepcopy(slot.appearance_validator),
        "consistency_validator": copy.deepcopy(slot.consistency_validator),
        "scale_detector":        copy.deepcopy(slot.scale_detector),
        "tracker":               copy.deepcopy(slot.tracker),
        "pose_suspicious":       slot.pose_suspicious,
        "pose_param_score":      slot.pose_param_score,
        "kin_predicted":         copy.deepcopy(slot.kin_predicted),
        "relaxed_count":         slot.relaxed_count,
    }


def _restore_slot_validators(slot, snap: dict) -> None:
    """Obnoví stav slotu ze snapshotu."""
    slot.motion_validator      = snap["motion_validator"]
    slot.appearance_validator  = snap["appearance_validator"]
    slot.consistency_validator = snap["consistency_validator"]
    slot.scale_detector        = snap["scale_detector"]
    slot.tracker               = snap["tracker"]
    slot.pose_suspicious       = snap["pose_suspicious"]
    slot.pose_param_score      = snap["pose_param_score"]
    slot.kin_predicted         = snap["kin_predicted"]
    slot.relaxed_count         = snap["relaxed_count"]


def _extract_crop_px(frame: np.ndarray, crop: tuple) -> np.ndarray | None:
    """Extrahuje oblast framu jako pixel array. Vrátí None pro příliš malý crop."""
    h, w = frame.shape[:2]
    cx1, cy1, cx2, cy2 = crop
    x1 = max(0, int(cx1 * w)); y1 = max(0, int(cy1 * h))
    x2 = min(w, int(cx2 * w)); y2 = min(h, int(cy2 * h))
    if x2 - x1 < _MIN_CROP_PX or y2 - y1 < _MIN_CROP_PX:
        return None
    return frame[y1:y2, x1:x2]


def _crop_to_fullframe(lm: np.ndarray, crop: tuple) -> np.ndarray:
    """Převede landmarks z crop-normalizovaných souřadnic na full-frame [0,1]."""
    cx1, cy1, cx2, cy2 = crop
    out = lm.copy()
    out[:, 0] = lm[:, 0] * (cx2 - cx1) + cx1
    out[:, 1] = lm[:, 1] * (cy2 - cy1) + cy1
    return out


# ── PersonSlot ─────────────────────────────────────────────────────────────────

class PersonSlot:
    """Drží stav jedné sledované osoby."""

    EMPTY    = "EMPTY"
    TRACKING = "TRACKING"
    GHOST    = "GHOST"   # pipeline selhalo, tracker extrapoluje pozici
    LOST     = "LOST"    # tracker i pipeline selhaly, hledáme v frozen_crop

    def __init__(self, slot_id: int) -> None:
        self.slot_id = slot_id

        # Per-slot pipeline komponenty
        self.tracker              = PersonTracker()
        self.pose_validator       = PoseValidator(
            visibility_threshold=_POSE_VIS_AVG_THR,
            min_key_joint_visibility=_POSE_VIS_MIN_JOINT_THR,
            min_torso_height=_POSE_GEO_MIN_TORSO,
        )
        self.motion_validator       = MotionValidator()
        self.appearance_validator   = AppearanceValidator()
        self.consistency_validator  = PoseConsistencyValidator()
        self.scale_detector         = ScaleChangeDetector()

        # Stav
        self.state: str                 = self.EMPTY
        self.crop: tuple | None         = None   # aktivní crop (TRACKING)
        self.frozen_crop: tuple | None  = None   # zmrazený crop (LOST)
        self.lost_frames: int           = 0

        # Relaxed pose validation counter
        self.relaxed_count: int         = 0

        # Hystereze TRACKING→GHOST: počet po sobě jdoucích selhání
        self.consecutive_failures: int  = 0
        # Počitač snímků ve stavu GHOST
        self.ghost_frames: int          = 0

        # EMA filtr velikosti crop čtverce (okno ~6 vzorků, alpha=2/7)
        self.crop_side_ema: float | None = None
        # Maximum EMA hodnoty – používá se při ghost (aby se okno nesmřšťovalo)
        self.crop_side_max: float | None = None

        # Kinematická predikce: expected hip_center pro TENTO snímek
        # (vypočtená z tracked_pos + velocity z předchozího snímku)
        self.kin_predicted: np.ndarray | None = None

        # Výsledky pose consistency z posledního snímku
        self.pose_suspicious:  bool  = False
        self.pose_param_score: float = 0.0

    def reset(self) -> None:
        self.tracker.reset()
        self.pose_validator.reset()
        self.motion_validator.reset()
        self.appearance_validator.reset()
        self.consistency_validator.reset()
        self.scale_detector.reset()
        self.state         = self.EMPTY
        self.crop          = None
        self.frozen_crop   = None
        self.lost_frames          = 0
        self.relaxed_count        = 0
        self.kin_predicted        = None
        self.consecutive_failures = 0
        self.ghost_frames         = 0
        self.crop_side_ema        = None
        self.crop_side_max        = None
        self.pose_suspicious   = False
        self.pose_param_score = 0.0


# ── Kandidát pro P2 ────────────────────────────────────────────────────────────

@dataclass
class _Candidate:
    lm:    np.ndarray
    pos:   np.ndarray   # hip_center (full-frame norm)
    count: int = 0


# ── PersonManager ──────────────────────────────────────────────────────────────

class PersonManager:
    """
    Koordinátor sledování max 2 osob.

    Metoda update() vrátí:
        (results, slot0_lost_transition)

        results[0] – Person 1 (slot 0)
        results[1] – Person 2 (slot 1)

    Klíče ve výsledkovém diktě:
        person_present    bool   – finální rozhodnutí (pipeline OR ghost)
        valid_pose        bool   – prošla early validace
        landmarks         arr|None – landmarks pokud pipeline SUCCESS
        _raw_lm           arr|None – surové landmarks (i při pipeline FAIL)
        state             str    – TRACKING / LOST / EMPTY
        crop              tuple|None
        frozen_crop       tuple|None
        final_conf        float
        presence_prob     float  – PersonTracker EMA
        sim_score         float  – MotionValidator (0=pohyb, 1=statické)
        appearance_score  float  – AppearanceValidator
        kin_score         float  – kinematic_score (1=přesně kde čekáme)
        track_info        dict
        motion_info       dict
        pipeline_used     str    – "crop" / "full_frame" / "ghost" / "none"
    """

    def __init__(self) -> None:
        self.slots: list[PersonSlot] = [PersonSlot(0), PersonSlot(1)]
        self._candidate: _Candidate | None = None
        self._frame_wh: tuple[int, int] = (1, 1)

    # ── Hlavní metoda ──────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        video_detector,    # PoseDetector (VIDEO mode) – P1 full-frame
        image_detector,    # PoseDetectorImage (IMAGE mode) – crop + scan
        prev_frame: np.ndarray | None = None,  # poslední přeskočený snímek před tímto
    ) -> tuple[list[dict], bool]:

        # Uložíme rozlišení framu pro pixel-správný čtvercový crop
        self._frame_wh = (frame.shape[1], frame.shape[0])

        # Pre-compute full-frame IMAGE scan jednou (sdíleno fallback + P2 scan)
        scan_all = image_detector.detect_all(frame)

        # ── Slot 0: Person 1 ──────────────────────────────────────────────
        r0, lost_transition = self._update_slot0(
            frame, timestamp_ms, video_detector, image_detector, scan_all, prev_frame
        )

        # ── Slot 1: Person 2 ──────────────────────────────────────────────
        # Pokud byl detekován scale switch na P1, okamžitě inicializuj P2
        # ze scan_all (bez čekání na _CANDIDATE_CONFIRM)
        if r0.get("scale_switch") and self.slots[1].state == self.slots[1].EMPTY:
            for lm in scan_all:
                conf = float(np.mean([lm[i, 3] for i in _KEY_IDX]))
                if conf < _MIN_KEY_VIS:
                    continue
                self.slots[1].state = self.slots[1].TRACKING
                self.slots[1].crop  = _compute_crop(lm, frame_wh=self._frame_wh)
                self.slots[1].lost_frames  = 0
                self.slots[1].kin_predicted = None
                logger.info("Slot 1: okamžitá init po scale switch P1")
                self._candidate = None
                break

        r1 = self._update_slot1(frame, image_detector, scan_all, r0)

        return [r0, r1], lost_transition

    # ── Person 1 ──────────────────────────────────────────────────────────────

    def _update_slot0(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        video_detector,
        image_detector,
        scan_all: list,
        prev_frame: np.ndarray | None = None,
    ) -> tuple[dict, bool]:
        slot = self.slots[0]

        # pipe_debug: zachytí výsledek obou pokusů (crop + full_frame) pro CSV
        pipe_debug = {
            "crop_stage": "skipped", "crop_val": None, "crop_ref": None,
            "full_stage": "skipped", "full_val": None, "full_ref": None,
        }

        if slot.state in (slot.TRACKING, slot.GHOST, slot.LOST):
            # TRACKING: aktivní crop
            # GHOST: aktivní crop (posouvá se dle kinematické predikce)
            # LOST: frozen crop (tracker již nefunguje, hledáme v poslední známé oblasti)
            crop = slot.crop if slot.state != slot.LOST else slot.frozen_crop
            detection_crop = crop   # crop použitý pro detekci v TOMTO snímku

            # 1. Pokus o crop detekci (IMAGE mode – stateless, přesný)
            # GHOST/LOST stav: uvolněný kinematický limit (osoba může skokem být daleko)
            is_relaxed = slot.state in (slot.GHOST, slot.LOST)
            lm_crop = self._detect_in_crop(frame, crop, image_detector)
            crop_ok, lm_c, kin_c, c_stage, c_val, c_ref = self._early_filter(
                slot, lm_crop, relaxed_kin=is_relaxed
            )
            pipe_debug["crop_stage"] = "pass" if crop_ok else c_stage
            pipe_debug["crop_val"]   = c_val if not crop_ok else None
            pipe_debug["crop_ref"]   = c_ref if not crop_ok else None

            # Vždy voláme VIDEO detektor (udržuje MediaPipe inter-frame stav)
            lm_video = video_detector.process_frame(frame, timestamp_ms)

            if crop_ok:
                effective_lm = lm_c
                kin_score    = kin_c
                pipe_used    = "crop"
            else:
                # 2. Full-frame fallback: VIDEO výsledek nebo nejbližší ze scanu
                # Uvolněný kinematický limit – crop selhal, person může být jinde (skok/pád)
                lm_full = lm_video if lm_video is not None else self._nearest_to(scan_all, slot)
                full_ok, lm_f, kin_f, f_stage, f_val, f_ref = self._early_filter(
                    slot, lm_full, relaxed_kin=True
                )
                pipe_debug["full_stage"] = "pass" if full_ok else f_stage
                pipe_debug["full_val"]   = f_val if not full_ok else None
                pipe_debug["full_ref"]   = f_ref if not full_ok else None
                effective_lm = lm_f  if full_ok else None
                kin_score    = kin_f if full_ok else 0.0
                pipe_used    = "full_frame" if full_ok else "none"

        else:  # EMPTY
            detection_crop = None   # žádný crop, detekce ve full-frame
            # VIDEO full-frame, nebo nejbližší ze scanu
            lm_video = video_detector.process_frame(frame, timestamp_ms)
            lm_full  = lm_video if lm_video is not None else self._nearest_to(scan_all, slot)
            full_ok, lm_v, kin_v, f_stage, f_val, f_ref = self._early_filter(slot, lm_full)
            pipe_debug["full_stage"] = "pass" if full_ok else f_stage
            pipe_debug["full_val"]   = f_val if not full_ok else None
            pipe_debug["full_ref"]   = f_ref if not full_ok else None
            effective_lm = lm_v  if full_ok else None
            kin_score    = kin_v if full_ok else 0.0
            pipe_used    = "full_frame" if full_ok else "none"

        # Soft validators + tracker (jednou za snímek)
        # Uložíme snapshot PŘED prvním průchodem – použijeme jej pokud dojde k hires retryi
        _snap0 = _snapshot_slot_validators(slot)
        result = self._score_and_decide(slot, frame, effective_lm, kin_score, pipe_used)

        # ── Víceúrovňový fallback pro suspicious snímek ────────────────────────
        # Spouští se pouze pokud je snímek suspicious a slot je aktivní.
        # Preferuje prev_frame (přeskočený snímek těsně před tímto) jako zdroj.
        # Level 1: prev_frame normální rozlišení (stejný detektor jako crop pipeline)
        # Level 2: prev_frame hires rozlišení (512×288, pomalejší ale přesnější)
        # backup_level: 0=není potřeba, 1=L1 stačil, 2=L2 stačil, 9=oboje selhalo
        # backup_trigger: "none" | "suspicious" | "no_detection"
        backup_level   = 0
        backup_trigger = "none"

        _run_suspicious    = result.get("pose_suspicious") and slot.state in (slot.TRACKING, slot.GHOST, slot.LOST)
        # no_detection fallback: spustí se jen pokud je stav TRACKING a pohyb předpovídaný
        # trackerem přesahuje 1/2 průměrné délky torsa – tichá staticka snímky nebudeme
        # zbytečně předetekovat drahým hires fallbackem.
        _pred_disp  = slot.tracker.predicted_displacement          # norm. souř. [0,1]
        _avg_torso  = slot.scale_detector.avg_torso_h              # norm. souř. nebo None
        _moving     = (_avg_torso is not None and _pred_disp > 0.5 * _avg_torso)
        _run_no_det = (effective_lm is None) and (slot.state == slot.TRACKING) and _moving

        if _run_suspicious or _run_no_det:
            backup_trigger = "suspicious" if result.get("pose_suspicious") else "no_detection"
            fb_crop        = slot.crop if slot.state != slot.LOST else slot.frozen_crop
            fb_frame       = prev_frame if prev_frame is not None else frame
            fb_label       = "prev_frame" if prev_frame is not None else "current_frame"
            is_relaxed_fb  = slot.state in (slot.GHOST, slot.LOST)
            backup_level   = 9  # pesimistický výchozí stav

            # Level 1: normální rozlišení na fb_frame
            logger.info(
                "Slot 0: suspicious (pose_param=%.3f) – backup L1 normální rozlišení na %s",
                result.get("pose_param", 0.0), fb_label,
            )
            lm_l1 = self._detect_in_crop(fb_frame, fb_crop, image_detector)
            if lm_l1 is not None:
                l1_ok, lm_l1v, kin_l1, _, _, _ = self._early_filter(
                    slot, lm_l1, relaxed_kin=is_relaxed_fb
                )
                if l1_ok:
                    logger.info("Slot 0: backup L1 ÚSPĚŠNÝ")
                    _restore_slot_validators(slot, _snap0)
                    result       = self._score_and_decide(slot, frame, lm_l1v, kin_l1, "crop")
                    backup_level = 1

            # Level 2: hires rozlišení – pouze pokud L1 selhal
            if backup_level == 9:
                logger.info("Slot 0: L1 selhal – zkouším backup L2 hires (512×288) na %s", fb_label)
                lm_l2 = self._detect_in_crop_hires(fb_frame, fb_crop, image_detector)
                if lm_l2 is not None:
                    l2_ok, lm_l2v, kin_l2, _, _, _ = self._early_filter(
                        slot, lm_l2, relaxed_kin=is_relaxed_fb
                    )
                    if l2_ok:
                        logger.info("Slot 0: backup L2 hires ÚSPĚŠNÝ")
                        _restore_slot_validators(slot, _snap0)
                        result       = self._score_and_decide(slot, frame, lm_l2v, kin_l2, "crop_hires")
                        backup_level = 2
                    else:
                        logger.debug("Slot 0: backup L2 selhal (early filter zamítl)")
                else:
                    logger.debug("Slot 0: backup L2 – detekce vrátila None")

            if backup_level == 9:
                logger.info("Slot 0: všechny backup úrovně selhaly")

        # Přechod stavů
        lost_transition = self._apply_state(slot, result)

        r0 = {
            **result,
            "slot_id":       0,
            "state":         slot.state,
            "crop":          slot.crop,
            "frozen_crop":   slot.frozen_crop,
            "detection_crop": detection_crop,
            "pipe_debug":    pipe_debug,
            "backup_level":  backup_level,
            "backup_trigger": backup_trigger,
            "kin_predicted": (float(slot.kin_predicted[0]), float(slot.kin_predicted[1])) if slot.kin_predicted is not None else None,
        }
        return r0, lost_transition

    # ── Person 2 ──────────────────────────────────────────────────────────────

    def _update_slot1(
        self,
        frame: np.ndarray,
        image_detector,
        scan_all: list,
        r0: dict,
    ) -> dict:
        slot = self.slots[1]

        if slot.state == slot.EMPTY:
            self._handle_p2_scan(scan_all, r0)
            return self._empty_result(1)

        # P2 je TRACKING, GHOST nebo LOST – detekce pouze v crop
        crop = slot.crop if slot.state != slot.LOST else slot.frozen_crop
        detection_crop = crop   # crop použitý pro detekci v TOMTO snímku
        is_relaxed = slot.state in (slot.GHOST, slot.LOST)
        lm_crop = self._detect_in_crop(frame, crop, image_detector)
        full_ok, lm_v, kin_v, _, _, _ = self._early_filter(slot, lm_crop, relaxed_kin=is_relaxed)
        effective_lm = lm_v  if full_ok else None
        kin_score    = kin_v if full_ok else 0.0
        pipe_used    = "crop" if full_ok else "none"

        # ── Overlap check: zamítni slot 1 pokud detekuje stejnou osobu jako slot 0 ──
        # Slot 0 má už bboxes předpočítané v "limb_bboxes_px" – žádný extra výpočet.
        # Pro slot 1 počítáme bboxes jen když máme co porovnávat.
        if effective_lm is not None:
            bboxes0: dict[str, tuple] = r0.get("limb_bboxes_px") or {}
            bboxes1 = _compute_limb_bboxes(effective_lm, self._frame_wh) if bboxes0 else {}
            overlaps = [
                _bbox_overlap_fraction(bboxes1[name], bboxes0[name])
                for name in bboxes1
                if name in bboxes0
            ]
            if overlaps:
                avg_overlap = float(np.mean(overlaps))
                if avg_overlap >= _OVERLAP_SAME_PERSON_THR:
                    logger.info(
                        "Slot 1: overlap check → stejná osoba jako slot 0 "
                        "(avg_overlap=%.2f >= %.2f) – detekce zamítnuta",
                        avg_overlap, _OVERLAP_SAME_PERSON_THR,
                    )
                    effective_lm = None
                    kin_score    = 0.0
                    pipe_used    = "none"

        _snap1 = _snapshot_slot_validators(slot)
        result = self._score_and_decide(slot, frame, effective_lm, kin_score, pipe_used)

        # Hires fallback (stejná logika jako u slot 0)
        if result.get("pose_suspicious") and slot.state in (slot.TRACKING, slot.GHOST, slot.LOST):
            hires_crop = slot.crop if slot.state != slot.LOST else slot.frozen_crop
            logger.info(
                "Slot 1: snímek suspicious (pose_param=%.3f) – spouštím hires fallback (512×288)",
                result.get("pose_param", 0.0),
            )
            lm_hires = self._detect_in_crop_hires(frame, hires_crop, image_detector)
            if lm_hires is not None:
                is_relaxed_hires = slot.state in (slot.GHOST, slot.LOST)
                hires_ok, lm_h, kin_h, _, _, _ = self._early_filter(
                    slot, lm_hires, relaxed_kin=is_relaxed_hires
                )
                if hires_ok:
                    logger.info("Slot 1: hires fallback ÚSPĚŠNÝ – výsledek přepsán hires detekcí")
                    _restore_slot_validators(slot, _snap1)
                    result = self._score_and_decide(slot, frame, lm_h, kin_h, "crop_hires")
                else:
                    logger.debug("Slot 1: hires fallback selhal (early filter zamítl)")
            else:
                logger.debug("Slot 1: hires fallback – detekce vrátila None")

        self._apply_state(slot, result)

        return {
            **result,
            "slot_id":        1,
            "state":          slot.state,
            "crop":           slot.crop,
            "frozen_crop":    slot.frozen_crop,
            "detection_crop": detection_crop,
            "kin_predicted": (float(slot.kin_predicted[0]), float(slot.kin_predicted[1])) if slot.kin_predicted is not None else None,
        }

    # ── Validační pipeline ─────────────────────────────────────────────────────

    def _early_filter(
        self,
        slot: PersonSlot,
        raw_lm: np.ndarray | None,
        relaxed_kin: bool = False,
    ) -> tuple[bool, np.ndarray | None, float, str, float, float]:
        """
        Kroky 1–4 pipeline (bez state mutace):
          1. Přítomnost landmarks
          2. Confidence check (viditelnost klíčových kloubů)
          3. Pose validation (geometrie)
          4. Kinematická brána (hard fail)

        Vrátí (ok, landmarks_or_None, kin_score, fail_stage, fail_value, fail_ref).
        fail_stage = "" pokud prošlo (ok=True).
        """
        if raw_lm is None:
            return False, None, 0.0, "no_landmarks", 0.0, 0.0

        # Confidence
        conf = float(np.mean([raw_lm[i, 3] for i in _KEY_IDX]))
        if conf < _MIN_KEY_VIS:
            return False, None, 0.0, "confidence", round(conf, 3), _MIN_KEY_VIS

        # Pose validation
        relaxed = slot.relaxed_count > 0
        valid, reason = slot.pose_validator.validate(raw_lm, relaxed=relaxed)
        if not valid:
            return False, None, 0.0, f"pose_val:{reason}", 0.0, 0.0

        # Kinematická vzdálenost — pouze soft penalizace do kin_score, žádný hard fail
        kin_score = 1.0   # neutrální bez historie
        if slot.kin_predicted is not None:
            curr_pos = _hip_center(raw_lm)
            dist = float(np.linalg.norm(curr_pos - slot.kin_predicted))
            ref_dist = _MAX_KIN_DIST_RELAXED if relaxed_kin else _MAX_KIN_DIST
            kin_score = max(0.0, 1.0 - (dist / ref_dist))

        return True, raw_lm, kin_score, "", 0.0, 0.0

    def _score_and_decide(
        self,
        slot: PersonSlot,
        frame: np.ndarray,
        effective_lm: np.ndarray | None,
        kin_score: float,
        pipe_used: str,
    ) -> dict:
        """
        Kroky 5–8: soft validators + tracker + váhovaná kombinace.
        Mutuje stav: motion_validator, appearance_validator, tracker, relaxed_count,
        kin_predicted.
        """
        valid_pose = effective_lm is not None

        # 5. Motion validator (soft)
        sim_score, motion_info = slot.motion_validator.update(frame, effective_lm)

        # Tvrdý motion práh (různý pro crop vs full-frame)
        if valid_pose:
            motion_thr = _MOTION_HARD_THR_CROP if pipe_used in ("crop", "crop_hires") else _MOTION_HARD_THR_FULL
            if (1.0 - sim_score) < motion_thr:
                valid_pose   = False
                effective_lm = None

        # 6. Appearance validator (soft)
        appearance_score = slot.appearance_validator.update(frame, effective_lm)

        # 6b. Pose consistency (suspicious flag)
        suspicious, pose_param_score = slot.consistency_validator.update(
            effective_lm if valid_pose else None
        )
        slot.pose_suspicious  = suspicious
        slot.pose_param_score = pose_param_score
        slot.scale_detector.update(effective_lm if valid_pose else None)

        # 6c. Detekce přeskoku na jinou osobu (scale switch)
        # Scale switch neznamená okamžitý reset – jen zamítneme tuto detekci jako nevalidní.
        # Historie (appearance, scale buffer) zůstává nedotčena pro příští snímek.
        scale_err    = slot.scale_detector.last_scale_err
        scale_switch = False
        if (
            valid_pose
            and not suspicious
            and scale_err >= _SCALE_SWITCH_THR
            and appearance_score < _SCALE_SWITCH_APPEAR
        ):
            scale_switch = True
            logger.info(
                "Slot %d: scale switch detekován (scale_err=%.3f appear=%.3f) – detekce zamítnuta",
                slot.slot_id, scale_err, appearance_score,
            )
            valid_pose   = False
            effective_lm = None

        # 7. PersonTracker (kinematika + temporální EMA)
        tracker_present, track_info = slot.tracker.update(
            valid_pose=valid_pose,
            landmarks=effective_lm,
        )
        presence_prob = track_info.get("presence_prob", 0.0)
        ghost_active  = track_info.get("ghost_active", False)

        # Uložit predikci pro PŘÍŠTÍ snímek
        if valid_pose:
            tp  = track_info.get("tracked_pos", (0.0, 0.0))
            vel = track_info.get("velocity",     (0.0, 0.0))
            slot.kin_predicted = np.array([tp[0] + vel[0], tp[1] + vel[1]])
        elif ghost_active:
            # Pipeline selhal, ale tracker stále extrapoluje pohyb přes ghost —
            # použij jeho predikci aby se střed cropu pohyboval spolu s osobou
            pp = track_info.get("predicted_pos")
            if pp is not None:
                slot.kin_predicted = np.array([float(pp[0]), float(pp[1])])

        # Aktualizace relaxed_count
        if valid_pose:
            slot.relaxed_count = _RELAXED_MAX
        else:
            slot.relaxed_count = max(slot.relaxed_count - 1, 0)

        # 8. Váhovaná kombinace – zahrnuje jen složky, které mají skutečná data.
        # Složky s defaultní hodnotou 1.0 (kin, motion, appearance) se vynechají
        # pokud ještě nemají historii, aby neuměle nafukaly final_conf.
        #
        # Hodnoty komponent jsou remapovány z [0.3, 1.0] → [0.0, 1.0] před vstupem
        # do váhové kombinace – hodnoty pod 0.3 se zahodí jako 0.0, čímž se zvyšuje
        # citlivost a snižuje počet falešně pozitivních detekcí.
        def _remap(v: float) -> float:
            return max(0.0, (v - 0.5) / 0.5)

        if valid_pose:
            components: list[tuple[float, float]] = [
                (_W_TRACKER, _remap(presence_prob)),   # vždy relevantní
            ]
            if slot.kin_predicted is not None:
                components.append((_W_KIN, _remap(kin_score)))
            if slot.motion_validator.has_history:
                components.append((_W_MOTION, _remap(1.0 - sim_score)))
            if slot.appearance_validator.has_history:
                components.append((_W_APPEARANCE, _remap(appearance_score)))
            total_w    = sum(w for w, _ in components)
            final_conf = sum(w * v for w, v in components) / total_w
        else:
            final_conf = 0.0

        # Rozhodnutí
        pipeline_success = valid_pose and tracker_present and (final_conf >= _FINAL_THR)
        # Ghost: pipeline selhal, ale tracker stále drží osobu
        person_present = pipeline_success or ghost_active

        # Pipe_used upřesnění dle ghost
        if not pipeline_success and ghost_active:
            pipe_used = "ghost"

        return {
            "person_present":   person_present,
            "valid_pose":       valid_pose,
            "landmarks":        effective_lm if pipeline_success else None,
            "_raw_lm":          effective_lm,
            "final_conf":       round(final_conf, 3),
            "presence_prob":    round(presence_prob, 3),
            "sim_score":        round(sim_score, 3),
            "appearance_score": round(appearance_score, 3),
            "kin_score":        round(kin_score, 3),
            "track_info":       track_info,
            "motion_info":      motion_info,
            "limb_bboxes_px":   {
                name: info["roi_orig"]
                for name, info in motion_info.get("limb_debug", {}).items()
                if info.get("roi_orig") is not None
            },
            "pipeline_used":    pipe_used,
            "pose_suspicious":  suspicious,
            "pose_param":       round(pose_param_score, 3),
            "pose_len_score":   round(slot.consistency_validator.last_len_score, 3),
            "pose_ang_score":   round(slot.consistency_validator.last_ang_score, 3),
            "pose_scale_err":   round(slot.scale_detector.last_scale_err, 3),
            "pose_scale_detail": slot.scale_detector.last_scale_detail,
            "scale_switch":     scale_switch,
        }

    # ── State machine ─────────────────────────────────────────────────────────

    def _apply_state(self, slot: PersonSlot, result: dict) -> bool:
        """
        Aktualizuje slot.state, crop, frozen_crop, lost_frames.
        Vrátí True pokud slot přešel do stavu GHOST (signál pro reset temporal).

        Stavové přechody:
          TRACKING → GHOST   po _LOST_HYSTERESIS po sobě jdoucích selhání
          GHOST    → TRACKING při pipeline_success
          GHOST    → LOST     po _GHOST_FRAMES selhání (tracker už nefunguje)
          LOST     → TRACKING při pipeline_success (reacquired)
          LOST     → EMPTY    po _LOST_EXPIRE snímcích
        """
        pipeline_success = result["valid_pose"] and result["final_conf"] >= _FINAL_THR
        effective_lm     = result["_raw_lm"]
        track_info       = result["track_info"]
        ghost_active     = track_info.get("ghost_active", False)

        lost_transition = False

        if pipeline_success:
            # Pipeline uspěl → reset čítačů, libovolný stav → TRACKING
            was_ghost_or_lost = slot.state in (slot.GHOST, slot.LOST)
            prev_ghost_frames             = slot.ghost_frames
            slot.consecutive_failures = 0
            slot.ghost_frames         = 0

            # Aktualizuj crop z detekovaného skeletu
            if effective_lm is not None:
                vis = effective_lm[effective_lm[:, 3] > 0.2]
                if len(vis) >= 3:
                    raw_side = (
                        max(float(vis[:, 0].max()) - float(vis[:, 0].min()), 0.05)
                        + max(float(vis[:, 1].max()) - float(vis[:, 1].min()), 0.05)
                    )
                    # Po ≥3 ghost snímcích a pokud je EMA menší než bbox+10 %
                    # → pravděpodobně jiná osoba → reset EMA
                    if (
                        prev_ghost_frames >= 3
                        and slot.crop_side_ema is not None
                        and raw_side > slot.crop_side_ema
                    ):
                        slot.crop_side_ema = None
                        slot.crop_side_max = None

                    if slot.crop_side_ema is None:
                        slot.crop_side_ema = raw_side * (1.0 + _CROP_MARGIN)
                    else:
                        slot.crop_side_ema = (
                            _CROP_EMA_ALPHA * raw_side
                            + (1.0 - _CROP_EMA_ALPHA) * slot.crop_side_ema
                        )
                    if slot.crop_side_max is None or slot.crop_side_ema > slot.crop_side_max:
                        slot.crop_side_max = slot.crop_side_ema
                    effective_side = max(raw_side, slot.crop_side_ema)
                else:
                    effective_side = slot.crop_side_ema
                nc = _compute_crop(effective_lm, slot.kin_predicted, effective_side,
                                   frame_wh=self._frame_wh)
                if nc:
                    slot.crop = nc

            if was_ghost_or_lost:
                slot.frozen_crop = None
                slot.lost_frames = 0
                prev = slot.state
                logger.info("Slot %d: %s → TRACKING (reacquired)", slot.slot_id, prev)

            slot.state       = slot.TRACKING
            slot.lost_frames = 0

        else:
            # Pipeline selhal

            # Ghost crop: posun středu podle trackerovy predikce
            if slot.kin_predicted is not None and slot.crop_side_max is not None:
                fw, fh = self._frame_wh if self._frame_wh[0] > 1 else (1, 1)
                side_px = slot.crop_side_max * (fw * fh) ** 0.5
                half_x = (side_px / 2.0) / fw
                half_y = (side_px / 2.0) / fh
                cx, cy = float(slot.kin_predicted[0]), float(slot.kin_predicted[1])
                slot.crop = (
                    max(0.0, cx - half_x),
                    max(0.0, cy - half_y),
                    min(1.0, cx + half_x),
                    min(1.0, cy + half_y * 2.0),
                )

            if slot.state == slot.TRACKING:
                slot.consecutive_failures += 1
                if slot.consecutive_failures >= _LOST_HYSTERESIS:
                    slot.consecutive_failures = 0
                    slot.ghost_frames         = 0
                    slot.state                = slot.GHOST
                    slot.frozen_crop          = slot.crop
                    lost_transition           = True
                    logger.info("Slot %d: TRACKING → GHOST", slot.slot_id)

            elif slot.state == slot.GHOST:
                slot.ghost_frames += 1
                if slot.ghost_frames >= _GHOST_FRAMES:
                    slot.ghost_frames  = 0
                    slot.state         = slot.LOST
                    slot.lost_frames   = 0
                    slot.crop_side_ema = None
                    slot.crop_side_max = None
                    slot.tracker.reset()
                    slot.pose_validator.reset()
                    logger.info("Slot %d: GHOST → LOST", slot.slot_id)

            elif slot.state == slot.LOST:
                slot.lost_frames += 1
                if slot.lost_frames >= _LOST_EXPIRE:
                    logger.info("Slot %d: LOST → EMPTY (timeout)", slot.slot_id)
                    slot.reset()

        return lost_transition

    # ── P2 candidate ──────────────────────────────────────────────────────────

    def _handle_p2_scan(self, scan_all: list, r0: dict) -> None:
        """Zpracuje výsledky full-frame scanu pro hledání nové P2."""
        slot1 = self.slots[1]
        if slot1.state != slot1.EMPTY:
            self._candidate = None
            return

        # Nehledáme P2 pokud P1 pipeline právě selhal (nestabilní situace —
        # kandidát by mohl být ta samá osoba detekovaná jinak)
        if not (r0.get("valid_pose") and r0.get("final_conf", 0.0) >= _FINAL_THR):
            self._candidate = None
            return

        # Referenční pozice P1
        p1_lm = r0.get("_raw_lm")
        p1_pos: np.ndarray | None = None
        if p1_lm is not None:
            p1_pos = _hip_center(p1_lm)
        elif self.slots[0].crop is not None:
            cx1, cy1, cx2, cy2 = self.slots[0].crop
            p1_pos = np.array([(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0])

        # Najdi kandidáta: dobrá visibility + dostatečně daleko od P1
        best_lm: np.ndarray | None = None
        best_pos: np.ndarray | None = None
        for lm in scan_all:
            conf = float(np.mean([lm[i, 3] for i in _KEY_IDX]))
            if conf < _MIN_KEY_VIS:
                continue
            pos = _hip_center(lm)
            if p1_pos is not None:
                if float(np.linalg.norm(pos - p1_pos)) < _MIN_DIST_P2:
                    continue
            best_lm  = lm
            best_pos = pos
            break

        if best_lm is None:
            self._candidate = None
            return

        # Overlap check: zamítni kandidáta pokud se překrývá s P1
        bboxes0: dict[str, tuple] = r0.get("limb_bboxes_px") or {}
        if bboxes0:
            bboxes_cand = _compute_limb_bboxes(best_lm, self._frame_wh)
            overlaps = [
                _bbox_overlap_fraction(bboxes_cand[name], bboxes0[name])
                for name in bboxes_cand
                if name in bboxes0
            ]
            if overlaps and float(np.mean(overlaps)) >= _OVERLAP_SAME_PERSON_THR:
                logger.debug(
                    "P2 scan: kandidát zamítnut – překryv s P1 (avg_overlap=%.2f)",
                    float(np.mean(overlaps)),
                )
                self._candidate = None
                return

        # Aktualizace / reset kandidáta
        if self._candidate is None:
            self._candidate = _Candidate(lm=best_lm, pos=best_pos, count=1)
        else:
            dist = float(np.linalg.norm(best_pos - self._candidate.pos))
            if dist < 0.20:
                self._candidate.count += 1
                self._candidate.lm    = best_lm
                self._candidate.pos   = best_pos
            else:
                self._candidate = _Candidate(lm=best_lm, pos=best_pos, count=1)

        # Potvrzení → inicializace P2
        if self._candidate.count >= _CANDIDATE_CONFIRM:
            slot1.state = slot1.TRACKING
            slot1.crop  = _compute_crop(self._candidate.lm, frame_wh=self._frame_wh)
            slot1.lost_frames  = 0
            slot1.kin_predicted = None
            logger.info(
                "Slot 1: nová P2 potvrzena pos=(%.2f, %.2f)",
                self._candidate.pos[0], self._candidate.pos[1],
            )
            self._candidate = None

    # ── Utility ───────────────────────────────────────────────────────────────

    def _detect_in_crop(
        self,
        frame: np.ndarray,
        crop: tuple | None,
        image_detector,
    ) -> np.ndarray | None:
        """Detekuje pózu v crop oblasti, přepočítá na full-frame souřadnice."""
        if crop is None:
            return None
        crop_px = _extract_crop_px(frame, crop)
        if crop_px is None:
            return None
        detected = image_detector.detect_all(crop_px)
        if not detected:
            return None
        return _crop_to_fullframe(detected[0], crop)

    def _detect_in_crop_hires(
        self,
        frame: np.ndarray,
        crop: tuple | None,
        image_detector,
    ) -> np.ndarray | None:
        """Stejné jako _detect_in_crop, ale používá detect_all_hires (512×288)."""
        if crop is None:
            return None
        crop_px = _extract_crop_px(frame, crop)
        if crop_px is None:
            return None
        detected = image_detector.detect_all_hires(crop_px)
        if not detected:
            return None
        return _crop_to_fullframe(detected[0], crop)

    def _nearest_to(
        self,
        scan_all: list,
        slot: PersonSlot,
    ) -> np.ndarray | None:
        """Z full-frame scan_all vybere detekci nejbližší k predikované poloze slotu."""
        if not scan_all:
            return None

        # Referenční bod: predikovaná pozice nebo střed crop
        ref: np.ndarray | None = slot.kin_predicted
        if ref is None and slot.crop is not None:
            cx1, cy1, cx2, cy2 = slot.crop
            ref = np.array([(cx1 + cx2) / 2.0, (cy1 + cy2) / 2.0])
        if ref is None:
            return scan_all[0]   # bez reference vezmi první

        best = min(scan_all, key=lambda lm: float(np.linalg.norm(_hip_center(lm) - ref)))
        dist = float(np.linalg.norm(_hip_center(best) - ref))
        # Zamítnout pokud příliš daleko (pravděpodobně jiná osoba)
        if dist > _MIN_DIST_P2:
            return None
        return best

    def _empty_result(self, slot_id: int) -> dict:
        slot = self.slots[slot_id]
        return {
            "slot_id":          slot_id,
            "person_present":   False,
            "valid_pose":       False,
            "landmarks":        None,
            "_raw_lm":          None,
            "final_conf":       0.0,
            "presence_prob":    0.0,
            "sim_score":        0.0,
            "appearance_score": 1.0,
            "kin_score":        0.0,
            "track_info":       {},
            "motion_info":      {},
            "pipeline_used":    "none",
            "state":            slot.EMPTY,
            "crop":             None,
            "frozen_crop":      None,
        }

    def log_stats(self) -> None:
        self.slots[0].pose_validator.log_stats()

    def reset(self) -> None:
        for slot in self.slots:
            slot.reset()
        self._candidate = None
