"""
person_tracker.py
-----------------
Temporální konzistence + tracking osoby.

Staví na existující valid_pose vrstvě (PoseValidator) a přidává:
  - Tracking hip_center s predikcí pohybu (kinematický model)
  - Matching: nový skeleton přiřazen ke sledované osobě jen pokud je blízko
  - Exponenciální vyhlazení přítomnosti (presence_prob)
  - Stavový automat: NO_PERSON / PERSON_UNCERTAIN / PERSON_PRESENT
  - Grace period: osoba nezmizí okamžitě po několika invalid snímcích
  - Ghost tracking: předpověď pozice během absence (max ghost_frames)
  - Edge logic: povolí zmizení jen pokud osoba odchází ven z záběru

Výstup každého snímku:
  person_present (bool)  – použít místo valid_pose v dalších algoritmech
  debug_info     (dict)  – pro debug panel
"""

from __future__ import annotations
import logging
from enum import Enum, auto

import numpy as np
from pose_detector import LANDMARK_INDEX

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Stavový automat
# ─────────────────────────────────────────────────────────────────────────────

class PersonState(Enum):
    NO_PERSON        = auto()
    PERSON_UNCERTAIN = auto()
    PERSON_PRESENT   = auto()


# ─────────────────────────────────────────────────────────────────────────────
# Pomocné funkce
# ─────────────────────────────────────────────────────────────────────────────

def _hip_center(landmarks: np.ndarray) -> np.ndarray:
    """Vrátí (x, y) hip_center jako numpy pole tvaru (2,)."""
    I = LANDMARK_INDEX
    l = landmarks[I["left_hip"],  :2]
    r = landmarks[I["right_hip"], :2]
    return (l + r) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# PersonTracker
# ─────────────────────────────────────────────────────────────────────────────

class PersonTracker:
    """
    Sleduje jednu osobu v normalizovaných souřadnicích [0, 1].

    Parametry:
        alpha              -- EMA koeficient vyhlazení presence_prob (0–1)
                              vyšší = pomalejší reakce, méně blikání
        presence_high      -- prah pro PERSON_PRESENT (> X)
        presence_low       -- prah pro NO_PERSON (< X)
        grace_frames       -- počet invalid snímků než se presence_prob sníží
        ghost_frames       -- max. snímků predikce bez detekce (ghost tracking)
        match_threshold    -- max. vzdálenost pro přiřazení skeletonu [norm. 0–1]
        edge_margin        -- x < edge_margin nebo x > 1-edge_margin = near edge
        velocity_alpha     -- EMA koeficient vyhlazení velocity
    """

    def __init__(
        self,
        alpha: float             = 0.70,
        presence_high: float     = 0.60,
        presence_low: float      = 0.30,
        grace_frames: int        = 3,
        ghost_frames: int        = 5,
        match_threshold: float   = 0.25,
        edge_margin: float       = 0.20,
        velocity_alpha: float    = 0.50,
    ):
        self.alpha           = alpha
        self.presence_high   = presence_high
        self.presence_low    = presence_low
        self.grace_frames    = grace_frames
        self.ghost_frames    = ghost_frames
        self.match_threshold = match_threshold
        self.edge_margin     = edge_margin
        self.velocity_alpha  = velocity_alpha

        # Sledovaná pozice a rychlost
        self._position: np.ndarray | None = None     # (x, y) poslední validní
        self._velocity: np.ndarray         = np.zeros(2)
        self._predicted: np.ndarray | None = None    # predikovaná pozice

        # Vyhlazená pravděpodobnost přítomnosti
        self._presence_prob: float = 0.0

        # Stav
        self._state: PersonState = PersonState.NO_PERSON

        # Čítače
        self._missing_frames: int = 0   # po sobě jdoucí snímky bez match
        self._ghost_count:    int = 0   # kolik snímků jsme v ghost módu
        self._stable_frames:  int = 0   # po sobě jdoucí snímky ve stavu PERSON_PRESENT

        # Reacquire mode – ochrana proti statickým objektům po opuštění záběru.
        # Po vypršení ghost trackingu (pozice resetována) vstoupíme do reacquire:
        # Nová detekce je přijata jen tehdy, pokud se POHYBUJE (není static objekt).
        # Pokud reacquire_still >= reacquire_still_max snímků beze změny pozice
        # → zamítnutí jako statický falešný pozitiv, reset.
        self._reacquire_mode:    bool                  = False
        self._reacquire_anchor:  np.ndarray | None     = None  # pozice při vstupu do reacquire
        self._reacquire_still:   int                   = 0     # snímky beze změny
        self.reacquire_still_max: int                  = 4     # max povoleno stojících snímků
        self.reacquire_move_thr:  float                = 0.018 # min pohyb od anchor [norm. souřadnice]

    # ── Hlavní metoda ─────────────────────────────────────────────────────────

    def update(
        self,
        valid_pose: bool,
        landmarks: np.ndarray | None,
    ) -> tuple[bool, dict]:
        """
        Zpracuje jeden snímek a vrátí (person_present, debug_info).

        Parametry:
            valid_pose  -- výstup PoseValidator.validate()
            landmarks   -- (33, 4) pole nebo None (pokud nebylo valid_pose)

        Vrátí:
            person_present (bool)
            debug_info     (dict) – pro zobrazení v debug panelu
        """
        matched = False

        if valid_pose and landmarks is not None:
            new_pos = _hip_center(landmarks)

            # ── Vrstva 2: Matching – je nový skeleton naše osoba? ─────────
            if self._position is None:
                # První detekce (nebo po resetu po dlouhé pauze): přiřadíme vždy
                matched = True
            else:
                predicted = self._get_predicted()
                dist = float(np.linalg.norm(new_pos - predicted))
                # Adaptivní threshold: base + 3× aktuální rychlost
                # Po první detekci (velocity≈0) zdvojíme base, aby se nestalo
                # že osoba vstoupí do záběru a ihned je odmítnuta.
                speed = float(np.linalg.norm(self._velocity))
                base_thr = (
                    self.match_threshold * 2.0
                    if self._missing_frames == 0 and self._ghost_count == 0
                       and speed < 1e-4
                    else self.match_threshold
                )
                effective_thr = base_thr + 3.0 * speed
                if dist <= effective_thr:
                    matched = True
                else:
                    # Vzdálenost příliš velká → asi jiná osoba / false positive
                    # Ale: pokud ghost tracking vypršel, resetujeme a přijmeme
                    if self._ghost_count > self.ghost_frames:
                        matched = True
                        # Reset stavu – přiřadíme novou trackovanou osobu
                        self._position   = None
                        self._velocity   = np.zeros(2)
                        self._ghost_count = 0
                    else:
                        logger.debug(
                            "PersonTracker: skeleton odmítnut, dist=%.3f > thr=%.3f",
                            dist, self.match_threshold,
                        )

            if matched:
                # ── Reacquire: ochrana proti statickým objektům ───────────
                # V reacquire módu (po vypršení ghost trackingu) ověřujeme,
                # zda nová detekce skutečně vykazuje pohyb. Statické objekty
                # jako nábytek/vybavení se nepohybují → zamítnutí.
                if self._reacquire_mode:
                    if self._reacquire_anchor is None:
                        # První detekce po resetu – zapamatujeme si pozici
                        self._reacquire_anchor = new_pos.copy()
                        self._reacquire_still  = 1
                    else:
                        move = float(np.linalg.norm(new_pos - self._reacquire_anchor))
                        if move >= self.reacquire_move_thr:
                            # Pohyb zaznamenán → potvrzená reálná osoba
                            self._reacquire_mode   = False
                            self._reacquire_anchor = None
                            self._reacquire_still  = 0
                        else:
                            self._reacquire_still += 1
                            if self._reacquire_still >= self.reacquire_still_max:
                                # Příliš mnoho staticných snímků → statický objekt
                                logger.debug(
                                    "PersonTracker: statický objekt zamítnut "
                                    "(pohyb=%.3f < %.3f po %d snímcích)",
                                    move, self.reacquire_move_thr, self._reacquire_still,
                                )
                                matched              = False
                                self._position       = None
                                self._velocity       = np.zeros(2)
                                self._reacquire_mode   = False
                                self._reacquire_anchor = None
                                self._reacquire_still  = 0

                if matched:
                    # ── Vrstva 1: Aktualizace pozice a velocity ───────────────
                    if self._position is not None:
                        raw_vel = new_pos - self._position
                        self._velocity = (
                            self.velocity_alpha * raw_vel
                            + (1.0 - self.velocity_alpha) * self._velocity
                        )
                    else:
                        # První detekce – velocity neznáme, ale pokud máme
                        # ghost predikci (extrapolaci), odhadneme posun z ní.
                        # Bez toho by velocity zůstala [0,0] a příští snímek
                        # by s pohybující se osobou mohl matcher odmítnout.
                        if self._predicted is not None:
                            self._velocity = new_pos - self._predicted
                    self._position = new_pos.copy()
                    self._predicted = None
                    self._missing_frames = 0
                    self._ghost_count = 0

        # ── vrstva 3: pose_confidence ─────────────────────────────────────
        pose_confidence = 1.0 if matched else 0.0

        # ── Grace period: po dobu grace_frames počítáme confidence = 1 ───
        if not matched:
            self._missing_frames += 1
            if self._missing_frames <= self.grace_frames:
                pose_confidence = 1.0   # grace period – nesnižujeme prob

        # ── Vrstva 4: Temporal smoothing (EMA) ───────────────────────────
        self._presence_prob = (
            self.alpha * self._presence_prob
            + (1.0 - self.alpha) * pose_confidence
        )

        # ── Vrstva 5: State machine ───────────────────────────────────────
        p = self._presence_prob
        if p > self.presence_high:
            self._state = PersonState.PERSON_PRESENT
        elif p > self.presence_low:
            self._state = PersonState.PERSON_UNCERTAIN
        else:
            self._state = PersonState.NO_PERSON

        # Čítač stabilní přítomnosti (pro MotionValidator: long_tracked)
        if self._state == PersonState.PERSON_PRESENT:
            self._stable_frames += 1
        else:
            self._stable_frames = 0

        # ── Vrstva 6–8: Edge + směr pohybu ───────────────────────────────
        near_edge       = False
        allow_disappear = False
        if self._position is not None:
            cx = float(self._position[0])
            near_edge = cx < self.edge_margin or cx > (1.0 - self.edge_margin)
            if near_edge:
                # dx > 0 = pohyb doprava. Pokud osoba je u pravého okraje a
                # pohybuje se doprava → allow_disappear
                dx = float(self._velocity[0])
                moving_out = (cx < self.edge_margin and dx < 0) or \
                             (cx > (1.0 - self.edge_margin) and dx > 0)
                allow_disappear = moving_out

        # ── Vrstva 9: Logika zmizení ──────────────────────────────────────
        # Pokud není matched a osoba není u okraje → udržíme PRESENT
        if not matched and self._state != PersonState.NO_PERSON:
            if not near_edge:
                self._state = PersonState.PERSON_UNCERTAIN  # nepřepnem na NO
            # Pokud je u okraje a pohybuje se ven → dovolíme zmizení (necháme stav)

        # ── Vrstva 10: Ghost tracking ─────────────────────────────────────
        ghost_active = False
        if not matched:
            if self._position is not None:
                self._ghost_count += 1
                if self._ghost_count <= self.ghost_frames:
                    ghost_active = True
                    self._predicted = self._get_predicted()
                elif self._ghost_count > self.ghost_frames:
                        # Ghost period vypršel → resetuj pozici aby příští match
                        # byl vždy akceptován bez distance check.
                        # Vstoupíme do reacquire módu – příští detekce musí prokázat pohyb.
                        self._position         = None
                        self._velocity         = np.zeros(2)
                        self._ghost_count      = 0
                        self._reacquire_mode   = True
                        self._reacquire_anchor = None
                        self._reacquire_still  = 0

        # ── Vrstva 11: Finální rozhodnutí ─────────────────────────────────
        grace_active = self._missing_frames <= self.grace_frames and self._missing_frames > 0

        person_present = (
            self._state == PersonState.PERSON_PRESENT
            or (self._state == PersonState.PERSON_UNCERTAIN and (grace_active or matched))
            # Pokud je skeleton úspěšně přiřazen k osobě, vždy propustíme.
            # Tracker slouží primárně k filtraci VZDÁLENÝCH skeletonů (impostor)
            # a ke ghost trackingu během absence – ne k odmítání vlastních matchů.
            or matched
            or (ghost_active and self._state != PersonState.NO_PERSON)
        )

        # ── Debug info ────────────────────────────────────────────────────
        pos  = self._position  if self._position  is not None else np.zeros(2)
        pred = self._predicted if self._predicted is not None else pos
        debug_info = {
            "tracked_pos":    (float(pos[0]),  float(pos[1])),
            "predicted_pos":  (float(pred[0]), float(pred[1])),
            "velocity":       (float(self._velocity[0]), float(self._velocity[1])),
            "presence_prob":  round(self._presence_prob, 3),
            "state":          self._state.name,
            "ghost_active":   ghost_active,
            "near_edge":      near_edge,
            "matched":        matched,
            "reacquire":      self._reacquire_mode,
            "stable_frames":  self._stable_frames,
        }

        return person_present, debug_info

    def _get_predicted(self) -> np.ndarray:
        """Vrátí predikovanou pozici: last_pos + velocity (kinematický model 1. řádu)."""
        if self._position is None:
            return np.zeros(2)
        return self._position + self._velocity

    def reset(self) -> None:
        """Reset stavu při přechodu na nové video."""
        self._position       = None
        self._velocity       = np.zeros(2)
        self._predicted      = None
        self._presence_prob  = 0.0
        self._state          = PersonState.NO_PERSON
        self._missing_frames = 0
        self._ghost_count    = 0
        self._reacquire_mode   = False
        self._reacquire_anchor = None
        self._reacquire_still  = 0
        self._stable_frames    = 0
