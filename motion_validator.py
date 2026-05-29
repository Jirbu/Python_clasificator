"""
motion_validator.py
-------------------
Per-limb validátor pohybu v oblasti skeletu (Motion-per-Limb Consistency).

Princip:
  - Udržuje buffer posledních N framů (grayscale, downscaled 256x144)
  - Pro aktuální frame t porovnává s historickými t-4, t-8, t-12, t-16, t-20
  - Analýza probíhá PER-LIMB: 5 oblastí (torso, levá paže, pravá paže,
    levá noha, pravá noha)
  - Pro každou končetinu:
      1. ROI = bounding box viditelných landmarků + padding
      2. phaseCorrelate LOKÁLNĚ NA ROI  zobrazuje (kompenzace pohybu)
      3. TM_CCOEFF_NORMED (normalizovaná křížová korelace) -> similarity
      4. Průměr přes časové kroky -> limb_similarity
  - Vážená agregace: torso=0.40, paže×2=0.15, nohy×2=0.15
  - Vysoká similarity -> statický objekt -> invalid
  - Výjimka: long-tracked osoba může mít vysoké similarity (stojí)
  - Temporální smoothing přes posledních 3 hodnot

Vrstvy:
  L1  Buffer framů
  L2  Výběr historických framů (kroky 4,8,12,16,20)
  L3  Definice ROI per-limb z bounding boxu viditelných landmarků + padding
  L4  Globální zarovnání kamerového pohybu: phaseCorrelate na celém framu (256×144),
       jednou za historický krok → sdíleno pro všechny končetiny
  L5  Downscale ROI na 32x32 pro výkon
  L6  Normalizovaná křížová korelace (TM_CCOEFF_NORMED) -> similarity
  L7  Průměr přes historické kroky -> limb_similarity
  L8  Vážená agregace přes končetiny (torso=0.4, ostatní=0.15)
  L9  Temporální smoothing (průměr posledních 3)
  L10 Prahové rozhodnutí: sim > threshold_static -> statický objekt
  L11 Výjimka long_tracked -> vždy dynamic
  L12 Debug info (motion_score, region_dynamic, roi_orig, limb_debug)
"""

from __future__ import annotations

import logging
from collections import deque

import cv2
import numpy as np
from pose_detector import LANDMARK_INDEX

logger = logging.getLogger(__name__)

# Definice končetin: skupiny landmarků + váhy pro agregaci
_LIMB_DEFS = [
    {
        "name":   "torso",
        "keys":   ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "weight": 0.40,
    },
    {
        "name":   "left_arm",
        "keys":   ["left_shoulder", "left_elbow", "left_wrist"],
        "weight": 0.15,
    },
    {
        "name":   "right_arm",
        "keys":   ["right_shoulder", "right_elbow", "right_wrist"],
        "weight": 0.15,
    },
    {
        "name":   "left_leg",
        "keys":   ["left_hip", "left_knee", "left_ankle"],
        "weight": 0.15,
    },
    {
        "name":   "right_leg",
        "keys":   ["right_hip", "right_knee", "right_ankle"],
        "weight": 0.15,
    },
]

# Předpočítáme indexy landmarků z LANDMARK_INDEX
_LIMBS: list[dict] = [
    {
        "name":    ld["name"],
        "indices": [LANDMARK_INDEX[k] for k in ld["keys"]],
        "weight":  ld["weight"],
    }
    for ld in _LIMB_DEFS
]


class MotionValidator:
    """
    Per-limb validátor pohybu v oblasti skeletu.

    Skóre (similarity_score) v [0, 1]:
      - Blizko 1.0 -> regiony se nemení -> pravdepodobné statický objekt
      - Blizko 0.0 -> regiony se mení  -> pravdepodobné pohybující se osoba

    Parametry:
        buffer_size      -- pocet framů v historickém bufferu (min. 25)
        compare_steps    -- vzdálenosti snímků pro srovnání
        roi_padding      -- padding kolem bounding boxu každé konecetiny [px v downscale]
        threshold_static -- práh similarity pro oznacení jako statický objekt
        gray_w / gray_h  -- rozlišení downscale bufferu
    """

    def __init__(
        self,
        buffer_size: int          = 25,
        compare_steps: list[int]  = None,
        roi_padding: int          = 13,
        threshold_static: float   = 0.71,
        gray_w: int               = 256,
        gray_h: int               = 144,
    ):
        self.compare_steps         = compare_steps or [4, 8, 12, 16, 20]
        self.roi_padding           = roi_padding
        self.threshold_static      = threshold_static
        self.gray_w                = gray_w
        self.gray_h                = gray_h

        # L1: Buffer framů (grayscale, downscaled)
        self._frame_buffer: deque[np.ndarray] = deque(maxlen=buffer_size)

    # -- Pomocné --

    def _limb_roi(
        self,
        landmarks: np.ndarray,
        indices: list[int],
    ) -> tuple[int, int, int, int] | None:
        """Vrátí (x1, y1, x2, y2) ROI v downscale souřadnicích pro danou skupinu
        landmarků. Vrátí None pokud žádný landmark není dostatecne viditelný."""
        xs = [landmarks[i, 0] * self.gray_w for i in indices if landmarks[i, 3] > 0.2]
        ys = [landmarks[i, 1] * self.gray_h for i in indices if landmarks[i, 3] > 0.2]
        if not xs:
            return None
        pad = self.roi_padding
        x1 = max(0,            int(min(xs)) - pad)
        y1 = max(0,            int(min(ys)) - pad)
        x2 = min(self.gray_w,  int(max(xs)) + pad)
        y2 = min(self.gray_h,  int(max(ys)) + pad)
        if (x2 - x1) < 12 or (y2 - y1) < 12:
            return None
        return x1, y1, x2, y2

    def _ds_to_orig(
        self,
        box_ds: tuple[int, int, int, int],
        w_orig: int,
        h_orig: int,
    ) -> tuple[int, int, int, int]:
        """Prevede souřadnice z downscale prostoru do originálního framu."""
        x1, y1, x2, y2 = box_ds
        return (
            int(x1 / self.gray_w * w_orig),
            int(y1 / self.gray_h * h_orig),
            int(x2 / self.gray_w * w_orig),
            int(y2 / self.gray_h * h_orig),
        )

    # -- Hlavní metoda --

    def update(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> tuple[float, dict]:
        """
        Zpracuje jeden snímek.

        Parametry:
            frame     -- BGR frame v originální velikosti
            landmarks -- (33, 4) pole nebo None (pokud pose invalid)

        Vrátí:
            sim_score  (float) -- 0–1; vysoké = statické (penalizace), nízké = pohyb
            debug_info (dict)  -- pro debug overlay (obsahuje také region_dynamic pro vizualizaci)
        """
        # L1: Přidat frame do bufferu
        gray       = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_small = cv2.resize(gray, (self.gray_w, self.gray_h))
        self._frame_buffer.append(gray_small)

        h_orig, w_orig = frame.shape[:2]

        # Bez skeletu -> neutral (0.0 = žádná penalizace, jako by se pohyboval)
        if landmarks is None:
            return 0.0, {
                "motion_score":   0.0,
                "region_dynamic": True,
                "roi_orig":       None,
                "limb_debug":     {},
            }

        buf_len       = len(self._frame_buffer)
        limb_results: list[tuple[float, float]] = []   # (weight, similarity)
        limb_debug:   dict[str, dict]           = {}
        torso_roi_orig: tuple | None            = None

        # ── Torso ROI pro phaseCorrelate ──────────────────────────────────
        # Střed torsa = průměr ramen + kyčlí (indexy 11,12,23,24)
        torso_kp_idx = [
            LANDMARK_INDEX["left_shoulder"], LANDMARK_INDEX["right_shoulder"],
            LANDMARK_INDEX["left_hip"],      LANDMARK_INDEX["right_hip"],
        ]
        torso_vis = [(landmarks[i, 0], landmarks[i, 1]) for i in torso_kp_idx
                     if landmarks[i, 3] > 0.2]
        if torso_vis:
            cx_n = float(np.mean([p[0] for p in torso_vis]))  # 0–1
            cy_n = float(np.mean([p[1] for p in torso_vis]))  # 0–1
        else:
            cx_n, cy_n = 0.5, 0.5  # fallback na střed obrazu

        # Bounding box ±0.15 od středu torsa v downscale souřadnicích
        half = 0.15
        tc_x1 = max(0,            int((cx_n - half) * self.gray_w))
        tc_y1 = max(0,            int((cy_n - half) * self.gray_h))
        tc_x2 = min(self.gray_w,  int((cx_n + half) * self.gray_w))
        tc_y2 = min(self.gray_h,  int((cy_n + half) * self.gray_h))

        gray_f       = gray_small.astype(np.float32)
        torso_curr_f = gray_f[tc_y1:tc_y2, tc_x1:tc_x2]  # ROI aktuálního framu

        # ── Zarovnání historických framů na torso ROI ─────────────────────
        # Pro každý historický krok: phaseCorrelate na torso ROI → shift →
        # warpAffine celého historického framu → zarovnaný frame pro NCC
        _max_shift   = min(self.gray_w, self.gray_h) * 0.25
        aligned_hists: dict[int, np.ndarray | None] = {}

        for step in self.compare_steps:
            hist_idx = buf_len - 1 - step
            if hist_idx < 0:
                aligned_hists[step] = None
                continue
            hist_raw  = self._frame_buffer[hist_idx]
            hist_roi  = hist_raw[tc_y1:tc_y2, tc_x1:tc_x2].astype(np.float32)

            if torso_curr_f.size == 0 or hist_roi.shape != torso_curr_f.shape:
                aligned_hists[step] = hist_raw
                continue

            try:
                shift, _ = cv2.phaseCorrelate(torso_curr_f, hist_roi)
                dx = float(np.clip(shift[0], -_max_shift, _max_shift))
                dy = float(np.clip(shift[1], -_max_shift, _max_shift))
            except Exception:
                dx, dy = 0.0, 0.0

            shift_mag = (dx * dx + dy * dy) ** 0.5
            if shift_mag > 2.0:
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                aligned_hists[step] = cv2.warpAffine(
                    hist_raw, M, (self.gray_w, self.gray_h),
                    flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                aligned_hists[step] = hist_raw

        # ── Per-limb NCC + medián přes časové kroky ───────────────────────
        for limb in _LIMBS:
            roi_box = self._limb_roi(landmarks, limb["indices"])
            if roi_box is None:
                continue
            x1, y1, x2, y2 = roi_box
            roi_curr = gray_small[y1:y2, x1:x2]

            roi_orig_limb = self._ds_to_orig(roi_box, w_orig, h_orig)
            if limb["name"] == "torso":
                torso_roi_orig = roi_orig_limb

            similarities: list[float] = []
            for step in self.compare_steps:
                hist_aligned = aligned_hists.get(step)
                if hist_aligned is None:
                    continue

                roi_hist = hist_aligned[y1:y2, x1:x2]
                if roi_hist.shape != roi_curr.shape or roi_hist.size == 0:
                    continue

                r_curr = cv2.resize(roi_curr,  (32, 32))
                r_hist = cv2.resize(roi_hist,  (32, 32))

                res = cv2.matchTemplate(
                    r_curr.astype(np.float32),
                    r_hist.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED,
                )
                sim = float(np.clip(res[0, 0], 0.0, 1.0))
                similarities.append(sim)

            # Medián přes historické kroky (robustnější než průměr)
            if similarities:
                limb_sim = float(np.median(similarities))
                limb_results.append((limb["weight"], limb_sim))
                limb_debug[limb["name"]] = {
                    "sim":      round(limb_sim, 4),
                    "roi_orig": roi_orig_limb,
                }

        # L8: Vážená agregace přes končetiny
        if limb_results:
            total_w          = sum(w for w, _ in limb_results)
            similarity_score = sum(w * s for w, s in limb_results) / total_w
        else:
            similarity_score = 0.0

        # L9: Prahové rozhodnutí
        # SIM >= threshold_static (0.80) → statický objekt → odmítnout.
        region_dynamic = similarity_score < self.threshold_static

        if not region_dynamic:
            logger.debug(
                "MotionValidator: sim=%.4f >= %.2f -> STATIC",
                similarity_score, self.threshold_static,
            )

        # L12: Debug info
        debug_info = {
            "motion_score":   round(similarity_score, 4),
            "region_dynamic": region_dynamic,
            "roi_orig":       torso_roi_orig,
            "limb_debug":     limb_debug,
        }

        # Vracíme sim_score (0–1, vyšší = statický = větší motion penalizace).
        # region_dynamic zůstává v debug_info pro zpracovatávat vizualizace.
        return similarity_score, debug_info

    @property
    def has_history(self) -> bool:
        """True pokud buffer obsahuje dost snímků pro smysluplné porovnání."""
        return len(self._frame_buffer) > self.compare_steps[0]

    def reset(self) -> None:
        """Reset při přechodu na nové video."""
        self._frame_buffer.clear()