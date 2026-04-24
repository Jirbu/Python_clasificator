"""
appearance_validator.py
-----------------------
Validátor vzhledu osoby na základě barvy jednotlivých segmentů těla (HSV).

Princip (dle barva.md):
  - 5 segmentů: torso, left/right paže, left/right stehno
  - Pro každý segment: 3 vnitřní body na úsečce (t=0.25, 0.5, 0.75)
  - Každý bod: oblast 5×5 px → mean H, S (V ignorujeme)
  - Buffer 10 snímků per segment (update jen při high confidence)
  - Referenční barva = průměr bufferu
  - Odchylka: wH * circular_H_dist + wS * |S_curr - S_avg|
  - Váhy segmentů: torso=0.40, paže×2=0.15, stehna×2=0.15
  - Výstup: score 0–1 (1.0 = barva odpovídá historii, neutrální pokud buffer prázdný)
"""

from __future__ import annotations

from collections import deque

import cv2
import numpy as np

from pose_detector import LANDMARK_INDEX

# ── Konstanty ────────────────────────────────────────────────────────────────

_BUFFER_SIZE = 10
_MIN_BUFFER  = 3     # min snímků v bufferu než začneme skórovat
_W_H         = 0.7  # váha hue složky při výpočtu odchylky
_W_S         = 0.3  # váha saturation složky

# ── Definice segmentů ────────────────────────────────────────────────────────
# A_keys / B_keys: landmark jména jejichž průměr tvoří krajní body segmentu

_SEGMENT_DEFS = [
    {
        "name":   "torso",
        "A_keys": ["left_shoulder", "right_shoulder"],
        "B_keys": ["left_hip",      "right_hip"     ],
        "weight": 0.40,
    },
    {
        "name":   "left_arm",
        "A_keys": ["left_shoulder"],
        "B_keys": ["left_elbow"  ],
        "weight": 0.15,
    },
    {
        "name":   "right_arm",
        "A_keys": ["right_shoulder"],
        "B_keys": ["right_elbow"  ],
        "weight": 0.15,
    },
    {
        "name":   "left_leg",
        "A_keys": ["left_hip" ],
        "B_keys": ["left_knee"],
        "weight": 0.15,
    },
    {
        "name":   "right_leg",
        "A_keys": ["right_hip" ],
        "B_keys": ["right_knee"],
        "weight": 0.15,
    },
]

# Předpočítáme indexy landmarků
_SEGMENTS: list[dict] = [
    {
        "name":   sd["name"],
        "A_idx":  [LANDMARK_INDEX[k] for k in sd["A_keys"]],
        "B_idx":  [LANDMARK_INDEX[k] for k in sd["B_keys"]],
        "weight": sd["weight"],
    }
    for sd in _SEGMENT_DEFS
]

# ── Pomocné funkce ───────────────────────────────────────────────────────────

def _circular_hue_dist(h1: float, h2: float) -> float:
    """Kruhová vzdálenost hue v rozsahu OpenCV [0, 179]. Vrátí 0–90."""
    d = abs(h1 - h2)
    return min(d, 180.0 - d)


def _sample_segment_hs(
    frame_hsv: np.ndarray,
    pt_a: np.ndarray,
    pt_b: np.ndarray,
) -> tuple[float, float] | None:
    """
    Vzorkuje 3 vnitřní body na segmentu A→B (t=0.25, 0.5, 0.75).
    Pro každý bod vezme oblast 5×5 pixelů.
    Vrátí (mean_H, mean_S) ze všech pixelů, nebo None pokud nejsou data.
    """
    h, w = frame_hsv.shape[:2]
    h_vals: list[float] = []
    s_vals: list[float] = []

    for t in (0.25, 0.50, 0.75):
        pt = pt_a + t * (pt_b - pt_a)
        px = int(pt[0] * w)
        py = int(pt[1] * h)

        x1, x2 = max(0, px - 2), min(w, px + 3)
        y1, y2 = max(0, py - 2), min(h, py + 3)
        roi = frame_hsv[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        h_vals.extend(roi[:, :, 0].flatten().tolist())
        s_vals.extend(roi[:, :, 1].flatten().tolist())

    if not h_vals:
        return None
    return float(np.mean(h_vals)), float(np.mean(s_vals))


# ── Hlavní třída ─────────────────────────────────────────────────────────────

class AppearanceValidator:
    """
    Validátor vzhledu osoby na základě barevných příznaků (HSV segmenty těla).

    Score 0–1:
        1.0 = barva odpovídá historii (nebo buffer ještě není plný)
        0.0 = výrazná odchylka od historické barvy
    """

    def __init__(
        self,
        buffer_size: int = _BUFFER_SIZE,
        min_buffer:  int = _MIN_BUFFER,
    ) -> None:
        self._min_buffer = min_buffer
        # Per-segment buffer: deque (H, S) dvojic
        self._buffers: dict[str, deque] = {
            seg["name"]: deque(maxlen=buffer_size)
            for seg in _SEGMENTS
        }

    def update(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> float:
        """
        Zpracuje jeden snímek.

        Parametry:
            frame     -- BGR frame v originální velikosti
            landmarks -- (33, 4) pole nebo None

        Vrátí:
            appearance_score (float) -- 0–1
        """
        if landmarks is None:
            return 1.0  # neutrální – žádná data

        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Confidence proxy: průměrná viditelnost klíčových kloubů
        key_idx = [
            LANDMARK_INDEX["left_shoulder"],  LANDMARK_INDEX["right_shoulder"],
            LANDMARK_INDEX["left_hip"],        LANDMARK_INDEX["right_hip"],
        ]
        conf      = float(np.mean([landmarks[i, 3] for i in key_idx]))
        high_conf = conf > 0.8

        seg_scores: list[tuple[float, float]] = []  # (weight, error)

        for seg in _SEGMENTS:
            # Krajní body segmentu: průměr viditelných landmarků
            a_pts = [landmarks[i, :2] for i in seg["A_idx"] if landmarks[i, 3] > 0.3]
            b_pts = [landmarks[i, :2] for i in seg["B_idx"] if landmarks[i, 3] > 0.3]
            if not a_pts or not b_pts:
                continue

            pt_a = np.mean(a_pts, axis=0)
            pt_b = np.mean(b_pts, axis=0)

            hs = _sample_segment_hs(frame_hsv, pt_a, pt_b)
            if hs is None:
                continue
            h_curr, s_curr = hs

            # Aktualizuj buffer jen při vysoké confidence
            if high_conf:
                self._buffers[seg["name"]].append((h_curr, s_curr))

            # Skóruj jen pokud je dost dat
            buf = self._buffers[seg["name"]]
            if len(buf) < self._min_buffer:
                continue

            h_avg = float(np.mean([v[0] for v in buf]))
            s_avg = float(np.mean([v[1] for v in buf]))

            # Normalizovaná odchylka (0–1)
            delta_h = _circular_hue_dist(h_curr, h_avg) / 90.0
            delta_s = abs(s_curr - s_avg) / 255.0

            error = _W_H * delta_h + _W_S * delta_s
            seg_scores.append((seg["weight"], error))

        if not seg_scores:
            return 1.0  # buffer ještě není plný – neutrální

        total_w     = sum(w for w, _ in seg_scores)
        final_error = sum(w * e for w, e in seg_scores) / total_w

        return float(np.clip(1.0 - final_error, 0.0, 1.0))

    def reset(self) -> None:
        """Reset při přechodu na nové video."""
        for buf in self._buffers.values():
            buf.clear()
