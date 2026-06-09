"""
torso_angle.py
--------------
Výpočet úhlu natočení těla od svislé osy.

Definice:
    0°   = osoba stojí přímo (hlava nahoře)
    90°  = osoba je vodorovně
    180° = osoba je hlavou dolů (stojka / přemet)

Prioritní řetězec výpočtu osy (od nejlepšího k fallbacku):
    1. base=střed_kyčlí   → top=nos
    2. base=střed_ramen   → top=nos        (kyčle nejsou viditelné)
    3. base=střed_kyčlí   → top=střed_ramen (nos není viditelný)
    4. base=jeden_bok     → top=nos         (jen jeden bok viditelný)
    5. base=jedno_rameno  → top=nos         (jen jedno rameno viditelné)
    6. base=jeden_bok     → top=střed_ramen (nos není viditelný, jen jeden bok)
    7. base=jeden_bok     → top=jedno_rameno(nos není viditelný, jen jeden z každého páru)
    8. base=střed_ramen   → top=jedno_rameno – nedává smysl, vynecháno

    Výsledek je vždy v [0°, 180°].
"""

from __future__ import annotations

import numpy as np

from pose_detector import LANDMARK_INDEX

_VIS_THR = 0.20


def _mid_or_single(lm: np.ndarray, key_l: str, key_r: str) -> np.ndarray | None:
    """Vrátí střed páru kloubů, nebo jeden pokud druhý není viditelný, nebo None."""
    vl = lm[LANDMARK_INDEX[key_l], 3]
    vr = lm[LANDMARK_INDEX[key_r], 3]
    pl = lm[LANDMARK_INDEX[key_l], :2]
    pr = lm[LANDMARK_INDEX[key_r], :2]
    if vl >= _VIS_THR and vr >= _VIS_THR:
        return (pl + pr) / 2.0
    if vl >= _VIS_THR:
        return pl.copy()
    if vr >= _VIS_THR:
        return pr.copy()
    return None


def compute_torso_angle(lm: np.ndarray) -> float | None:
    """
    Vrátí úhel osy těla od svislice v stupních [0.0, 180.0], nebo None.

    Parametry:
        lm – landmarks array tvaru (N, 4): x, y, z, visibility (x,y normalizované 0–1)
    """
    nose_vis = lm[LANDMARK_INDEX["nose"], 3]
    nose_xy  = lm[LANDMARK_INDEX["nose"], :2] if nose_vis >= _VIS_THR else None

    hip_pt      = _mid_or_single(lm, "left_hip",      "right_hip")
    shoulder_pt = _mid_or_single(lm, "left_shoulder", "right_shoulder")

    # Vyber dvojici (base, top) podle priority
    if nose_xy is not None:
        if hip_pt is not None:
            base, top = hip_pt, nose_xy          # 1. / 4.
        elif shoulder_pt is not None:
            base, top = shoulder_pt, nose_xy     # 2. / 5.
        else:
            return None
    else:
        # Nos není viditelný → torzo kyčle→ramena
        if hip_pt is not None and shoulder_pt is not None:
            base, top = hip_pt, shoulder_pt      # 3. / 6. / 7.
        else:
            return None

    axis = top - base
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None

    axis_norm = axis / length
    cos_angle = float(np.clip(np.dot(axis_norm, np.array([0.0, -1.0])), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
