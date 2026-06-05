"""
torso_angle.py
--------------
Výpočet úhlu natočení těla od svislé osy.

Definice:
    0°   = osoba stojí přímo (hlava nad kyčlemi)
    90°  = osoba je vodorovně
    180° = osoba je hlavou dolů (stojka / přemet)

Výpočet (priorita):
    1. Primární:  osa = střed_kyčlí → nos
    2. Fallback:  osa = střed_ramen → nos  (pokud kyčle nejsou dostatečně viditelné)

    angle = arccos(dot(osa_norm, (0, -1)))
    Výsledek je vždy v [0°, 180°].
"""

from __future__ import annotations

import numpy as np

from pose_detector import LANDMARK_INDEX

_VIS_THR = 0.20   # min visibility pro použití kloubu


def compute_torso_angle(lm: np.ndarray) -> float | None:
    """
    Vrátí úhel osy těla od svislice v stupních [0.0, 180.0], nebo None.

    Parametry:
        lm – landmarks array tvaru (N, 4): x, y, z, visibility (x,y normalizované 0–1)
    """
    nose_vis = lm[LANDMARK_INDEX["nose"], 3]
    if nose_vis < _VIS_THR:
        return None

    nose_xy = lm[LANDMARK_INDEX["nose"], :2]

    # Primární: střed kyčlí → nos
    lh_vis = lm[LANDMARK_INDEX["left_hip"],  3]
    rh_vis = lm[LANDMARK_INDEX["right_hip"], 3]
    if lh_vis >= _VIS_THR and rh_vis >= _VIS_THR:
        lh = lm[LANDMARK_INDEX["left_hip"],  :2]
        rh = lm[LANDMARK_INDEX["right_hip"], :2]
        base = (lh + rh) / 2.0
    else:
        # Fallback: střed ramen → nos
        ls_vis = lm[LANDMARK_INDEX["left_shoulder"],  3]
        rs_vis = lm[LANDMARK_INDEX["right_shoulder"], 3]
        if ls_vis < _VIS_THR or rs_vis < _VIS_THR:
            return None
        ls = lm[LANDMARK_INDEX["left_shoulder"],  :2]
        rs = lm[LANDMARK_INDEX["right_shoulder"], :2]
        base = (ls + rs) / 2.0

    axis = nose_xy - base   # vektor od základny k hlavě

    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return None

    axis_norm = axis / length
    # "Nahoru" v obrazových souřadnicích = (0, -1)
    cos_angle = float(np.clip(np.dot(axis_norm, np.array([0.0, -1.0])), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos_angle)))
