"""
scale_change_detector.py
-------------------------
Detekuje uniformní změnu velikosti těla mezi snímky.

Logika:
  - Udržuje klouzavý buffer absolutních délek segmentů (torso, ramena, paže, stehna)
  - Pro každý nový snímek spočítá rel_err = |curr - avg_prev| / avg_prev
  - y_k = -A^(-rel_err) + 1  ... malé změny → malý příspěvek, velké → velký (saturuje k 1)
  - y_k_sign = y_k * sign(curr - avg_prev)
  - scale_err = |mean(y_k_sign)|  ... koherentní změny → vysoké, chaotické → nízké

Pravidla resetu: TODO – budou definována samostatně.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from pose_detector import LANDMARK_INDEX

_I = LANDMARK_INDEX

_LEN_KEYS: list[str] = [
    "torso_h",
    "shoulder_w",
    "left_upper_arm",
    "right_upper_arm",
    "left_forearm",
    "right_forearm",
    "left_thigh",
    "right_thigh",
    "left_shin",
    "right_shin",
]

_SEG_VIS = 0.30   # min visibility obou konců segmentu


def _extract_lengths(lm: np.ndarray) -> dict[str, float] | None:
    """
    Extrahuje délky segmentů z landmarks.
    Vrátí None pokud trup (ramena + kyčle) není dostatečně viditelný.
    Neviditelné segmenty (vis < _SEG_VIS) jsou vynechány ze slovníku.
    """
    torso_idx = [
        _I["left_shoulder"], _I["right_shoulder"],
        _I["left_hip"],      _I["right_hip"],
    ]
    if any(lm[i, 3] < _SEG_VIS for i in torso_idx):
        return None

    ls = lm[_I["left_shoulder"],  :2];  ls_v = lm[_I["left_shoulder"],  3]
    rs = lm[_I["right_shoulder"], :2];  rs_v = lm[_I["right_shoulder"], 3]
    lh = lm[_I["left_hip"],       :2];  lh_v = lm[_I["left_hip"],       3]
    rh = lm[_I["right_hip"],      :2];  rh_v = lm[_I["right_hip"],      3]
    le = lm[_I["left_elbow"],     :2];  le_v = lm[_I["left_elbow"],     3]
    re = lm[_I["right_elbow"],    :2];  re_v = lm[_I["right_elbow"],    3]
    lk = lm[_I["left_knee"],      :2];  lk_v = lm[_I["left_knee"],      3]
    rk = lm[_I["right_knee"],     :2];  rk_v = lm[_I["right_knee"],     3]
    lw = lm[_I["left_wrist"],     :2];  lw_v = lm[_I["left_wrist"],     3]
    rw = lm[_I["right_wrist"],    :2];  rw_v = lm[_I["right_wrist"],    3]
    la = lm[_I["left_ankle"],     :2];  la_v = lm[_I["left_ankle"],     3]
    ra = lm[_I["right_ankle"],    :2];  ra_v = lm[_I["right_ankle"],    3]

    sc = (ls + rs) / 2.0
    hc = (lh + rh) / 2.0

    def L(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    result: dict[str, float] = {
        "torso_h":    L(sc, hc),
        "shoulder_w": L(ls, rs),
    }
    l_uarm = L(ls, le) if ls_v >= _SEG_VIS and le_v >= _SEG_VIS else None
    r_uarm = L(rs, re) if rs_v >= _SEG_VIS and re_v >= _SEG_VIS else None
    l_fore = L(le, lw) if le_v >= _SEG_VIS and lw_v >= _SEG_VIS else None
    r_fore = L(re, rw) if re_v >= _SEG_VIS and rw_v >= _SEG_VIS else None
    l_thig = L(lh, lk) if lh_v >= _SEG_VIS and lk_v >= _SEG_VIS else None
    r_thig = L(rh, rk) if rh_v >= _SEG_VIS and rk_v >= _SEG_VIS else None
    l_shin = L(lk, la) if lk_v >= _SEG_VIS and la_v >= _SEG_VIS else None
    r_shin = L(rk, ra) if rk_v >= _SEG_VIS and ra_v >= _SEG_VIS else None

    # Fallback: pokud jeden z páru chybí, použij hodnotu druhého
    if l_uarm is None: l_uarm = r_uarm
    if r_uarm is None: r_uarm = l_uarm
    if l_fore is None: l_fore = r_fore
    if r_fore is None: r_fore = l_fore
    if l_thig is None: l_thig = r_thig
    if r_thig is None: r_thig = l_thig
    if l_shin is None: l_shin = r_shin
    if r_shin is None: r_shin = l_shin

    if l_uarm is not None: result["left_upper_arm"]  = l_uarm
    if r_uarm is not None: result["right_upper_arm"] = r_uarm
    if l_fore is not None: result["left_forearm"]    = l_fore
    if r_fore is not None: result["right_forearm"]   = r_fore
    if l_thig is not None: result["left_thigh"]      = l_thig
    if r_thig is not None: result["right_thigh"]     = r_thig
    if l_shin is not None: result["left_shin"]       = l_shin
    if r_shin is not None: result["right_shin"]      = r_shin

    return result


class ScaleChangeDetector:
    """
    Detekuje uniformní změnu velikosti těla.

    Parametry:
        buf_size   -- počet snímků v bufferu (výchozí 4);
                      scale_err se počítá jako curr vs. průměr předchozích buf_size-1 snímků
        scale_base -- základ exponenciály A v y=-A^(-rel_err)+1 (výchozí 10.0)
    """

    def __init__(
        self,
        buf_size:   int   = 4,
        scale_base: float = 10.0,
    ) -> None:
        self.buf_size   = buf_size
        self.scale_base = scale_base

        self._buffer: deque[dict[str, float]] = deque(maxlen=buf_size)

        # Diagnostika
        self.last_scale_err:    float = 0.0
        self.last_scale_detail: dict[str, dict] = {}   # {seg: {"rel": float, "exp": float}}

    def update(self, lm: np.ndarray | None) -> float:
        """
        Zpracuje nový snímek a vrátí scale_err ∈ [0, 1].

        0.0 = žádná koherentní změna (nebo buffer ještě není plný)
        1.0 = všechny segmenty se změnily stejným směrem o velkou hodnotu
        """
        if lm is None:
            return self.last_scale_err

        lens = _extract_lengths(lm)
        if lens is None:
            return self.last_scale_err

        prev_list = list(self._buffer)   # PŘED append
        self._buffer.append(lens)

        if len(prev_list) < self.buf_size - 1:
            # Buffer ještě není plný
            self.last_scale_err    = 0.0
            self.last_scale_detail = {}
            return 0.0

        A = self.scale_base
        y_vals: list[float] = []
        detail: dict[str, dict] = {}

        for k in _LEN_KEYS:
            if k not in lens:
                continue
            prev_vals = [d[k] for d in prev_list if k in d]
            if len(prev_vals) < len(prev_list):
                continue   # segment chyběl v některém předchozím snímku
            avg_v = float(np.mean(prev_vals))
            if avg_v < 1e-6:
                continue
            diff = lens[k] - avg_v
            rel_err = abs(diff) / avg_v
            y_k      = -A ** (-rel_err) + 1
            y_k_sign = y_k * (diff / abs(diff)) if abs(diff) > 1e-9 else 0.0
            detail[k] = {"rel": round(rel_err, 3), "exp": round(y_k, 3)}
            y_vals.append(y_k_sign)

        self.last_scale_detail = detail
        if y_vals:
            self.last_scale_err = float(abs(np.mean(y_vals)))
        else:
            self.last_scale_err = 0.0

        return self.last_scale_err

    def reset(self) -> None:
        self._buffer.clear()
        self.last_scale_err    = 0.0
        self.last_scale_detail = {}
