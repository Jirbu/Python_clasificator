"""
pose_consistency_validator.py
------------------------------
Detekuje náhlé nerealistické změny parametrů pózy mezi snímky.

Pipeline:
  1. Pro každý snímek spočítá strukturu parametrů (délky segmentů + úhly kloubů)
  2. Uloží absolutní rozdíl oproti předchozímu snímku do klouzavého bufferu (5 snímků)
  3. Při plném bufferu:
       avg = průměr prvních N-1 rozdílů (bez aktuálního)
       err_k = 0           pokud curr_k <= avg_k
             = lin. interp. pokud avg_k < curr_k < ratio * avg_k
             = 1.0          pokud curr_k >= ratio * avg_k
  4. Vážená suma err_k přes všechny parametry → score
  5. score > suspicious_thr → SUSPICIOUS

Segmenty:  torso_h, shoulder_w, left/right upper_arm, left/right thigh
Úhly:      left/right shoulder (hip_center–shoulder–elbow),
           left/right hip      (shoulder_center–hip–knee)
"""

from __future__ import annotations

from collections import deque

import numpy as np

from pose_detector import LANDMARK_INDEX

_I = LANDMARK_INDEX

# ── Váhy jednotlivých parametrů ─────────────────────────────────────────────
_WEIGHTS_LEN: dict[str, float] = {
    "torso_h":            1.0,
    "shoulder_w":         0.8,
    "left_upper_arm":     1.0,
    "right_upper_arm":    1.0,
    "left_thigh":         1.0,
    "right_thigh":        1.0,
}
_WEIGHTS_ANG: dict[str, float] = {
    "left_shoulder_ang":  1.2,
    "right_shoulder_ang": 1.2,
    "left_hip_ang":       1.2,
    "right_hip_ang":      1.2,
}
_WEIGHTS: dict[str, float] = {**_WEIGHTS_LEN, **_WEIGHTS_ANG}

# Klíče délek pro scale buffer
_LEN_KEYS = list(_WEIGHTS_LEN.keys())


def _compute_params(lm: np.ndarray) -> dict[str, float] | None:
    """
    Spočítá strukturu parametrů z landmarks.
    Vrátí None pokud klíčové klouby trupu (ramena, kyčle) nejsou dostatečně viditelné.
    Délky končetin a úhly jsou volitelné – počítají se pouze pokud oba konce segmentu
    mají visibility >= _SEG_VIS_THR. Chybějící parametry mají hodnotu None.
    """
    # Trup musí být vždy viditelný – jinak vůbec nepočítáme
    torso_idx = [
        _I["left_shoulder"], _I["right_shoulder"],
        _I["left_hip"],      _I["right_hip"],
    ]
    if any(lm[i, 3] < 0.30 for i in torso_idx):
        return None

    ls = lm[_I["left_shoulder"],  :2];  ls_v = lm[_I["left_shoulder"],  3]
    rs = lm[_I["right_shoulder"], :2];  rs_v = lm[_I["right_shoulder"], 3]
    lh = lm[_I["left_hip"],       :2];  lh_v = lm[_I["left_hip"],       3]
    rh = lm[_I["right_hip"],      :2];  rh_v = lm[_I["right_hip"],      3]
    le = lm[_I["left_elbow"],     :2];  le_v = lm[_I["left_elbow"],     3]
    re = lm[_I["right_elbow"],    :2];  re_v = lm[_I["right_elbow"],    3]
    lk = lm[_I["left_knee"],      :2];  lk_v = lm[_I["left_knee"],      3]
    rk = lm[_I["right_knee"],     :2];  rk_v = lm[_I["right_knee"],     3]

    sc = (ls + rs) / 2.0   # střed ramen
    hc = (lh + rh) / 2.0   # střed boků

    SEG_VIS = 0.30   # min visibility obou konců segmentu

    def length(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def angle(p1: np.ndarray, vertex: np.ndarray, p2: np.ndarray) -> float:
        """Úhel ve stupních: p1–vertex–p2."""
        v1 = p1 - vertex
        v2 = p2 - vertex
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        return float(np.degrees(np.arccos(cos)))

    # Trupové délky – vždy dostupné (torzo kontrolováno výše)
    params: dict[str, float | None] = {
        "torso_h":   length(sc, hc),
        "shoulder_w": length(ls, rs),
        # Délky končetin – jen pokud oba konce jsou viditelné
        "left_upper_arm":  length(ls, le) if ls_v >= SEG_VIS and le_v >= SEG_VIS else None,
        "right_upper_arm": length(rs, re) if rs_v >= SEG_VIS and re_v >= SEG_VIS else None,
        "left_thigh":      length(lh, lk) if lh_v >= SEG_VIS and lk_v >= SEG_VIS else None,
        "right_thigh":     length(rh, rk) if rh_v >= SEG_VIS and rk_v >= SEG_VIS else None,
        # Úhly – jen pokud všechny tři body jsou viditelné
        "left_shoulder_ang":  angle(hc, ls, le) if ls_v >= SEG_VIS and le_v >= SEG_VIS else None,
        "right_shoulder_ang": angle(hc, rs, re) if rs_v >= SEG_VIS and re_v >= SEG_VIS else None,
        "left_hip_ang":       angle(sc, lh, lk) if lh_v >= SEG_VIS and lk_v >= SEG_VIS else None,
        "right_hip_ang":      angle(sc, rh, rk) if rh_v >= SEG_VIS and rk_v >= SEG_VIS else None,
    }
    return params


class PoseConsistencyValidator:
    """
    Klouzavý validátor konzistence parametrů pózy.

    Parametry:
        buffer_size      -- délka diff bufferu (výchozí 5)
        err_ratio        -- curr > ratio * avg → err = 1.0 (výchozí 2.0)
        suspicious_thr   -- vážená suma err pro SUSPICIOUS (výchozí 3.0)
    """

    def __init__(
        self,
        buffer_size:    int   = 5,
        err_ratio:      float = 2.0,
        suspicious_thr: float = 3.0,
    ) -> None:
        self.buffer_size    = buffer_size
        self.err_ratio      = err_ratio
        self.suspicious_thr = suspicious_thr

        self._diff_buffer: deque[dict[str, float]] = deque(maxlen=buffer_size)
        self._last_params: dict[str, float] | None = None

        # Diagnostika
        self.last_score:     float = 0.0
        self.last_len_score: float = 0.0
        self.last_ang_score: float = 0.0
        self.last_avg_diff:  dict[str, float] = {}
        self.last_curr_diff: dict[str, float] = {}

    def update(self, lm: np.ndarray | None) -> tuple[bool, float]:
        """
        Vrátí (is_suspicious, score).

        is_suspicious = True pokud score > suspicious_thr a buffer je plný.
        score = 0.0 pokud buffer ještě není plný nebo landmarks chybí.
        """
        if lm is None:
            self._last_params = None
            self._diff_buffer.clear()
            self.last_score = 0.0
            self.last_len_score = 0.0
            self.last_ang_score = 0.0
            return False, 0.0

        params = _compute_params(lm)
        if params is None:
            self._last_params = None
            self._diff_buffer.clear()
            self.last_score = 0.0
            self.last_len_score = 0.0
            self.last_ang_score = 0.0
            return False, 0.0

        if self._last_params is None:
            self._last_params = params
            self.last_score = 0.0
            self.last_len_score = 0.0
            self.last_ang_score = 0.0
            return False, 0.0

        # Absolutní diff oproti předchozímu snímku — jen společné klíče (oba snímky měly hodnotu)
        diff = {
            k: abs(params[k] - self._last_params[k])
            for k in _WEIGHTS
            if params.get(k) is not None and self._last_params.get(k) is not None
        }
        self._last_params = params
        if not diff:
            return False, 0.0
        self._diff_buffer.append(diff)

        if len(self._diff_buffer) < self.buffer_size:
            self.last_score = 0.0
            return False, 0.0

        buf = list(self._diff_buffer)
        prev  = buf[:-1]   # první N-1 záznamů (bez aktuálního)
        curr  = buf[-1]    # aktuální diff

        # Průměrná odchylka předchozích snímků (jen klíče přítonné ve všech předchozích záznamech)
        common_keys = set(diff.keys()).intersection(*[set(d.keys()) for d in prev])
        if not common_keys:
            self.last_score = 0.0
            return False, 0.0
        avg = {k: float(np.mean([d[k] for d in prev if k in d])) for k in common_keys}

        # Err per parametr — oddělené pro délky a úhly
        ratio = self.err_ratio
        total_err = 0.0
        len_err   = 0.0
        ang_err   = 0.0
        for k, w in _WEIGHTS.items():
            if k not in common_keys or k not in diff:
                continue   # segment nebyl viditelný v tomto nebo předchozích snímkách
            a = avg[k]
            c = curr[k]
            if a < 1e-6:
                err = 0.0
            elif c <= a:
                err = 0.0
            elif c >= ratio * a:
                err = 1.0
            else:
                err = (c - a) / ((ratio - 1.0) * a)
            total_err += w * err
            if k in _WEIGHTS_LEN:
                len_err += w * err
            else:
                ang_err += w * err

        self.last_score     = total_err
        self.last_len_score = len_err
        self.last_ang_score = ang_err
        self.last_avg_diff  = avg
        self.last_curr_diff = curr

        is_suspicious = total_err > self.suspicious_thr
        return is_suspicious, total_err

    def reset(self) -> None:
        self._diff_buffer.clear()
        self._last_params   = None
        self.last_score     = 0.0
        self.last_len_score = 0.0
        self.last_ang_score = 0.0
        self.last_avg_diff  = {}
        self.last_curr_diff = {}
