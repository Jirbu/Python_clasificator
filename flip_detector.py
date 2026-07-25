"""
flip_detector.py
----------------
Detekce přemetu (flip) z průběhu úhlu torsa.

Vstup: sekvence úhlů torsa v rozmezí 0–360° (výstup torso_angle.compute_torso_angle).
Výstup: True pokud v posledních `window_size` snímcích docházelo k rotaci (flip).

Stavy:
    IDLE  – bez výrazné rotace (|Δθ| < rot_threshold)
    CW    – rotace po směru hodinových ručiček  (Δθ >=  rot_threshold)
    CCW   – rotace proti směru hodinových ručiček (Δθ <= -rot_threshold)

Flip je detekován pokud alespoň `flip_threshold` (výchozí 70 %) snímků v okně
je ve stavu CW nebo CCW.

Bez externích závislostí – používá pouze numpy.
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ── Pomocná funkce ─────────────────────────────────────────────────────────

def angle_diff(a: float, b: float) -> float:
    """
    Vrátí orientovaný rozdíl úhlů v intervalu (-180, 180].
    Správně zachází s přechodem přes 0°/360°.
    """
    return ((b - a + 180.0) % 360.0) - 180.0


# ── Hlavní detektor ────────────────────────────────────────────────────────

class FlipDetector:
    """
    Detekuje přemet (flip) z průběhu úhlu torsa.

    Parametry:
        window_size     – počet posledních snímků pro vyhodnocení (výchozí 8)
        flip_threshold  – minimální podíl snímků v rotačním stavu pro detekci (0–1)
        rot_threshold   – minimální |Δθ| [stupně/snímek] pro klasifikaci jako CW/CCW
    """

    STATE_NAMES = ["IDLE", "CW", "CCW"]

    def __init__(
        self,
        window_size:    int   = 8,
        flip_threshold: float = 0.70,
        rot_threshold:  float = 7.5,
    ):
        self.window_size    = window_size
        self.flip_threshold = flip_threshold
        self.rot_threshold  = rot_threshold

        # Buffer ukládá až window_size+1 úhlů (pro výpočet window_size delt)
        self._angle_buf: deque[float] = deque(maxlen=window_size + 1)

        # Diagnostika
        self.last_is_flip:        bool      = False
        self.last_flip_prob:      float     = 0.0
        self.last_dominant_state: str       = "IDLE"
        self.last_states:         list[str] = []

    # ── Veřejné API ────────────────────────────────────────────────────────

    def update(self, angle: float | None) -> bool:
        """
        Přidá nový úhel torsa do bufferu a vyhodnotí přítomnost flipu.

        Parametry:
            angle – úhel torsa 0–360° (None = osoba není viditelná)

        Vrátí:
            True pokud je detekován flip v aktuálním okně.
        """
        if angle is None:
            self._angle_buf.clear()
            self.last_is_flip        = False
            self.last_flip_prob      = 0.0
            self.last_dominant_state = "IDLE"
            self.last_states         = []
            return False

        self._angle_buf.append(angle)

        if len(self._angle_buf) < 2:
            self.last_is_flip = False
            return False

        angles = list(self._angle_buf)
        deltas = [angle_diff(angles[i], angles[i + 1]) for i in range(len(angles) - 1)]

        thr = self.rot_threshold
        states = []
        for d in deltas:
            if d >= thr:
                states.append(1)   # CW
            elif d <= -thr:
                states.append(2)   # CCW
            else:
                states.append(0)   # IDLE

        n = len(states)
        rotation_count = sum(1 for s in states if s != 0)
        flip_prob = rotation_count / n

        counts   = [states.count(i) for i in range(3)]
        dominant = int(np.argmax(counts))

        self.last_flip_prob      = float(flip_prob)
        self.last_dominant_state = self.STATE_NAMES[dominant]
        self.last_states         = [self.STATE_NAMES[s] for s in states]
        self.last_is_flip        = flip_prob >= self.flip_threshold
        return self.last_is_flip

    def reset(self) -> None:
        """Reset bufferu (při přechodu na nové video nebo ztrátu osoby)."""
        self._angle_buf.clear()
        self.last_is_flip        = False
        self.last_flip_prob      = 0.0
        self.last_dominant_state = "IDLE"
        self.last_states         = []
