"""
temporal_model.py
-----------------
Sliding window buffer pro temporální analýzu sekvencí snímků.

Zodpovědnost:
  - Udržuje kruhový buffer posledních N feature vektorů
  - Vrátí concatenovaný temporal feature vektor jakmile je buffer plný
  - Reset între videi
"""

from collections import deque
import numpy as np


class TemporalWindow:
    """
    Sliding window nad posledními `window_size` feature vektory.

    Parametry:
        window_size -- počet snímků v okně (default 6)
    """

    def __init__(self, window_size: int = 6):
        self.window_size = window_size
        self._buffer: deque = deque(maxlen=window_size)

    def add_frame_features(self, features: np.ndarray) -> None:
        """
        Přidá feature vektor aktuálního snímku do bufferu.

        features -- 1D numpy vektor příznaků jednoho snímku
        """
        self._buffer.append(features.astype(np.float32))

    def is_ready(self) -> bool:
        """
        Vrátí True, pokud je buffer plný (máme alespoň window_size snímků).
        Klasifikátor volat jen pokud is_ready() == True.
        """
        return len(self._buffer) == self.window_size

    def get_temporal_features(self) -> np.ndarray:
        """
        Vrátí concatenovaný temporal feature vektor:
          [F(t - window_size + 1), ..., F(t-1), F(t)]

        Celková délka = window_size × features_per_frame

        Vyvolá ValueError pokud buffer není plný – vždy kontrolovat is_ready().
        """
        if not self.is_ready():
            raise ValueError(
                f"Buffer není plný ({len(self._buffer)}/{self.window_size}). "
                "Volej is_ready() před get_temporal_features()."
            )
        # deque je řazený od nejstaršího po nejnovější
        return np.concatenate(list(self._buffer))

    def reset(self) -> None:
        """Vyprázdní buffer. Volat při přechodu na nové video."""
        self._buffer.clear()

    def __len__(self) -> int:
        return len(self._buffer)
