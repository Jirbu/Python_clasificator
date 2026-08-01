"""
flip_detector.py

Kontinuální detektor salta.

Nevyužívá stavový automat ani HMM.

Místo toho integruje:

- energii rotace
- důvěryhodnou otočku

a průběžně odhaduje očekávanou úhlovou rychlost.

Autor: ChatGPT
"""

from __future__ import annotations

import math


def angle_diff(a: float, b: float) -> float:
    """
    Rozdíl dvou úhlů <-180,180>.
    """
    return ((b - a + 180.0) % 360.0) - 180.0


class FlipDetector:

    def __init__(
        self,

        # jak rychle zapomíná evidence
        energy_decay=0.97,

        # jak rychle zapomíná akumulovaná rotace
        rotation_decay=0.998,

        # EMA očekávané úhlové rychlosti
        expected_alpha=0.05,

        # jak rychle roste evidence
        velocity_scale=20.0,

        # rozptyl podobnosti
        similarity_sigma=15.0,

        # minimální potřebná evidence
        energy_threshold=8.0,

        # minimální akumulovaná otočka
        rotation_threshold=90.0,
    ):

        self.energy_decay = energy_decay
        self.rotation_decay = rotation_decay

        self.expected_alpha = expected_alpha

        self.velocity_scale = velocity_scale
        self.similarity_sigma = similarity_sigma

        self.energy_threshold = energy_threshold
        self.rotation_threshold = rotation_threshold

        self.last_angle = None

        self.expected_delta = 0.0

        self.energy = 0.0
        self.rotation = 0.0

        self.flip = False

    # -----------------------------------------------------

    def update(self, angle):

        if angle is None:
            self.reset()
            return False

        if self.last_angle is None:
            self.last_angle = angle
            return False

        delta = angle_diff(self.last_angle, angle)
        self.last_angle = angle

        #
        # 1)
        # síla rotace
        #

        rotation_strength = math.tanh(
            abs(delta) / self.velocity_scale
        )

        #
        # 2)
        # podobnost s očekáváním
        #

        similarity = math.exp(
            -abs(delta - self.expected_delta)
            / self.similarity_sigma
        )

        #
        # 3)
        # výsledná evidence
        #

        evidence = rotation_strength * similarity

        #
        # 4)
        # aktualizace očekávání
        #

        self.expected_delta = (
            (1.0 - self.expected_alpha) * self.expected_delta
            + self.expected_alpha * delta
        )

        #
        # 5)
        # integrace evidence
        #

        self.energy *= self.energy_decay
        self.energy += evidence

        #
        # 6)
        # integrace otočky
        #

        self.rotation *= self.rotation_decay
        self.rotation += evidence * delta

        #
        # 7)
        # rozhodnutí
        #

        self.flip = (
            self.energy > self.energy_threshold
            and
            abs(self.rotation) > self.rotation_threshold
        )

        return self.flip

    # -----------------------------------------------------

    def confidence(self):

        return min(
            self.energy / self.energy_threshold,
            1.0,
        )

    # -----------------------------------------------------

    def reset(self):

        self.last_angle = None

        self.expected_delta = 0.0

        self.energy = 0.0
        self.rotation = 0.0

        self.flip = False