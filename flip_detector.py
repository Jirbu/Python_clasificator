import math
import numpy as np


def angle_diff(a, b):
    """Rozdíl úhlů <-180,180>."""
    return ((b - a + 180) % 360) - 180


class FlipDetector:

    IDLE = 0
    CW = 1
    CCW = 2

    STATE_NAMES = ["IDLE", "CW", "CCW"]

    def __init__(self):

        self.last_angle = None

        # počáteční pravděpodobnosti
        self.p = np.array([
            0.98,
            0.01,
            0.01
        ], dtype=float)

        # přechody mezi stavy
        self.A = np.array([
            [0.97, 0.02, 0.01],
            [0.02, 0.97, 0.01],
            [0.02, 0.01, 0.97]
        ], dtype=float)

        # očekávané Δθ
        self.mu = np.array([
            0.0,
            15.0,
            -15.0
        ])

        # směrodatná odchylka měření
        self.sigma = np.array([
            8.0,
            10.0,
            10.0
        ])

    def gaussian(self, x, mu, sigma):
        return math.exp(
            -0.5 * ((x - mu) / sigma) ** 2
        )

    def update(self, angle):

        if angle is None:
            self.last_angle = None
            self.p[:] = [0.98, 0.01, 0.01]
            return False

        if self.last_angle is None:
            self.last_angle = angle
            return False

        dtheta = angle_diff(self.last_angle, angle)
        self.last_angle = angle

        # -----------------------------------------
        # Prediction
        # -----------------------------------------

        pred = self.p @ self.A

        # -----------------------------------------
        # Likelihood
        # -----------------------------------------

        likelihood = np.array([
            self.gaussian(dtheta, self.mu[0], self.sigma[0]),
            self.gaussian(dtheta, self.mu[1], self.sigma[1]),
            self.gaussian(dtheta, self.mu[2], self.sigma[2]),
        ])

        # -----------------------------------------
        # Bayes update
        # -----------------------------------------

        self.p = pred * likelihood
        self.p /= np.sum(self.p)

        flip_probability = self.p[self.CW] + self.p[self.CCW]

        return flip_probability > 0.8

    @property
    def flip_probability(self):
        return self.p[self.CW] + self.p[self.CCW]

    @property
    def state(self):
        return self.STATE_NAMES[np.argmax(self.p)]

    def reset(self) -> None:
        """Reset stavu filtru (nové video nebo ztráta osoby)."""
        self.last_angle = None
        self.p[:] = [0.98, 0.01, 0.01]