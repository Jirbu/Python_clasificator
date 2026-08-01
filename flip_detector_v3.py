import math


def angle_diff(a, b):
    """Rozdíl úhlů <-180,180>"""
    return ((b - a + 180.0) % 360.0) - 180.0


class RotationEvidence:

    def __init__(
        self,
        score_decay=0.97,
        dir_alpha=0.05,
        velocity_scale=20.0,
        score_threshold=6.0,
        rotation_threshold=90.0,
    ):

        self.score_decay = score_decay
        self.dir_alpha = dir_alpha
        self.velocity_scale = velocity_scale

        self.score_threshold = score_threshold
        self.rotation_threshold = rotation_threshold

        self.last_angle = None

        #
        # spojitý odhad směru (-1 ... +1)
        #
        self.expected_dir = 0.0

        #
        # integrovaná evidence
        #
        self.score = 0.0

        #
        # důvěryhodná otočka
        #
        self.rotation = 0.0

    def update(self, angle):

        if angle is None:
            return False

        if self.last_angle is None:
            self.last_angle = angle
            return False

        delta = angle_diff(self.last_angle, angle)
        self.last_angle = angle

        #
        # ------------------------------------
        # 1) Síla rotace
        # ------------------------------------
        #

        strength = math.tanh(abs(delta) / self.velocity_scale)

        #
        # ------------------------------------
        # 2) Směr rotace
        # ------------------------------------
        #

        direction = math.tanh(delta / self.velocity_scale)

        #
        # ------------------------------------
        # 3) Jak moc souhlasí se směrem,
        #    který se buduje z historie
        # ------------------------------------
        #

        agreement = 0.5 * (
            1.0 +
            direction * self.expected_dir
        )

        #
        # ------------------------------------
        # 4) Evidence tohoto framu
        # ------------------------------------
        #

        evidence = strength * agreement

        #
        # ------------------------------------
        # 5) Integrátor evidence
        # ------------------------------------
        #

        self.score *= self.score_decay
        self.score += evidence

        #
        # ------------------------------------
        # 6) Aktualizace očekávaného směru
        # ------------------------------------
        #

        self.expected_dir = (
            (1.0 - self.dir_alpha)
            * self.expected_dir
            +
            self.dir_alpha
            * direction
        )

        #
        # ------------------------------------
        # 7) Integrace otočky (s útlumem)
        # ------------------------------------
        #

        confidence = 1.0 / (
            1.0 +
            math.exp(-(self.score - self.score_threshold))
        )

        self.rotation *= self.score_decay   # stejný decay jako score
        self.rotation += confidence * delta

        #
        # ------------------------------------
        # 8) Výsledek
        # ------------------------------------
        #

        return (
            confidence > 0.8
            and
            abs(self.rotation) > self.rotation_threshold
        )

    def reset(self):
        self.last_angle = None
        self.expected_dir = 0.0
        self.score = 0.0
        self.rotation = 0.0


# Alias pro kompatibilitu s pipeline
FlipDetector = RotationEvidence