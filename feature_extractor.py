"""
feature_extractor.py
--------------------
Modul pro extrakci příznaků z normalizovaných landmarků kostry.

Pipeline:
  1. Normalizace landmarků (translace na hip_center, scale dle torso délky)
  2. Výpočet úhlů kloubů (kolena, lokty, kyčle, ramena)
  3. Výpočet vzdáleností (ruce, nohy, šíře ramen)
  4. Orientace těla (naklon trupu)
  5. Pohybové příznaky (rychlost a zrychlení kloubů)

Výstup: 1D numpy vektor ~50 příznaků na snímek.
"""

import numpy as np
from pose_detector import LANDMARK_INDEX


class FeatureExtractor:
    """
    Transformuje raw MediaPipe landmarks na numerický feature vektor.

    Parametry:
        visibility_threshold -- landmark s nižší visibility se považuje za skrytý
    """

    def __init__(self, visibility_threshold: float = 0.5):
        self.visibility_threshold = visibility_threshold
        # Uložíme předchozí snímek pro výpočet velocity
        self._prev_landmarks: np.ndarray | None = None
        self._prev_velocity: np.ndarray | None = None

    # ─────────────────────────────────────────────────────────────────────────
    # 1. NORMALIZACE LANDMARKŮ
    # ─────────────────────────────────────────────────────────────────────────

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Normalizuje souřadnice kostry tak, aby byla invariantní vůči
        posunu a měřítku.

        Kroky:
          a) hip_center = průměr levého a pravého kyčle
          b) odečtení hip_center od všech bodů
          c) torso_length = průměrná vzdálenost ramen a kyčlí od hip_center
          d) dělení torso_length (scale normalizace)

        Vrátí pole (33, 3) – pouze x, y, z (bez visibility).
        """
        pts = landmarks[:, :3].copy()  # (33, 3) x, y, z

        # a) hip_center
        l_hip = pts[LANDMARK_INDEX["left_hip"]]
        r_hip = pts[LANDMARK_INDEX["right_hip"]]
        hip_center = (l_hip + r_hip) / 2.0

        # b) translace
        pts -= hip_center

        # c) torso_length
        l_shoulder = pts[LANDMARK_INDEX["left_shoulder"]]
        r_shoulder = pts[LANDMARK_INDEX["right_shoulder"]]
        shoulder_center = (l_shoulder + r_shoulder) / 2.0
        torso_length = np.linalg.norm(shoulder_center)  # vzdálenost od origin

        # d) scale normalizace – ochrana před dělením nulou
        if torso_length > 1e-6:
            pts /= torso_length

        return pts  # (33, 3)

    # ─────────────────────────────────────────────────────────────────────────
    # 2. ÚHLY KLOUBŮ
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _angle_between(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
        """
        Vypočítá úhel (stupně) v bodě B vzniklý trojicí bodů A-B-C.
        """
        ba = a - b
        bc = c - b
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)
        if norm_ba < 1e-6 or norm_bc < 1e-6:
            return 0.0
        cos_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # numerická stabilita
        return float(np.degrees(np.arccos(cos_angle)))

    def extract_joint_angles(self, pts: np.ndarray) -> np.ndarray:
        """
        Vypočítá 8 úhlů kloubů (levá+pravá strana × 4 klouby):
          - loket  (shoulder - elbow - wrist)
          - koleno (hip - knee - ankle)
          - kyčel  (shoulder - hip - knee)
          - rameno (elbow - shoulder - hip)

        Vrátí vektor délky 8.
        """
        I = LANDMARK_INDEX
        angles = []

        for side in ("left", "right"):
            # loket
            angles.append(self._angle_between(
                pts[I[f"{side}_shoulder"]],
                pts[I[f"{side}_elbow"]],
                pts[I[f"{side}_wrist"]],
            ))
            # koleno
            angles.append(self._angle_between(
                pts[I[f"{side}_hip"]],
                pts[I[f"{side}_knee"]],
                pts[I[f"{side}_ankle"]],
            ))
            # kyčel
            angles.append(self._angle_between(
                pts[I[f"{side}_shoulder"]],
                pts[I[f"{side}_hip"]],
                pts[I[f"{side}_knee"]],
            ))
            # rameno
            angles.append(self._angle_between(
                pts[I[f"{side}_elbow"]],
                pts[I[f"{side}_shoulder"]],
                pts[I[f"{side}_hip"]],
            ))

        return np.array(angles, dtype=np.float32)  # (8,)

    # ─────────────────────────────────────────────────────────────────────────
    # 3. VZDÁLENOSTI
    # ─────────────────────────────────────────────────────────────────────────

    def extract_distances(self, pts: np.ndarray) -> np.ndarray:
        """
        Vypočítá 5 klíčových vzdáleností:
          - vzdálenost zápěstí (hand distance)
          - vzdálenost kotníků (foot distance)
          - šíře ramen
          - vzdálenost levé ruky od kyčle
          - vzdálenost pravé ruky od kyčle

        Vrátí vektor délky 5.
        """
        I = LANDMARK_INDEX
        dists = [
            np.linalg.norm(pts[I["left_wrist"]]  - pts[I["right_wrist"]]),   # ruce
            np.linalg.norm(pts[I["left_ankle"]]  - pts[I["right_ankle"]]),   # nohy
            np.linalg.norm(pts[I["left_shoulder"]] - pts[I["right_shoulder"]]),  # ramena
            np.linalg.norm(pts[I["left_wrist"]]  - pts[I["left_hip"]]),      # L ruka-kyčel
            np.linalg.norm(pts[I["right_wrist"]] - pts[I["right_hip"]]),     # P ruka-kyčel
        ]
        return np.array(dists, dtype=np.float32)  # (5,)

    # ─────────────────────────────────────────────────────────────────────────
    # 4. ORIENTACE TĚLA
    # ─────────────────────────────────────────────────────────────────────────

    def extract_body_orientation(self, pts: np.ndarray) -> np.ndarray:
        """
        Vypočítá orientaci trupu:
          - torso_tilt_x: naklon vlevo/vpravo (osa X, stupně)
          - torso_tilt_z: naklon dopředu/dozadu (osa Z, stupně)
          - shoulder_angle: rotace ramen vůči horizontále
          - hip_angle: rotace kyčlí vůči horizontále

        Vrátí vektor délky 4.
        """
        I = LANDMARK_INDEX
        l_sh = pts[I["left_shoulder"]]
        r_sh = pts[I["right_shoulder"]]
        l_hip = pts[I["left_hip"]]
        r_hip = pts[I["right_hip"]]

        sh_center = (l_sh + r_sh) / 2.0  # střed ramen (origin je hip_center = 0)

        # Naklon trupu kolem osy X (vertikála v rovině Y-Z)
        torso_tilt_x = float(np.degrees(np.arctan2(sh_center[0], sh_center[1])))
        # Naklon trupu kolem osy Z (dopředu/dozadu)
        torso_tilt_z = float(np.degrees(np.arctan2(sh_center[2], sh_center[1])))

        # Úhel ramen vůči horizontální ose
        sh_vec = r_sh - l_sh
        shoulder_angle = float(np.degrees(np.arctan2(sh_vec[1], sh_vec[0])))

        # Úhel kyčlí vůči horizontální ose
        hip_vec = r_hip - l_hip
        hip_angle = float(np.degrees(np.arctan2(hip_vec[1], hip_vec[0])))

        return np.array(
            [torso_tilt_x, torso_tilt_z, shoulder_angle, hip_angle],
            dtype=np.float32,
        )  # (4,)

    # ─────────────────────────────────────────────────────────────────────────
    # 5. VÝŠKA KLÍČOVÝCH BODŮ (absolutní Y-souřadnice)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_keypoint_heights(self, pts: np.ndarray) -> np.ndarray:
        """
        Y-souřadnice (výška) klíčových kloubů.
        Po normalizaci jsou tyto hodnoty relativní k torso délce.
        Informují o tom, zda je člověk ve vzduchu, dřepí apod.

        Vrátí vektor délky 10.
        """
        I = LANDMARK_INDEX
        keys = [
            "nose",
            "left_shoulder", "right_shoulder",
            "left_wrist",    "right_wrist",
            "left_hip",      "right_hip",
            "left_knee",     "right_knee",
            "left_ankle",    # pozn.: right_ankle se přidá níže
        ]
        heights = [pts[I[k]][1] for k in keys[:-1]]
        heights.append(pts[LANDMARK_INDEX["right_ankle"]][1])
        return np.array(heights, dtype=np.float32)  # (10,)

    # ─────────────────────────────────────────────────────────────────────────
    # 6. POHYBOVÉ PŘÍZNAKY (velocity, acceleration)
    # ─────────────────────────────────────────────────────────────────────────

    def extract_motion_features(
        self,
        current_pts: np.ndarray,
        prev_pts: np.ndarray | None,
    ) -> np.ndarray:
        """
        Vypočítá průměrnou rychlost a zrychlení kloubů.

        Pro každý ze 13 klíčových kloubů:
          - velocity_magnitude   = ||pos(t) - pos(t-1)||
          - acceleration_magnitude = ||vel(t) - vel(t-1)||

        Vrátí vektor délky 26 (13 velocity + 13 acceleration).
        Pokud nejsou předchozí data, vrátí nulový vektor.
        """
        key_joints = [
            "nose",
            "left_shoulder",  "right_shoulder",
            "left_elbow",     "right_elbow",
            "left_wrist",     "right_wrist",
            "left_hip",       "right_hip",
            "left_knee",      "right_knee",
            "left_ankle",     "right_ankle",
        ]
        indices = [LANDMARK_INDEX[j] for j in key_joints]

        zero_vel = np.zeros(len(indices), dtype=np.float32)
        zero_acc = np.zeros(len(indices), dtype=np.float32)

        if prev_pts is None:
            current_vel = zero_vel.copy()
            current_acc = zero_acc.copy()
        else:
            diff = current_pts[indices] - prev_pts[indices]
            current_vel = np.linalg.norm(diff, axis=1).astype(np.float32)

            if self._prev_velocity is None:
                current_acc = zero_acc.copy()
            else:
                current_acc = np.abs(
                    current_vel - self._prev_velocity
                ).astype(np.float32)

        self._prev_velocity = current_vel
        return np.concatenate([current_vel, current_acc])  # (26,)

    # ─────────────────────────────────────────────────────────────────────────
    # HLAVNÍ METODA
    # ─────────────────────────────────────────────────────────────────────────

    def extract_features(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Kompletní extrakce příznaků z jednoho snímku.

        Vstupy:
          landmarks -- raw MediaPipe landmarks (33, 4): x, y, z, visibility

        Výstup:
          feature vektor délky ~53:
            8  úhlů kloubů
            5  vzdáleností
            4  orientace těla
            10 výšek klíčových bodů
            26 pohybových příznaků
          Celkem: 53 příznaků
        """
        # Normalizace
        pts = self.normalize_landmarks(landmarks)

        # Extrakce skupin příznaků
        angles    = self.extract_joint_angles(pts)       # (8,)
        distances = self.extract_distances(pts)          # (5,)
        orient    = self.extract_body_orientation(pts)   # (4,)
        heights   = self.extract_keypoint_heights(pts)   # (10,)
        motion    = self.extract_motion_features(pts, self._prev_landmarks)  # (26,)

        # Uložení aktuálního snímku jako předchozí pro příští iteraci
        self._prev_landmarks = pts

        return np.concatenate([angles, distances, orient, heights, motion])  # (53,)

    def reset(self):
        """Resetuje historii (použít při přechodu na nové video)."""
        self._prev_landmarks = None
        self._prev_velocity = None
