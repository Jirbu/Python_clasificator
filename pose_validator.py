"""
pose_validator.py
-----------------
Vícevrstvý validační systém pro detekci false positive póz.

Pro každý frame rozhodne: valid_pose = True / False

Vrstvy:
  1. Visibility score  – průměrná visibility klíčových kloubů
  2. Geometrická       – proporce těla (šíře ramen/boků, výška trupu)
  3. Strukturální      – logické pořadí těla (hlava nad rameny, ramena nad boky...)
  4. Temporální        – náhlý skok skeletu po absenci = false positive
  5. Minimální počet   – dostatečný počet viditelných landmarků

Každá vrstva vrací (passed: bool, reason: str | None).
"""

from collections import deque
import logging
import numpy as np
from pose_detector import LANDMARK_INDEX

logger = logging.getLogger(__name__)


# ── Klíčové klouby pro validaci ───────────────────────────────────────────────
_KEY_JOINTS = [
    LANDMARK_INDEX["left_shoulder"],
    LANDMARK_INDEX["right_shoulder"],
    LANDMARK_INDEX["left_hip"],
    LANDMARK_INDEX["right_hip"],
    LANDMARK_INDEX["left_knee"],
    LANDMARK_INDEX["right_knee"],
]

# Joints sledované pro temporální stabilitu
_TEMPORAL_JOINTS = [
    LANDMARK_INDEX["left_shoulder"],
    LANDMARK_INDEX["right_shoulder"],
    LANDMARK_INDEX["left_hip"],
    LANDMARK_INDEX["right_hip"],
    LANDMARK_INDEX["left_knee"],
    LANDMARK_INDEX["right_knee"],
    LANDMARK_INDEX["left_wrist"],
    LANDMARK_INDEX["right_wrist"],
]


class PoseValidator:
    """
    Validuje každý detekovaný skeleton přes 5 nezávislých vrstev.

    Parametry:
        visibility_threshold      -- min. průměrná visibility klíčových kloubů
        min_key_joint_visibility  -- min. visibility KAŽDÉHO klíčového kloubu (nejhorší kloub)
                                     Hlavní diskriminátor: skryté koleno/kyčel → unreliable pose
        min_shoulder_width        -- min. šíře ramen v norm. souřadnicích [0-1]
        min_hip_width             -- min. šíře boků v norm. souřadnicích
        min_torso_height          -- min. výška trupu v norm. souřadnicích
        structural_tolerance      -- tolerance pro strukturální kontrolu pořadí (v norm. souřadnicích)
        temporal_buffer_size      -- počet posledních validních snímků pro stabilitu
        temporal_jump_threshold   -- max. průměrný pohyb mezi snímky (norm. souřadnice)
        min_visible_landmarks     -- min. počet landmarků s visibility > 0.5
    """

    def __init__(
        self,
        visibility_threshold: float = 0.25,
        min_key_joint_visibility: float = 0.25,
        min_shoulder_width: float = 0.01,
        min_hip_width: float = 0.01,
        min_torso_height: float = 0.03,
        structural_tolerance: float = 0.35,
        temporal_buffer_size: int = 3,
        temporal_jump_threshold: float = 0.75,
        min_visible_landmarks: int = 7,
    ):
        self.visibility_threshold     = visibility_threshold
        self.min_key_joint_visibility = min_key_joint_visibility
        self.min_shoulder_width       = min_shoulder_width
        self.min_hip_width            = min_hip_width
        self.min_torso_height         = min_torso_height
        self.structural_tolerance     = structural_tolerance
        self.temporal_jump_threshold  = temporal_jump_threshold
        self.min_visible_landmarks    = min_visible_landmarks

        # Buffer posledních N validních skeletonů (pro temporální vrstvu)
        self._valid_skeleton_buffer: deque = deque(maxlen=temporal_buffer_size)
        # Počet po sobě jdoucích nevalidních snímků
        self._consecutive_invalid: int = 0
        # Statistiky odmitnutí pro každou vrstvu (pro ladění)
        self.rejection_stats: dict[str, int] = {
            "L1_visibility": 0,
            "L2_geometry":   0,
            "L3_structure":  0,
            "L4_temporal":   0,
            "L5_landmarks":  0,
        }
    # ─────────────────────────────────────────────────────────────────────────
    # VRSTVA 1: Visibility score
    # ─────────────────────────────────────────────────────────────────────────

    def _check_visibility(self, landmarks: np.ndarray) -> tuple[bool, str | None]:
        """
        Průměrná visibility klíčových kloubů musí být >= visibility_threshold.
        Navíc KAŽDÝ klíčový kloub musí mít visibility >= min_key_joint_visibility.

        Druhá podmínka je hlavní diskriminátor: pokud je třeba jedno koleno nebo
        kyčel skrytá (MediaPipe hádá polohu), cela póza je nespolehlivá.
        """
        scores = [landmarks[i, 3] for i in _KEY_JOINTS]
        pose_score = float(np.mean(scores))
        if pose_score < self.visibility_threshold:
            return False, f"L1 avg visibility={pose_score:.2f} < {self.visibility_threshold}"
        min_score = float(min(scores))
        if min_score < self.min_key_joint_visibility:
            joint_names = ["l_sh", "r_sh", "l_hip", "r_hip", "l_knee", "r_knee"]
            worst_name = joint_names[scores.index(min_score)]
            return False, f"L1 min joint {worst_name}={min_score:.2f} < {self.min_key_joint_visibility}"
        return True, None

    # ─────────────────────────────────────────────────────────────────────────
    # VRSTVA 2: Geometrická validace
    # ─────────────────────────────────────────────────────────────────────────

    def _check_geometry(self, landmarks: np.ndarray) -> tuple[bool, str | None]:
        """
        Zkontroluje základní proporce těla v normalizovaných souřadnicích [0-1].

        Logika OR: selže jen pokud je CURRENT scéna zcela degenerovaná:
          - torso_height je příliš malá (skeleton je jen bod)
          - ZÁROVEŇ ani šíře ramen ani šíře boků nepřekračují minimum
            (oba znaky degenerace musí být přítomny najednou)

        Tato volná logika propustí lidi snímaní z boku kde shoulder_width → 0.
        """
        I = LANDMARK_INDEX
        l_sh  = landmarks[I["left_shoulder"],  :2]
        r_sh  = landmarks[I["right_shoulder"], :2]
        l_hip = landmarks[I["left_hip"],       :2]
        r_hip = landmarks[I["right_hip"],      :2]

        shoulder_width = float(np.linalg.norm(l_sh - r_sh))
        hip_width      = float(np.linalg.norm(l_hip - r_hip))
        torso_height   = float(np.linalg.norm((l_sh + r_sh) / 2.0 - (l_hip + r_hip) / 2.0))

        # Torso musí mít alespoň minimální výšku (skeleton není kolapsovaný na bod)
        if torso_height < self.min_torso_height:
            return False, f"L2 torso_height={torso_height:.3f} < {self.min_torso_height}"

        # Šíře: stačí alespoň JEDNO kritérium (ramena NEBO boky mají rozestup)
        # → propustí boční záběry kde jedno z nich degeneruje
        if shoulder_width < self.min_shoulder_width and hip_width < self.min_hip_width:
            return False, (
                f"L2 oba rozestupy moc male "
                f"(sh={shoulder_width:.3f}, hip={hip_width:.3f})"
            )

        return True, None

    # ─────────────────────────────────────────────────────────────────────────
    # VRSTVA 3: Strukturální kontrola
    # ─────────────────────────────────────────────────────────────────────────

    def _check_structure(self, landmarks: np.ndarray) -> tuple[bool, str | None]:
        """
        Ověří logické pořadí těla pomocí Y-souřadnic.

        V MediaPipe y=0 je nahoře, y=1 je dole.
        Normální pozice: nos < ramena < boky < kolena

        Tolerance kompenzuje legitimní rotace (akrobacie, naklonění).
        Stojka (handstand) by měla projít díky toleranci.
        Selže jen extrémně nesmyslné konfigurace.
        """
        I   = LANDMARK_INDEX
        tol = self.structural_tolerance

        nose_y     = float(landmarks[I["nose"],           1])
        sh_y       = float((landmarks[I["left_shoulder"], 1] + landmarks[I["right_shoulder"], 1]) / 2.0)
        hip_y      = float((landmarks[I["left_hip"],      1] + landmarks[I["right_hip"],      1]) / 2.0)
        knee_y     = float((landmarks[I["left_knee"],     1] + landmarks[I["right_knee"],     1]) / 2.0)

        # Nos musí být nad rameny (nose_y < sh_y) s tolerancí
        if nose_y > sh_y + tol:
            return False, f"L3 nos pod rameny (nose_y={nose_y:.2f} sh_y={sh_y:.2f})"

        # Ramena musí být nad boky
        if sh_y > hip_y + tol:
            return False, f"L3 ramena pod boky (sh_y={sh_y:.2f} hip_y={hip_y:.2f})"

        # Boky musí být nad koleny
        if hip_y > knee_y + tol:
            return False, f"L3 boky pod koleny (hip_y={hip_y:.2f} knee_y={knee_y:.2f})"

        return True, None

    # ─────────────────────────────────────────────────────────────────────────
    # VRSTVA 4: Temporální stabilita
    # ─────────────────────────────────────────────────────────────────────────

    def _check_temporal(self, landmarks: np.ndarray) -> tuple[bool, str | None]:
        """
        Pokud skeleton náhle zmizí a pak se zjeví na zcela jiném místě,
        jde pravděpodobně o false positive.

        Podmínka selže, pokud:
          - buffer má alespoň 1 předchozí validní skeleton
          - průměrný posun klíčových kloubů > temporal_jump_threshold
          - a to po sérii nevalidních snímků (skeleton "zmizel a zase se objevil")
        """
        if len(self._valid_skeleton_buffer) == 0:
            # Nemáme historii → nelze posoudit → pass
            return True, None

        prev = self._valid_skeleton_buffer[-1]  # poslední validní skeleton
        curr_pts = landmarks[_TEMPORAL_JOINTS, :2]
        prev_pts = prev[_TEMPORAL_JOINTS, :2]

        movement = float(np.mean(np.linalg.norm(curr_pts - prev_pts, axis=1)))

        # Penalizace pouze pokud se skeleton "znovu objevuje" po absenci
        # (po >= 2 po sobě jdoucích nevalidních snímcích)
        if self._consecutive_invalid >= 2 and movement > self.temporal_jump_threshold:
            return False, f"L4 náhlý skok={movement:.3f} po {self._consecutive_invalid} inv. sn."

        return True, None

    # ─────────────────────────────────────────────────────────────────────────
    # VRSTVA 5: Minimální počet viditelných landmarků
    # ─────────────────────────────────────────────────────────────────────────

    def _check_landmark_count(self, landmarks: np.ndarray) -> tuple[bool, str | None]:
        """
        Alespoň min_visible_landmarks musí mít visibility > 0.5.
        """
        count = int(np.sum(landmarks[:, 3] > 0.5))
        if count < self.min_visible_landmarks:
            return False, f"L5 visible_landmarks={count} < {self.min_visible_landmarks}"
        return True, None

    # ─────────────────────────────────────────────────────────────────────────
    # HLAVNÍ METODA
    # ─────────────────────────────────────────────────────────────────────────

    def validate(self, landmarks: np.ndarray, relaxed: bool = False) -> tuple[bool, str | None]:
        """
        Spustí všechny validační vrstvy.

        relaxed=True  -- zmírní thresholdy pro všechny vrstvy a přeskočí L4.
                         Použij když PersonTracker s vysokou jistotou říká, že
                         osoba je v záběru (predikce z předchozího snímku).
                         V tomto режimu validátor ověří jen že pose je použitelná,
                         nikoliv že je "perfektní".

        Vrátí:
            (True,  None)          – skeleton je validní
            (False, "důvod selhání") – skeleton je nevalidní
        """
        # Multiplier-based relaxace (neměníme instance parametry)
        vis_thr   = 0.12  if relaxed else self.visibility_threshold
        key_thr   = 0.12  if relaxed else self.min_key_joint_visibility
        torso_min = 0.015 if relaxed else self.min_torso_height
        struct_tol = 0.60 if relaxed else self.structural_tolerance
        min_lm    = 4     if relaxed else self.min_visible_landmarks

        # Dočasně přemapujeme atributy (volání sub-metod je čte přímo)
        _orig = (
            self.visibility_threshold, self.min_key_joint_visibility,
            self.min_torso_height, self.structural_tolerance, self.min_visible_landmarks
        )
        self.visibility_threshold     = vis_thr
        self.min_key_joint_visibility = key_thr
        self.min_torso_height         = torso_min
        self.structural_tolerance     = struct_tol
        self.min_visible_landmarks    = min_lm

        try:
            layer_map = [
                ("L5_landmarks", self._check_landmark_count),
                ("L1_visibility", self._check_visibility),
                ("L2_geometry",   self._check_geometry),
                ("L3_structure",  self._check_structure),
            ]
            if not relaxed:
                layer_map.append(("L4_temporal", self._check_temporal))

            for stat_key, check in layer_map:
                passed, reason = check(landmarks)
                if not passed:
                    self.rejection_stats[stat_key] += 1
                    self._consecutive_invalid += 1
                    logger.debug("Zamítnuto [%s]%s: %s", stat_key,
                                 " (relaxed)" if relaxed else "", reason)
                    return False, reason

        finally:
            # Vždy obnovíme původní hodnoty
            (
                self.visibility_threshold, self.min_key_joint_visibility,
                self.min_torso_height, self.structural_tolerance, self.min_visible_landmarks
            ) = _orig

        # Všechny vrstvy prošly → validní
        self._consecutive_invalid = 0
        self._valid_skeleton_buffer.append(landmarks.copy())
        return True, None

    def log_stats(self) -> None:
        """Vypíše statistiky odmitnutí pro každou vrstvu do logu."""
        total = sum(self.rejection_stats.values())
        if total == 0:
            logger.info("  Validace: žádná odmitnutí.")
            return
        logger.info("  Validace – odmitnutí po vrstvach:")
        for layer, count in self.rejection_stats.items():
            if count > 0:
                logger.info("    %-18s %d snímků", layer, count)

    def reset(self) -> None:
        """Reset při přechodu na nové video."""
        self._valid_skeleton_buffer.clear()
        self._consecutive_invalid = 0
        for key in self.rejection_stats:
            self.rejection_stats[key] = 0
