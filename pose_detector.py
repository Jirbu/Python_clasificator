"""
pose_detector.py
----------------
Modul pro detekci pozice těla pomocí MediaPipe Pose (Tasks API, v0.10+).

Zodpovědnost:
  - Stažení modelu pose_landmarker_full.task při prvním spuštění
  - Inicializace MediaPipe PoseLandmarker (Tasks API – VIDEO mód)
  - Resize snímku na cílové rozlišení před detekcí
  - Extrakce 33 landmarků (x, y, z, visibility) z každého snímku
  - Vrácení None pokud pose nebyla detekována
"""

import os
import logging
import urllib.request
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np

logger = logging.getLogger(__name__)

# Cílové rozlišení snímku před pose detekcí (standardní)
TARGET_WIDTH = 256
TARGET_HEIGHT = 144

# Vyšší rozlišení pro hires fallback detekci (2× standardní)
HIRES_WIDTH = 512
HIRES_HEIGHT = 288

# Počet landmarků MediaPipe Pose
NUM_LANDMARKS = 33

# URL a lokální cesta modelu (Tasks API vyžaduje .task soubor)
_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task"
)
_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "models", "pose_landmarker_full.task"
)


def _ensure_model(model_path: str = _DEFAULT_MODEL_PATH) -> str:
    """
    Pokud model ještě neexistuje, stáhne ho z Google Storage.
    Vrátí ověřenou cestu k modelu.
    """
    if not os.path.exists(model_path):
        os.makedirs(os.path.dirname(os.path.abspath(model_path)), exist_ok=True)
        logger.info("Stahuji MediaPipe model: %s", _MODEL_URL)
        urllib.request.urlretrieve(_MODEL_URL, model_path)
        logger.info("Model uložen: %s", model_path)
    return model_path


class PoseDetector:
    """
    Obaluje MediaPipe PoseLandmarker (Tasks API) a poskytuje
    jednoduché rozhraní pro detekci pózy ve video módu.

    Parametry:
        min_detection_confidence -- minimální jistota detekce (0.0 - 1.0)
        min_tracking_confidence  -- minimální jistota trackingu (0.0 - 1.0)
        model_path               -- cesta k .task souboru modelu
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        model_path: str = _DEFAULT_MODEL_PATH,
    ):
        model_path = _ensure_model(model_path)

        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,   # tracking přes snímky
            num_poses=1,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Zmenší snímek na TARGET_WIDTH x TARGET_HEIGHT a převede BGR -> RGB.
        MediaPipe Tasks API očekává RGB vstup.
        """
        # KROK 1: Downscale – vstupní BGR snímek (libovolné rozlišení z videa)
        # se zmenší na pevné 256×144 px. To snižuje výpočetní náklady MediaPipe.
        resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

        # KROK 2: BGR → RGB konverze. Snímek zůstává barevný (3 kanály).
        # Žádný grayscale se zde neaplikuje – MediaPipe dostává plný barevný obraz.
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        return rgb

    def detect_pose(self, frame: np.ndarray, timestamp_ms: float):
        """
        Spustí pose detection na snímku (BGR formát, libovolné rozlišení).
        timestamp_ms musí být monotónně rostoucí (odpovídá pozici ve videu).
        Vrátí PoseLandmarkerResult nebo None.
        """
        preprocessed = self.preprocess_frame(frame)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=preprocessed)
        return self._landmarker.detect_for_video(mp_image, int(timestamp_ms))

    def extract_landmarks(self, pose_results) -> np.ndarray | None:
        """
        Z PoseLandmarkerResult extrahuje numpy pole tvaru (33, 4):
          sloupce: [x, y, z, visibility]

        Souřadnice x, y jsou normalizovány do [0, 1] relativně k rozlišení.
        Pokud pose nebyla detekována, vrátí None.
        """
        if pose_results is None or not pose_results.pose_landmarks:
            return None

        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        for i, lm in enumerate(pose_results.pose_landmarks[0]):
            # visibility je Optional[float] v Tasks API
            visibility = lm.visibility if lm.visibility is not None else 1.0
            landmarks[i] = [lm.x, lm.y, lm.z, visibility]

        return landmarks

    def process_frame(self, frame: np.ndarray, timestamp_ms: float) -> np.ndarray | None:
        """
        Zkrácené rozhraní: detekce + extrakce v jednom volání.
        Vrátí landmarks (33, 4) nebo None.
        """
        results = self.detect_pose(frame, timestamp_ms)
        return self.extract_landmarks(results)

    def close(self):
        """Uvolní MediaPipe zdroje."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ─────────────────────────────────────────────────────────────────────────────
# Stateless IMAGE mode detektor – pro crop a full-frame scan
# ─────────────────────────────────────────────────────────────────────────────

class PoseDetectorImage:
    """
    Stateless pose detector v IMAGE módu (nevyžaduje monotónní timestamps).

    Použití:
      - Detekce v crop oblasti (LOST recovery pro Person 1)
      - Full-frame scan pro hledání Person 2
      - Detection pro Person 2 v jejím tracking crop

    Parametry:
        num_poses      -- max počet osob k detekci (typicky 2)
        min_confidence -- minimální confidence pro detekci pózy
        model_path     -- cesta k .task souboru modelu
    """

    def __init__(
        self,
        num_poses: int       = 2,
        min_confidence: float = 0.6,
        model_path: str       = _DEFAULT_MODEL_PATH,
    ):
        model_path = _ensure_model(model_path)
        base_options = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=num_poses,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    def detect_all(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Detekuje všechny pózy v snímku.

        Vrátí list (33, 4) numpy polí [x, y, z, visibility] pro každou
        detekovanou osobu. Souřadnice jsou normalizovány do [0, 1]
        relativně k rozměrům vstupního snímku.

        Prázdný list pokud nebyla detekována žádná póza.
        """
        resized = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = self._landmarker.detect(mp_img)

        out: list[np.ndarray] = []
        if not result.pose_landmarks:
            return out
        for pose in result.pose_landmarks:
            lm = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            for i, pt in enumerate(pose):
                lm[i] = [
                    pt.x, pt.y, pt.z,
                    pt.visibility if pt.visibility is not None else 1.0,
                ]
            out.append(lm)
        return out

    def detect_all_hires(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Stejné jako detect_all, ale před detekcí zvětší vstup na HIRES_WIDTH×HIRES_HEIGHT
        (512×288) místo standardních 256×144.
        Použití: hires fallback pokud standardní detekce vrátila suspicious result.
        """
        resized = cv2.resize(frame, (HIRES_WIDTH, HIRES_HEIGHT))
        rgb     = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        mp_img  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result  = self._landmarker.detect(mp_img)

        out: list[np.ndarray] = []
        if not result.pose_landmarks:
            return out
        for pose in result.pose_landmarks:
            lm = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
            for i, pt in enumerate(pose):
                lm[i] = [
                    pt.x, pt.y, pt.z,
                    pt.visibility if pt.visibility is not None else 1.0,
                ]
            out.append(lm)
        return out

    def close(self):
        """Uvolní MediaPipe zdroje."""
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


# ── MediaPipe indexy pro klíčové klouby ──────────────────────────────────────
# Tyto konstanty jsou sdíleny s feature_extractor.py
LANDMARK_INDEX = {
    "nose":          0,
    "left_shoulder":  11,
    "right_shoulder": 12,
    "left_elbow":     13,
    "right_elbow":    14,
    "left_wrist":     15,
    "right_wrist":    16,
    "left_hip":       23,
    "right_hip":      24,
    "left_knee":      25,
    "right_knee":     26,
    "left_ankle":     27,
    "right_ankle":    28,
    "left_heel":      29,
    "right_heel":     30,
}
