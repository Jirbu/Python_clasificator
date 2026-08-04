"""
pose_backends.py
----------------
Přepínatelné pose-estimation backendy pro srovnávací test (MediaPipe vs.
MoveNet vs. YOLOv8-pose). Python nemá podmíněný překlad jako C (#ifdef) –
ekvivalent je běhová volba přes konstantu POSE_MODEL + factory funkce
create_pose_detectors(), která vrátí (video_detector, image_detector) se
stejným rozhraním jako pose_detector.PoseDetector / PoseDetectorImage
(process_frame / detect_all / detect_all_hires). Zbytek pipeline
(person_manager.py, main.py) se tedy vůbec nemusí starat o to, který model
zrovna běží.

Poznámka ke kloubům: MoveNet i YOLOv8-pose vrací standardních 17 COCO bodů
(bez pat), zatímco MediaPipe má 33 landmarků (BlazePose topologie). Zbytek
pipeline (torso_angle.py, person_manager.py, feature_extractor.py, ...)
používá výhradně nos/ramena/lokty/zápěstí/kyčle/kolena/kotníky přes
LANDMARK_INDEX – všechny se z COCO 17 bodů namapují 1:1. Paty (29/30) a
oči/uši (COCO 1-4) se nikde dál nepoužívají, zůstanou v poli s visibility=0.
"""

from __future__ import annotations

import os
import logging

import cv2
import numpy as np

from pose_detector import (
    PoseDetector, PoseDetectorImage, NUM_LANDMARKS,
    _letterbox_resize, _unletterbox_landmarks,
)

logger = logging.getLogger(__name__)

# ── Přepínač modelu ───────────────────────────────────────────────────────────
POSE_MODEL = "yolov8"   # "mediapipe" | "movenet" | "yolov8"

_MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

# COCO-17 index -> mediapipe (BlazePose) index
_COCO_TO_MP = {
    0:  0,   # nose
    5:  11,  # left_shoulder
    6:  12,  # right_shoulder
    7:  13,  # left_elbow
    8:  14,  # right_elbow
    9:  15,  # left_wrist
    10: 16,  # right_wrist
    11: 23,  # left_hip
    12: 24,  # right_hip
    13: 25,  # left_knee
    14: 26,  # right_knee
    15: 27,  # left_ankle
    16: 28,  # right_ankle
}


def _coco17_to_mp33(xy: np.ndarray, score: np.ndarray) -> np.ndarray:
    """
    xy    -- (17, 2) normalizované [0,1] souřadnice (x, y)
    score -- (17,) confidence/visibility [0,1]
    Vrátí (33, 4) pole [x, y, z=0, visibility] v mediapipe konvenci.
    """
    lm = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
    for coco_i, mp_i in _COCO_TO_MP.items():
        lm[mp_i, 0] = xy[coco_i, 0]
        lm[mp_i, 1] = xy[coco_i, 1]
        lm[mp_i, 2] = 0.0
        lm[mp_i, 3] = score[coco_i]
    return lm


# ─────────────────────────────────────────────────────────────────────────────
# MoveNet (TFLite, přes ai_edge_litert – lehký runtime bez celého tensorflow)
# ─────────────────────────────────────────────────────────────────────────────

_MOVENET_URLS = {
    "lightning": "https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite",
    "thunder":   "https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite",
}
_MOVENET_INPUT_SIZE = {"lightning": 192, "thunder": 256}


def _ensure_movenet_model(variant: str) -> str:
    path = os.path.join(_MODELS_DIR, f"movenet_{variant}.tflite")
    if not os.path.exists(path):
        os.makedirs(_MODELS_DIR, exist_ok=True)
        logger.info("Stahuji MoveNet (%s): %s", variant, _MOVENET_URLS[variant])
        import urllib.request
        urllib.request.urlretrieve(_MOVENET_URLS[variant], path)
    return path


class MoveNetBackend:
    """
    Implementuje stejné rozhraní jako PoseDetector + PoseDetectorImage
    (process_frame / detect_all / detect_all_hires), jedním modelem –
    MoveNet nemá samostatný "video" režim, jde vždy o nezávislou detekci
    na jednom snímku.
    """

    def __init__(self, variant: str = "thunder"):
        from ai_edge_litert.interpreter import Interpreter
        model_path = _ensure_movenet_model(variant)
        self._interp = Interpreter(model_path=model_path)
        self._interp.allocate_tensors()
        self._input_idx  = self._interp.get_input_details()[0]["index"]
        self._output_idx = self._interp.get_output_details()[0]["index"]
        self._size = _MOVENET_INPUT_SIZE[variant]

    def _run(self, frame: np.ndarray) -> np.ndarray | None:
        # Letterbox (zachová poměr stran, viz pose_detector._letterbox_resize),
        # BGR->RGB, uint8 vstup (model očekává quantized uint8, ne float).
        canvas, new_w, new_h, pad_x, pad_y = _letterbox_resize(frame, self._size, self._size)
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.uint8)
        inp = np.expand_dims(rgb, axis=0)

        self._interp.set_tensor(self._input_idx, inp)
        self._interp.invoke()
        out = self._interp.get_tensor(self._output_idx)[0, 0]  # (17, 3): [y, x, score]

        xy    = out[:, [1, 0]]   # MoveNet vrací (y, x) -> přehodíme na (x, y)
        score = out[:, 2]

        lm = _coco17_to_mp33(xy, score)
        lm = _unletterbox_landmarks(lm, self._size, self._size, new_w, new_h, pad_x, pad_y)
        return lm

    # -- rozhraní shodné s PoseDetector --
    def process_frame(self, frame: np.ndarray, timestamp_ms: float) -> np.ndarray | None:
        return self._run(frame)

    # -- rozhraní shodné s PoseDetectorImage --
    def detect_all(self, frame: np.ndarray) -> list[np.ndarray]:
        lm = self._run(frame)
        return [lm] if lm is not None else []

    def detect_all_hires(self, frame: np.ndarray) -> list[np.ndarray]:
        return self.detect_all(frame)   # MoveNet nemá samostatný hires vstup

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# YOLOv8-Pose (ultralytics)
# ─────────────────────────────────────────────────────────────────────────────

_YOLO_WEAK_BOX_CONF_THR = 0.7   # pod tímto prahem box confidence (jistota "je tu osoba")
                                 # nejlepší detekce se navíc zkusí i otočený snímek o 180°
                                 # (validováno na referenčních snímcích – viz konverzace/testy)


class YoloPoseBackend:
    """Stejné rozhraní jako MoveNetBackend, přes ultralytics YOLOv8-pose."""

    def __init__(self, weights: str = "yolov8n-pose.pt"):
        from ultralytics import YOLO
        self._model = YOLO(weights)

    def _run_all_raw(self, frame: np.ndarray) -> tuple[list[np.ndarray], float | None]:
        """Detekce bez rotace. Vrátí (landmarks všech osob, box confidence první/top osoby)."""
        results = self._model(frame, verbose=False)
        if not results or results[0].keypoints is None or results[0].boxes is None:
            return [], None
        kp = results[0].keypoints
        boxes = results[0].boxes
        out: list[np.ndarray] = []
        for i in range(len(kp.xyn)):
            xy = kp.xyn[i].cpu().numpy()
            score = kp.conf[i].cpu().numpy() if kp.conf is not None else np.ones(17, dtype=np.float32)
            out.append(_coco17_to_mp33(xy, score))
        top_box_conf = float(boxes.conf[0].cpu().numpy()) if len(boxes.conf) > 0 else None
        return out, top_box_conf

    def _run_all(self, frame: np.ndarray) -> list[np.ndarray]:
        """
        Vrátí landmarks VŠECH detekovaných osob (ne jen první) – výběr té
        správné (nejblíž predikci, bez překryvu s jinou osobou, ...) dělá
        stávající logika v person_manager.py stejně jako pro mediapipe scan.

        Pokud je box confidence nejlepší detekce slabá/žádná (< _YOLO_WEAK_BOX_CONF_THR),
        zkusí se navíc detekce na snímku otočeném o 180° (postava vzhůru nohama
        může být pro model čitelnější "vzpřímeně") a použije se ta lepší.
        """
        detected, box_conf = self._run_all_raw(frame)
        is_weak = box_conf is None or box_conf < _YOLO_WEAK_BOX_CONF_THR
        if not is_weak:
            return detected

        rot_frame = cv2.rotate(frame, cv2.ROTATE_180)
        rot_detected, rot_box_conf = self._run_all_raw(rot_frame)
        if rot_box_conf is None or (box_conf is not None and rot_box_conf <= box_conf):
            return detected

        # Otočení pomohlo – souřadnice z otočeného snímku převedeme zpět
        # (x,y -> 1-x, 1-y), ať odpovídají orientaci vstupního `frame`.
        fixed: list[np.ndarray] = []
        for lm in rot_detected:
            lm2 = lm.copy()
            lm2[:, 0] = 1.0 - lm2[:, 0]
            lm2[:, 1] = 1.0 - lm2[:, 1]
            fixed.append(lm2)
        return fixed

    def process_frame(self, frame: np.ndarray, timestamp_ms: float) -> np.ndarray | None:
        detected = self._run_all(frame)
        return detected[0] if detected else None

    def detect_all(self, frame: np.ndarray) -> list[np.ndarray]:
        return self._run_all(frame)

    def detect_all_hires(self, frame: np.ndarray) -> list[np.ndarray]:
        return self.detect_all(frame)

    def close(self):
        pass


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def create_pose_detectors(
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.45,
    min_confidence: float = 0.5,
):
    """
    Vrátí (video_detector, image_detector) podle konstanty POSE_MODEL.
    Oba mají stejné rozhraní bez ohledu na zvolený model – zbytek pipeline
    (main.py, person_manager.py) se nemusí měnit.
    """
    if POSE_MODEL == "mediapipe":
        video_detector = PoseDetector(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        image_detector = PoseDetectorImage(min_confidence=min_confidence)
        return video_detector, image_detector

    if POSE_MODEL == "movenet":
        # video_detector/image_detector jsou stateless wrappery nad stejnou
        # síť – sdílená instance šetří zbytečné zdvojené načtení modelu
        # (identický výpočet, žádný vliv na výsledky).
        backend = MoveNetBackend(variant="thunder")
        return backend, backend

    if POSE_MODEL == "yolov8":
        backend = YoloPoseBackend()
        return backend, backend

    raise ValueError(f"Neznámý POSE_MODEL: {POSE_MODEL!r}")
