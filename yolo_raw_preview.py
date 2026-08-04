"""
yolo_raw_preview.py
--------------------
Diagnostický skript: pustí čistě YOLOv8-pose na video (bez person_manager
pipeline, bez klasifikace, bez validačních prahů) a vykreslí syrový výstup
modelu (bbox + skeleton) do debug videa.

Vylepšení (otočení o 180°):
  - Rotace se zkouší JEN když je normální (vzpřímená) detekce slabá/žádná
    (confidence < WEAK_CONF_THR) – ne na každém snímku, ať to zbytečně
    nezdvojnásobí výpočet všude tam, kde je detekce v pořádku.
  - Pokud otočení POMŮŽE aktuálnímu (slabému) snímku, zkusí se retroaktivně
    otočení i na PŘEDCHOZÍ snímek – i když jeho normální detekce sama o sobě
    slabá nebyla (takže se pro něj rotace jinak vůbec nezkoušela). Pokud
    zlepší jeho dosud vybranou confidence, přepíše se i jeho výstup.
    (Heuristika: když je jeden snímek uprostřed salta lépe vidět otočený,
    je pravděpodobné, že sousední snímek téže rotace na tom bude podobně.)
  - Kvůli tomuhle zpětnému přepisu se výstup do videa zapisuje se zpožděním
    jednoho snímku (drží se v bufferu, dokud není jasné, jestli se nemá
    opravit).

Použití:
    python yolo_raw_preview.py --video videos/IMG_6497.MOV --output output_debug/IMG_6497_yolo_raw.mp4
    python yolo_raw_preview.py --video videos_clip_test/IMG_6497_clip.mp4 --output output_debug/clip_yolo_raw.mp4
"""

from __future__ import annotations

import argparse
import os

import cv2
import numpy as np
from ultralytics import YOLO

from video_loader import VideoLoader

_KEY_COCO_IDX = [5, 6, 11, 12]  # ramena + kyčle (COCO indexování)


def _box_confidence(results) -> float | None:
    """
    Confidence KONKRÉTNÍ OSOBY – detekční (box) confidence první detekované
    osoby, stejné číslo, jaké ultralytics vykresluje u bounding boxu
    (label "person 0.66"). Tohle je jiný signál než confidence jednotlivých
    kloubů (_key_confidence) – box confidence říká "je tu vůbec člověk",
    keypoint confidence říká "sedí přesně poloha kloubů".
    """
    boxes = results[0].boxes
    if boxes is None or len(boxes.conf) == 0:
        return None
    return float(boxes.conf[0].cpu().numpy())


def _key_confidence(results) -> float | None:
    """Průměrná confidence klíčových kloubů (ramena+kyčle) první detekované osoby."""
    kp = results[0].keypoints
    if kp is None or len(kp.xyn) == 0 or kp.conf is None:
        return None
    conf = kp.conf[0].cpu().numpy()
    return float(np.mean(conf[_KEY_COCO_IDX]))


def _detect(model, frame: np.ndarray):
    """Vrátí (results, box_confidence, key_joint_confidence, anotovaný_obraz)."""
    results = model(frame, verbose=False)
    box_conf = _box_confidence(results)
    key_conf = _key_confidence(results)
    annotated = results[0].plot()
    return results, box_conf, key_conf, annotated


def _detect_rotated(model, frame: np.ndarray):
    """Detekce na snímku otočeném o 180° – anotovaný obraz se otočí zpět,
    aby seděl na normální orientaci videa."""
    rot = cv2.rotate(frame, cv2.ROTATE_180)
    results, box_conf, key_conf, annotated_rot = _detect(model, rot)
    annotated = cv2.rotate(annotated_rot, cv2.ROTATE_180)
    return results, box_conf, key_conf, annotated


def _better(a: float | None, b: float | None) -> bool:
    """True pokud je a lepší než b (None = nejhorší možná hodnota)."""
    if a is None:
        return False
    if b is None:
        return True
    return a > b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Cesta ke vstupnímu videu")
    ap.add_argument("--output", required=True, help="Cesta pro výstupní debug video (.mp4)")
    ap.add_argument("--fps", type=int, default=8, help="Vzorkovací FPS (default 8, stejné jako hlavní pipeline)")
    ap.add_argument("--weights", default="yolov8n-pose.pt", help="YOLO pose váhy (default nano)")
    ap.add_argument("--weak-conf-thr", type=float, default=0.7, help="Pod tímto prahem (nebo bez detekce) se zkouší i otočení o 180°")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    model = YOLO(args.weights)
    loader = VideoLoader(args.video, target_fps=args.fps)

    writer = None
    n = 0
    n_rot_helped = 0
    n_retro_fixed = 0
    prev = None  # {"frame", "box_conf", "key_conf", "annotated", "ts"} – čeká na případnou zpětnou opravu

    def _label_and_write(state):
        cv2.putText(
            state["annotated"],
            f"t={state['ts']:.0f}ms  box_conf={state['box_conf']}  key_conf={state['key_conf']}",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA,
        )
        writer.write(state["annotated"])

    for timestamp_ms, frame, _prev_frame in loader.frame_generator():
        _, box_conf_normal, key_conf_normal, annotated_normal = _detect(model, frame)

        # Rotaci zkoušíme jen když je box confidence (jistota "je tu člověk")
        # slabá/žádná – ušetří to výpočet na většině snímků, kde je detekce v pořádku.
        is_weak = box_conf_normal is None or box_conf_normal < args.weak_conf_thr
        box_conf_rot, key_conf_rot, annotated_rot = None, None, None
        if is_weak:
            _, box_conf_rot, key_conf_rot, annotated_rot = _detect_rotated(model, frame)

        rotation_helped = is_weak and _better(box_conf_rot, box_conf_normal)
        if rotation_helped:
            n_rot_helped += 1
            chosen_box, chosen_key, chosen_annotated = box_conf_rot, key_conf_rot, annotated_rot
        else:
            chosen_box, chosen_key, chosen_annotated = box_conf_normal, key_conf_normal, annotated_normal

        cur = {
            "frame": frame, "ts": timestamp_ms,
            "box_conf": chosen_box, "key_conf": chosen_key, "annotated": chosen_annotated,
        }

        if writer is None:
            h, w = chosen_annotated.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(args.output, fourcc, args.fps, (w, h))

        # Rotace pomohla TOMUTO (slabému) snímku -> zkus zpětně i na PŘEDCHOZÍ
        # (dosud nezapsaný) snímek, i kdyby jeho box confidence nebyla
        # vyhodnocená jako slabá – rotaci jsme pro něj jinak vůbec nezkoušeli.
        if rotation_helped and prev is not None:
            _, prev_box_rot, prev_key_rot, prev_annotated_rot = _detect_rotated(model, prev["frame"])
            if _better(prev_box_rot, prev["box_conf"]):
                prev["box_conf"] = prev_box_rot
                prev["key_conf"] = prev_key_rot
                prev["annotated"] = prev_annotated_rot
                n_retro_fixed += 1

        if prev is not None:
            _label_and_write(prev)
        prev = cur

        n += 1
        if n % 50 == 0:
            print(f"  ... {n} snímků zpracováno (t={timestamp_ms:.0f}ms)")

    if prev is not None:
        _label_and_write(prev)

    if writer is not None:
        writer.release()
    print(f"Hotovo: {n} snímků -> {args.output}")
    print(f"  rotace pomohla u {n_rot_helped} snímků, zpětně opraven předchozí snímek {n_retro_fixed}x")


if __name__ == "__main__":
    main()
