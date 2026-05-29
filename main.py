"""
main.py
-------
Hlavní skript pipeline pro analýzu lidského pohybu z videí.

Postup:
  1. Načte všechna videa z /videos/
  2. Pro každé video:
      a) Iteruje snímky s frame samplingem (target 8 FPS)
      b) Detekuje pózu (MediaPipe)
      c) Extrahuje příznaky (klouby, vzdálenosti, pohyb)
      d) Plní temporální okno (6 snímků)
      e) Klasifikuje akci
      f) Zapisuje řádek do CSV
      g) Volitelně generuje debug video s overlays (/output_debug/)
  3. Výstupní CSV soubory uloží do /output/

Spuštění:
    python main.py
    python main.py --debug
    python main.py --videos ./videos --output ./output --fps 8 --model ./models/rf_model.pkl --debug
"""

import argparse
import csv
import logging
import os
import sys
import time
from pathlib import Path

from video_loader import VideoLoader, get_video_files
from pose_detector import PoseDetector, PoseDetectorImage
from feature_extractor import FeatureExtractor
from temporal_model import TemporalWindow
from classifier import ActionClassifier
from visualizer import Visualizer
from jump_detector import JumpDetector
from person_manager import PersonManager

# ── Konfigurace loggeru ───────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# ZPRACOVÁNÍ JEDNOHO VIDEA
# ─────────────────────────────────────────────────────────────────────────────

def _pipe_stage_label(stage: str) -> str:
    """Převede interní název stage na zkratku dle specifikace pipeline."""
    if stage in ("pass", "skipped"):
        return stage
    if stage == "no_landmarks":
        return "pose_conf"      # žádná poze detekována
    if stage == "confidence":
        return "pose_conf"      # průměrná confidence pod prahem
    if stage.startswith("pose_val:L1"):
        return "pose_vis"       # L1 = viditelnost kloubů
    if stage.startswith("pose_val:L2"):
        return "pose_geo"       # L2 = geometrie (torso výška, rozestupy)
    if stage.startswith("pose_val:"):
        return "pose_val"
    if stage == "kinematics":
        return "kin_score"
    if stage == "motion":
        return "mot_sim"
    if stage == "appearance":
        return "appear"
    if stage == "final_conf":
        return "final_conf"
    return stage                # fallback – neznámý stage


def process_video(
    video_path: str,
    output_path: str,
    target_fps: int,
    model_path: str | None,
    debug: bool = False,
    debug_output_path: str | None = None,
) -> int:
    """
    Zpracuje jedno video a zapíše výstupní CSV soubor.
    Pokud debug=True, vygeneruje také video s overlays do debug_output_path.

    Vrátí počet zapsaných řádků (klasifikovaných snímků).
    """
    video_name = Path(video_path).name
    logger.info("── Zpracovávám: %s", video_name)
    t_start = time.time()

    # Detektory pózy
    pose_detector  = PoseDetector(min_detection_confidence=0.5, min_tracking_confidence=0.45)   # VIDEO mode: Person 1 TRACKING (full frame)
    image_detector = PoseDetectorImage(min_confidence=0.5)   # IMAGE mode: Person 1 LOST crop + Person 2 + scan

    # Multi-person koordinátor
    multi_manager = PersonManager()

    # Klasifikace – pouze Person 1
    feature_extractor = FeatureExtractor()
    temporal_window   = TemporalWindow(window_size=6)
    classifier        = ActionClassifier(model_path=model_path)
    jump_detector     = JumpDetector()

    rows_written      = 0
    highlight_timestamps: list[float] = []
    frames_processed = 0
    frames_no_pose   = 0
    frames_invalid   = 0

    fps_timer   = time.time()
    fps_counter = 0
    current_fps = 0.0

    visualizer = None

    try:
        pipe_csv_path = output_path.replace(".csv", "_pipeline_debug.csv")
        with (
            VideoLoader(video_path, target_fps=target_fps) as loader,
            open(output_path, "w", newline="", encoding="utf-8") as csv_file,
            open(pipe_csv_path, "w", newline="", encoding="utf-8") as pipe_csv_file,
        ):
            info = loader.get_video_info()
            logger.info(
                "  Video: %dx%d px | %.1f FPS | %d snímků | krok: každý %d. snímek",
                info["width"], info["height"],
                info["fps"], info["total_frames"],
                info["frame_step"],
            )

            if debug and debug_output_path:
                visualizer = Visualizer(
                    output_path=debug_output_path,
                    frame_width=info["width"],
                    frame_height=info["height"],
                    output_fps=float(target_fps),
                )
                logger.info("  Debug video: %s", debug_output_path)

            writer = csv.writer(csv_file)
            writer.writerow(["timestamp_ms", "highlight", "is_jump", "action"])  # CSV hlavička

            pipe_writer = csv.writer(pipe_csv_file)
            pipe_writer.writerow([
                "time",
                "cropframe", "cropframe_num", "cropframe_ref",
                "fullframe", "fullframe_num", "fullframe_ref",
            ])

            # ── Hlavní smyčka ─────────────────────────────────────────────
            for timestamp_ms, frame, prev_frame in loader.frame_generator():
                frames_processed += 1

                # FPS counter: aktualizuj každou sekundu
                fps_counter += 1
                now = time.time()
                if now - fps_timer >= 1.0:
                    current_fps = fps_counter / (now - fps_timer)
                    fps_counter = 0
                    fps_timer = now

                # ── Multi-person tracking ──────────────────────────────────
                # results[0] = Person 1, results[1] = Person 2
                # slot0_lost = True pokud Person 1 právě přešla TRACKING→LOST
                results, slot0_lost = multi_manager.update(
                    frame, timestamp_ms, pose_detector, image_detector, prev_frame
                )
                r0 = results[0]  # Person 1
                r1 = results[1]  # Person 2

                # Pipeline debug CSV — jeden řádek na každý snímek
                pd_ = r0.get("pipe_debug", {})
                pipe_writer.writerow([
                    f"{timestamp_ms:.0f}",
                    _pipe_stage_label(pd_.get("crop_stage", "skipped")),
                    pd_.get("crop_val", ""),
                    pd_.get("crop_ref", ""),
                    _pipe_stage_label(pd_.get("full_stage", "skipped")),
                    pd_.get("full_val", ""),
                    pd_.get("full_ref", ""),
                ])

                # Statistiky (Person 1)
                if r0["_raw_lm"] is None:
                    frames_no_pose += 1
                elif not r0["valid_pose"]:
                    frames_invalid += 1

                # Reset temporal okna při ztrátě Person 1
                if slot0_lost:
                    temporal_window.reset()

                # Person 1 stav pro klasifikaci
                person_present = r0["person_present"]
                landmarks      = r0["landmarks"]
                valid_pose     = r0["valid_pose"]

                # Pokud osoba není přítomna → zapsat debug a přeskočit klasifikaci
                if not person_present:
                    jump_detector.update_missing(timestamp_ms)
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                        )
                    continue

                # Ghost frame: tracker říká present, ale nejsou platná landmarks
                if not valid_pose or landmarks is None:
                    jump_detector.update_missing(timestamp_ms)
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                        )
                    continue

                features = feature_extractor.extract_features(landmarks)

                # Fyzikální validace skoku (na každém validním snímku)
                physics_is_jump = jump_detector.update(frame, landmarks, timestamp_ms)

                # Naplnění temporálního okna
                temporal_window.add_frame_features(features)

                if not temporal_window.is_ready():
                    # Warm-up: skelet ano, akce ještě ne
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                            jump_detector=jump_detector,
                        )
                    continue

                temporal_features = temporal_window.get_temporal_features()
                action     = classifier.predict(temporal_features)
                conf_dict  = classifier.predict_proba(temporal_features)

                confidence = conf_dict.get(action, 0.0)

                # Highlight = ne-normální akce A zároveň fyzikálně detekovaný skok
                highlight = (action not in (None, "normal", "unknown")) and physics_is_jump
                if highlight:
                    highlight_timestamps.append(timestamp_ms)

                # Zápis do CSV
                writer.writerow([f"{timestamp_ms:.0f}", str(highlight), str(physics_is_jump), action])
                rows_written += 1

                # Debug: skelet + akce
                if visualizer:
                    visualizer.write_frame(
                        frame, action, timestamp_ms, current_fps, p1=r0, p2=r1,
                        jump_detector=jump_detector,
                    )

    finally:
        pose_detector.close()
        image_detector.close()
        multi_manager.log_stats()
        multi_manager.reset()
        jump_detector.reset()
        if visualizer:
            visualizer.release()

    elapsed = time.time() - t_start
    logger.info(
        "  ✓ Hotovo: %d snímků | %d validních | %d klasifikováno | "
        "%d bez pózy | %d zamítnuto validací | %.1f s",
        frames_processed, frames_processed - frames_no_pose - frames_invalid,
        rows_written, frames_no_pose, frames_invalid, elapsed,
    )

    # Výpis highlight timestampů
    if highlight_timestamps:
        print(f"\n{'='*50}")
        print(f"HIGHLIGHTS ({len(highlight_timestamps)} událostí):")
        for ts in highlight_timestamps:
            secs = ts / 1000.0
            mins = int(secs // 60)
            print(f"  {mins:02d}:{secs % 60:06.3f}  ({ts:.0f} ms)")
        print('='*50)
    else:
        print("\nHIGHLIGHTS: žádné")
    return rows_written


# ─────────────────────────────────────────────────────────────────────────────
# BATCH ZPRACOVÁNÍ VŠECH VIDEÍ
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    videos_dir: str = "./videos",
    output_dir: str = "./output",
    output_debug_dir: str = "./output_debug",
    target_fps: int = 8,
    model_path: str | None = None,
    debug: bool = False,
) -> None:
    """
    Zpracuje všechna videa v `videos_dir` a zapíše CSV do `output_dir`.
    Pokud debug=True, zapíše také debug videa do `output_debug_dir`.
    """
    os.makedirs(output_dir, exist_ok=True)
    if debug:
        os.makedirs(output_debug_dir, exist_ok=True)

    try:
        video_files = get_video_files(videos_dir)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    if not video_files:
        logger.warning("Ve složce '%s' nebyla nalezena žádná videa.", videos_dir)
        return

    logger.info("Nalezeno %d videí v '%s'.", len(video_files), videos_dir)
    if debug:
        logger.info("Debug režim zapnut – debug videa → '%s'", output_debug_dir)

    total_rows = 0
    total_time = time.time()

    for video_path in video_files:
        stem = Path(video_path).stem
        output_path       = os.path.join(output_dir, f"{stem}.csv")
        debug_output_path = os.path.join(output_debug_dir, f"{stem}_debug.mp4") if debug else None

        try:
            rows = process_video(
                video_path, output_path, target_fps, model_path,
                debug=debug, debug_output_path=debug_output_path,
            )
            total_rows += rows
        except Exception as exc:
            logger.error("Chyba při zpracování '%s': %s", video_path, exc)
            continue

    elapsed_total = time.time() - total_time
    logger.info(
        "══ Pipeline dokončena: %d videí | %d řádků CSV | celkem %.1f s ══",
        len(video_files), total_rows, elapsed_total,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI ROZHRANÍ
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pipeline pro analýzu lidského pohybu z videí."
    )
    parser.add_argument(
        "--videos", default="./videos",
        help="Složka se vstupními videi (default: ./videos)"
    )
    parser.add_argument(
        "--output", default="./output",
        help="Složka pro výstupní CSV soubory (default: ./output)"
    )
    parser.add_argument(
        "--output-debug", default="./output_debug",
        help="Složka pro debug videa (default: ./output_debug)"
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="Cílové FPS pro analýzu (default: 8)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Cesta k .pkl souboru natrénovaného modelu (volitelné)"
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Generovat debug video s overlays do --output-debug složky"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(
        videos_dir=args.videos,
        output_dir=args.output,
        output_debug_dir=args.output_debug,
        target_fps=args.fps,
        model_path=args.model,
        debug=args.debug,
    )
