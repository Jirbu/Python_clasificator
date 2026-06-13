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
from torso_angle import compute_torso_angle, compute_torso_angle_debug, FREERUN_ANGLE_THR

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

    # Statistiky backup fallbacku a timing
    # backup_level: 0=bez backupu, 1=L1 stačil, 2=L2 stačil, 9=vše selhalo
    backup_counts: dict[int, int]         = {0: 0, 1: 0, 2: 0, 9: 0}
    backup_times:  dict[int, list[float]] = {0: [], 1: [], 2: [], 9: []}
    # Per-trigger statistiky: {"suspicious": {0:0,1:0,...}, "no_detection": {...}}
    trigger_counts: dict[str, dict[int, int]] = {
        "suspicious":   {1: 0, 2: 0, 9: 0},
        "no_detection": {1: 0, 2: 0, 9: 0},
    }

    # Buffer řádků pro post-processing highlight s ±2 oknem
    frame_rows: list[dict] = []

    fps_timer   = time.time()
    fps_counter = 0
    current_fps = 0.0

    visualizer = None

    try:
        pipe_csv_path = output_path.replace(".csv", "_pipeline_debug.csv")
        jbuff_csv_path = output_path.replace(".csv", "_jump_buff.csv")
        torso_csv_path = output_path.replace(".csv", "_torso_debug.csv")
        with (
            VideoLoader(video_path, target_fps=target_fps) as loader,
            open(output_path, "w", newline="", encoding="utf-8") as csv_file,
            open(pipe_csv_path, "w", newline="", encoding="utf-8") as pipe_csv_file,
            open(jbuff_csv_path, "w", newline="", encoding="utf-8") as jbuff_file,
            open(torso_csv_path, "w", newline="", encoding="utf-8") as torso_csv_file,
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
            # CSV hlavička bude zapsána při post-processingu na konci

            pipe_writer = csv.writer(pipe_csv_file)
            pipe_writer.writerow([
                "time",
                "cropframe", "cropframe_num", "cropframe_ref",
                "fullframe", "fullframe_num", "fullframe_ref",
            ])

            jbuff_writer = csv.writer(jbuff_file)
            # Sloupce: timestamp_ms, debug_ms, buf_1 (nejnovější) .. buf_5 (nejstarší)
            # debug_ms = čas v ms který zobrazuje přehrávač debug videa (= pořadí snímku / target_fps)
            jbuff_writer.writerow(["timestamp_ms", "debug_ms", "buf_1", "buf_2", "buf_3", "buf_4", "buf_5"])

            torso_writer = csv.writer(torso_csv_file)
            # rejection_code: 0=OK, 1=lm None, 2=nos viditelný/žádný kloub, 3=nos neviditelný/chybí kyčle nebo ramena, 4=nulová osa
            #                 5=osoba není přítomna (person_present=False), 6=ghost/neplatná poze
            torso_writer.writerow(["timestamp_ms", "torso_angle", "rejection_code"])

            # ── Hlavní smyčka ─────────────────────────────────────────────
            for timestamp_ms, frame, prev_frame in loader.frame_generator():
                frames_processed += 1
                # Pořadí snímku v debug videu (0-based) + čas který ukáže přehrávač
                _debug_frame_idx = frames_processed - 1
                _debug_ms = _debug_frame_idx * (1000.0 / target_fps)
                t_frame_start = time.perf_counter()

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

                # Zaznamenej backup level a čas zpracování tohoto snímku
                backup_level   = r0.get("backup_level", 0)
                backup_trigger = r0.get("backup_trigger", "none")
                frame_duration_ms = (time.perf_counter() - t_frame_start) * 1000.0
                backup_counts[backup_level] = backup_counts.get(backup_level, 0) + 1
                backup_times.setdefault(backup_level, []).append(frame_duration_ms)
                if backup_trigger in trigger_counts and backup_level != 0:
                    trigger_counts[backup_trigger][backup_level] = \
                        trigger_counts[backup_trigger].get(backup_level, 0) + 1

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

                # Záloha bufferu jump_detectoru PŘED zápisem tohoto snímku.
                # Pokud byl backup triggerován z důvodu no_detection (= landmarks pochází
                # z prev_frame, ne z aktuálního snímku), restaurujeme snapshot a zapíšeme
                # update_missing – aby trajektorie v bufferu byla časově konzistentní.
                _jd_snap = jump_detector.snapshot()

                # Pokud osoba není přítomna → zapsat debug a přeskočit klasifikaci
                if not person_present:
                    jump_detector.update_missing(timestamp_ms)
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      False,
                        "backup":       0,
                        "action":       "",
                        "is_acrobatic": False,
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 5: _jbuf_v.append("")
                    jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_v)
                    torso_writer.writerow([f"{timestamp_ms:.0f}", "", 5])
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                        )
                    continue

                # Ghost frame: tracker říká present, ale nejsou platná landmarks
                if not valid_pose or landmarks is None:
                    jump_detector.update_missing(timestamp_ms)
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      False,
                        "backup":       0,
                        "action":       "",
                        "is_acrobatic": False,
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 5: _jbuf_v.append("")
                    jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_v)
                    torso_writer.writerow([f"{timestamp_ms:.0f}", "", 6])
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                        )
                    continue

                features = feature_extractor.extract_features(landmarks)

                # Fyzikální validace skoku (na každém validním snímku).
                # Výjimka: no_detection backup → landmarks jsou z prev_frame,
                # nikoli z aktuálního snímku. Restaurujeme snapshot bufferu a
                # zapíšeme update_missing, aby trajektorie zůstala časově konzistentní.
                _no_det_backup = (
                    r0.get("backup_trigger") == "no_detection"
                    and r0.get("backup_level") in (1, 2)
                )
                if _no_det_backup:
                    jump_detector.restore(_jd_snap)
                    physics_is_jump = jump_detector.update_missing(timestamp_ms)
                else:
                    physics_is_jump = jump_detector.update(frame, landmarks, timestamp_ms)

                # Naplnění temporálního okna
                # Enkóduj backup level + trigger do jediného CSV čísla:
                #   no backup → 0
                #   suspicious: L1→1, L2→2, failed→5
                #   no_detection: L1→6, L2→7, failed→8
                _CSV_BACKUP_MAP = {
                    "suspicious":   {0: 0, 1: 1, 2: 2, 9: 5},
                    "no_detection": {0: 0, 1: 6, 2: 7, 9: 8},
                }
                csv_backup = _CSV_BACKUP_MAP.get(backup_trigger, {}).get(backup_level, backup_level)

                temporal_window.add_frame_features(features)

                if not temporal_window.is_ready():
                    # Warm-up: skelet ano, akce ještě ne
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      physics_is_jump,
                        "backup":       csv_backup,
                        "action":       "",
                        "is_acrobatic": False,
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 5: _jbuf_v.append("")
                    jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_v)
                    _ta_warmup, _tr_warmup = compute_torso_angle_debug(r0.get("_raw_lm"))
                    torso_writer.writerow([
                        f"{timestamp_ms:.0f}",
                        f"{_ta_warmup:.2f}" if _ta_warmup is not None else "",
                        _tr_warmup,
                    ])
                    if visualizer:
                        visualizer.write_frame(
                            frame, None, timestamp_ms, current_fps, p1=r0, p2=r1,
                            jump_detector=jump_detector,
                            torso_angle=_ta_warmup,
                        )
                    continue

                temporal_features = temporal_window.get_temporal_features()
                action     = classifier.predict(temporal_features)
                conf_dict  = classifier.predict_proba(temporal_features)

                confidence = conf_dict.get(action, 0.0)

                # Výpočet úhlu torza (pro debug vizualizaci)
                # Používáme _raw_lm – obsahuje landmarks i při fallbacku nebo nízkém final_conf
                _raw_lm_for_angle = r0.get("_raw_lm")
                torso_angle, _torso_rej = compute_torso_angle_debug(_raw_lm_for_angle)
                torso_writer.writerow([
                    f"{timestamp_ms:.0f}",
                    f"{torso_angle:.2f}" if torso_angle is not None else "",
                    _torso_rej,
                ])

                freerun = physics_is_jump and torso_angle is not None and torso_angle > FREERUN_ANGLE_THR
                print(f"FR,{timestamp_ms:.0f},{physics_is_jump},{torso_angle is not None},{torso_angle is not None and torso_angle > FREERUN_ANGLE_THR},{torso_angle},{freerun}")
                # Highlight – předběžné vyhodnocení (bude přepočítáno s ±2 oknem na konci)
                is_acrobatic = action not in (None, "normal", "unknown")
                highlight = is_acrobatic and physics_is_jump

                # Buffering řádku pro post-processing ±2 okno
                frame_rows.append({
                    "timestamp_ms": f"{timestamp_ms:.0f}",
                    "debug_ms":     f"{_debug_ms:.0f}",
                    "is_jump":      physics_is_jump,
                    "backup":       csv_backup,
                    "action":       action,
                    "is_acrobatic": is_acrobatic,
                    "freerun":      freerun,
                })
                rows_written += 1

                # Jump buffer debug CSV – stav bufferu po dokončení tohoto snímku
                # buf_1 = nejnovější, buf_5 = nejstarší; prázdné sloty = ""
                _jbuf = list(jump_detector._buffer)  # nejstarší → nejnovější
                _jbuf_vals = [
                    f"{e['y_corrected']:.5f}" if e["valid"] else ""
                    for e in reversed(_jbuf)          # otočíme: buf_1 = [-1], buf_5 = [0]
                ]
                # Doplň na přesně 5 hodnot (pokud buffer ještě není plný)
                while len(_jbuf_vals) < 5:
                    _jbuf_vals.append("")
                jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_vals)

                # Debug: skelet + akce
                if visualizer:
                    visualizer.write_frame(
                        frame, action, timestamp_ms, current_fps, p1=r0, p2=r1,
                        jump_detector=jump_detector,
                        torso_angle=torso_angle,
                        freerun=freerun,
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

    # ── Post-processing: highlight ────────────────────────────────────────────
    # highlight = (is_acrobatic AND is_jump) OR freerun
    for row in frame_rows:
        row["highlight"] = (row["is_acrobatic"] and row["is_jump"]) or row.get("freerun", False)

    # Zápis do CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        w = csv.writer(csv_file)
        w.writerow(["timestamp_ms", "debug_ms", "highlight", "is_jump", "freerun", "backup", "action"])
        for row in frame_rows:
            w.writerow([
                row["timestamp_ms"],
                row["debug_ms"],
                str(row["highlight"]),
                str(row["is_jump"]),
                str(row.get("freerun", False)),
                str(row["backup"]),
                row["action"],
            ])
            if row["highlight"]:
                highlight_timestamps.append(float(row["timestamp_ms"]))

    # ── Sumarizace backup fallbacku ──────────────────────────────────────────
    total_frames = sum(backup_counts.values())
    total_backup = total_frames - backup_counts.get(0, 0)

    def _avg_ms(level: int) -> str:
        times = backup_times.get(level, [])
        return f"{sum(times)/len(times):.1f} ms" if times else "n/a"

    print(f"\n{'='*55}")
    print("BACKUP FALLBACK – SUMARIZACE")
    print(f"{'='*55}")
    print(f"  Celkem snímků zpracováno : {total_frames}")
    print(f"  Bez backupu  (0) : {backup_counts.get(0,0):5d}  ({backup_counts.get(0,0)/max(1,total_frames)*100:.1f} %)  prům. čas: {_avg_ms(0)}")
    print(f"  Backup L1    (1) : {backup_counts.get(1,0):5d}  ({backup_counts.get(1,0)/max(1,total_frames)*100:.1f} %)  prům. čas: {_avg_ms(1)}")
    print(f"  Backup L2    (2) : {backup_counts.get(2,0):5d}  ({backup_counts.get(2,0)/max(1,total_frames)*100:.1f} %)  prům. čas: {_avg_ms(2)}")
    print(f"  Vše selhalo  (9) : {backup_counts.get(9,0):5d}  ({backup_counts.get(9,0)/max(1,total_frames)*100:.1f} %)  prům. čas: {_avg_ms(9)}")
    print(f"  ── Snímků vyžadujících backup: {total_backup} ({total_backup/max(1,total_frames)*100:.1f} %)")
    if total_backup > 0:
        print(f"  ── Z nich zachráněno L1 : {backup_counts.get(1,0)/total_backup*100:.1f} %")
        print(f"  ── Z nich zachráněno L2 : {backup_counts.get(2,0)/total_backup*100:.1f} %")
        print(f"  ── Z nich nezachráněno  : {backup_counts.get(9,0)/total_backup*100:.1f} %")

    # Per-trigger detail
    for trigger_key, trigger_label in [
        ("suspicious",   "pose_suspicious (geom. chyba)"),
        ("no_detection", "no_detection    (detekce selhala)"),
    ]:
        tc = trigger_counts[trigger_key]
        t_total = sum(tc.values())
        if t_total == 0:
            continue
        print(f"  {'─'*51}")
        print(f"  Příčina: {trigger_label}  → celkem {t_total} snímků")
        print(f"    zachráněno L1 : {tc.get(1,0):4d}  ({tc.get(1,0)/t_total*100:.1f} %)")
        print(f"    zachráněno L2 : {tc.get(2,0):4d}  ({tc.get(2,0)/t_total*100:.1f} %)")
        print(f"    nezachráněno  : {tc.get(9,0):4d}  ({tc.get(9,0)/t_total*100:.1f} %)")
    print('='*55)

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
