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
import contextlib
import csv
import logging
import os
import sys
import time
from collections import deque
from pathlib import Path

import pose_backends
from video_loader import VideoLoader, get_video_files
from pose_backends import create_pose_detectors
from feature_extractor import FeatureExtractor
from temporal_model import TemporalWindow
from classifier import ActionClassifier
from visualizer import Visualizer
from jump_detector import JumpDetector
from flip_detector import FlipDetector, CoverageFlipDetector
from person_manager import PersonManager
from torso_angle import compute_torso_angle, compute_torso_angle_debug, compute_torso_angle_full, FREERUN_ANGLE_THR, torso_deviation, LANDMARK_INDEX

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

class _NullWriter:
    """No-op náhrada za csv.writer – použita mimo --debug, aby se diagnostické
    CSV vůbec nezapisovaly (ušetří I/O při měření realtime výkonu)."""
    def writerow(self, row) -> None:
        pass


def _build_full_clusters(
    timestamps: list[float], target_fps: int, gap_ms: float = 3000.0, expand_frames: int = 2,
) -> list[tuple[float, float]]:
    """
    Seskupí seřazené timestampy do clusterů: sousední časy patří do stejného
    clusteru, pokud je mezera mezi nimi <= gap_ms. Výsledná hranice clusteru se
    pak rozšíří o `expand_frames` snímků (dle target_fps) před první a za
    poslední hranici – to je "full_cluster".

    Vrací seznam (full_start_ms, full_end_ms) seřazený vzestupně.
    """
    if not timestamps:
        return []
    ts_sorted = sorted(timestamps)
    clusters: list[list[float]] = [[ts_sorted[0]]]
    for t in ts_sorted[1:]:
        if t - clusters[-1][-1] <= gap_ms:
            clusters[-1].append(t)
        else:
            clusters.append([t])

    frame_interval_ms = 1000.0 / target_fps
    pad = expand_frames * frame_interval_ms
    return [(c[0] - pad, c[-1] + pad) for c in clusters]


def _fmt_ms(ms: float) -> str:
    secs = ms / 1000.0
    mins = int(secs // 60)
    return f"{mins:02d}:{secs % 60:06.3f} ({ms:.0f} ms)"


def _print_full_clusters(name: str, timestamps: list[float], target_fps: int) -> None:
    clusters = _build_full_clusters(timestamps, target_fps)
    if clusters:
        print(f"\n{name} – FULL_CLUSTERY ({len(clusters)}):")
        for i, (start, end) in enumerate(clusters, 1):
            print(f"  #{i}: {_fmt_ms(start)}  →  {_fmt_ms(end)}")
    else:
        print(f"\n{name} – FULL_CLUSTERY: žádné")


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

    # Detektory pózy – scale-adaptivní přepínání MediaPipe/YOLOv8 podle toho,
    # jak blízko kamery je sledovaná osoba (bbox_frac = max rozměr bbox osoby /
    # výška obrazu). Blízko kamery → MediaPipe (přesnější i rychlejší nablízko,
    # ověřeno: 5/5 referencí, 52 % reálného času). Daleko → YOLOv8 (přesnější na
    # malé/vzdálené postavy, ověřeno: 9/9 referencí přes cov_flip_prob).
    # Kalibrováno na dvou referenčních videích: IMG_6497 (daleko) bbox_frac
    # max ~0.30, testovaci_1 (blízko, špička v okamžiku skoku) ~0.72 – hystereze
    # 0.35/0.50 dává bezpečnou rezervu k oběma stranám.
    pose_backends.POSE_MODEL = "yolov8"
    _yolo_video, _yolo_image = create_pose_detectors(
        min_detection_confidence=0.5, min_tracking_confidence=0.45, min_confidence=0.5,
    )
    # MediaPipe se načítá LÍNĚ – teprve při první skutečné potřebě přepnout
    # (scéna se přiblíží). Video, které nikdy nepřepne (jako IMG_6497), tak
    # nezaplatí ani startovní náklad na jeho načtení.
    _mp_video: object | None = None
    _mp_image: object | None = None
    _SCALE_CLOSE_LO    = 0.35
    _SCALE_CLOSE_HI    = 0.50
    _SCALE_WINDOW_SIZE = 7
    _scale_window: deque[float] = deque(maxlen=_SCALE_WINDOW_SIZE)
    active_pose_model = "yolov8"   # výchozí – obecnější/vzdálenější scéna
    pose_backends.POSE_MODEL = active_pose_model

    def _backend_pair(model: str):
        nonlocal _mp_video, _mp_image
        if model == "mediapipe":
            if _mp_video is None:
                _saved = pose_backends.POSE_MODEL
                pose_backends.POSE_MODEL = "mediapipe"
                _mp_video, _mp_image = create_pose_detectors(
                    min_detection_confidence=0.5, min_tracking_confidence=0.45, min_confidence=0.5,
                )
                pose_backends.POSE_MODEL = _saved
            return (_mp_video, _mp_image)
        return (_yolo_video, _yolo_image)

    def _bbox_frac(landmarks, frame_shape) -> float | None:
        """max(šířka, výška) bbox viditelných klíčových bodů / výška obrazu."""
        if landmarks is None:
            return None
        vis_mask = landmarks[:, 3] > 0.5
        if vis_mask.sum() < 4:
            return None
        frame_h, frame_w = frame_shape[0], frame_shape[1]
        xs = landmarks[vis_mask, 0] * frame_w
        ys = landmarks[vis_mask, 1] * frame_h
        return float(max(xs.max() - xs.min(), ys.max() - ys.min()) / frame_h)

    pose_detector, image_detector = _backend_pair(active_pose_model)

    # Multi-person koordinátor
    multi_manager = PersonManager()

    # Klasifikace – pouze Person 1
    feature_extractor = FeatureExtractor()
    temporal_window   = TemporalWindow(window_size=6)
    classifier        = ActionClassifier(model_path=model_path)
    jump_detector     = JumpDetector()
    flip_detector     = FlipDetector()
    coverage_detector = CoverageFlipDetector()

    rows_written      = 0
    highlight_timestamps: list[float] = []
    acrobe_timestamps: list[float] = []
    skok_timestamps: list[float] = []
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
        # Diagnostické CSV (pipeline_debug/jump_buff/torso_debug) se otevírají a
        # zapisují jen v --debug módu – v realtime/produkčním běhu jde o zbytečný I/O.
        _pipe_cm  = open(pipe_csv_path, "w", newline="", encoding="utf-8") if debug else contextlib.nullcontext()
        _jbuff_cm = open(jbuff_csv_path, "w", newline="", encoding="utf-8") if debug else contextlib.nullcontext()
        _torso_cm = open(torso_csv_path, "w", newline="", encoding="utf-8") if debug else contextlib.nullcontext()
        with (
            VideoLoader(video_path, target_fps=target_fps) as loader,
            open(output_path, "w", newline="", encoding="utf-8") as csv_file,
            _pipe_cm as pipe_csv_file,
            _jbuff_cm as jbuff_file,
            _torso_cm as torso_csv_file,
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

            pipe_writer = csv.writer(pipe_csv_file) if debug else _NullWriter()
            pipe_writer.writerow([
                "time",
                "cropframe", "cropframe_num", "cropframe_ref",
                "fullframe", "fullframe_num", "fullframe_ref",
            ])

            jbuff_writer = csv.writer(jbuff_file) if debug else _NullWriter()
            # Sloupce: timestamp_ms, debug_ms, buf_1 (nejnovější) .. buf_5 (nejstarší)
            # debug_ms = čas v ms který zobrazuje přehrávač debug videa (= pořadí snímku / target_fps)
            jbuff_writer.writerow(["timestamp_ms", "debug_ms", "buf_1", "buf_2", "buf_3", "buf_4", "buf_5", "buf_6"])

            torso_writer = csv.writer(torso_csv_file) if debug else _NullWriter()
            # rejection_code: 0=OK, 1=lm None, 2=nos viditelný/žádný kloub, 3=nos neviditelný/chybí kyčle nebo ramena, 4=nulová osa
            #                 5=osoba není přítomna (person_present=False), 6=ghost/neplatná poze
            torso_writer.writerow(["timestamp_ms", "torso_angle", "rejection_code"])

            # ── Hlavní smyčka ─────────────────────────────────────────────
            for timestamp_ms, frame, prev_frame, pending_before, is_regular in loader.frame_generator():
                if not is_regular:
                    # Extra snímek by sem dorazil jen po loader.request_densify() (aktuálně
                    # nikde nevoláno – viz poznámka u _densify_trigger níže). Ponecháno beze
                    # zpracování, aby přesně odpovídalo původnímu chování (frame skip).
                    continue

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
                    frame, timestamp_ms, pose_detector, image_detector, prev_frame,
                    track_second_person=debug,
                )
                r0 = results[0]  # Person 1
                r1 = results[1]  # Person 2

                # ── Scale-adaptivní přepínání pose modelu ──────────────────
                # bbox_frac tohoto snímku (pokud je platná póza) → klouzavé MAX
                # přes posledních _SCALE_WINDOW_SIZE snímků → hystereze mezi
                # _SCALE_CLOSE_LO/_SCALE_CLOSE_HI rozhodne o aktivním modelu
                # pro PŘÍŠTÍ snímek (tenhle už byl zpracován současným).
                _frac = _bbox_frac(r0.get("_raw_lm"), frame.shape)
                if _frac is not None:
                    _scale_window.append(_frac)
                if _scale_window:
                    _scale_now = max(_scale_window)
                    if active_pose_model == "yolov8" and _scale_now >= _SCALE_CLOSE_HI:
                        active_pose_model = "mediapipe"
                    elif active_pose_model == "mediapipe" and _scale_now <= _SCALE_CLOSE_LO:
                        active_pose_model = "yolov8"
                    multi_manager.set_pose_model(active_pose_model)
                    pose_detector, image_detector = _backend_pair(active_pose_model)

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
                    hmm_flip_prob = flip_detector.update(None, 0.0, timestamp_ms)
                    hmm_flip = hmm_flip_prob > 0.85
                    cov_flip_prob = coverage_detector.update(None, 0.0, timestamp_ms)
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      False,
                        "backup":       0,
                        "action":       "",
                        "is_acrobatic": False,
                        "hmm_flip":     hmm_flip,
                        "hmm_flip_prob": round(hmm_flip_prob, 3),
                        "cov_flip_prob": round(cov_flip_prob, 3),
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 6: _jbuf_v.append("")
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
                    hmm_flip_prob = flip_detector.update(None, 0.0, timestamp_ms)
                    hmm_flip = hmm_flip_prob > 0.85
                    cov_flip_prob = coverage_detector.update(None, 0.0, timestamp_ms)
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      False,
                        "backup":       0,
                        "action":       "",
                        "is_acrobatic": False,
                        "hmm_flip":     hmm_flip,
                        "hmm_flip_prob": round(hmm_flip_prob, 3),
                        "cov_flip_prob": round(cov_flip_prob, 3),
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 6: _jbuf_v.append("")
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

                # Flip detektor (Kalman) – aktualizace torso úhlem + confidence
                _ta_for_flip, _ta_conf_for_flip, _ = compute_torso_angle_full(r0.get("_raw_lm"))

                # POZNÁMKA k dynamickému zahuštění vzorkování (vyzkoušeno a zavrženo):
                # Záměr byl při is_jump AND slabé torso confidence dosytit flip detektory
                # o jinak zahazované mezilehlé nativní snímky (pending_before / loader.
                # request_densify()). Pokus ukázal, že volání multi_manager.update() na
                # těchto mezilehlých snímcích PERTURBUJE stavový tracker (video_detector /
                # image_detector uvnitř PersonManager čekají přibližně pravidelný 8fps
                # interval pro svou predikci pozice/rychlosti) – na ref4 to způsobilo NOVÝ
                # ghost frame na oficiálním vzorku, který předtím fungoval (cov_flip_prob
                # kleslo z 0.507 na 0.000). Proto se zde multi_manager.update NEVOLÁ mimo
                # pravidelnou 8fps kadenci. Řešení by vyžadovalo stavově izolovanou
                # (stateless) detekci na zamrzlém crop okně, ne polling živého trackeru.

                hmm_flip_prob = flip_detector.update(_ta_for_flip, _ta_conf_for_flip, timestamp_ms)
                hmm_flip = hmm_flip_prob > 0.85
                cov_flip_prob = coverage_detector.update(_ta_for_flip, _ta_conf_for_flip, timestamp_ms)

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
                    _ta_warmup, _tr_warmup = compute_torso_angle_debug(r0.get("_raw_lm"))
                    _freerun_warmup = (
                        physics_is_jump
                        and _ta_warmup is not None
                        and torso_deviation(_ta_warmup) > FREERUN_ANGLE_THR
                    )
                    frame_rows.append({
                        "timestamp_ms": f"{timestamp_ms:.0f}",
                        "debug_ms":     f"{_debug_ms:.0f}",
                        "is_jump":      physics_is_jump,
                        "backup":       csv_backup,
                        "action":       "",
                        "is_acrobatic": False,
                        "freerun":      _freerun_warmup,
                        "hmm_flip":     hmm_flip,
                        "hmm_flip_prob": round(hmm_flip_prob, 3),
                        "cov_flip_prob": round(cov_flip_prob, 3),
                    })
                    _jbuf_w = list(jump_detector._buffer)
                    _jbuf_v = [f"{e['y_corrected']:.5f}" if e["valid"] else "" for e in reversed(_jbuf_w)]
                    while len(_jbuf_v) < 6: _jbuf_v.append("")
                    jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_v)
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
                            freerun=_freerun_warmup,
                            hmm_flip=hmm_flip_prob,
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

                freerun = physics_is_jump and torso_angle is not None and torso_deviation(torso_angle) > FREERUN_ANGLE_THR

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
                    "hmm_flip":     hmm_flip,
                    "hmm_flip_prob": round(hmm_flip_prob, 3),
                    "cov_flip_prob": round(cov_flip_prob, 3),
                })
                rows_written += 1

                # Jump buffer debug CSV – stav bufferu po dokončení tohoto snímku
                # buf_1 = nejnovější, buf_6 = nejstarší; prázdné sloty = ""
                _jbuf = list(jump_detector._buffer)  # nejstarší → nejnovější
                _jbuf_vals = [
                    f"{e['y_corrected']:.5f}" if e["valid"] else ""
                    for e in reversed(_jbuf)          # otočíme: buf_1 = [-1], buf_6 = [0]
                ]
                # Doplň na přesně 6 hodnot (pokud buffer ještě není plný)
                while len(_jbuf_vals) < 6:
                    _jbuf_vals.append("")
                jbuff_writer.writerow([f"{timestamp_ms:.0f}", f"{_debug_ms:.0f}"] + _jbuf_vals)

                # Debug: skelet + akce
                if visualizer:
                    visualizer.write_frame(
                        frame, action, timestamp_ms, current_fps, p1=r0, p2=r1,
                        jump_detector=jump_detector,
                        torso_angle=torso_angle,
                        freerun=freerun,
                        hmm_flip=hmm_flip_prob,
                    )

    finally:
        # mediapipe pár se zavírá jen pokud byl (líně) skutečně načten.
        if _mp_video is not None:
            _mp_video.close()
        _yolo_video.close()
        multi_manager.log_stats()
        multi_manager.reset()
        jump_detector.reset()
        flip_detector.reset()
        coverage_detector.reset()
        if visualizer:
            visualizer.release()

    elapsed = time.time() - t_start
    logger.info(
        "  ✓ Hotovo: %d snímků | %d validních | %d klasifikováno | "
        "%d bez pózy | %d zamítnuto validací | %.1f s",
        frames_processed, frames_processed - frames_no_pose - frames_invalid,
        rows_written, frames_no_pose, frames_invalid, elapsed,
    )

    # ── Post-processing: zarovnání hmm_flip ───────────────────────────────────
    # flip_detector.update() vrací hodnotu zpožděnou o lookahead_frames snímků
    # (pseudo-online lookahead buffer), takže patří snímku o `lag` řádků zpět,
    # ne aktuálnímu řádku, na který se zapsala za běhu. Poslední `lag` řádků
    # nemá k dispozici vyřešenou budoucí hodnotu – zůstane jim poslední známá.
    lag = flip_detector.lookahead_frames
    if 0 < lag < len(frame_rows):
        hmm_flip_vals      = [row["hmm_flip"] for row in frame_rows]
        hmm_flip_prob_vals = [row["hmm_flip_prob"] for row in frame_rows]
        n = len(frame_rows)
        for i in range(n - lag):
            frame_rows[i]["hmm_flip"]      = hmm_flip_vals[i + lag]
            frame_rows[i]["hmm_flip_prob"] = hmm_flip_prob_vals[i + lag]

    # ── Post-processing: flip_window – zpětné označení snímků rotace ─────────
    # hmm_flip se aktivuje až po nahromadění dostatečné rotace, takže samotný
    # náběh (prvních pár desítek stupňů rotace) mu unikne – proto se vzestupná
    # hrana zpětně promítne na FLIP_BACKFILL_FRAMES předchozích snímků.
    # Celé plató hmm_flip=True (ne jen náběh) se počítá taky – proti dopřednému
    # "doznívání" do snímků bez platné pózy (osoba mimo záběr při dopadu) chrání
    # už samotné is_jump, které je v takových snímcích napevno False.
    FLIP_BACKFILL_FRAMES = 4
    for row in frame_rows:
        row["flip_window"] = bool(row["hmm_flip"])
    for i, row in enumerate(frame_rows):
        prev_flip = frame_rows[i - 1]["hmm_flip"] if i > 0 else False
        if row["hmm_flip"] and not prev_flip:
            for j in range(max(0, i - FLIP_BACKFILL_FRAMES), i + 1):
                frame_rows[j]["flip_window"] = True

    # ── Post-processing: freerun + highlight ──────────────────────────────────
    # freerun = fyzicky ve vzduchu A probíhá (zpětně označená) rotace –
    # nahrazuje dřívější okamžitý test torso_deviation > FREERUN_ANGLE_THR,
    # který je náchylný na šum jednoho snímku.
    # highlight = (is_acrobatic AND is_jump) OR freerun
    for row in frame_rows:
        row["freerun"] = bool(row.get("is_jump", False)) and row["flip_window"]
        row["highlight"] = row.get("freerun", False)

    # ── Post-processing: acrobe / skok – jen pro konzolový výpis ──────────────
    # Stejná zpětná logika jako flip_window, ale s nižším prahem 0.6 (ACROBE_PROB_THR)
    # – zachytí i slabší/méně jisté otočky.
    # acrobe = proběhla (slabší) otočka, ale BEZ skoku
    # skok   = proběhl skok, ale BEZ (jisté, 0.85) otočky
    ACROBE_PROB_THR = 0.6
    for row in frame_rows:
        row["_flip_lo"] = row.get("hmm_flip_prob", 0.0) >= ACROBE_PROB_THR
    for row in frame_rows:
        row["flip_window_lo"] = bool(row["_flip_lo"])
    for i, row in enumerate(frame_rows):
        prev_flip_lo = frame_rows[i - 1]["_flip_lo"] if i > 0 else False
        if row["_flip_lo"] and not prev_flip_lo:
            for j in range(max(0, i - FLIP_BACKFILL_FRAMES), i + 1):
                frame_rows[j]["flip_window_lo"] = True

    for row in frame_rows:
        row["acrobe"] = row["flip_window_lo"] and not row.get("is_jump", False)
        row["skok"]   = bool(row.get("is_jump", False)) and not row["flip_window"]

    # Zápis do CSV
    with open(output_path, "w", newline="", encoding="utf-8") as csv_file:
        w = csv.writer(csv_file)
        w.writerow(["timestamp_ms", "debug_ms", "highlight", "is_jump", "freerun", "hmm_flip", "hmm_flip_prob", "flip_window", "cov_flip_prob", "backup", "action"])
        for row in frame_rows:
            w.writerow([
                row["timestamp_ms"],
                row["debug_ms"],
                str(row["highlight"]),
                str(row["is_jump"]),
                str(row.get("freerun", False)),
                str(row.get("hmm_flip", False)),
                row.get("hmm_flip_prob", 0.0),
                str(row.get("flip_window", False)),
                row.get("cov_flip_prob", 0.0),
                str(row["backup"]),
                row["action"],
            ])
            if row["highlight"]:
                highlight_timestamps.append(float(row["timestamp_ms"]))
            if row["acrobe"]:
                acrobe_timestamps.append(float(row["timestamp_ms"]))
            if row["skok"]:
                skok_timestamps.append(float(row["timestamp_ms"]))

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

    # Výpis acrobe timestampů (otočka bez skoku, práh 0.6)
    if acrobe_timestamps:
        print(f"\n{'='*50}")
        print(f"ACROBE ({len(acrobe_timestamps)} událostí):")
        for ts in acrobe_timestamps:
            secs = ts / 1000.0
            mins = int(secs // 60)
            print(f"  {mins:02d}:{secs % 60:06.3f}  ({ts:.0f} ms)")
        print('='*50)
    else:
        print("\nACROBE: žádné")

    # Výpis skok timestampů (skok bez otočky)
    if skok_timestamps:
        print(f"\n{'='*50}")
        print(f"SKOK ({len(skok_timestamps)} událostí):")
        for ts in skok_timestamps:
            secs = ts / 1000.0
            mins = int(secs // 60)
            print(f"  {mins:02d}:{secs % 60:06.3f}  ({ts:.0f} ms)")
        print('='*50)
    else:
        print("\nSKOK: žádné")

    # ── Clustery: sousední časy (v rámci kategorie) se sloučí, pokud je mezera
    # <= 3s. Full_cluster = hranice clusteru rozšířená o 2 snímky před/po. ─────
    print(f"\n{'#'*55}")
    print("CLUSTERY (spojení časů <= 3s do jednoho okna, hranice ±2 snímky)")
    print(f"{'#'*55}")
    _print_full_clusters("HIGHLIGHTS", highlight_timestamps, target_fps)
    _print_full_clusters("ACROBE", acrobe_timestamps, target_fps)
    _print_full_clusters("SKOK", skok_timestamps, target_fps)
    print(f"{'#'*55}")

    # ── Realtime rezerva: čas zpracování vs. reálná délka videa ───────────────
    video_duration_s = info["total_frames"] / info["fps"] if info["fps"] > 0 else 0.0
    if video_duration_s > 0:
        pct_of_realtime = elapsed / video_duration_s * 100.0
        headroom_pct = 100.0 - pct_of_realtime
        print(f"\n{'='*55}")
        print("VÝKON – realtime rezerva")
        print(f"{'='*55}")
        print(f"  Zpracování trvalo   : {elapsed:.1f} s")
        print(f"  Délka videa         : {video_duration_s:.1f} s")
        print(f"  Poměr k reál. času  : {pct_of_realtime:.1f} %")
        print(f"  Rezerva             : {headroom_pct:.1f} %")
        print('='*55)

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
