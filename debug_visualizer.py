"""
debug_visualizer.py
-------------------
Modul pro debug vizualizaci pipeline - generuje výstupní video s overlays.

Pro každý zpracovaný snímek kreslí:
  - Skelet pózy (landmarks + connections)
  - Debug panel v pravém horním rohu (akce, confidence, timestamp, FPS)

Záměrně nepoužívá mp.solutions.drawing_utils (nefunguje v MediaPipe 0.10+).
Vše je kresleno přímo přes OpenCV.
"""

import cv2
import numpy as np

# ── Definice spojení kostry (MediaPipe Pose, 33 landmarks) ───────────────────
# Každá dvojice je (start_index, end_index) dle MediaPipe topologie
POSE_CONNECTIONS = [
    # obličej
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    # horní tělo
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),
    # trup
    (11, 23), (12, 24), (23, 24),
    # nohy
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

# Barvy pro vizualizaci
COLOR_LEFT   = (0, 230, 100)    # zelená – levá strana těla
COLOR_RIGHT  = (0, 120, 255)    # oranžová – pravá strana těla
COLOR_CENTER = (200, 200, 200)  # šedá – střed (trup, obličej)
COLOR_POINT  = (255, 255, 255)  # bílá – landmark body

# Indexy pravé strany (sudá MediaPipe konvence: pravá = liché indexy kloubů)
_RIGHT_INDICES = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}
_LEFT_INDICES  = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}

# Barvy akcí pro přehlednost panelu
ACTION_COLORS = {
    "normal":     (100, 230, 100),   # zelená
    "jump":       (0, 200, 255),     # žlutá
    "acrobatics": (0, 80, 255),      # červená
    "handstand":  (255, 180, 0),     # modrá
    "spin":       (200, 0, 255),     # fialová
    "unknown":    (150, 150, 150),   # šedá
    None:         (100, 100, 100),   # tmavá šedá
}


def _connection_color(start_idx: int, end_idx: int) -> tuple:
    """Vybere barvu linky podle toho, na které straně těla je."""
    if start_idx in _LEFT_INDICES and end_idx in _LEFT_INDICES:
        return COLOR_LEFT
    if start_idx in _RIGHT_INDICES and end_idx in _RIGHT_INDICES:
        return COLOR_RIGHT
    return COLOR_CENTER


class DebugVisualizer:
    """
    Spravuje výstupní debug video a kreslí overlays na každý snímek.

    Parametry:
        output_path  -- cesta k výstupnímu video souboru
        frame_width  -- šířka snímku (pixely)
        frame_height -- výška snímku (pixely)
        output_fps   -- FPS výstupního videa (typicky target_fps = 8)
    """

    def __init__(
        self,
        output_path: str,
        frame_width: int,
        frame_height: int,
        output_fps: float,
    ):
        self.frame_width  = frame_width
        self.frame_height = frame_height
        self.output_fps   = output_fps

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            output_path, fourcc, output_fps, (frame_width, frame_height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Nelze otevřít VideoWriter pro: {output_path}")

    # ── Kreslení skeletu ─────────────────────────────────────────────────────

    def draw_skeleton(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None,
    ) -> np.ndarray:
        """
        Nakreslí skelet pózy na snímek.

        landmarks -- (33, 4) numpy pole [x, y, z, visibility], nebo None.
                     x, y jsou normalizovány do [0, 1].
        """
        if landmarks is None:
            return frame

        h, w = frame.shape[:2]

        # Převod normalizovaných souřadnic na pixely
        pts_px = []
        for lm in landmarks:
            px = int(lm[0] * w)
            py = int(lm[1] * h)
            vis = float(lm[3])
            pts_px.append((px, py, vis))

        # Kreslení linek (connections)
        for start_idx, end_idx in POSE_CONNECTIONS:
            px_s, py_s, vis_s = pts_px[start_idx]
            px_e, py_e, vis_e = pts_px[end_idx]

            # Přeskočit skryté landmarks (nízká visibility)
            if vis_s < 0.3 or vis_e < 0.3:
                continue

            color = _connection_color(start_idx, end_idx)
            cv2.line(frame, (px_s, py_s), (px_e, py_e), color, 2, cv2.LINE_AA)

        # Kreslení bodů (landmarks)
        for px, py, vis in pts_px:
            if vis < 0.3:
                continue
            cv2.circle(frame, (px, py), 4, COLOR_POINT, -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (0, 0, 0), 1, cv2.LINE_AA)  # obrys

        # Label "P1" na střed torsa
        torso_idx = [11, 12, 23, 24]
        torso_vis = [(pts_px[i][0], pts_px[i][1]) for i in torso_idx if pts_px[i][2] >= 0.3]
        if torso_vis:
            tx = int(sum(p[0] for p in torso_vis) / len(torso_vis))
            ty = int(sum(p[1] for p in torso_vis) / len(torso_vis))
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize("P1", font, 0.6, 2)
            cv2.rectangle(frame, (tx - tw//2 - 3, ty - th - 3), (tx + tw//2 + 3, ty + 3), (0, 0, 0), -1)
            cv2.putText(frame, "P1", (tx - tw//2, ty), font, 0.6, (0, 230, 80), 2, cv2.LINE_AA)

        return frame

    def draw_tracking_overlay(
        self,
        frame: np.ndarray,
        track_info: dict,
    ) -> np.ndarray:
        """
        Vykreslí křížek na tracked_pos a prázdný kroužek na predicted_pos.
        Zobrazí se jen pokud jsou souřadnice nenulové.
        """
        h, w = frame.shape[:2]

        tp = track_info.get("tracked_pos", (0.0, 0.0))
        pp = track_info.get("predicted_pos", (0.0, 0.0))
        ghost = track_info.get("ghost_active", False)

        # Tracked position – zlatý křížek
        tx_px = int(tp[0] * w)
        ty_px = int(tp[1] * h)
        if 0 < tx_px < w and 0 < ty_px < h:
            color = (0, 200, 255) if not ghost else (0, 120, 255)
            cv2.line(frame, (tx_px - 8, ty_px), (tx_px + 8, ty_px), color, 2, cv2.LINE_AA)
            cv2.line(frame, (tx_px, ty_px - 8), (tx_px, ty_px + 8), color, 2, cv2.LINE_AA)

        # Predicted position – prázdný kroužek (jen při ghost tracking)
        if ghost:
            px_px = int(pp[0] * w)
            py_px = int(pp[1] * h)
            if 0 < px_px < w and 0 < py_px < h:
                cv2.circle(frame, (px_px, py_px), 10, (0, 120, 255), 1, cv2.LINE_AA)

        return frame

    # Barvy ROI boxů pro jednotlivé končetiny (BGR)
    _LIMB_ROI_COLORS = {
        "torso":     (0,   200, 255),   # žlutá
        "left_arm":  (0,   220,  80),   # zelená
        "right_arm": (255, 220,   0),   # cyan
        "left_leg":  (100, 100, 255),   # světlá červená
        "right_leg": (220,  80, 200),   # fialová
    }
    # Krátké popisky pro badge
    _LIMB_SHORT = {
        "torso":     "TORSO",
        "left_arm":  "L.ARM",
        "right_arm": "R.ARM",
        "left_leg":  "L.LEG",
        "right_leg": "R.LEG",
    }

    def draw_motion_overlay(
        self,
        frame: np.ndarray,
        motion_info: dict,
    ) -> np.ndarray:
        """
        Vykreslí per-limb ROI boxy + motion badge (spodní pravý roh).

        ROI boxy: každá končetina má svoji barvu.
        Badge: zelený rámeček = pohyb, oranžový = statický region.
               Zobrazuje celkové SIM skóre + skóre pro každou končetinu.
        """
        h, w = frame.shape[:2]

        motion_score   = motion_info.get("motion_score",   0.0)
        region_dynamic = motion_info.get("region_dynamic", True)
        limb_debug     = motion_info.get("limb_debug",     {})
        roi            = motion_info.get("roi_orig")

        # Barva statusu: zelená = pohyb/valid, oranžová = statický
        dyn_color = (0, 200, 80) if region_dynamic else (0, 120, 255)

        # Per-limb ROI boxy – kreslit pouze pokud je region dynamic (valid)
        if region_dynamic:
            if limb_debug:
                for limb_name, limb_info in limb_debug.items():
                    lroi = limb_info.get("roi_orig")
                    if lroi is None:
                        continue
                    color = self._LIMB_ROI_COLORS.get(limb_name, (200, 200, 200))
                    lx1, ly1, lx2, ly2 = lroi
                    cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), color, 1, cv2.LINE_AA)
            elif roi is not None:
                rx1, ry1, rx2, ry2 = roi
                cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), dyn_color, 1, cv2.LINE_AA)

        # Badge – spodní pravý roh
        # Výška: 20 (header) + počet_limbs * 16 + 22 (status)
        n_limbs  = len(limb_debug)
        badge_w  = 200
        badge_h  = 20 + n_limbs * 16 + 22
        badge_h  = max(badge_h, 50)
        bx1 = w - badge_w - 10
        by2 = h - 10
        bx2 = w - 10
        by1 = by2 - badge_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), dyn_color, 1)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cy = by1 + 15

        # Celkové SIM skóre
        cv2.putText(frame, f"SIM: {motion_score:.4f}",
                    (bx1 + 8, cy), font, 0.48, (180, 180, 180), 1, cv2.LINE_AA)
        cy += 18

        # Per-limb skóre (ve fixním pořadí)
        for limb_name in ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]:
            if limb_name not in limb_debug:
                continue
            sim_val = limb_debug[limb_name]["sim"]
            color   = self._LIMB_ROI_COLORS.get(limb_name, (180, 180, 180))
            label   = self._LIMB_SHORT.get(limb_name, limb_name[:6].upper())
            cv2.putText(frame, f"  {label}: {sim_val:.3f}",
                        (bx1 + 8, cy), font, 0.40, color, 1, cv2.LINE_AA)
            cy += 16

        # Status
        dyn_text = "DYNAMIC" if region_dynamic else "STATIC"
        cv2.putText(frame, dyn_text,
                    (bx1 + 8, cy + 4), font, 0.52, dyn_color, 1, cv2.LINE_AA)

        return frame

    # ── Debug panel ──────────────────────────────────────────────────────────

    def draw_debug_panel(
        self,
        frame: np.ndarray,
        action: str | None,
        timestamp_ms: float,
        current_fps: float,
        person_present: bool = True,
        valid_pose: bool = True,
    ) -> np.ndarray:
        """
        Nakreslí info panel v pravém horním rohu snímku.

        Fixní layout – VŽDY stejná výška, 4 řádky:
          ID:      timestamp v ms
          ACTION:  klasifikovaná akce nebo NULL
          PRESENT: YES / NO
          FPS:     aktuální FPS
        """
        h, w = frame.shape[:2]

        panel_w = 260
        panel_h = 110
        margin  = 10
        x1 = w - panel_w - margin
        y1 = margin
        x2 = w - margin
        y2 = y1 + panel_h

        # Poloprůhledné černé pozadí
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        # Rámeček – barva dle stavu přítomnosti osoby + validita pózy
        if person_present and valid_pose:
            border_color = (0, 200, 80)    # zelená: klasifikace proběhla
        elif person_present:
            border_color = (0, 170, 255)   # oranžová: ghost / nevalidní póza
        else:
            border_color = (0, 50, 220)    # červená: osoba není přítomna
        cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.52
        thickness  = 1
        line_h     = 24
        tx         = x1 + 8
        ty         = y1 + 22

        # ── Řádek 1: ID ────────────────────────────────────────────────────
        cv2.putText(frame, f"ID:      {timestamp_ms:.0f} ms", (tx, ty), font, font_scale,
                    (180, 180, 180), thickness, cv2.LINE_AA)

        # ── Řádek 2: ACTION ────────────────────────────────────────────────
        ty += line_h
        action_label = action.upper() if action is not None else "NULL"
        action_color = ACTION_COLORS.get(action, (200, 200, 200)) \
                       if (person_present and valid_pose and action is not None) \
                       else (100, 100, 100)
        cv2.putText(frame, "ACTION:  ", (tx, ty), font, font_scale,
                    (200, 200, 200), thickness, cv2.LINE_AA)
        label_x = tx + cv2.getTextSize("ACTION:  ", font, font_scale, thickness)[0][0]
        cv2.putText(frame, action_label, (label_x, ty), font, font_scale,
                    action_color, thickness + 1, cv2.LINE_AA)

        # ── Řádek 3: PRESENT ───────────────────────────────────────────────
        ty += line_h
        present_text  = "YES" if person_present else "NO"
        present_color = (0, 230, 80) if person_present else (0, 60, 220)
        cv2.putText(frame, "PRESENT: ", (tx, ty), font, font_scale,
                    (200, 200, 200), thickness, cv2.LINE_AA)
        lx = tx + cv2.getTextSize("PRESENT: ", font, font_scale, thickness)[0][0]
        cv2.putText(frame, present_text, (lx, ty), font, font_scale,
                    present_color, thickness + 1, cv2.LINE_AA)

        # ── Řádek 4: FPS ───────────────────────────────────────────────────
        ty += line_h
        cv2.putText(frame, f"FPS:     {current_fps:.1f}", (tx, ty), font, font_scale,
                    (150, 220, 255), thickness, cv2.LINE_AA)

        return frame

    # ── Zápis snímku ─────────────────────────────────────────────────────────

    def write_frame(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray | None,
        action: str | None,
        timestamp_ms: float,
        current_fps: float,
        person_present: bool = True,
        valid_pose: bool = True,
        track_info: dict | None = None,
        motion_info: dict | None = None,
        person2_info: dict | None = None,
        pipeline_scores: dict | None = None,
    ) -> None:
        """
        Nakreslí všechny overlays na kopii snímku a zapíše do výstupního videa.
        Pracuje na kopii – původní frame není modifikován.

        pipeline_scores -- dict s Person 1 pipeline daty:
            {"sim_score", "appearance_score", "presence_prob", "crop", "frozen_crop", "state"}
        person2_info -- volitelný dict s Person 2 daty:
            {"landmarks", "person_present", "state", "crop", "frozen_crop", "valid_pose",
             "track_info", "sim_score", "appearance_score", "presence_prob"}
        """
        vis_frame = frame.copy()
        if pipeline_scores is not None:
            self._draw_crop_overlay(vis_frame, pipeline_scores, track_info or {})
        if valid_pose and landmarks is not None:
            self.draw_skeleton(vis_frame, landmarks)
        if track_info is not None:
            self.draw_tracking_overlay(vis_frame, track_info)
        if motion_info is not None:
            self.draw_motion_overlay(vis_frame, motion_info)
        if person2_info is not None:
            self._draw_person2_overlay(vis_frame, person2_info)
        self.draw_debug_panel(
            vis_frame, action, timestamp_ms, current_fps,
            person_present=person_present, valid_pose=valid_pose,
        )
        self._writer.write(vis_frame)

    def _draw_crop_overlay(
        self,
        frame: np.ndarray,
        ps: dict,
        track_info: dict,
    ) -> None:
        """
        Kreslí žlutý bounding box okolo Person 1 a zobrazuje pipeline skóre.

        Primárně kreslí tight bbox z raw_lm (přesná poloha detekovaných kloubů).
        Pokud raw_lm není k dispozici, bounding box se nevykresluje (jen text u края).
        Také zobrazuje crop region šedou přerušovanou čarou (kde tracker hledá).
        """
        h, w = frame.shape[:2]
        box_color = (0, 220, 255)  # žlutá (BGR)

        # ── Tight bbox z aktuálních raw landmarks ────────────────────────
        raw_lm = ps.get("raw_lm")
        bx1 = by1 = bx2 = by2 = None
        if raw_lm is not None:
            vis = raw_lm[raw_lm[:, 3] > 0.2]
            if len(vis) >= 3:
                bx1 = max(0, int(vis[:, 0].min() * w))
                by1 = max(0, int(vis[:, 1].min() * h))
                bx2 = min(w, int(vis[:, 0].max() * w))
                by2 = min(h, int(vis[:, 1].max() * h))
                # Žlutý bounding box (tučný, 2px) okolo detekované osoby
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, 2, cv2.LINE_AA)
                # Žlutý label "P1" nad boxem
                cv2.putText(frame, "P1", (bx1 + 4, by1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2, cv2.LINE_AA)

        # ── Crop region (kde tracker hledá) – šedá přerušovaná čára ─────
        state = ps.get("state", "")
        crop  = ps.get("crop") if state == "TRACKING" else ps.get("frozen_crop")
        if crop is not None:
            cx1, cy1, cx2, cy2 = crop
            rx1 = max(0, int(cx1 * w))
            ry1 = max(0, int(cy1 * h))
            rx2 = min(w, int(cx2 * w))
            ry2 = min(h, int(cy2 * h))
            # Vykreslíme přerušovaný obdélník ručně (OpenCV nemá dashed rect)
            gray = (120, 120, 120)
            dash = 12
            for x in range(rx1, rx2, dash * 2):
                cv2.line(frame, (x, ry1), (min(x + dash, rx2), ry1), gray, 1)
                cv2.line(frame, (x, ry2), (min(x + dash, rx2), ry2), gray, 1)
            for y in range(ry1, ry2, dash * 2):
                cv2.line(frame, (rx1, y), (rx1, min(y + dash, ry2)), gray, 1)
                cv2.line(frame, (rx2, y), (rx2, min(y + dash, ry2)), gray, 1)

        # ── Pipeline skóre – text uvnitř nebo vlevo od boxu ───────────────
        # Kinematická odchylka: vzdálenost tracked_pos a predicted_pos
        tp = track_info.get("tracked_pos", (0.0, 0.0))
        pp = track_info.get("predicted_pos", (0.0, 0.0))
        kin_dist = float(((tp[0] - pp[0])**2 + (tp[1] - pp[1])**2) ** 0.5)

        sim_score   = ps.get("sim_score", 0.0)
        color_score = ps.get("appearance_score", 1.0)
        pres_prob   = ps.get("presence_prob", 0.0)

        lines = [
            f"motion sim: {sim_score:.2f}",
            f"color:      {color_score:.2f}",
            f"pose conf:  {pres_prob:.2f}",
            f"kinematics: {kin_dist:.3f}",
        ]

        font       = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.42
        thickness  = 1
        line_h     = 17

        # Umístění textu: uvnitř boxu nebo levý horní roh framu jako fallback
        if bx1 is not None:
            tx = bx1 + 4
            ty = by1 + line_h + 4
        else:
            tx = 6
            ty = line_h + 4

        for i, line in enumerate(lines):
            (tw, th), _ = cv2.getTextSize(line, font, font_scale, thickness)
            cur_ty = ty + i * line_h
            cv2.rectangle(frame, (tx - 2, cur_ty - th - 1), (tx + tw + 2, cur_ty + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, line, (tx, cur_ty), font, font_scale,
                        box_color, thickness, cv2.LINE_AA)

    def _draw_skeleton_muted(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        """Nakreslí skeleton šedivě – signalizuje zamítnutou detekci."""
        h, w = frame.shape[:2]
        pts_px = [(int(lm[0] * w), int(lm[1] * h), float(lm[3])) for lm in landmarks]
        muted_color = (80, 80, 80)
        for start_idx, end_idx in POSE_CONNECTIONS:
            px_s, py_s, vis_s = pts_px[start_idx]
            px_e, py_e, vis_e = pts_px[end_idx]
            if vis_s < 0.3 or vis_e < 0.3:
                continue
            cv2.line(frame, (px_s, py_s), (px_e, py_e), muted_color, 1, cv2.LINE_AA)
        """Nakreslí skeleton šedivě – signalizuje zamítnutou detekci."""
        h, w = frame.shape[:2]
        pts_px = [(int(lm[0] * w), int(lm[1] * h), float(lm[3])) for lm in landmarks]
        muted_color = (80, 80, 80)
        for start_idx, end_idx in POSE_CONNECTIONS:
            px_s, py_s, vis_s = pts_px[start_idx]
            px_e, py_e, vis_e = pts_px[end_idx]
            if vis_s < 0.3 or vis_e < 0.3:
                continue
            cv2.line(frame, (px_s, py_s), (px_e, py_e), muted_color, 1, cv2.LINE_AA)
        for px, py, vis in pts_px:
            if vis < 0.3:
                continue
            cv2.circle(frame, (px, py), 3, muted_color, -1, cv2.LINE_AA)
    def _draw_person2_overlay(self, frame: np.ndarray, p2: dict) -> None:
        """
        Kresli Person 2 overlay:
          - skeleton modrou barvou (je-li p\u0159\u00edtomna)
          - crop box: zelen\u00fd = TRACKING, or\u00e1n\u017eov\u00fd = LOST
          - mal\u00fd badge \u201eP2\u201c v lev\u00e9m doln\u00edm rohu crop boxu
        """
        h, w = frame.shape[:2]
        state         = p2.get("state", "EMPTY")
        landmarks2    = p2.get("landmarks")
        valid_pose2   = p2.get("valid_pose", False)
        person2_pres  = p2.get("person_present", False)

        if state == "EMPTY":
            return  # Person 2 nen\u00ed aktivn\u00ed

        # Vybere kter\u00fd crop zobrazit (aktivn\u00ed nebo frozen)
        crop = p2.get("crop") if state == "TRACKING" else p2.get("frozen_crop")
        crop_color = (0, 230, 180) if state == "TRACKING" else (0, 120, 255)  # cyan / ora

        # Crop box
        if crop is not None:
            cx1, cy1, cx2, cy2 = crop
            bx1 = max(0, int(cx1 * w))
            by1 = max(0, int(cy1 * h))
            bx2 = min(w, int(cx2 * w))
            by2 = min(h, int(cy2 * h))
            # Plonn\u00fd (\u010derkovan\u00fd efekt: 2 px \u010d\u00e1ra + 2 px mezera)
            for seg_start in range(bx1, bx2, 8):
                seg_end = min(seg_start + 4, bx2)
                cv2.line(frame, (seg_start, by1), (seg_end, by1), crop_color, 2)
                cv2.line(frame, (seg_start, by2), (seg_end, by2), crop_color, 2)
            for seg_start in range(by1, by2, 8):
                seg_end = min(seg_start + 4, by2)
                cv2.line(frame, (bx1, seg_start), (bx1, seg_end), crop_color, 2)
                cv2.line(frame, (bx2, seg_start), (bx2, seg_end), crop_color, 2)
            # Badge \u201eP2\u201c
            font = cv2.FONT_HERSHEY_SIMPLEX
            label = f"P2 {state[:4]}"
            (tw, th), _ = cv2.getTextSize(label, font, 0.45, 1)
            lx, ly = bx1 + 4, by1 + th + 4
            cv2.rectangle(frame, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                          (0, 0, 0), -1)
            cv2.putText(frame, label, (lx, ly), font, 0.45, crop_color, 1, cv2.LINE_AA)

        # Person 2 skeleton (zlaté schéma)
        if person2_pres and valid_pose2 and landmarks2 is not None:
            pts2 = [(int(lm[0] * w), int(lm[1] * h), float(lm[3])) for lm in landmarks2]
            for si, ei in POSE_CONNECTIONS:
                _, _, vis_s = pts2[si]
                _, _, vis_e = pts2[ei]
                if vis_s < 0.3 or vis_e < 0.3:
                    continue
                cv2.line(frame, pts2[si][:2], pts2[ei][:2], (255, 180, 60), 2, cv2.LINE_AA)
            for px, py, vis in pts2:
                if vis < 0.3:
                    continue
                cv2.circle(frame, (px, py), 4, (255, 220, 80), -1, cv2.LINE_AA)
                cv2.circle(frame, (px, py), 4, (0, 0, 0),      1, cv2.LINE_AA)

            # Label "P2" na střed torsa
            torso_idx = [11, 12, 23, 24]
            torso_vis = [(pts2[i][0], pts2[i][1]) for i in torso_idx if pts2[i][2] >= 0.3]
            if torso_vis:
                tx = int(sum(p[0] for p in torso_vis) / len(torso_vis))
                ty = int(sum(p[1] for p in torso_vis) / len(torso_vis))
                font = cv2.FONT_HERSHEY_SIMPLEX
                (tw, th), _ = cv2.getTextSize("P2", font, 0.6, 2)
                cv2.rectangle(frame, (tx - tw//2 - 3, ty - th - 3), (tx + tw//2 + 3, ty + 3), (0, 0, 0), -1)
                cv2.putText(frame, "P2", (tx - tw//2, ty), font, 0.6, (255, 220, 80), 2, cv2.LINE_AA)
    # ── Cleanup ───────────────────────────────────────────────────────────────

    def release(self) -> None:
        """Uvolní VideoWriter. Musí být zavoláno na konci zpracování videa."""
        if self._writer.isOpened():
            self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
