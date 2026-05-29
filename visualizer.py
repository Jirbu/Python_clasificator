"""
visualizer.py
-------------
Vizualizační modul pro debug video pipeline.

Na každý snímek kreslí pro každou osobu:
  ┌─ Per-osoba ──────────────────────────────────────────────────────────┐
  │  - Skeleton (P1: barevný, P2: zlatý)                                 │
  │  - Těsný bounding box z landmarks (P1: žlutý tučný, P2: cyan)       │
  │  - Crop / search region (přerušovaná čára, P1: šedá, P2: cyan)      │
  │  - Label osoby na trupu (P1, P2)                                     │
  │  - Scores panel vedle bounding boxu:                                  │
  │      STATE | pipe | conf | tracker | motion | appear | kin            │
  │  - Křížek na tracked_pos, kroužek na ghost predicted_pos             │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Globální panel (vpravo nahoře) ─────────────────────────────────────┐
  │  ID (timestamp), ACTION (klasifikace P1), PRESENT, FPS               │
  └──────────────────────────────────────────────────────────────────────┘
  ┌─ Motion ROI (vpravo dole, jen P1) ───────────────────────────────────┐
  │  Per-limb barevné ROI boxy + SIM skóre badge                         │
  └──────────────────────────────────────────────────────────────────────┘

API:
    visualizer = Visualizer(output_path, frame_width, frame_height, output_fps)
    visualizer.write_frame(frame, action, timestamp_ms, current_fps, p1=r0, p2=r1)
    visualizer.release()
"""

from __future__ import annotations

import cv2
import numpy as np

# ── Skeletal connections (MediaPipe Pose, 33 landmarks) ───────────────────────
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12),
    (11, 13), (13, 15), (15, 17), (15, 19), (17, 19), (15, 21),
    (12, 14), (14, 16), (16, 18), (16, 20), (18, 20), (16, 22),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
]

_LEFT_IDX  = {11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31}
_RIGHT_IDX = {12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32}

# Barvy P1 (barevný skelet)
_P1_LEFT   = (0, 230, 100)    # zelená – levá strana
_P1_RIGHT  = (0, 120, 255)    # oranžová – pravá strana
_P1_CENTER = (200, 200, 200)  # šedá – střed
_P1_PT     = (255, 255, 255)  # bílá – body
_P1_BOX    = (0, 220, 255)    # žlutá – bounding box

# Barvy P2 (zlatý skelet)
_P2_LINE   = (255, 180, 60)
_P2_PT     = (255, 220, 80)
_P2_BOX    = (0, 230, 200)   # cyan

# Barva stavového textu
_COLOR_TRACKING = (0, 220, 80)   # zelená
_COLOR_GHOST    = (0, 200, 255)  # žlutá/cyan – ghost predikce
_COLOR_LOST     = (0, 80, 255)   # oranžová
_COLOR_GHOST    = (0, 160, 255)  # světlá oranžová

# Motion limb ROI barvy
_LIMB_COLORS = {
    "torso":     (0,   200, 255),
    "left_arm":  (0,   220,  80),
    "right_arm": (255, 220,   0),
    "left_leg":  (100, 100, 255),
    "right_leg": (220,  80, 200),
}
_LIMB_SHORT = {
    "torso": "TORSO", "left_arm": "L.ARM", "right_arm": "R.ARM",
    "left_leg": "L.LEG", "right_leg": "R.LEG",
}

# Akce → barva
_ACTION_COLORS = {
    "normal": (100, 230, 100), "jump": (0, 200, 255),
    "acrobatics": (0, 80, 255), "handstand": (255, 180, 0),
    "spin": (200, 0, 255), "unknown": (150, 150, 150), None: (100, 100, 100),
}

_FONT      = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SM   = 0.40
_FONT_MD   = 0.50
_FONT_LG   = 0.60


def _lh(scale: float = _FONT_SM, thick: int = 1) -> int:
    """Výška řádku textu."""
    return cv2.getTextSize("A", _FONT, scale, thick)[0][1] + 8


def _text_bg(
    frame: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float,
    fg: tuple,
    thick: int = 1,
    alpha: float = 0.65,
) -> int:
    """Nakreslí text s poloprůhledným černým pozadím. Vrátí šířku textu."""
    (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
    pad = 3
    overlay = frame.copy()
    cv2.rectangle(overlay, (x - pad, y - th - pad), (x + tw + pad, y + pad), (0, 0, 0), -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    cv2.putText(frame, text, (x, y), _FONT, scale, fg, thick, cv2.LINE_AA)
    return tw


def _dashed_rect(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    color: tuple,
    thick: int = 1,
    dash: int = 10,
) -> None:
    """Nakreslí přerušovaný obdélník."""
    for sx in range(x1, x2, dash * 2):
        ex = min(sx + dash, x2)
        cv2.line(frame, (sx, y1), (ex, y1), color, thick)
        cv2.line(frame, (sx, y2), (ex, y2), color, thick)
    for sy in range(y1, y2, dash * 2):
        ey = min(sy + dash, y2)
        cv2.line(frame, (x1, sy), (x1, ey), color, thick)
        cv2.line(frame, (x2, sy), (x2, ey), color, thick)


# ── Hlavní třída ──────────────────────────────────────────────────────────────

class Visualizer:
    """
    Vizualizátor pro debug video.

    Parametry:
        output_path  – cesta k výstupnímu .mp4 souboru
        frame_width  – šířka snímku [px]
        frame_height – výška snímku [px]
        output_fps   – FPS výstupního videa
    """

    def __init__(
        self,
        output_path: str,
        frame_width: int,
        frame_height: int,
        output_fps: float,
    ) -> None:
        self.frame_width  = frame_width
        self.frame_height = frame_height

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(
            output_path, fourcc, output_fps, (frame_width, frame_height)
        )
        if not self._writer.isOpened():
            raise IOError(f"Nelze otevřít VideoWriter: {output_path}")

    # ── Hlavní metoda ─────────────────────────────────────────────────────────

    def write_frame(
        self,
        frame: np.ndarray,
        action: str | None,
        timestamp_ms: float,
        current_fps: float,
        p1: dict | None = None,
        p2: dict | None = None,
        jump_detector=None,
    ) -> None:
        """
        Nakreslí všechny overlays a zapíše snímek do videa.

        Parametry:
            frame        – originální BGR snímek (nebude modifikován)
            action       – klasifikovaná akce Person 1 pro tento snímek (nebo None)
            timestamp_ms – timestamp snímku
            current_fps  – aktuální FPS zpracování
            p1           – result dict pro Person 1 (z PersonManager.update)
            p2           – result dict pro Person 2
            jump_detector – JumpDetector instance pro trajectory vizualizaci (nebo None)
        """
        vis = frame.copy()

        # Uložit referenci pro scores panel (musí být před _draw_person)
        self._last_jump_detector = jump_detector

        # P1 overlays
        if p1 is not None:
            self._draw_person(vis, p1, person_id=1)
            self._draw_motion_limbs(vis, p1.get("motion_info", {}))

        # P2 overlays
        if p2 is not None and p2.get("state") != "EMPTY":
            self._draw_person(vis, p2, person_id=2)

        # Jump trajectory vizualizace (pohybuje se s osobou)
        if jump_detector is not None and p1 is not None:
            traj = jump_detector.get_trajectory_for_viz()
            if traj is not None:
                self._draw_jump_trajectory(vis, traj, p1)

        # Globální panel (vpravo nahoře)
        self._draw_global_panel(vis, action, timestamp_ms, current_fps, p1)

        # HIGH RES badge (levý dolní roh) – zobrazí se pokud byl použit hires fallback
        if p1 is not None and p1.get("pipeline_used") == "crop_hires":
            self._draw_hires_badge(vis)

        self._writer.write(vis)

    def release(self) -> None:
        self._writer.release()

    def _draw_hires_badge(self, frame: np.ndarray) -> None:
        """Nakreslí 'HIGH RES' badge v levém dolním rohu snímku."""
        h, w = frame.shape[:2]
        text = "HIGH RES"
        scale = _FONT_MD
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, _FONT, scale, thick)
        pad = 8
        margin = 10
        x1 = margin
        y2 = h - margin
        x2 = x1 + tw + pad * 2
        y1 = y2 - th - pad * 2
        # Poloprůhledné pozadí
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.70, frame, 0.30, 0, frame)
        # Rámeček + text (fialová barva)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (220, 60, 220), 1, cv2.LINE_AA)
        cv2.putText(
            frame, text,
            (x1 + pad, y2 - pad),
            _FONT, scale, (220, 60, 220), thick, cv2.LINE_AA,
        )

    # ── Jump trajectory ────────────────────────────────────────────────────────

    def _draw_jump_trajectory(
        self,
        frame: np.ndarray,
        traj: dict,
        p1: dict,
    ) -> None:
        """
        Kreslí trajektorii skoku:
          - Modré tečky na skutečných pozicích boků v snímku
          - Mřížka vpravo od osoby (pohybuje se s osobou)
          - Zrcadlové body v mřížce (x=čas, y=y_corrected) spojené horizontální čarou s originálem
          - Fialová parabola přes zrcadlové body
        """
        import numpy as np

        h, w = frame.shape[:2]

        hip_xy       = traj["hip_xy"]       # list[(x_norm, y_norm)]
        y_corr       = traj["y_corrected"]  # list[float]
        t_norm       = traj["t_norm"]       # list[float] 0..1
        parabola_abc = traj.get("parabola_abc")
        n            = len(hip_xy)

        if n == 0:
            return

        # Aktuální hip v pixelech (poslední bod)
        cur_hip_x = int(hip_xy[-1][0] * w)
        cur_hip_y = int(hip_xy[-1][1] * h)

        # ── Mřížka: anchor = 80px vpravo od aktuálního hip ─────────────────
        GRID_OFFSET_X  = 80          # px od hip ke kraji mřížky
        GRID_W         = 5 * 28      # 5 sloupců × 28 px
        GRID_H         = 120         # výška mřížky v px
        GRID_COLS      = 5           # počet časových sloupců
        GRID_ROWS      = 4           # počet horizontálních referenčních čar

        gx0 = cur_hip_x + GRID_OFFSET_X
        gx1 = gx0 + GRID_W
        gy0 = cur_hip_y - GRID_H // 2
        gy1 = gy0 + GRID_H

        col_xs = [gx0 + int(i * GRID_W / (GRID_COLS - 1)) for i in range(GRID_COLS)]

        # Pevné měřítko: GRID_Y_SCALE = kolik jednotek y_corrected odpovídá celé výšce mřížky
        # 0.15 = pohyb o 15 % výšky snímku zaplní celou mřížku (odpovídá cca výšce skoku)
        GRID_Y_SCALE = 0.15
        y_center = sum(y_corr) / len(y_corr)   # střed rozsahu = průměr bufferu

        def map_y(yc: float) -> int:
            """Mapuje y_corrected na pixel v mřížce s pevným měřítkem (bez ořezu)."""
            t = (yc - y_center) / GRID_Y_SCALE + 0.5   # 0.5 = střed mřížky
            return int(gy0 + t * GRID_H)

        # ── Polotransparentní pozadí mřížky ────────────────────────────────
        overlay = frame.copy()
        cv2.rectangle(overlay, (gx0 - 4, gy0 - 4), (gx1 + 4, gy1 + 4), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # ── Vertikální čáry (čas) ───────────────────────────────────────────
        for cx in col_xs:
            cv2.line(frame, (cx, gy0), (cx, gy1), (60, 60, 60), 1)

        # ── Horizontální referenční čáry ────────────────────────────────────
        for ri in range(GRID_ROWS + 1):
            ry = gy0 + int(ri * GRID_H / GRID_ROWS)
            cv2.line(frame, (gx0, ry), (gx1, ry), (50, 50, 50), 1)

        # ── Parabola přes zrcadlové body ────────────────────────────────────
        if parabola_abc is not None:
            a, b, c = parabola_abc
            pts = []
            for px in range(gx0, gx1 + 1, 2):
                t = (px - gx0) / GRID_W
                yc = a * t * t + b * t + c
                py = map_y(yc)
                pts.append((px, py))
            if len(pts) >= 2:
                for i in range(len(pts) - 1):
                    cv2.line(frame, pts[i], pts[i + 1], (200, 50, 220), 2)

        # ── Body a spojovací čáry ────────────────────────────────────────────
        for i in range(n):
            # Skutečná pozice v snímku
            ox = int(hip_xy[i][0] * w)
            oy = int(hip_xy[i][1] * h)

            # Zrcadlový bod v mřížce
            col_idx = int(t_norm[i] * (GRID_COLS - 1) + 0.5)
            col_idx = max(0, min(GRID_COLS - 1, col_idx))
            mx = col_xs[col_idx]
            my = map_y(y_corr[i])

            # Horizontální spojovací čára (tenká, modrá)
            cv2.line(frame, (ox, oy), (mx, my), (200, 120, 30), 1)

            # Modrý kruh na skutečné pozici
            alpha_val = 120 + int(135 * (i / max(n - 1, 1)))  # starší = průhledněj
            color_blue = (alpha_val, 80, 30)   # BGR: modrá s tmaváním pro starší
            cv2.circle(frame, (ox, oy), 5, color_blue, -1)

            # Zrcadlový bod (bílý)
            cv2.circle(frame, (mx, my), 4, (230, 230, 230), -1)

    # ── Per-osoba ─────────────────────────────────────────────────────────────

    def _draw_person(self, frame: np.ndarray, result: dict, person_id: int) -> None:
        """Kreslí vše pro jednu osobu: skeleton, bbox, crop region, scores."""
        h, w = frame.shape[:2]

        is_p1      = (person_id == 1)
        box_color  = _P1_BOX if is_p1 else _P2_BOX
        state      = result.get("state", "EMPTY")
        raw_lm     = result.get("_raw_lm")
        landmarks  = result.get("landmarks")        # pouze pokud pipeline SUCCESS
        valid_pose = result.get("valid_pose", False)
        track_info = result.get("track_info", {})
        ghost      = track_info.get("ghost_active", False)
        person_id_label = f"P{person_id}"

        # ── Skeleton ─────────────────────────────────────────────────────
        # landmarks = validní pipeline success; raw_lm = cokoliv co MediaPipe detekoval
        draw_lm = landmarks if landmarks is not None else raw_lm
        pipeline_ok = landmarks is not None
        if draw_lm is not None:
            if is_p1:
                self._draw_skeleton_p1(frame, draw_lm,
                                       dimmed=(not pipeline_ok and valid_pose),
                                       invalid=(not valid_pose))
            else:
                self._draw_skeleton_p2(frame, draw_lm)

        # ── Detection crop (fialový) — oblast ve které probíhala detekce v TOMTO snímku ──
        det_crop = result.get("detection_crop")
        if det_crop is not None:
            dc1, dy1, dc2, dy2 = det_crop
            dx1 = max(0, int(dc1 * w)); ddy1 = max(0, int(dy1 * h))
            dx2 = min(w, int(dc2 * w)); ddy2 = min(h, int(dy2 * h))
            cv2.rectangle(frame, (dx1, ddy1), (dx2, ddy2), (200, 0, 200), 1, cv2.LINE_AA)

        # ── Crop bbox (pose + 40% margin) — žlutý/cyan plný obdélník ────
        # TRACKING/GHOST: aktivní crop; LOST: frozen crop
        # (toto je crop vypočtený v TOMTO snímku, použitý pro detekci v PŘÍŠTÍM snímku)
        crop = result.get("crop") if state in ("TRACKING", "GHOST") else result.get("frozen_crop")
        bx1 = by1 = bx2 = by2 = None
        if crop is not None:
            cx1, cy1, cx2, cy2 = crop
            bx1 = max(0, int(cx1 * w)); by1 = max(0, int(cy1 * h))
            bx2 = min(w, int(cx2 * w)); by2 = min(h, int(cy2 * h))
            thick = 2 if is_p1 else 1
            cv2.rectangle(frame, (bx1, by1), (bx2, by2), box_color, thick, cv2.LINE_AA)
            # Label osoby nad boxem
            _text_bg(frame, person_id_label, bx1 + 4, by1 - 6, _FONT_MD, box_color, thick)
            # State label v rohu crop regionu
            if state == "TRACKING":
                state_color = _COLOR_TRACKING
            elif state == "GHOST":
                state_color = _COLOR_GHOST
            else:
                state_color = _COLOR_LOST
            state_short = state[:4]  # TRAC / GHOS / LOST
            _text_bg(frame, state_short, bx1 + 4, by1 + 16, _FONT_SM, state_color)

        # ── Tracking markers ─────────────────────────────────────────────
        tp = track_info.get("tracked_pos", (0.0, 0.0))
        pp = track_info.get("predicted_pos", (0.0, 0.0))
        tx_px = int(tp[0] * w); ty_px = int(tp[1] * h)
        if 0 < tx_px < w and 0 < ty_px < h:
            c = _COLOR_GHOST if ghost else box_color
            cv2.line(frame, (tx_px - 8, ty_px), (tx_px + 8, ty_px), c, 2, cv2.LINE_AA)
            cv2.line(frame, (tx_px, ty_px - 8), (tx_px, ty_px + 8), c, 2, cv2.LINE_AA)
        if ghost:
            px_px = int(pp[0] * w); py_px = int(pp[1] * h)
            if 0 < px_px < w and 0 < py_px < h:
                cv2.circle(frame, (px_px, py_px), 12, _COLOR_GHOST, 1, cv2.LINE_AA)

        # ── Predikovaný střed cropu pro PŘÍŠTÍ snímek (červený křížek) ──
        kp = result.get("kin_predicted")
        if kp is not None:
            kx = int(kp[0] * w); ky = int(kp[1] * h)
            if 0 <= kx < w and 0 <= ky < h:
                S = 10  # polovina délky ramene křížku
                cv2.line(frame, (kx - S, ky), (kx + S, ky), (0, 0, 220), 3, cv2.LINE_AA)
                cv2.line(frame, (kx, ky - S), (kx, ky + S), (0, 0, 220), 3, cv2.LINE_AA)

        # ── Scores panel na středu torsa ──────────────────────────────
        if is_p1:
            self._draw_p1_scores(frame, result, raw_lm)
        else:
            self._draw_p2_scores(frame, result, raw_lm)

    # ── Skeleton drawing ──────────────────────────────────────────────────────

    def _draw_skeleton_p1(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        dimmed: bool = False,
        invalid: bool = False,
    ) -> None:
        """Skelet P1: barevný (zelená/oranžová/šedá) nebo ztlumený při pipeline FAIL.
        invalid=True: celá pose červene – pose existuje ale byla odmítnuta validací."""
        h, w = frame.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h), float(lm[3])) for lm in landmarks]

        for si, ei in POSE_CONNECTIONS:
            if pts[si][2] < 0.3 or pts[ei][2] < 0.3:
                continue
            if invalid:
                color = (0, 0, 220)
            elif dimmed:
                color = (80, 80, 80)
            elif si in _LEFT_IDX and ei in _LEFT_IDX:
                color = _P1_LEFT
            elif si in _RIGHT_IDX and ei in _RIGHT_IDX:
                color = _P1_RIGHT
            else:
                color = _P1_CENTER
            cv2.line(frame, pts[si][:2], pts[ei][:2], color, 2, cv2.LINE_AA)

        pt_color = (0, 0, 200) if invalid else ((120, 120, 120) if dimmed else _P1_PT)
        for px, py, vis in pts:
            if vis < 0.3:
                continue
            cv2.circle(frame, (px, py), 4, pt_color,  -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (0, 0, 0),  1, cv2.LINE_AA)

        # Label "P1" na trupu
        torso_pts = [(pts[i][0], pts[i][1]) for i in [11, 12, 23, 24] if pts[i][2] >= 0.3]
        if torso_pts:
            tx = int(sum(p[0] for p in torso_pts) / len(torso_pts))
            ty = int(sum(p[1] for p in torso_pts) / len(torso_pts))
            lc = (0, 0, 200) if invalid else ((80, 80, 80) if dimmed else _P1_BOX)
            _text_bg(frame, "P1", tx - 10, ty, _FONT_MD, lc, thick=2)

    def _draw_skeleton_p2(self, frame: np.ndarray, landmarks: np.ndarray) -> None:
        """Skelet P2: zlatý."""
        h, w = frame.shape[:2]
        pts = [(int(lm[0] * w), int(lm[1] * h), float(lm[3])) for lm in landmarks]

        for si, ei in POSE_CONNECTIONS:
            if pts[si][2] < 0.3 or pts[ei][2] < 0.3:
                continue
            cv2.line(frame, pts[si][:2], pts[ei][:2], _P2_LINE, 2, cv2.LINE_AA)

        for px, py, vis in pts:
            if vis < 0.3:
                continue
            cv2.circle(frame, (px, py), 4, _P2_PT,    -1, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 4, (0, 0, 0),  1, cv2.LINE_AA)

        # Label "P2" na trupu
        torso_pts = [(pts[i][0], pts[i][1]) for i in [11, 12, 23, 24] if pts[i][2] >= 0.3]
        if torso_pts:
            tx = int(sum(p[0] for p in torso_pts) / len(torso_pts))
            ty = int(sum(p[1] for p in torso_pts) / len(torso_pts))
            _text_bg(frame, "P2", tx - 10, ty, _FONT_MD, _P2_BOX, thick=2)

    # ── helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _torso_center_px(
        raw_lm: np.ndarray | None,
        frame_w: int,
        frame_h: int,
    ) -> tuple[int, int] | None:
        """Střed torsa v pixelech z raw_lm (klouby 11,12,23,24). None pokud nelze."""
        if raw_lm is None:
            return None
        idxs = [i for i in (11, 12, 23, 24) if raw_lm[i, 3] >= 0.2]
        if not idxs:
            return None
        cx = int(np.mean([raw_lm[i, 0] for i in idxs]) * frame_w)
        cy = int(np.mean([raw_lm[i, 1] for i in idxs]) * frame_h)
        return (cx, cy)

    def _draw_scores_at(
        self,
        frame: np.ndarray,
        lines: list[tuple[str, tuple]],
        anchor_x: int,
        anchor_y: int,
        dot_color: tuple,
    ) -> None:
        """Kreslí seznam (text, barva) vertikálně od anchor bodu. Malá tečka = anchor."""
        lh     = _lh(_FONT_SM)
        pad    = 4
        max_w  = max(
            cv2.getTextSize(t, _FONT, _FONT_SM, 1)[0][0] for t, _ in lines
        ) if lines else 80
        panel_h = len(lines) * lh + pad * 2
        panel_w = max_w + pad * 2
        px = anchor_x + 8
        py = anchor_y - panel_h // 2  # vycentrováno kolem středu torsa
        # Ořez ke kraji snímku
        h, w = frame.shape[:2]
        if px + panel_w > w:
            px = anchor_x - panel_w - 8
        py = max(lh, min(py, h - panel_h - 4))

        # Poloprůhledné pozadí
        overlay = frame.copy()
        cv2.rectangle(
            overlay,
            (px - pad, py - pad),
            (px + panel_w, py + panel_h),
            (0, 0, 0), -1,
        )
        cv2.addWeighted(overlay, 0.60, frame, 0.40, 0, frame)

        # Text
        for i, (text, color) in enumerate(lines):
            cv2.putText(
                frame, text,
                (px, py + pad + (i + 1) * lh - 4),
                _FONT, _FONT_SM, color, 1, cv2.LINE_AA,
            )

        # Malá tečka u středu torsa
        cv2.circle(frame, (anchor_x, anchor_y), 4, dot_color, -1, cv2.LINE_AA)
        cv2.circle(frame, (anchor_x, anchor_y), 4, (0, 0, 0),  1, cv2.LINE_AA)
        # Spojovací čára
        line_target_x = px if px > anchor_x else px + panel_w
        cv2.line(
            frame,
            (anchor_x, anchor_y),
            (line_target_x, anchor_y),
            dot_color, 1, cv2.LINE_AA,
        )

    # ── P1 Scores panel ───────────────────────────────────────────────────────

    def _draw_p1_scores(
        self,
        frame: np.ndarray,
        result: dict,
        raw_lm: np.ndarray | None,
    ) -> None:
        """
        Scores panel pro Person 1 – ukotvený na středu torsa.
        Zobrazuje: tracker_pres | frame_conf | 1-motion_sim | kin_score | appear | final_conf | present
        """
        h, w = frame.shape[:2]

        final_conf     = result.get("final_conf",       0.0)
        pres_prob      = result.get("presence_prob",    0.0)
        sim_score      = result.get("sim_score",        0.0)
        appear         = result.get("appearance_score", 1.0)
        kin_score      = result.get("kin_score",        1.0)
        person_present = result.get("person_present",   False)
        track_info     = result.get("track_info",       {})
        ghost          = track_info.get("ghost_active",  False)

        # Per-frame pose confidence: průměrná visibility klíčových kloubů (ramena + kyčle)
        _KEY_IDX_VIS = [11, 12, 23, 24]
        if raw_lm is not None:
            frame_conf = float(np.mean([raw_lm[i, 3] for i in _KEY_IDX_VIS]))
        else:
            frame_conf = 0.0

        present_color = (0, 230, 80) if person_present else (0, 80, 255)
        conf_color    = (0, 255, 180) if final_conf >= 0.30 else (80, 80, 255)

        _SCALE_DEBUG = True   # ← přepni na False pro skrytí per-segment řádků

        pose_suspicious  = result.get("pose_suspicious", False)
        pose_param_score = result.get("pose_param", 0.0)
        pose_len_score   = result.get("pose_len_score", 0.0)
        pose_ang_score   = result.get("pose_ang_score", 0.0)
        pose_scale_err   = result.get("pose_scale_err", 0.0)
        pose_scale_detail = result.get("pose_scale_detail", {})
        susp_color     = (0, 60, 220) if pose_suspicious else (0, 230, 80)
        scale_color    = (0, 60, 220) if pose_scale_err > 0.30 else (0, 230, 80)

        # Jump L1 debug: delta_y vs threshold_amp v pixelech
        jd = getattr(self, "_last_jump_detector", None)
        jump_lines: list[tuple[str, tuple]] = []
        if jd is not None:
            dy_px  = int(jd.last_delta_y       * h)
            thr_px = int(jd.last_threshold_amp * h)
            ok     = dy_px >= thr_px
            jcolor = (0, 230, 80) if ok else (0, 80, 255)
            reason = jd.last_fail_reason if jd.last_fail_reason else "null"
            reason_color = (0, 230, 80) if not jd.last_fail_reason else (0, 80, 255)
            mse_ok    = jd.last_fit_error <= jd.max_fit_error
            mse_color = (0, 230, 80) if mse_ok else (0, 80, 255)
            lin4_err  = jd.last_lin4_err
            fifth_err = jd.last_fifth_err
            outlier_ok    = not (lin4_err > 1e-6 and fifth_err > jd.outlier_err_ratio * lin4_err)
            outlier_color = (0, 230, 80) if outlier_ok else (0, 80, 255)
            jump_lines = [
                (f"delta_y vs thrsh: {dy_px} > {thr_px}", jcolor),
                (f"mse > fit: {jd.last_fit_error:.4f} > {jd.max_fit_error:.4f}", mse_color),
                (f"4lin_err vs fifth_err: {lin4_err:.4f} vs {fifth_err:.4f}", outlier_color),
                (f"reason = {reason}",                     reason_color),
            ]

        lines = [
            (f"tracker_pres:{pres_prob:.2f}",                          _P1_BOX),
            (f"frame_conf:  {frame_conf:.2f}",                          _P1_BOX),
            (f"1-mot_sim:   {1.0 - sim_score:.2f}",                    _P1_BOX),
            (f"kin_score:   {kin_score:.2f}",       _COLOR_GHOST if ghost else _P1_BOX),
            (f"appear:      {appear:.2f}",                              _P1_BOX),
            (f"final_conf:  {final_conf:.2f}",                          conf_color),
            (f"present:     {'YES' if person_present else 'NO'}",       present_color),
            (f"pose_param:  {pose_param_score:.2f} {'SUSP' if pose_suspicious else 'ok'}", susp_color),
            (f"  len:{pose_len_score:.2f} ang:{pose_ang_score:.2f}",                       susp_color),
            (f"scale_err:   {pose_scale_err:.3f}",                                         scale_color),
            (f"  sc: " + " ".join(f"{k[:4]}={v['rel']:.2f}" if isinstance(v, dict) else f"{k[:4]}={v:.2f}" for k, v in pose_scale_detail.items()), scale_color),
        ]
        if _SCALE_DEBUG and pose_scale_detail:
            _SEG_SHORT = {
                "torso_h": "torso_h", "shoulder_w": "shou_w",
                "left_upper_arm": "L_uarm", "right_upper_arm": "R_uarm",
                "left_thigh": "L_thigh", "right_thigh": "R_thigh",
            }
            for seg_key, seg_val in pose_scale_detail.items():
                if isinstance(seg_val, dict):
                    rel_v = seg_val.get("rel", 0.0)
                    exp_v = seg_val.get("exp", 0.0)
                else:
                    rel_v, exp_v = seg_val, 0.0
                label = _SEG_SHORT.get(seg_key, seg_key)
                lines.append((f"  {label}: rel={rel_v:.3f} exp={exp_v:.3f}", scale_color))
        lines += jump_lines

        center = self._torso_center_px(raw_lm, w, h)
        if center is None:
            center = (6, h // 2)

        self._draw_scores_at(frame, lines, center[0], center[1], _P1_BOX)

    # ── P2 Scores panel ───────────────────────────────────────────────────────

    def _draw_p2_scores(
        self,
        frame: np.ndarray,
        result: dict,
        raw_lm: np.ndarray | None,
    ) -> None:
        """Kompaktní scores panel pro Person 2 – ukotvený na středu torsa."""
        h, w = frame.shape[:2]

        final_conf     = result.get("final_conf",       0.0)
        pres_prob      = result.get("presence_prob",    0.0)
        sim_score      = result.get("sim_score",        0.0)
        appear         = result.get("appearance_score", 1.0)
        kin_score      = result.get("kin_score",        1.0)
        person_present = result.get("person_present",   False)

        present_color = (0, 230, 80) if person_present else (0, 80, 255)

        lines = [
            (f"pose_conf:  {pres_prob:.2f}",                    _P2_BOX),
            (f"1-mot_sim:  {1.0 - sim_score:.2f}",              _P2_BOX),
            (f"kin_score:  {kin_score:.2f}",                    _P2_BOX),
            (f"appear:     {appear:.2f}",                       _P2_BOX),
            (f"final_conf: {final_conf:.2f}",                   _P2_BOX),
            (f"present:    {'YES' if person_present else 'NO'}", present_color),
        ]

        center = self._torso_center_px(raw_lm, w, h)
        if center is None:
            center = (w - 120, h // 2)

        self._draw_scores_at(frame, lines, center[0], center[1], _P2_BOX)

    # ── Motion ROI overlay ────────────────────────────────────────────────────

    def _draw_motion_limbs(self, frame: np.ndarray, motion_info: dict) -> None:
        """Nakreslí per-limb ROI boxy a badge se SIM skóre (P1, vpravo dole)."""
        if not motion_info:
            return

        h, w     = frame.shape[:2]
        limb_dbg = motion_info.get("limb_debug", {})
        total    = motion_info.get("motion_score", 0.0)
        dynamic  = motion_info.get("region_dynamic", True)

        dyn_color = (0, 200, 80) if dynamic else (0, 120, 255)

        # ROI boxy (jen pokud dynamic)
        if dynamic:
            for name, info in limb_dbg.items():
                roi = info.get("roi_orig")
                if roi is None:
                    continue
                lx1, ly1, lx2, ly2 = roi
                cv2.rectangle(frame, (lx1, ly1), (lx2, ly2),
                              _LIMB_COLORS.get(name, (200, 200, 200)), 1, cv2.LINE_AA)

        # Badge (vpravo dole)
        n_limbs = len(limb_dbg)
        badge_w = 185
        badge_h = 20 + n_limbs * 15 + 20
        bx1 = w - badge_w - 8
        by2 = h - 8
        bx2 = w - 8
        by1 = by2 - badge_h

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), dyn_color, 1)

        cy = by1 + 15
        cv2.putText(frame, f"SIM: {total:.4f}", (bx1 + 6, cy), _FONT, 0.45,
                    (180, 180, 180), 1, cv2.LINE_AA)
        cy += 17

        for name in ["torso", "left_arm", "right_arm", "left_leg", "right_leg"]:
            if name not in limb_dbg:
                continue
            sim = limb_dbg[name].get("sim", 0.0)
            color = _LIMB_COLORS.get(name, (180, 180, 180))
            label = _LIMB_SHORT.get(name, name[:5].upper())
            cv2.putText(frame, f"  {label}: {sim:.3f}", (bx1 + 6, cy), _FONT, 0.38,
                        color, 1, cv2.LINE_AA)
            cy += 15

        cv2.putText(frame, "DYNAMIC" if dynamic else "STATIC", (bx1 + 6, cy + 4),
                    _FONT, 0.48, dyn_color, 1, cv2.LINE_AA)

    # ── Globální panel ────────────────────────────────────────────────────────

    def _draw_global_panel(
        self,
        frame: np.ndarray,
        action: str | None,
        timestamp_ms: float,
        current_fps: float,
        p1: dict | None,
    ) -> None:
        """
        Debug panel vpravo nahoře:
            ID:      timestamp ms
            ACTION:  klasifikovaná akce nebo NULL
            PRESENT: YES / NO
            FPS:     aktuální FPS
        """
        h, w = frame.shape[:2]

        person_present = p1.get("person_present", False) if p1 else False
        valid_pose     = p1.get("valid_pose", False)     if p1 else False

        panel_w = 220
        panel_h = 128
        px = w - panel_w - 8
        py = 8
        lh = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + panel_w, py + panel_h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + panel_w, py + panel_h), (60, 60, 60), 1)

        tx = px + 8
        ty = py + lh

        # Řádek 1: Timestamp
        cv2.putText(frame, f"ID:      {timestamp_ms:.0f} ms", (tx, ty), _FONT, _FONT_SM,
                    (150, 220, 255), 1, cv2.LINE_AA)
        ty += lh

        # Řádek 2: ACTION
        action_label = action if action else "NULL"
        action_color = _ACTION_COLORS.get(action, _ACTION_COLORS[None])
        cv2.putText(frame, "ACTION:  ", (tx, ty), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
        label_x = tx + cv2.getTextSize("ACTION:  ", _FONT, _FONT_SM, 1)[0][0]
        cv2.putText(frame, action_label, (label_x, ty), _FONT, _FONT_SM,
                    action_color, 2, cv2.LINE_AA)
        ty += lh

        # Řádek 3: JUMP_CL (výsledek fyzikálního detektoru)
        jd = getattr(self, "_last_jump_detector", None)
        jump_cl = jd.last_is_jump if jd is not None else False
        jump_cl_text  = "TRUE" if jump_cl else "FALSE"
        jump_cl_color = (0, 230, 80) if jump_cl else (0, 60, 220)
        cv2.putText(frame, "jump_cl: ", (tx, ty), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
        lx = tx + cv2.getTextSize("jump_cl: ", _FONT, _FONT_SM, 1)[0][0]
        cv2.putText(frame, jump_cl_text, (lx, ty), _FONT, _FONT_SM, jump_cl_color, 2, cv2.LINE_AA)
        ty += lh

        # Řádek 4: FINAL (odvozený label)
        if person_present:
            if jump_cl and action == "acrobatics":
                final_label = "acrobatics"
                final_color = _ACTION_COLORS.get("acrobatics", (0, 200, 255))
            elif jump_cl:
                final_label = "jump"
                final_color = _ACTION_COLORS.get("jump", (0, 200, 255))
            else:
                final_label = "normal"
                final_color = _ACTION_COLORS.get("normal", (180, 180, 180))
        else:
            final_label = "---"
            final_color = (100, 100, 100)
        cv2.putText(frame, "final:   ", (tx, ty), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
        lx = tx + cv2.getTextSize("final:   ", _FONT, _FONT_SM, 1)[0][0]
        cv2.putText(frame, final_label, (lx, ty), _FONT, _FONT_SM, final_color, 2, cv2.LINE_AA)
        ty += lh

        # Řádek 3: PRESENT
        present_text  = "YES" if person_present else "NO"
        present_color = (0, 230, 80) if person_present else (0, 60, 220)
        cv2.putText(frame, "PRESENT: ", (tx, ty), _FONT, _FONT_SM, (200, 200, 200), 1, cv2.LINE_AA)
        lx = tx + cv2.getTextSize("PRESENT: ", _FONT, _FONT_SM, 1)[0][0]
        cv2.putText(frame, present_text, (lx, ty), _FONT, _FONT_SM,
                    present_color, 2, cv2.LINE_AA)
        ty += lh

        # Řádek 4: FPS
        cv2.putText(frame, f"FPS:     {current_fps:.1f}", (tx, ty), _FONT, _FONT_SM,
                    (150, 220, 255), 1, cv2.LINE_AA)
