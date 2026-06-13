"""
jump_detector.py
----------------
Fyzikální detekce skoku (jump) z trajektorie těžiště.

Pipeline:
  1. Výpočet těžiště (hip_center.y) + výška trupu z landmarků
  2. Korekce pohybu kamery (cv2.phaseCorrelate)
  3. Temporální buffer [(t_sec, y_corrected, torso_h), ...]
  4. Fit paraboly: y = at² + bt + c  (np.polyfit)
  5. Fyzikální validace (5 vrstev):
       L1 - delta_y > 0.1 * torso_height  (amplituda relativní k tělu)
       L2 - fit MSE < threshold + |a| dostatečné  (konzistentní zrychlení)
       L3 - existuje monotónní segment ≥3 bodů  (skutečný pohyb)
       L4 - max|velocity| > 0.15 * torso_height/s  (dynamika)
       L5 - bez symetrie, bez air-time podmínek
  6. Kombinace s klasifikátorem (AND logika)
  7. Reset bufferu po detekovaném skoku
"""

import logging
from collections import deque

import cv2
import numpy as np

from pose_detector import LANDMARK_INDEX

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# VRSTVA 1 – Výpočet těžiště
# ─────────────────────────────────────────────────────────────────────────────

def compute_hip_center_y(landmarks: np.ndarray) -> float:
    """Vrátí Y souřadnici středu boků v normalizovaných souřadnicích."""
    I = LANDMARK_INDEX
    return (float(landmarks[I["left_hip"], 1]) + float(landmarks[I["right_hip"], 1])) / 2.0


def compute_torso_height(landmarks: np.ndarray) -> float:
    """
    Vrátí výšku trupu (vzdálenost střed ramen – střed boků) v norm. souřadnicích.
    Používá se jako referenční velikost těla pro relativní thresholdy.
    """
    I = LANDMARK_INDEX
    sh_y  = (float(landmarks[I["left_shoulder"], 1]) + float(landmarks[I["right_shoulder"], 1])) / 2.0
    hip_y = (float(landmarks[I["left_hip"],      1]) + float(landmarks[I["right_hip"],      1])) / 2.0
    # abs() protože při stojce může být sh_y > hip_y
    return max(abs(sh_y - hip_y), 0.01)  # floor 0.01 zabrání dělení nulou


# ─────────────────────────────────────────────────────────────────────────────
# VRSTVA 2 – Korekce pohybu kamery
# ─────────────────────────────────────────────────────────────────────────────

class CameraMotionCorrector:
    """
    Odhadne vertikální pohyb kamery pomocí cv2.phaseCorrelate a kumulativně
    koriguje y_position.

    phaseCorrelate pracuje s float32 snímky ve frekvencní doméně (FFT).
    Pro rychlost pracuje s šedotonovou zmenšenou verzí snímku.

    Parametry:
        correction_scale  -- škálovací faktor: jak moc věříme fázovému odhadu
                             (0.0 = bez korekce, 1.0 = plná korekce)
        resize_factor     -- faktor zmenšení snímku pro odhad (rychlost)
        max_shift_per_frame -- maximální přijatelný posun za snímek [norm. 0–1]
                               větší posun se ignoruje (artifact, rychlý pan)
    """

    def __init__(
        self,
        correction_scale: float = 1.0,
        resize_factor: float = 0.25,
        max_shift_per_frame: float = 0.05,
    ):
        self.correction_scale    = correction_scale
        self.resize_factor       = resize_factor
        self.max_shift_per_frame = max_shift_per_frame

        self._prev_gray: np.ndarray | None = None
        self._cumulative_shift_y: float    = 0.0

    def update(self, frame: np.ndarray) -> float:
        """
        Zpracuje aktuální snímek, vrátí y_corrected_offset.

        Vrátí:
            Kumulativní korekční posun v normalizovaných souřadnicích (subtrahuješ od y_raw).
        """
        h, w = frame.shape[:2]
        new_h = max(1, int(h * self.resize_factor))
        new_w = max(1, int(w * self.resize_factor))

        # Šedotonový zmenšený snímek (float32 pro phaseCorrelate)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (new_w, new_h)).astype(np.float32)

        if self._prev_gray is not None:
            try:
                # phaseCorrelate vrací (shift, response)
                (shift_x, shift_y), response = cv2.phaseCorrelate(self._prev_gray, small)
            except cv2.error:
                shift_y = 0.0
                response = 0.0

            # Převod: pixely small → normalizované souřadnice původního snímku
            shift_y_norm = (shift_y / new_h) * (new_h / h)  # = shift_y / h (v orig. px)

            # Odmítnout extrémní posuny (kamera se nezachvěje o 5 % výšky za snímek)
            if abs(shift_y_norm) <= self.max_shift_per_frame and response > 0.1:
                self._cumulative_shift_y += shift_y_norm * self.correction_scale

        self._prev_gray = small
        return self._cumulative_shift_y

    def reset(self) -> None:
        """Reset při přechodu na nové video."""
        self._prev_gray = None
        self._cumulative_shift_y = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HLAVNÍ DETEKTOR
# ─────────────────────────────────────────────────────────────────────────────

class JumpDetector:
    """
    Fyzikální detektor skoku na základě 5-vrstvové analýzy trajektorie těžiště.

    Thresholdy amplitudy a rychlosti jsou relativní vůči výšce trupu, takže
    fungují pro různé vzdálenosti kamery.

    Parametry:
        buffer_size          -- počet snímků v temporálním bufferu (6–10)
        amplitude_factor     -- L1: min. delta_y = amplitude_factor * torso_height
        min_abs_a            -- L2: min. |koeficient a| paraboly (zrychlení)
        max_fit_error        -- L2: max. MSE fitu paraboly (konzistence)
        min_monotone_len     -- L3: min. délka monotónního segmentu
        velocity_factor      -- L4: min. max_velocity = velocity_factor * torso_h/s
        camera_correction    -- zda použít korekci pohybu kamery
    """

    def __init__(
        self,
        buffer_size: int         = 5,
        amplitude_factor: float  = 0.5,
        min_abs_a: float         = 0.10,
        max_fit_error: float     = 0.0020,
        min_monotone_len: int    = 3,
        velocity_factor: float   = 0.15,
        camera_correction: bool  = True,
    ):
        self.buffer_size      = buffer_size
        self.amplitude_factor = amplitude_factor
        self.min_abs_a        = min_abs_a
        self.max_fit_error    = max_fit_error
        self.min_monotone_len = min_monotone_len
        self.velocity_factor  = velocity_factor

        # Buffer prvků: dict s klíči t_sec, y_corrected, torso_h, hip_x, hip_y
        self._buffer: deque[dict] = deque(maxlen=buffer_size)

        self._camera = CameraMotionCorrector() if camera_correction else None

        # Diagnostické hodnoty (pro debug panel)
        self.last_is_jump:       bool  = False
        self.last_a:             float = 0.0
        self.last_delta_y:       float = 0.0
        self.last_threshold_amp: float = 0.0
        self.last_torso_h:       float = 0.0
        self.last_max_vel:       float = 0.0
        self.last_fit_error:     float = 0.0
        self.last_fail_reason:   str   = ""
        self.last_lin4_err:      float = 0.0
        self.last_fifth_err:     float = 0.0
        self.outlier_err_ratio:  float = 4.0   # fifth_err > ratio * lin4_err → outlier

    # ── Platný snímek ────────────────────────────────────────────────────────

    def update(
        self,
        frame: np.ndarray,
        landmarks: np.ndarray,
        timestamp_ms: float,
    ) -> bool:
        """
        Zpracuje platný snímek: uloží validní záznam do bufferu.
        """
        t_sec = timestamp_ms / 1000.0

        I = LANDMARK_INDEX
        y_raw       = compute_hip_center_y(landmarks)
        torso_h     = compute_torso_height(landmarks)
        hip_x_raw   = (float(landmarks[I["left_hip"], 0]) + float(landmarks[I["right_hip"], 0])) / 2.0

        if self._camera is not None:
            camera_shift = self._camera.update(frame)
            y_corrected  = y_raw - camera_shift
        else:
            y_corrected = y_raw

        self._buffer.append({
            "valid":       True,
            "t_sec":       t_sec,
            "y_corrected": y_corrected,
            "torso_h":     torso_h,
            "hip_x":       hip_x_raw,
            "hip_y":       y_raw,
        })

        return self._evaluate()

    # ── Chybějící snímek ─────────────────────────────────────────────────────

    def update_missing(self, timestamp_ms: float) -> bool:
        """
        Zaregistruje snímek bez platné pózy (uloží neplatný slot do bufferu).
        Buffer tak zachovává správné časové pozice.
        """
        self._buffer.append({
            "valid":       False,
            "t_sec":       timestamp_ms / 1000.0,
            "y_corrected": 0.0,
            "torso_h":     0.0,
            "hip_x":       0.0,
            "hip_y":       0.0,
        })
        return self._evaluate()

    # ── Společné vyhodnocení ──────────────────────────────────────────────────

    def _evaluate(self) -> bool:
        if len(self._buffer) < self.buffer_size:
            self.last_is_jump       = False
            self.last_fail_reason   = "buf"
            self.last_delta_y       = 0.0
            self.last_threshold_amp = 0.0
            return False

        buf = list(self._buffer)

        # Krajní body (0 a -1) musí být platné
        if not buf[-1]["valid"]:
            self.last_is_jump       = False
            self.last_fail_reason   = "buf"
            self.last_delta_y       = 0.0
            self.last_threshold_amp = 0.0
            return False

        # Oba prostřední nesměí být zároveň neplatné
        middle = buf[1:-1]
        if middle and all(not e["valid"] for e in middle):
            self.last_is_jump       = False
            self.last_fail_reason   = "buf"
            self.last_delta_y       = 0.0
            self.last_threshold_amp = 0.0
            return False

        is_jump = self._analyse_trajectory(buf)
        self.last_is_jump = is_jump
        return is_jump

    # ── Analýza trajektorie ───────────────────────────────────────────────────

    def _analyse_trajectory(self, buf: list[dict]) -> bool:
        """
        5-vrstvá fyzikální analýza.
        Pracuje pouze s validními body z bufferu.
        Vrátí True pouze pokud VŠECHNY vrstvy projdou.
        """
        pts     = [e for e in buf if e["valid"]]
        t_arr   = np.array([p["t_sec"]       for p in pts], dtype=np.float64)
        y_arr   = np.array([p["y_corrected"] for p in pts], dtype=np.float64)
        torso_h = float(np.median([p["torso_h"] for p in pts]))

        t_norm = t_arr - t_arr[0]   # relativní čas (první bod = 0)
        dt     = np.diff(t_norm)    # časové kroky mezi snímky

        # ── L1: Minimální amplituda ──────────────────────────────────────
        delta_y = float(np.max(y_arr) - np.min(y_arr))
        self.last_delta_y = delta_y
        threshold_amp = self.amplitude_factor * torso_h
        self.last_threshold_amp = threshold_amp
        self.last_torso_h       = torso_h
        if delta_y < threshold_amp:
            self.last_fail_reason = "amp"
            return False

        # ── L2: Fit paraboly – konzistentní zrychlení ────────────────────
        try:
            coeffs  = np.polyfit(t_norm, y_arr, 2)
            y_fit   = np.polyval(coeffs, t_norm)
            mse     = float(np.mean((y_arr - y_fit) ** 2))
        except (np.linalg.LinAlgError, ValueError):
            self.last_fail_reason = "fit"
            return False

        a = float(coeffs[0])
        self.last_a         = a
        self.last_fit_error = mse

        if abs(a) < self.min_abs_a:
            self.last_fail_reason = "curve"
            return False
        if mse > self.max_fit_error:
            self.last_fail_reason = "fit"
            return False

        # ── L3: Outlier fit – poslední bod vs lineární fit prvních N-1 ──
        self.last_lin4_err  = 0.0
        self.last_fifth_err = 0.0
        if len(t_norm) >= 3:
            t_head = t_norm[:-1]
            y_head = y_arr[:-1]
            # Outlier check má smysl jen pokud máme dost bodů pro spolehlivý
            # lineární fit – s méně než 3 body je fit vždy (téměř) dokonalý
            # a lin4_err ≈ 0, takže by check nesprávně prošel nebo zamítl.
            if len(t_head) >= 3:
                try:
                    lin_coeffs = np.polyfit(t_head, y_head, 1)
                    y_head_fit  = np.polyval(lin_coeffs, t_head)
                    lin4_err    = float(np.mean(np.abs(y_head - y_head_fit)))
                    fifth_err   = float(abs(y_arr[-1] - np.polyval(lin_coeffs, t_norm[-1])))
                    self.last_lin4_err  = lin4_err
                    self.last_fifth_err = fifth_err
                    if lin4_err > 1e-6 and fifth_err > 10.0 * lin4_err:
                        # Extrémní outlier → označíme poslední bod jako nevalidní
                        self._buffer[-1]["valid"] = False
                    if lin4_err > 1e-6 and fifth_err > self.outlier_err_ratio * lin4_err:
                        self.last_fail_reason = "outlier"
                        return False
                except (np.linalg.LinAlgError, ValueError):
                    pass

        # ── L4: Minimální rychlost ───────────────────────────────────────
        dy = np.diff(y_arr)
        # Ochrana proti nulovým dt
        safe_dt = np.where(dt > 1e-6, dt, 1e-6)
        velocity = dy / safe_dt
        max_vel  = float(np.max(np.abs(velocity)))
        self.last_max_vel = max_vel

        threshold_vel = self.velocity_factor * torso_h  # per second (dt je v sekundách)
        if max_vel < threshold_vel:
            self.last_fail_reason = "vel"
            return False

        self.last_fail_reason = ""
        return True

    @staticmethod
    def _has_monotone_segment(y: np.ndarray, min_len: int) -> bool:
        """
        Vrátí True pokud existuje sekvence min_len po sobě jdoucích bodů,
        které jsou monotónně rostoucí NEBO monotónně klesající.

        Toleruje jitter: porovnává jen sousední dvojice (sign(diff)).
        """
        if len(y) < min_len:
            return True   # nedostatek dat → nezamítat

        signs = np.sign(np.diff(y))  # +1 roste, -1 klesá, 0 stejné

        # Počítáme délku aktuálního monotónního běhu
        run = 1
        for i in range(1, len(signs)):
            if signs[i] != 0 and signs[i] == signs[i - 1]:
                run += 1
                if run >= min_len:
                    return True
            else:
                run = 1

        return False

    # ── Diagnostika pro debug panel ───────────────────────────────────────────

    def get_debug_info(self) -> dict:
        """Vrátí slovník s hodnotami pro debug panel."""
        return {
            "is_jump":       self.last_is_jump,
            "a":             self.last_a,
            "delta_y":       self.last_delta_y,
            "threshold_amp": self.last_threshold_amp,
            "torso_h":       self.last_torso_h,
            "max_vel":       self.last_max_vel,
            "fit_error":     self.last_fit_error,
        }

    # ── Trajectory data pro vizualizaci ───────────────────────────────────────

    def get_trajectory_for_viz(self) -> dict | None:
        """
        Vrátí data pro vizualizaci trajektorie skoku.

        Klíče výstupu:
          hip_xy      : list[(x_norm, y_norm)]  – surové hip_center ze snímků
          y_corrected : list[float]              – výška korigovaná kamerou (0=top, 1=bottom)
          t_norm      : list[float]              – normalizovaný čas 0..1
          parabola_abc: (a, b, c) | None         – koeficienty fitu, nebo None při chybě
          is_jump     : bool
        """
        buf = list(self._buffer)
        n = len(buf)
        if n < 2:
            return None

        hip_xy      = [(e["hip_x"], e["hip_y"]) for e in buf if e["valid"]]
        y_corrected = [e["y_corrected"] for e in buf if e["valid"]]
        t_norm      = [i / (len(hip_xy) - 1) if len(hip_xy) > 1 else 0.0 for i in range(len(hip_xy))]

        parabola_abc: tuple[float, float, float] | None = None
        try:
            import numpy as np
            coeffs = np.polyfit(t_norm, y_corrected, 2)
            parabola_abc = (float(coeffs[0]), float(coeffs[1]), float(coeffs[2]))
        except Exception:
            pass

        return {
            "hip_xy":       hip_xy,
            "y_corrected":  y_corrected,
            "t_norm":       t_norm,
            "parabola_abc": parabola_abc,
            "is_jump":      self.last_is_jump,
        }

    # ── Snapshot / Restore ─────────────────────────────────────────────────

    def snapshot(self) -> object:
        """
        Vrátí záložní kopii bufferu (hloubková kopie).
        Použij těsně před volání update() / update_missing() daného snímku.
        """
        import copy
        return copy.deepcopy(self._buffer)

    def restore(self, snap: object) -> None:
        """
        Obnoví buffer ze zálohy pořízené metodou snapshot().
        Zavolej před opakovaným zpracováním snímku (např. po fallbacku).
        """
        from collections import deque
        self._buffer = deque(snap, maxlen=self.buffer_size)

    # ── Reset ──────────────────────────────────────────────────────────────────

    def reset(self) -> None:
        """Reset bufferu a stavu (při přechodu na nové video)."""
        self._buffer.clear()
        self.last_is_jump       = False
        self.last_a             = 0.0
        self.last_delta_y       = 0.0
        self.last_threshold_amp = 0.0
        self.last_torso_h       = 0.0
        self.last_max_vel       = 0.0
        self.last_fit_error     = 0.0
        self.last_fail_reason   = ""
        if self._camera:
            self._camera.reset()
