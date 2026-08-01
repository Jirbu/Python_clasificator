"""
flip_detector.py
----------------
Detekce plného vertikálního přetočení (salto) z torso úhlu.

Princip:
  0) Pseudo-online rozhodování o outlierech přes Kalman: vstup se zpozdí o
     `lookahead_frames` snímků, takže má detektor pro každý snímek k dispozici
     i pár budoucích hodnot. Místo porovnávání syrových úhlů (které je u plné
     otočky nerozeznatelné od "skoku a návratu" – po 360° se úhel taky vrátí
     na start) se společně vyzkouší VŠECHNY kombinace přijmout/zahodit napříč
     centrem i budoucími snímky a vybere se ta s nejnižší celkovou "cenou"
     (součet čtverců inovací + penalizace za zahození). Tím se odhalí i víc
     osamocených vadných snímků blízko sebe (typicky mid-air pose glitch),
     aniž by to zaměňovalo skutečnou rychlou/plnou rotaci za outlier.
  1) Kalmanův filtr (stav = [unwrapped_angle, angular_velocity]) vyhlazuje
     zbylý šum a řeší wrap-around (0<->360). Measurement noise se škáluje podle
     confidence vstupu, navíc je tu gate proti hrubým outlierům jednoho snímku.
  2) Přes klouzavé časové okno (window_ms) se sleduje kumulativní rotace
     (unwrapped_angle[t] - unwrapped_angle[t-window]). Pokud je v okně
     dostatečně velká, dostatečně monotónní (nejde o kmitání tam a zpět)
     a dostatečně důvěryhodná, roste flip_probability.

Výpočetně jde jen o pár skalárních Kalman kroků navíc na snímek (lookahead
hypotézy) – pořád zanedbatelné vedle běhu pose modelu.

Pozn.: kvůli lookahead bufferu vrací update() hodnotu zpožděnou o
`lookahead_frames` snímků oproti právě předanému vstupu (pseudo-online).
"""

from __future__ import annotations

import itertools
import math
from collections import deque

KalmanState = tuple[float, float, float, float, float]  # x_angle, x_vel, p00, p01, p11


def angle_diff(a: float, b: float) -> float:
    """Rozdíl úhlů <-180,180>: nejkratší cesta z a do b."""
    return ((b - a + 180.0) % 360.0) - 180.0


def _smoothstep(x: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 1.0 if x >= hi else 0.0
    t = min(max((x - lo) / (hi - lo), 0.0), 1.0)
    return t * t * (3.0 - 2.0 * t)


class FlipDetector:

    def __init__(
        self,
        lookahead_frames: int = 3,       # kolik budoucích snímků se čeká na potvrzení (pseudo-online zpoždění)
        window_ms: float = 950.0,       # délka klouzavého okna pro kumulativní rotaci (~délka reálného salta, ať pravděpodobnost rychle spadne po dopadu)
        rot_threshold_lo: float = 220.0,  # od kolika stupňů v okně roste pravděpodobnost
        rot_threshold_hi: float = 300.0,  # od kolika stupňů je pravděpodobnost saturovaná (~1 otočka)
        monotonic_lo: float = 0.45,      # pod touto mírou "přímočarosti" rotace se nepočítá jako flip
        monotonic_hi: float = 0.75,
        conf_lo: float = 0.15,           # pod touto průměrnou confidence v okně je flip nevěrohodný
        conf_hi: float = 0.5,            # nad touto průměrnou confidence už confidence dál nelimituje
        q_angle: float = 4.0,            # process noise (deg^2 / s), nejistota ve vývoji úhlu
        q_vel: float = 150000.0,         # process noise (deg^2/s^2) – dost vysoké, aby filtr po klidu rychle "uvěřil" náhlé rotaci
        r_min: float = 300.0,            # measurement noise při confidence=1 (~17° std – confidence z visibility je skoro binární, nekryje geometrickou nejistotu 2bodového úhlu)
        r_max: float = 4000.0,           # measurement noise při confidence=0
        outlier_gate_deg: float = 150.0, # minimální gate v stupních (musí unést i rychlé salto, ne jen šum)
        outlier_gate_sigma: float = 5.0, # gate = max(outlier_gate_deg, sigma * innovation_std)
        reject_penalty_nis: float = 30.0,  # "cena" zahození měření v joint vyhodnocení (v NIS jednotkách, ~3-sigma gate)
        reject_penalty_nis_no_future: float = 6.0,  # přísnější práh, když je celé budoucí okno prázdné (žádná reálná korekce k dispozici)
        max_vel_dps: float = 900.0,      # tvrdý strop úhlové rychlosti (deg/s) – drží v*dt bezpečně pod 180° (hranice, kde je +delta k nerozeznání od -delta), jinak filtr může "aliasovat" na půl/celou otočku za snímek a vypadat stejně jako klid
        max_gap_ms: float = 400.0,       # po delší mezeře bez platné detekce se rychlost vynuluje – extrapolovat rotaci přes "slepé" období není fyzikálně opodstatněné
        dt_min: float = 1.0 / 120.0,
        dt_max: float = 1.0 / 5.0,
    ):
        self.lookahead_frames    = lookahead_frames
        self.window_ms          = window_ms
        self.rot_threshold_lo   = rot_threshold_lo
        self.rot_threshold_hi   = rot_threshold_hi
        self.monotonic_lo       = monotonic_lo
        self.monotonic_hi       = monotonic_hi
        self.conf_lo            = conf_lo
        self.conf_hi            = conf_hi
        self.q_angle            = q_angle
        self.q_vel              = q_vel
        self.r_min              = r_min
        self.r_max              = r_max
        self.outlier_gate_deg   = outlier_gate_deg
        self.outlier_gate_sigma = outlier_gate_sigma
        self.reject_penalty_nis = reject_penalty_nis
        self.reject_penalty_nis_no_future = reject_penalty_nis_no_future
        self.max_vel_dps        = max_vel_dps
        self.max_gap_ms         = max_gap_ms
        self.dt_min             = dt_min
        self.dt_max             = dt_max

        self._prob = 0.0
        self._net  = 0.0
        self._reset_state()

    # ------------------------------------------------------------------
    def _reset_state(self) -> None:
        self.last_ts_ms = None
        self._last_input_ts = None
        self._gap_ms = 0.0

        # stav Kalmanova filtru: unwrapped úhel [deg], úhlová rychlost [deg/s]
        self.x_angle = None
        self.x_vel   = 0.0

        # kovariance 2x2 (symetrická): p00, p01(=p10), p11
        self.p00 = 1.0e4
        self.p01 = 0.0
        self.p11 = 1.0e4

        # klouzavé okno (timestamp_ms, unwrapped_angle, confidence)
        self._window: deque[tuple[float, float, float]] = deque()

        # pseudo-online lookahead buffer (timestamp_ms, angle, confidence)
        self._buf: deque[tuple[float, float | None, float]] = deque()

    # ------------------------------------------------------------------
    def update(self, angle: float | None, confidence: float = 1.0, timestamp_ms: float | None = None) -> float:
        """
        Vrací flip_probability (float 0-1). Hodnota > 0.85 = detekovaný flip.
        Kvůli lookahead bufferu odpovídá vrácená hodnota snímku
        `lookahead_frames` zpět (dokud buffer nenaběhne, vrací poslední známou hodnotu).

        angle        – torso úhel [0,360) nebo None (žádná detekce v tomto snímku)
        confidence   – důvěryhodnost úhlu [0,1] (např. z visibility landmarků)
        timestamp_ms – čas snímku; pokud None, dopočítá se jako +33ms od posledního
        """
        if timestamp_ms is None:
            timestamp_ms = (self._last_input_ts + 33.0) if self._last_input_ts is not None else 0.0
        self._last_input_ts = timestamp_ms

        self._buf.append((timestamp_ms, angle, confidence))
        if len(self._buf) <= self.lookahead_frames:
            return self._prob

        center = self._buf.popleft()
        future_items = list(self._buf)  # zbylých `lookahead_frames` snímků "z budoucnosti"

        angle_to_use, conf_to_use = self._resolve_center(center, future_items)
        return self._process(angle_to_use, conf_to_use, center[0])

    # ------------------------------------------------------------------
    def _resolve_center(
        self,
        center: tuple[float, float | None, float],
        future_items: list[tuple[float, float | None, float]],
    ) -> tuple[float | None, float]:
        """
        Rozhodne, jestli centrální (nejstarší bufferovaný) snímek předat do
        Kalmanu jako platné měření, nebo ho přeskočit jako osamocený outlier.

        Vyzkouší SPOLEČNĚ všechny kombinace přijmout/zahodit napříč centrem
        i budoucími snímky (2^m, m = 1+lookahead_frames – řádově jednotky až
        desítky, triviální výpočet) a vybere kombinaci s nejnižší celkovou
        "cenou" (součet čtverců inovací + penalizace za každé zahození).
        Jen takhle společné vyhodnocení správně zvládne i dva blízké vadné
        snímky v jednom okně – porovnání centra jen proti "budoucnosti brané
        jako pravda" by se tou druhou vadnou hodnotou nechalo zmást.
        """
        _, angle_c, conf_c = center
        if angle_c is None:
            return None, 0.0

        if self.x_angle is None or not future_items:
            return angle_c, conf_c  # není vůči čemu porovnávat – přijmi tak, jak je

        items = [center] + future_items
        state0: KalmanState = (self.x_angle, self.x_vel, self.p00, self.p01, self.p11)
        ts0 = self.last_ts_ms

        has_future_evidence = any(a is not None for _, a, _ in future_items)
        penalty = self.reject_penalty_nis if has_future_evidence else self.reject_penalty_nis_no_future

        best_cost = None
        best_accept_center = True

        for mask in itertools.product((False, True), repeat=len(items)):
            state = state0
            last_ts = ts0
            cost = 0.0
            for (ts_i, angle_i, conf_i), accept in zip(items, mask):
                dt = min(max((ts_i - last_ts) / 1000.0, self.dt_min), self.dt_max)
                pred = self._kf_predict(state, dt)
                use_angle = angle_i if (accept and angle_i is not None) else None
                state, innovation, s = self._kf_update(pred, use_angle, conf_i)
                if use_angle is not None:
                    cost += (innovation * innovation) / max(s, 1e-6)  # NIS – "překvapivost" relativně k nejistotě predikce
                elif angle_i is not None:
                    cost += penalty  # měření existovalo, ale zahodili jsme ho
                last_ts = ts_i

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_accept_center = mask[0]

        if best_accept_center:
            return angle_c, conf_c
        return None, 0.0

    # ------------------------------------------------------------------
    def _kf_predict(self, state: KalmanState, dt: float) -> KalmanState:
        x_angle, x_vel, p00, p01, p11 = state
        angle_pred = x_angle + x_vel * dt
        vel_pred   = x_vel
        p00_pred = p00 + dt * 2.0 * p01 + dt * dt * p11 + self.q_angle * dt
        p01_pred = p01 + dt * p11
        p11_pred = p11 + self.q_vel * dt
        return (angle_pred, vel_pred, p00_pred, p01_pred, p11_pred)

    def _kf_update(
        self, pred: KalmanState, angle: float | None, confidence: float,
    ) -> tuple[KalmanState, float, float]:
        """Vrací (nový stav, inovace, S = kovariance inovace) – S umožňuje volajícímu
        normalizovat "překvapivost" měření podle aktuální nejistoty predikce
        (NIS – normalized innovation squared), místo posuzování surových stupňů."""
        angle_pred, vel_pred, p00_pred, p01_pred, p11_pred = pred

        c = min(max(confidence, 0.0), 1.0)
        r = self.r_min + (self.r_max - self.r_min) * (1.0 - c) ** 2
        s = p00_pred + r

        if angle is None:
            return (angle_pred, vel_pred, p00_pred, p01_pred, p11_pred), 0.0, s

        innovation = angle_diff(angle_pred % 360.0, angle % 360.0)  # = z_unwrapped - angle_pred

        gate = max(self.outlier_gate_deg, self.outlier_gate_sigma * math.sqrt(max(s, 1e-6)))

        if abs(innovation) > gate:
            # outlier jednoho snímku – nedůvěřuj měření, jen predikce
            return (angle_pred, vel_pred, p00_pred, p01_pred, p11_pred), innovation, s

        k0 = p00_pred / s
        k1 = p01_pred / s

        angle_new = angle_pred + k0 * innovation
        vel_new   = min(max(vel_pred + k1 * innovation, -self.max_vel_dps), self.max_vel_dps)

        p00_new = p00_pred * (1.0 - k0)
        p01_new = p01_pred * (1.0 - k0)
        p11_new = p11_pred - k1 * p01_pred

        return (angle_new, vel_new, p00_new, p01_new, p11_new), innovation, s

    # ------------------------------------------------------------------
    def _process(self, angle: float | None, confidence: float, timestamp_ms: float) -> float:
        if self.x_angle is None:
            # první validní inicializace stavu
            if angle is None:
                self.last_ts_ms = timestamp_ms
                return self._prob
            self.x_angle = angle
            self.x_vel   = 0.0
            self.last_ts_ms = timestamp_ms
            self._push_window(timestamp_ms, self.x_angle, confidence)
            return self._prob

        dt = (timestamp_ms - self.last_ts_ms) / 1000.0
        dt = min(max(dt, self.dt_min), self.dt_max)
        self.last_ts_ms = timestamp_ms

        self._gap_ms += dt * 1000.0
        if self._gap_ms > self.max_gap_ms:
            # delší dobu žádná platná detekce – rychlost dál extrapolovat nemá smysl
            # (mohla se stihnout dokončit, otočit, cokoliv), takže ji "vynulujeme"
            # a přerušíme i vazbu úhel-rychlost, ať ji jeden nový snímek hned neobnoví.
            self.x_vel = 0.0
            self.p01 = 0.0

        state = (self.x_angle, self.x_vel, self.p00, self.p01, self.p11)
        pred = self._kf_predict(state, dt)
        (angle_new, vel_new, p00_new, p01_new, p11_new), innovation, s = self._kf_update(pred, angle, confidence)

        gate = max(self.outlier_gate_deg, self.outlier_gate_sigma * math.sqrt(max(s, 1e-6)))
        used_conf = confidence if (angle is not None and abs(innovation) <= gate) else 0.0
        if used_conf > 0.0:
            self._gap_ms = 0.0

        self.x_angle, self.x_vel = angle_new, vel_new
        self.p00, self.p01, self.p11 = p00_new, p01_new, p11_new

        self._push_window(timestamp_ms, self.x_angle, used_conf)
        self._prob = self._compute_probability()
        return self._prob

    # ------------------------------------------------------------------
    def _push_window(self, timestamp_ms: float, unwrapped_angle: float, confidence: float) -> None:
        self._window.append((timestamp_ms, unwrapped_angle, confidence))
        cutoff = timestamp_ms - self.window_ms
        while len(self._window) > 1 and self._window[0][0] < cutoff:
            self._window.popleft()

    def _compute_probability(self) -> float:
        w = self._window
        if len(w) < 3:
            self._net = 0.0
            return 0.0

        net = w[-1][1] - w[0][1]
        total = 0.0
        conf_sum = 0.0
        for i in range(1, len(w)):
            total += abs(w[i][1] - w[i - 1][1])
        for _, _, c in w:
            conf_sum += c
        avg_conf = conf_sum / len(w)

        monotonic = abs(net) / total if total > 1e-6 else 0.0
        monotonic_score = _smoothstep(monotonic, self.monotonic_lo, self.monotonic_hi)
        rotation_score = _smoothstep(abs(net), self.rot_threshold_lo, self.rot_threshold_hi)
        conf_score = _smoothstep(avg_conf, self.conf_lo, self.conf_hi)

        self._net = net
        return rotation_score * monotonic_score * conf_score

    # ------------------------------------------------------------------
    @property
    def flip_probability(self) -> float:
        return float(self._prob)

    @property
    def state(self) -> str:
        if self._prob > 0.1:
            return "CW" if self._net > 0 else "CCW"
        return "IDLE"

    def reset(self) -> None:
        """Reset stavu filtru (nové video nebo ztráta osoby)."""
        self._prob = 0.0
        self._net  = 0.0
        self._reset_state()
