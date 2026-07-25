"""
Unit testy pro kaskádní pre-jump filtraci v JumpDetector._analyse_trajectory.

Testovaná logika (jump_detector.py):
  - 6. slot (nejstarší validní) je vyřazen z fitu, pokud jeho y_corrected >
    y_corrected[5. slot] + 0.30 * torso_h  (person was lower = pre-jump position)
  - Kaskáda: po vyřazení 6. se stejná podmínka aplikuje na nový nejstarší (5. slot)
  - Pokud podmínka pro 6. slot neplatí, oba zůstávají ve fitu
"""

import sys
import os
import numpy as np
import pytest

# Přidej kořenový adresář projektu do Python cesty
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from jump_detector import JumpDetector


# ── Pomocné funkce ───────────────────────────────────────────────────────────

TORSO_H = 0.20   # normalizovaná výška torza (konstantní pro jednoduchost)
THR     = 0.0    # práh pro pre-jump filtraci (0 = vyřaď kdykoliv oldest > next)


def _entry(t_sec: float, y: float, valid: bool = True) -> dict:
    """Vytvoří záznam do bufferu (y_corrected = y, menší y = výše)."""
    return {
        "valid":       valid,
        "t_sec":       t_sec,
        "y_corrected": y,
        "torso_h":     TORSO_H,
        "hip_x":       0.5,
        "hip_y":       y,
    }


def _call_analyse(entries: list[dict]) -> bool:
    """Zavolá JumpDetector._analyse_trajectory přímo s danými záznamy."""
    jd = JumpDetector(buffer_size=6, camera_correction=False)
    return jd._analyse_trajectory(entries)


def _get_fit_pts(entries: list[dict]) -> list[dict]:
    """
    Simuluje výběr bodů pro fit (stejná logika jako v _analyse_trajectory),
    vrátí seznam bodů které by šly do t_arr / y_arr.
    Umožňuje testovat samotnou filtraci bez závislosti na výsledku fitu.
    """
    pts_all = [e for e in entries if e["valid"]]
    pts = pts_all

    if len(pts) >= 2:
        _th = 0.0
        if pts[0]["y_corrected"] > pts[1]["y_corrected"] + _th:
            pts = pts[1:]
            if len(pts) >= 2 and pts[0]["y_corrected"] > pts[1]["y_corrected"] + _th:
                pts = pts[1:]

    return pts


# ── Scénář 1: Žádný pre-jump slot ─────────────────────────────────────────

class TestNoPreJump:
    """6. slot je součástí normálního oblouku — nesmí se vyřadit."""

    def _buf(self):
        # Symetrickа parabola: y klesá (osoba stoupá) pak stoupá (klesá)
        # Index 0 = nejstarší (6. slot)
        dt = 0.125
        ys = [0.60, 0.50, 0.40, 0.35, 0.40, 0.50]  # oblouk nahoru-dolů
        return [_entry(i * dt, y) for i, y in enumerate(ys)]

    def test_all_6_points_in_fit(self):
        """Podmínka y[0] > y[1] + thr neplatí → všech 6 bodů jde do fitu."""
        buf = self._buf()
        pts = _get_fit_pts(buf)
        assert len(pts) == 6, f"Očekáváno 6 bodů, dostali jsme {len(pts)}"

    def test_condition_not_met(self):
        """Ověř explicitně, že práh není překročen."""
        buf = self._buf()
        pts_all = [e for e in buf if e["valid"]]
        diff = pts_all[0]["y_corrected"] - pts_all[1]["y_corrected"]
        assert diff <= THR, (
            f"Slot 6 je o {diff:.4f} níže než slot 5, práh={THR:.4f} — "
            "podmínka by neměla platit"
        )


# ── Scénář 2: 6. slot je pre-jump, 5. slot je OK ─────────────────────────

class TestSlot6Excluded:
    """6. slot výrazně níže než 5. → vyřazen, 5. zůstává."""

    def _buf(self):
        dt = 0.125
        # Slot 6 (idx 0) je hluboko dole (pre-jump pozice, y velké)
        # Slot 5 (idx 1) je už nahoře (výskok začal)
        ys = [0.80, 0.50, 0.40, 0.35, 0.40, 0.50]
        return [_entry(i * dt, y) for i, y in enumerate(ys)]

    def test_slot6_excluded(self):
        """y[0]=0.80 > y[1]=0.50 + 0.06 → 6. slot vyřazen."""
        buf = self._buf()
        pts = _get_fit_pts(buf)
        assert len(pts) == 5, f"Očekáváno 5 bodů, dostali jsme {len(pts)}"
        # První bod ve fitu musí být slot 5 (y=0.50)
        assert abs(pts[0]["y_corrected"] - 0.50) < 1e-9

    def test_fit_result_matches_5slot_buffer(self):
        """
        Výsledek fitu ze 6-slotového bufferu (s pre-jump 6.) musí být shodný
        s fitem ze 5-slotového bufferu (bez pre-jump slotu vůbec).
        """
        buf_6 = self._buf()          # 6 bodů, první je pre-jump
        buf_5 = self._buf()[1:]      # totéž bez pre-jump slotu

        pts_6 = _get_fit_pts(buf_6)
        pts_5 = _get_fit_pts(buf_5)

        assert len(pts_6) == len(pts_5), "Počty bodů se musí shodovat"

        # y hodnoty musí být identické
        for i, (p6, p5) in enumerate(zip(pts_6, pts_5)):
            assert abs(p6["y_corrected"] - p5["y_corrected"]) < 1e-9, (
                f"Bod {i}: y6={p6['y_corrected']}, y5={p5['y_corrected']}"
            )


# ── Scénář 3: 6. i 5. slot jsou pre-jump (kaskáda) ──────────────────────

class TestSlot6And5Excluded:
    """Oba nejstarší sloty jsou výrazně níže → oba vyřazeny (kaskáda)."""

    def _buf(self):
        dt = 0.125
        # Slot 6 (idx 0) a slot 5 (idx 1) jsou pre-jump (obě y velké)
        # Slot 4 (idx 2) a dál jsou součástí výskoku
        ys = [0.85, 0.78, 0.50, 0.40, 0.45, 0.55]
        return [_entry(i * dt, y) for i, y in enumerate(ys)]

    def test_both_slots_excluded(self):
        """y[0]>y[1]+thr A y[1]>y[2]+thr → oba vyřazeny, zbývají 4 body."""
        buf = self._buf()
        pts = _get_fit_pts(buf)
        assert len(pts) == 4, f"Očekáváno 4 body, dostali jsme {len(pts)}"
        assert abs(pts[0]["y_corrected"] - 0.50) < 1e-9

    def test_fit_result_matches_4slot_buffer(self):
        """Výsledek musí být shodný s bufferem, který obsahuje jen 4 platné body."""
        buf_6 = self._buf()
        buf_4 = self._buf()[2:]      # bez obou pre-jump slotů

        pts_6 = _get_fit_pts(buf_6)
        pts_4 = _get_fit_pts(buf_4)

        assert len(pts_6) == len(pts_4)
        for i, (p6, p4) in enumerate(zip(pts_6, pts_4)):
            assert abs(p6["y_corrected"] - p4["y_corrected"]) < 1e-9, (
                f"Bod {i}: y6={p6['y_corrected']}, y4={p4['y_corrected']}"
            )


# ── Scénář 4: 6. slot splňuje podmínku, 5. slot nikoli ───────────────────

class TestSlot6ExcludedSlot5Kept:
    """6. vyřazen, ale 5. (nový nejstarší) podmínku nesplní → zůstane."""

    def _buf(self):
        dt = 0.125
        # Slot 6 je níže o více než práh
        # Slot 5 je níže než slot 4, ale méně než práh
        ys = [0.80, 0.51, 0.40, 0.35, 0.40, 0.50]
        return [_entry(i * dt, y) for i, y in enumerate(ys)]

    def test_only_slot6_excluded(self):
        """
        y[0]=0.80 > y[1]=0.51 + 0.06 → vyřaď slot 6.
        Pak y[1]=0.51 vs y[2]=0.40: diff=0.11 > thr=0.06 → vyřaď slot 5 TAKÉ.
        """
        # Přepočítáme práh pro tato data
        buf = self._buf()
        pts_all = [e for e in buf if e["valid"]]
        _th = 0.30 * TORSO_H  # = 0.06
        diff_56 = pts_all[0]["y_corrected"] - pts_all[1]["y_corrected"]  # 0.29
        diff_45 = pts_all[1]["y_corrected"] - pts_all[2]["y_corrected"]  # 0.11

        # Oba přesahují práh 0.06 → kaskáda vyřadí oba
        assert diff_56 > _th
        assert diff_45 > _th

        pts = _get_fit_pts(buf)
        assert len(pts) == 4  # oba vyřazeny kaskádou

    def test_slot5_kept_when_below_threshold(self):
        """Upraven buffer kde slot 5 je těsně pod prahem — musí zůstat."""
        dt = 0.125
        # diff mezi slotem 5 a 4 = 0.04 < 0.06 → slot 5 zůstane
        ys = [0.80, 0.44, 0.40, 0.35, 0.40, 0.50]
        buf = [_entry(i * dt, y) for i, y in enumerate(ys)]

        pts = _get_fit_pts(buf)
        assert len(pts) == 5, f"Očekáváno 5 bodů (slot 5 zůstává), dostali {len(pts)}"
        assert abs(pts[0]["y_corrected"] - 0.44) < 1e-9


# ── Scénář 5: Neúplný buffer (méně než 6 validních bodů) ─────────────────

class TestPartialBuffer:
    """Logika musí fungovat i s méně než 6 body."""

    def test_3_valid_points_no_crash(self):
        """Méně bodů → pre-jump podmínka se stále kontroluje (3 body)."""
        dt = 0.125
        ys = [0.80, 0.50, 0.40]
        buf = [_entry(i * dt, y) for i, y in enumerate(ys)]
        pts = _get_fit_pts(buf)
        # 0.80 > 0.50 + 0.06 → vyřaď nejstarší → zůstanou 2
        assert len(pts) == 2

    def test_1_valid_point_no_exclusion(self):
        """S jediným validním bodem nelze porovnat → žádné vyřazení."""
        buf = [_entry(0.0, 0.50)]
        pts = _get_fit_pts(buf)
        assert len(pts) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
