import numpy as np
import pandas as pd
import pytest

from screener.levels import (
    FALSE_BREAK,
    PULLBACK,
    RESISTANCE,
    SUPPORT,
    VALID_BREAK,
    levels_from_swings,
    track_level,
)


def _series(values):
    return pd.Series(values, dtype=float)


class TestTrackLevelBreakRule:
    """Prinsip sama seperti trendline: tembus sah hanya bila Close di luar level."""

    def test_intact_when_price_respects_support(self):
        closes = _series([100, 103, 105, 102, 104, 106])
        status = track_level(100, SUPPORT, 0, closes, closes - 1, closes + 1)
        assert status.role == SUPPORT
        assert status.events == []
        assert status.age_bars == 5

    def test_intraday_pierce_is_false_break_and_keeps_role(self):
        closes = _series([100, 103, 105, 102, 104, 106])
        lows = closes - 1
        lows.iloc[3] = 98  # Low menembus 100, Close 102 kembali di atas
        status = track_level(100, SUPPORT, 0, closes, lows, closes + 1)
        assert status.role == SUPPORT
        assert [(e.index, e.kind) for e in status.events] == [(3, FALSE_BREAK)]

    def test_close_exactly_on_level_is_not_a_break(self):
        closes = _series([100, 103, 100, 102])
        status = track_level(100, SUPPORT, 0, closes, closes, closes)
        assert status.events == []

    def test_origin_bar_itself_is_not_checked(self):
        # Bar 0 = bar tempat level terbentuk (Low = level); jangan dianggap tembus.
        closes = _series([101, 103, 105])
        lows = _series([100, 102, 104])
        status = track_level(100, SUPPORT, 0, closes, lows, closes + 1)
        assert status.events == []


class TestRoleReversal:
    """Support yang ditembus jadi resistance, dan sebaliknya; usia terbawa."""

    def test_support_broken_by_close_becomes_resistance(self):
        closes = _series([100, 103, 97, 95, 96])
        status = track_level(100, SUPPORT, 0, closes, closes - 1, closes + 1)
        assert status.role == RESISTANCE
        assert [(e.index, e.kind, e.role_before) for e in status.events] == [
            (2, VALID_BREAK, SUPPORT)
        ]

    def test_after_reversal_pierces_are_judged_against_new_role(self):
        # Setelah jadi resistance: High > 100 tapi Close < 100 -> level baru
        # diuji dari bawah dan bertahan = pullback pada peran resistance.
        closes = _series([100, 103, 97, 95, 99, 98])
        highs = closes + 1
        highs.iloc[4] = 101
        status = track_level(100, SUPPORT, 0, closes, closes - 1, highs)
        assert status.role == RESISTANCE
        assert [(e.index, e.kind, e.role_before) for e in status.events] == [
            (2, VALID_BREAK, SUPPORT),
            (4, PULLBACK, RESISTANCE),
        ]

    def test_resistance_broken_by_close_becomes_support_and_back(self):
        closes = _series([100, 98, 102, 104, 99, 97])
        status = track_level(100, RESISTANCE, 0, closes, closes - 1, closes + 1)
        assert status.initial_role == RESISTANCE
        assert status.role == RESISTANCE  # tembus naik (jadi support) lalu tembus turun lagi
        assert [(e.index, e.role_before) for e in status.valid_breaks] == [
            (2, RESISTANCE),
            (4, SUPPORT),
        ]

    def test_age_is_counted_from_origin_not_reset_by_reversal(self):
        closes = _series([100, 103, 97, 95, 96, 94, 93])
        status = track_level(100, SUPPORT, 0, closes, closes - 1, closes + 1)
        assert status.role == RESISTANCE
        assert status.age_bars == 6  # bukan 4 (sejak berbalik)

    def test_older_level_reports_larger_age(self):
        """Faktor waktu: level yang lebih lama bertahan punya usia lebih besar;
        buku tidak memberi skala, jadi hanya usia yang dilaporkan."""
        closes = _series([100.0] * 30)
        old = track_level(90, SUPPORT, 0, closes, closes, closes)
        young = track_level(90, SUPPORT, 25, closes, closes, closes)
        assert old.age_bars > young.age_bars

    def test_invalid_role_is_rejected(self):
        with pytest.raises(ValueError):
            track_level(100, "pivot", 0, _series([100]), _series([100]), _series([100]))


class TestLevelsFromSwings:
    def test_uses_low_for_support_and_high_for_resistance(self):
        close = _series([100, 110, 100, 110, 100, 110, 100])
        df = pd.DataFrame({"Close": close, "Low": close - 2, "High": close + 3})
        highs_idx, lows_idx = np.array([1, 3, 5]), np.array([2, 4])
        levels = levels_from_swings(df, highs_idx, lows_idx, per_side=2)
        by_origin = {level.origin_index: level for level in levels}
        assert by_origin[2].initial_role == SUPPORT and by_origin[2].price == 98
        assert by_origin[3].initial_role == RESISTANCE and by_origin[3].price == 113
        # per_side=2 -> swing high tertua (indeks 1) tidak ikut.
        assert sorted(by_origin) == [2, 3, 4, 5]

    def test_sorted_oldest_first(self):
        close = _series([100, 110, 100, 110, 100, 110, 100])
        df = pd.DataFrame({"Close": close, "Low": close - 2, "High": close + 3})
        levels = levels_from_swings(df, np.array([1, 3, 5]), np.array([2, 4]), per_side=3)
        assert [level.origin_index for level in levels] == [1, 2, 3, 4, 5]
        assert levels[0].age_bars == 5


class TestPullback:
    """Pullback = harga kembali menguji level yang SUDAH dilewati."""

    def test_touch_after_break_with_close_holding_is_pullback(self):
        # Resistance 100 ditembus naik di bar 2 (jadi support); bar 4 Low
        # menyentuh 100 tapi Close 101 bertahan -> pullback, peran tetap support.
        closes = _series([100, 99, 102, 104, 101, 103])
        lows = closes - 1
        lows.iloc[4] = 100
        status = track_level(100, RESISTANCE, 0, closes, lows, closes + 1)
        assert status.role == SUPPORT
        assert [(e.index, e.kind, e.role_before) for e in status.events] == [
            (2, VALID_BREAK, RESISTANCE),
            (4, PULLBACK, SUPPORT),
        ]
        assert [e.index for e in status.pullbacks_held] == [4]
        assert status.pullbacks_failed == []

    def test_touch_before_any_break_is_not_a_pullback(self):
        # Level belum pernah dilewati: sentuhan tepat di level bukan kejadian,
        # tembusan intraday tetap false break (aturan lama tidak berubah).
        closes = _series([100, 103, 102, 104, 103])
        lows = closes - 1
        lows.iloc[2] = 100  # sentuh
        lows.iloc[3] = 99  # tembus intraday
        status = track_level(100, SUPPORT, 0, closes, lows, closes + 1)
        assert [(e.index, e.kind) for e in status.events] == [(3, FALSE_BREAK)]
        assert status.pullbacks_held == []

    def test_pullback_that_closes_through_is_a_failed_pullback(self):
        # Ditembus naik di bar 2, lalu bar 4 Close kembali di bawah 100:
        # valid break kedua = pullback gagal, peran balik jadi resistance.
        closes = _series([100, 99, 102, 104, 98, 97])
        status = track_level(100, RESISTANCE, 0, closes, closes - 1, closes + 1)
        assert status.role == RESISTANCE
        assert [e.index for e in status.valid_breaks] == [2, 4]
        assert [e.index for e in status.pullbacks_failed] == [4]  # tembusan pertama bukan pullback

    def test_intraday_pierce_after_break_counts_as_pullback_not_false_break(self):
        # Setelah dilewati, Low menembus 100 tapi Close bertahan: itu uji ulang
        # (pullback bertahan), bukan lagi "false break" level asli.
        closes = _series([100, 99, 102, 104, 101])
        lows = closes - 1
        lows.iloc[4] = 98
        status = track_level(100, RESISTANCE, 0, closes, lows, closes + 1)
        assert [(e.index, e.kind) for e in status.events] == [(2, VALID_BREAK), (4, PULLBACK)]
        assert status.false_breaks == []

    def test_support_mirror_uses_highs(self):
        # Support 100 ditembus turun di bar 2 (jadi resistance); bar 4 High
        # menyentuh 100 dari bawah, Close 99 bertahan -> pullback.
        closes = _series([100, 101, 97, 95, 99, 98])
        highs = closes + 0.5
        highs.iloc[4] = 100
        status = track_level(100, SUPPORT, 0, closes, closes - 1, highs)
        assert status.role == RESISTANCE
        assert [(e.index, e.kind, e.role_before) for e in status.events] == [
            (2, VALID_BREAK, SUPPORT),
            (4, PULLBACK, RESISTANCE),
        ]

    def test_tolerance_lets_near_miss_count_as_pullback(self):
        # Low hanya sampai 100.5 (0.5% di atas support baru): bukan pullback
        # dengan tol 0 (ASUMSI default: harus menyentuh), pullback dengan tol 1%.
        closes = _series([100, 99, 102, 104, 101.5, 103])
        lows = closes - 1
        lows.iloc[4] = 100.5
        strict = track_level(100, RESISTANCE, 0, closes, lows, closes + 1)
        loose = track_level(100, RESISTANCE, 0, closes, lows, closes + 1, pullback_tol=0.01)
        assert strict.pullbacks_held == []
        assert [e.index for e in loose.pullbacks_held] == [4]

    def test_negative_tolerance_is_rejected(self):
        with pytest.raises(ValueError):
            track_level(100, SUPPORT, 0, _series([100]), _series([100]), _series([100]), pullback_tol=-0.1)
