"""Unit test gap (Edianto Ong).

Dua tahap yang dibedakan buku diuji terpisah: lompatan saat pembukaan, dan
apakah celahnya TETAP kosong setelah sesi itu selesai - hanya yang kedua
yang "meninggalkan gap pada chart".
"""

import pandas as pd
import pytest

from screener.gaps import (
    COMMON_GAP_MAX_DAYS,
    GAP_COMMON,
    GAP_DOWN,
    GAP_UNCLASSIFIED,
    GAP_UNDETERMINED,
    GAP_UP,
    PREFERRED_INTERVAL,
    PRESSURE_BUYING,
    PRESSURE_SELLING,
    detect_gaps,
    is_preferred_interval,
)


def _frame(bars):
    """bars = daftar (Open, High, Low, Close)."""
    index = pd.date_range("2025-01-06", periods=len(bars), freq="B", tz="Asia/Jakarta")
    return pd.DataFrame(
        {
            "Open": [b[0] for b in bars], "High": [b[1] for b in bars],
            "Low": [b[2] for b in bars], "Close": [b[3] for b in bars],
        },
        index=index,
    ).astype(float)


# Sesi acuan: High 100, Low 90.
BASE = (95, 100, 90, 98)


class TestGapUp:
    """Pembukaan melonjak di atas High sesi sebelumnya."""

    def test_open_above_previous_high_is_a_gap_up(self):
        gaps = detect_gaps(_frame([BASE, (105, 108, 103, 107)]))
        assert len(gaps) == 1
        gap = gaps[0]
        assert gap.kind == GAP_UP and gap.is_up
        assert gap.previous_extreme == 100  # High sesi sebelumnya
        assert (gap.lower, gap.upper) == (100, 105)  # lompatan pembukaan
        assert gap.size == 5 and gap.size_pct == pytest.approx(5.0)

    def test_gap_remains_when_the_session_never_comes_back_down(self):
        gap = detect_gaps(_frame([BASE, (105, 108, 103, 107)]))[0]
        assert gap.remains
        assert gap.visible_zone == (100, 103)  # celah yang tersisa di chart

    def test_gap_is_covered_when_the_session_trades_back_through_it(self):
        # Buka di 105 tapi turun sampai 99: celahnya tertutup sesi itu juga.
        gap = detect_gaps(_frame([BASE, (105, 108, 99, 101)]))[0]
        assert not gap.remains
        assert gap.visible_zone is None

    def test_opening_exactly_at_the_previous_high_is_not_a_gap(self):
        assert detect_gaps(_frame([BASE, (100, 104, 99, 103)])) == []

    def test_opening_inside_the_previous_range_is_not_a_gap(self):
        assert detect_gaps(_frame([BASE, (95, 104, 94, 103)])) == []


class TestGapDown:
    """Cermin: pembukaan melonjak di bawah Low sesi sebelumnya."""

    def test_open_below_previous_low_is_a_gap_down(self):
        gap = detect_gaps(_frame([BASE, (85, 88, 83, 84)]))[0]
        assert gap.kind == GAP_DOWN and not gap.is_up
        assert gap.previous_extreme == 90  # Low sesi sebelumnya
        assert (gap.lower, gap.upper) == (85, 90)
        assert gap.remains and gap.visible_zone == (88, 90)

    def test_gap_down_covered_within_the_session(self):
        gap = detect_gaps(_frame([BASE, (85, 92, 83, 91)]))[0]
        assert not gap.remains and gap.visible_zone is None

    def test_opening_exactly_at_the_previous_low_is_not_a_gap(self):
        assert detect_gaps(_frame([BASE, (90, 95, 88, 94)])) == []


class TestFilling:
    """Pengamatan faktual: kapan harga kembali masuk zona celah."""

    def test_later_session_that_trades_back_fills_the_gap(self):
        gaps = detect_gaps(_frame([BASE, (105, 108, 103, 107), (104, 106, 99, 100)]))
        assert gaps[0].is_filled and gaps[0].filled_index == 2

    def test_gap_stays_open_when_price_never_returns(self):
        gaps = detect_gaps(_frame([BASE, (105, 108, 103, 107), (108, 112, 106, 110)]))
        assert not gaps[0].is_filled and gaps[0].filled_index is None

    def test_touching_the_edge_counts_as_filled(self):
        gaps = detect_gaps(_frame([BASE, (105, 108, 103, 107), (104, 106, 100, 102)]))
        assert gaps[0].filled_index == 2  # Low tepat di 100 = High sebelum gap

    def test_gap_down_is_filled_from_below(self):
        gaps = detect_gaps(_frame([BASE, (85, 88, 83, 84), (86, 91, 85, 90)]))
        assert gaps[0].kind == GAP_DOWN and gaps[0].filled_index == 2


class TestDetectionOptions:
    def test_only_remaining_hides_covered_gaps(self):
        frame = _frame([BASE, (105, 108, 99, 101)])  # gap up yang tertutup
        assert len(detect_gaps(frame)) == 1
        assert detect_gaps(frame, only_remaining=True) == []

    def test_minimum_size_filters_small_jumps(self):
        frame = _frame([BASE, (100.5, 104, 100.2, 103)])  # lompatan 0,5%
        assert len(detect_gaps(frame)) == 1
        assert detect_gaps(frame, min_size_pct=1.0) == []

    def test_negative_minimum_is_rejected(self):
        with pytest.raises(ValueError):
            detect_gaps(_frame([BASE, BASE]), min_size_pct=-1)

    def test_first_bar_cannot_be_a_gap(self):
        gaps = detect_gaps(_frame([BASE, (105, 108, 103, 107)]))
        assert all(g.index >= 1 for g in gaps)

    def test_several_gaps_are_reported_in_order(self):
        frame = _frame([BASE, (105, 108, 103, 107), (115, 118, 112, 117), (80, 85, 78, 79)])
        gaps = detect_gaps(frame)
        assert [g.index for g in gaps] == [1, 2, 3]
        assert [g.kind for g in gaps] == [GAP_UP, GAP_UP, GAP_DOWN]

    def test_empty_or_incomplete_frame_gives_nothing(self):
        assert detect_gaps(pd.DataFrame()) == []
        assert detect_gaps(pd.DataFrame({"Close": [1, 2, 3]})) == []


class TestTimeframe:
    """Buku: gap paling umum dianalisis pada daily chart."""

    def test_daily_is_the_preferred_interval(self):
        assert PREFERRED_INTERVAL == "1d"
        assert is_preferred_interval("1d")

    def test_other_intervals_are_flagged_as_less_common(self):
        assert not is_preferred_interval("1wk")
        assert not is_preferred_interval("1mo")


def _frame_with_volume(bars, volumes):
    df = _frame(bars)
    df["Volume"] = [float(v) for v in volumes]
    return df


class TestGapMeaning:
    """Buku: gap up = minat beli tinggi, gap down = tekanan jual kuat."""

    def test_gap_up_means_buying_interest(self):
        gap = detect_gaps(_frame([BASE, (105, 108, 103, 107)]))[0]
        assert gap.pressure == PRESSURE_BUYING

    def test_gap_down_means_selling_pressure(self):
        gap = detect_gaps(_frame([BASE, (85, 88, 83, 84)]))[0]
        assert gap.pressure == PRESSURE_SELLING


class TestVolumeOnGaps:
    """Buku: gap sering ditandai volume yang meningkat drastis."""

    def test_volume_ratio_is_measured_on_the_gap_bar(self):
        bars = [BASE] * 5 + [(105, 108, 103, 107)]
        gap = detect_gaps(_frame_with_volume(bars, [1_000] * 5 + [3_000]))[0]
        assert gap.volume_ratio == pytest.approx(3.0)
        assert gap.has_volume_surge is True

    def test_ordinary_volume_is_not_a_surge(self):
        bars = [BASE] * 5 + [(105, 108, 103, 107)]
        gap = detect_gaps(_frame_with_volume(bars, [1_000] * 6))[0]
        assert gap.volume_ratio == pytest.approx(1.0)
        assert gap.has_volume_surge is False

    def test_without_volume_column_nothing_is_assumed(self):
        gap = detect_gaps(_frame([BASE, (105, 108, 103, 107)]))[0]
        assert gap.volume_ratio is None and gap.has_volume_surge is None


class TestCommonGap:
    """Common gap: tanpa lonjakan volume DAN tertutup kurang dari 1 minggu."""

    def _quiet_gap(self, fill_offset_days, volumes=None):
        """Gap up pada hari ke-5, tertutup `fill_offset_days` hari kemudian."""
        bars = [BASE] * 5 + [(105, 108, 103, 107)]
        padding = max(fill_offset_days, 1)
        bars += [(106, 109, 104, 108)] * (padding - 1) + [(104, 106, 95, 96)]
        volumes = volumes or [1_000] * len(bars)
        df = _frame_with_volume(bars, volumes)
        # Tanggal dibuat harian kalender supaya tenggat "1 minggu" terukur.
        df.index = pd.date_range(
            "2025-01-06", periods=len(df), freq="D", tz="Asia/Jakarta"
        )
        return detect_gaps(df)[0]

    def test_quiet_gap_closed_within_a_week_is_common(self):
        gap = self._quiet_gap(fill_offset_days=3)
        assert gap.days_to_fill == 3
        assert gap.kind_label == GAP_COMMON and gap.is_common

    def test_gap_closed_after_a_week_is_not_common(self):
        gap = self._quiet_gap(fill_offset_days=10)
        assert gap.days_to_fill == 10
        assert gap.kind_label == GAP_UNCLASSIFIED

    def test_volume_surge_rules_out_common_even_if_closed_fast(self):
        volumes = [1_000] * 5 + [5_000] + [1_000] * 3
        gap = self._quiet_gap(fill_offset_days=3, volumes=volumes)
        assert gap.has_volume_surge
        assert gap.kind_label == GAP_UNCLASSIFIED  # jenisnya tidak ditebak

    def test_one_week_is_the_book_threshold(self):
        assert COMMON_GAP_MAX_DAYS == 7
        assert self._quiet_gap(fill_offset_days=6).kind_label == GAP_COMMON
        assert self._quiet_gap(fill_offset_days=7).kind_label == GAP_UNCLASSIFIED

    def test_threshold_is_adjustable(self):
        bars = [BASE] * 5 + [(105, 108, 103, 107)] + [(106, 109, 104, 108)] * 9
        bars += [(104, 106, 95, 96)]
        df = _frame_with_volume(bars, [1_000] * len(bars))
        df.index = pd.date_range("2025-01-06", periods=len(df), freq="D", tz="Asia/Jakarta")
        assert detect_gaps(df)[0].kind_label == GAP_UNCLASSIFIED
        assert detect_gaps(df, max_common_days=30)[0].kind_label == GAP_COMMON


class TestUndeterminedGap:
    """Gap muda yang belum tertutup belum bisa dinilai - jangan dipaksakan."""

    def test_young_open_gap_is_undetermined(self):
        bars = [BASE] * 5 + [(105, 108, 103, 107), (106, 109, 104, 108)]
        df = _frame_with_volume(bars, [1_000] * len(bars))
        df.index = pd.date_range("2025-01-06", periods=len(df), freq="D", tz="Asia/Jakarta")
        gap = detect_gaps(df)[0]
        assert not gap.is_filled
        assert gap.days_open == 1
        assert gap.kind_label == GAP_UNDETERMINED

    def test_old_open_gap_is_no_longer_a_common_candidate(self):
        bars = [BASE] * 5 + [(105, 108, 103, 107)] + [(106, 109, 104, 108)] * 10
        df = _frame_with_volume(bars, [1_000] * len(bars))
        df.index = pd.date_range("2025-01-06", periods=len(df), freq="D", tz="Asia/Jakarta")
        gap = detect_gaps(df)[0]
        assert gap.days_open >= COMMON_GAP_MAX_DAYS
        assert gap.kind_label == GAP_UNCLASSIFIED

    def test_other_gap_types_are_never_guessed(self):
        """breakaway/runaway/exhaustion belum diberikan buku."""
        bars = [BASE] * 5 + [(105, 108, 103, 107)]
        gap = detect_gaps(_frame_with_volume(bars, [1_000] * 5 + [9_000]))[0]
        assert gap.kind_label in (GAP_COMMON, GAP_UNCLASSIFIED, GAP_UNDETERMINED)
