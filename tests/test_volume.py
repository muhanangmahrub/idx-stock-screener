"""Unit test volume (Edianto Ong).

Aturan buku yang diuji: volume bar tinggi = perdagangan ramai, bar pendek =
minat kurang. Ambang angkanya belum diberikan buku, jadi yang dikunci di sini
adalah perilaku ukurannya - termasuk bahwa volume nol (pre-opening) dibedakan
dari volume kecil.
"""

import numpy as np
import pandas as pd
import pytest

from screener.volume import (
    DEFAULT_HIGH_RATIO,
    DEFAULT_LOW_RATIO,
    VOLUME_HIGH,
    VOLUME_LOW,
    VOLUME_NO_TRADE,
    VOLUME_NORMAL,
    average_volume,
    classify_volume,
    drop_tradeless_bars,
    is_tradeless,
    relative_volume,
    volume_profiles,
)


def _frame(volumes, close=1_000.0):
    index = pd.date_range("2025-01-06", periods=len(volumes), freq="B", tz="Asia/Jakarta")
    return pd.DataFrame(
        {
            "Open": close, "High": close + 10, "Low": close - 10, "Close": close,
            "Volume": [float(v) for v in volumes],
        },
        index=index,
    )


class TestTradelessBars:
    """Volume nol = tidak ada perdagangan, bukan minat rendah."""

    @pytest.mark.parametrize("volume", [0, 0.0, None, -1])
    def test_zero_or_missing_is_tradeless(self, volume):
        assert is_tradeless(volume)

    def test_positive_volume_is_not_tradeless(self):
        assert not is_tradeless(1)

    def test_tradeless_bar_gets_its_own_label(self):
        assert classify_volume(0, average=1_000) == VOLUME_NO_TRADE

    def test_tradeless_bars_can_be_dropped(self):
        df = _frame([100, 0, 200, 0, 300])
        cleaned = drop_tradeless_bars(df)
        assert len(cleaned) == 3
        assert (cleaned["Volume"] > 0).all()

    def test_dropping_is_safe_without_volume_column(self):
        df = pd.DataFrame({"Close": [1, 2, 3]})
        assert len(drop_tradeless_bars(df)) == 3

    def test_tradeless_bars_do_not_drag_the_average_down(self):
        with_gap = average_volume(_frame([100, 100, 0, 100, 100])["Volume"], window=5)
        without_gap = average_volume(_frame([100, 100, 100, 100, 100])["Volume"], window=5)
        assert with_gap.iloc[-1] == pytest.approx(without_gap.iloc[-1])


class TestAverageVolume:
    def test_average_uses_only_earlier_bars(self):
        # Bar terakhir melonjak; rata-ratanya tidak boleh ikut terangkat.
        volumes = _frame([100, 100, 100, 1_000])["Volume"]
        averages = average_volume(volumes, window=3)
        assert averages.iloc[-1] == pytest.approx(100)

    def test_including_current_bar_changes_the_average(self):
        volumes = _frame([100, 100, 100, 1_000])["Volume"]
        averages = average_volume(volumes, window=4, exclude_current=False)
        assert averages.iloc[-1] == pytest.approx(325)

    def test_first_bar_has_no_history(self):
        averages = average_volume(_frame([100, 200])["Volume"], window=5)
        assert pd.isna(averages.iloc[0])

    def test_window_must_be_positive(self):
        with pytest.raises(ValueError):
            average_volume(_frame([100])["Volume"], window=0)

    def test_relative_volume_is_one_when_ordinary(self):
        ratios = relative_volume(_frame([100] * 6)["Volume"], window=3)
        assert ratios.iloc[-1] == pytest.approx(1.0)

    def test_relative_volume_doubles_when_volume_doubles(self):
        ratios = relative_volume(_frame([100, 100, 100, 200])["Volume"], window=3)
        assert ratios.iloc[-1] == pytest.approx(2.0)


class TestClassification:
    """Tinggi vs pendek dinilai relatif terhadap kebiasaan emiten itu sendiri."""

    def test_tall_bar_is_high(self):
        assert classify_volume(200, average=100) == VOLUME_HIGH

    def test_short_bar_is_low(self):
        assert classify_volume(40, average=100) == VOLUME_LOW

    def test_ordinary_bar_is_normal(self):
        assert classify_volume(100, average=100) == VOLUME_NORMAL

    def test_thresholds_are_inclusive_at_the_boundary(self):
        assert classify_volume(150, average=100) == VOLUME_HIGH  # tepat 1,5x
        assert classify_volume(50, average=100) == VOLUME_LOW  # tepat 0,5x

    def test_defaults_are_marked_assumptions(self):
        assert DEFAULT_HIGH_RATIO == 1.5 and DEFAULT_LOW_RATIO == 0.5

    def test_thresholds_are_adjustable(self):
        assert classify_volume(120, 100, high_ratio=1.1, low_ratio=0.9) == VOLUME_HIGH
        assert classify_volume(120, 100, high_ratio=3.0, low_ratio=0.5) == VOLUME_NORMAL

    def test_inverted_thresholds_are_rejected(self):
        with pytest.raises(ValueError):
            classify_volume(100, 100, high_ratio=0.5, low_ratio=1.5)

    def test_without_an_average_nothing_is_guessed(self):
        assert classify_volume(100, average=None) == VOLUME_NORMAL
        assert classify_volume(100, average=np.nan) == VOLUME_NORMAL


class TestVolumeProfiles:
    def test_profile_per_bar_with_ratio(self):
        profiles = volume_profiles(_frame([100, 100, 100, 100, 300]), window=4)
        assert len(profiles) == 5
        last = profiles[-1]
        assert last.index == 4
        assert last.volume == 300
        assert last.average == pytest.approx(100)
        assert last.ratio == pytest.approx(3.0)
        assert last.is_high and not last.is_low

    def test_quiet_bar_is_flagged_as_low_interest(self):
        profiles = volume_profiles(_frame([100, 100, 100, 100, 20]), window=4)
        assert profiles[-1].label == VOLUME_LOW
        assert profiles[-1].is_low

    def test_tradeless_bar_is_neither_high_nor_low(self):
        profiles = volume_profiles(_frame([100, 100, 100, 100, 0]), window=4)
        assert profiles[-1].label == VOLUME_NO_TRADE
        assert not profiles[-1].is_high and not profiles[-1].is_low

    def test_frame_without_volume_gives_no_profiles(self):
        assert volume_profiles(pd.DataFrame({"Close": [1, 2, 3]})) == []

    def test_first_bar_has_no_average_to_compare_with(self):
        profiles = volume_profiles(_frame([100, 200, 300]), window=3)
        assert profiles[0].average is None and profiles[0].ratio is None
        assert profiles[0].label == VOLUME_NORMAL
