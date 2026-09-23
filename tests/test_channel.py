"""Unit test channeling (Edianto Ong): koridor dua garis sejajar.

Data sintetis dibuat supaya koridornya jelas: harga zigzag naik di antara
dua garis dengan kemiringan sama, lalu satu varian menembus basic trendline
(bearish) dan satu varian menembus channel line (akselerasi, bullish).
"""

import numpy as np
import pandas as pd
import pytest

from screener.channel import (
    BASIC_BREAK_TREND_CHANGE,
    BIAS_BEARISH,
    BIAS_BULLISH,
    CHANNEL_BREAK_ACCELERATION,
    CHANNEL_INTACT,
    build_channel,
    check_channel_break,
)
from screener.trend import DOWN_TRENDLINE, UP_TRENDLINE, AnchorPoint, Trendline


def _series(values):
    return pd.Series(values, dtype=float)


def _up_basic(slope=1.0, intercept=100.0, last_anchor=4, end_index=19):
    """Basic trendline naik: nilai 100 di bar 0, 119 di bar 19."""
    return Trendline(
        kind=UP_TRENDLINE,
        slope=slope,
        intercept=intercept,
        points=[AnchorPoint(0, intercept), AnchorPoint(last_anchor, intercept + slope * last_anchor)],
        end_index=end_index,
    )


def _down_basic(slope=-1.0, intercept=120.0, end_index=19):
    return Trendline(
        kind=DOWN_TRENDLINE,
        slope=slope,
        intercept=intercept,
        points=[AnchorPoint(0, intercept), AnchorPoint(4, intercept + slope * 4)],
        end_index=end_index,
    )


class TestBuildChannel:
    """Channel line = proyeksi SEJAJAR basic trendline lewat sisi seberang."""

    def test_uptrend_channel_line_is_parallel_and_above(self):
        basic = _up_basic()
        # Swing high 10 di atas garis pada bar 5, 11, 17.
        highs = _series([basic.value_at(i) + 3 for i in range(20)])
        for i in (5, 11, 17):
            highs.iloc[i] = basic.value_at(i) + 10
        lows = _series([basic.value_at(i) - 1 for i in range(20)])
        channel = build_channel(basic, highs, lows, np.array([5, 11, 17]), np.array([0, 4]))
        assert channel.slope == basic.slope  # sejajar
        assert channel.anchor.index in (5, 11, 17)
        assert channel.channel_value_at(0) == pytest.approx(110)
        assert channel.width_at(0) == pytest.approx(10)
        assert channel.width_at(19) == pytest.approx(10)  # lebar konstan
        assert channel.is_uptrend and channel.basic_position == "bawah"

    def test_anchor_is_the_swing_farthest_from_the_basic_line(self):
        basic = _up_basic()
        highs = _series([basic.value_at(i) + 2 for i in range(20)])
        highs.iloc[5] = basic.value_at(5) + 6
        highs.iloc[11] = basic.value_at(11) + 12  # paling jauh -> jadi titik acuan
        highs.iloc[17] = basic.value_at(17) + 12  # menyentuh garis yang sama
        lows = _series([basic.value_at(i) - 1 for i in range(20)])
        channel = build_channel(basic, highs, lows, np.array([5, 11, 17]), np.array([0, 4]))
        assert channel.anchor.index == 11
        assert channel.width_at(11) == pytest.approx(12)

    def test_channel_is_not_forced_when_only_one_swing_touches_the_line(self):
        """Satu titik sentuh belum membuktikan koridor - jangan dipaksakan."""
        basic = _up_basic()
        highs = _series([basic.value_at(i) + 2 for i in range(20)])
        highs.iloc[11] = basic.value_at(11) + 12  # hanya satu swing yang jauh
        lows = _series([basic.value_at(i) - 1 for i in range(20)])
        highs_idx, lows_idx = np.array([5, 11, 17]), np.array([0, 4])
        assert build_channel(basic, highs, lows, highs_idx, lows_idx) is None
        # Ambang bisa diatur bila pemilik ingin lebih longgar.
        loose = build_channel(basic, highs, lows, highs_idx, lows_idx, min_touches=1)
        assert loose is not None and loose.touches == 1

    def test_downtrend_channel_line_is_below_and_uses_lows(self):
        basic = _down_basic()
        lows = _series([basic.value_at(i) - 3 for i in range(20)])
        for i in (6, 12):
            lows.iloc[i] = basic.value_at(i) - 8
        highs = _series([basic.value_at(i) + 1 for i in range(20)])
        channel = build_channel(basic, highs, lows, np.array([0, 4]), np.array([6, 12]))
        assert channel.slope == basic.slope
        assert not channel.is_uptrend and channel.basic_position == "atas"
        assert channel.channel_value_at(6) < channel.basic_value_at(6)
        assert channel.width_at(6) == pytest.approx(8)

    def test_touches_count_swings_that_reach_the_channel_line(self):
        basic = _up_basic()
        highs = _series([basic.value_at(i) + 2 for i in range(20)])
        highs.iloc[5] = basic.value_at(5) + 10
        highs.iloc[11] = basic.value_at(11) + 10  # menyentuh channel line juga
        highs.iloc[17] = basic.value_at(17) + 4  # jauh dari channel line
        lows = _series([basic.value_at(i) - 1 for i in range(20)])
        channel = build_channel(basic, highs, lows, np.array([5, 11, 17]), np.array([0, 4]))
        assert channel.touches == 2

    def test_no_opposite_swing_in_range_gives_no_channel(self):
        basic = _up_basic()
        prices = _series([basic.value_at(i) for i in range(20)])
        assert build_channel(basic, prices, prices, np.array([]), np.array([0, 4])) is None


class TestCheckChannelBreak:
    """Buku: pada uptrend channeling, basic trendline tertembus = sinyal
    bearish (awal perubahan tren); channel line tertembus = akselerasi
    (sinyal bullish). Downtrend cermin."""

    def _uptrend_channel(self):
        basic = _up_basic()
        highs = _series([basic.value_at(i) + 10 for i in range(20)])
        lows = _series([basic.value_at(i) - 1 for i in range(20)])
        return build_channel(basic, highs, lows, np.array([5, 11, 17]), np.array([0, 4]))

    def test_price_inside_the_corridor_is_intact(self):
        channel = self._uptrend_channel()
        closes = _series([channel.basic_value_at(i) + 5 for i in range(20)])
        result = check_channel_break(channel, closes, tol=0.0)
        assert result.status == CHANNEL_INTACT
        assert result.bias is None and result.index is None

    def test_close_below_basic_trendline_is_bearish_trend_change(self):
        channel = self._uptrend_channel()
        closes = _series([channel.basic_value_at(i) + 5 for i in range(20)])
        closes.iloc[12] = channel.basic_value_at(12) - 2
        result = check_channel_break(channel, closes, tol=0.0)
        assert result.status == BASIC_BREAK_TREND_CHANGE
        assert result.bias == BIAS_BEARISH
        assert result.index == 12

    def test_close_above_channel_line_is_bullish_acceleration(self):
        channel = self._uptrend_channel()
        closes = _series([channel.basic_value_at(i) + 5 for i in range(20)])
        closes.iloc[14] = channel.channel_value_at(14) + 2
        result = check_channel_break(channel, closes, tol=0.0)
        assert result.status == CHANNEL_BREAK_ACCELERATION
        assert result.bias == BIAS_BULLISH
        assert result.index == 14

    def test_tolerance_band_suppresses_a_thin_break(self):
        channel = self._uptrend_channel()
        closes = _series([channel.basic_value_at(i) + 5 for i in range(20)])
        closes.iloc[12] = channel.basic_value_at(12) * 0.99  # 1% di bawah garis
        assert check_channel_break(channel, closes, tol=0.02).status == CHANNEL_INTACT
        assert check_channel_break(channel, closes, tol=0.005).status == BASIC_BREAK_TREND_CHANGE

    def test_downtrend_basic_break_is_bullish_and_channel_break_is_bearish(self):
        basic = _down_basic()
        lows = _series([basic.value_at(i) - 8 for i in range(20)])
        highs = _series([basic.value_at(i) + 1 for i in range(20)])
        channel = build_channel(basic, highs, lows, np.array([0, 4]), np.array([6, 12]))

        up = _series([channel.basic_value_at(i) - 4 for i in range(20)])
        up.iloc[10] = channel.basic_value_at(10) + 2
        breaking_up = check_channel_break(channel, up, tol=0.0)
        assert breaking_up.status == BASIC_BREAK_TREND_CHANGE
        assert breaking_up.bias == BIAS_BULLISH

        down = _series([channel.basic_value_at(i) - 4 for i in range(20)])
        down.iloc[10] = channel.channel_value_at(10) - 2
        breaking_down = check_channel_break(channel, down, tol=0.0)
        assert breaking_down.status == CHANNEL_BREAK_ACCELERATION
        assert breaking_down.bias == BIAS_BEARISH

    def test_bars_up_to_last_basic_anchor_are_skipped(self):
        channel = self._uptrend_channel()
        closes = _series([channel.basic_value_at(i) + 5 for i in range(20)])
        closes.iloc[3] = channel.basic_value_at(3) - 5  # sebelum titik acuan terakhir (bar 4)
        assert check_channel_break(channel, closes, tol=0.0).status == CHANNEL_INTACT
