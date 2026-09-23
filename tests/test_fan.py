"""Unit test The Fan Principle (Edianto Ong Bab 14).

Data sintetis dibangun dari spesifikasi kipas itu sendiri: titik pangkal +
tiga swing acuan + bar penembusan. Harga di antara kejadian sengaja ditaruh
di sisi "dalam" garis yang sedang berlaku, supaya penembusan yang terdeteksi
hanya yang memang dirancang.
"""

import numpy as np
import pandas as pd
import pytest

from screener.fan import (
    FAN_BEARISH,
    FAN_BULLISH,
    FAN_CONFIRMED,
    FAN_FORMING,
    FAN_LINE_COUNT,
    ROLE_RESISTANCE,
    ROLE_SUPPORT,
    detect_fan,
)
from screener.trend import DOWN_TRENDLINE, UP_TRENDLINE


def _line(origin, anchor):
    slope = (anchor[1] - origin[1]) / (anchor[0] - origin[0])
    return lambda i: origin[1] + slope * (i - origin[0])


def _fan_chart(origin, anchors, breaks, n_bars, bearish=True, cushion=6.0):
    """Bangun OHLC sintetis untuk kipas yang dirancang.

    Tiap bar diberi harga di sisi dalam garis yang sedang berlaku (di atas
    garis untuk fan bearish), kecuali di bar penembusan (di sisi luar) dan di
    bar acuan (tepat di garis, karena garis memang ditarik lewat titik itu).
    """
    lines = [_line(origin, a) for a in anchors]
    closes = np.zeros(n_bars)
    for i in range(n_bars):
        active = 0
        for k, anchor in enumerate(anchors):
            if i >= anchor[0]:
                active = k
        value = lines[active](i)
        if i in breaks:
            offset = -cushion if bearish else cushion
        else:
            offset = cushion if bearish else -cushion
        closes[i] = value + offset
    closes[origin[0]] = origin[1]
    for anchor in anchors:
        closes[anchor[0]] = anchor[1]

    closes = pd.Series(closes, dtype=float)
    highs = closes + 1.0
    lows = closes - 1.0
    # Swing acuan harus tepat di harga yang dirancang: Low untuk fan bearish,
    # High untuk fan bullish (sesuai kolom yang dipakai modul).
    for index, price in [origin] + list(anchors):
        if bearish:
            lows.iloc[index] = price
        else:
            highs.iloc[index] = price
    return pd.DataFrame({"Close": closes, "High": highs, "Low": lows})


def _bearish_fan():
    """Tren naik melemah: pangkal (0, 100); acuan makin landai; tiga tembusan."""
    origin = (0, 100.0)
    anchors = [(10, 140.0), (20, 150.0), (32, 155.0)]  # slope 4 -> 2,5 -> 1,72
    breaks = [15, 26, 40]
    df = _fan_chart(origin, anchors, breaks, n_bars=46)
    lows_idx = np.array([0, 10, 20, 32])
    highs_idx = np.array([5, 13, 24, 36])
    return df, highs_idx, lows_idx


def _bullish_fan():
    """Tren turun menguat: pangkal (0, 200); acuan makin landai; tiga tembusan."""
    origin = (0, 200.0)
    anchors = [(10, 160.0), (20, 150.0), (32, 145.0)]  # slope -4 -> -2,5 -> -1,72
    breaks = [15, 26, 40]
    df = _fan_chart(origin, anchors, breaks, n_bars=46, bearish=False)
    highs_idx = np.array([0, 10, 20, 32])
    lows_idx = np.array([5, 13, 24, 36])
    return df, highs_idx, lows_idx


def _retest_chart(touch_bar, close_above=False):
    """Kipas bearish yang setelah menembus garis pertama TETAP di bawahnya,
    lalu di `touch_bar` naik menyentuh garis itu lagi (uji ulang)."""
    line1 = _line((0, 100.0), (10, 140.0))  # nilai = 100 + 4i
    closes = pd.Series(
        [100, 108, 116, 124, 132, 138, 136, 138, 139, 140, 140,
         146, 140, 141, 142, 143, 144, 146, 150, 152, 154, 156, 158, 160, 162],
        dtype=float,
    )
    highs = closes + 1.0
    lows = closes - 1.0
    lows.iloc[0], lows.iloc[10], lows.iloc[18] = 100.0, 140.0, 150.0  # swing low acuan
    highs.iloc[touch_bar] = line1(touch_bar)  # High menyentuh garis pertama
    if close_above:
        closes.iloc[touch_bar] = line1(touch_bar) + 1
        highs.iloc[touch_bar] = line1(touch_bar) + 2
    df = pd.DataFrame({"Close": closes, "High": highs, "Low": lows})
    return df, np.array([5, touch_bar]), np.array([0, 10, 18])


def _detect(df, highs_idx, lows_idx, direction=FAN_BEARISH, **kwargs):
    kwargs.setdefault("tol", 0.0)
    return detect_fan(
        direction, df["Close"], df["High"], df["Low"], highs_idx, lows_idx, **kwargs
    )


class TestFanStructure:
    """Tiga garis memancar dari SATU titik pangkal dan makin melebar."""

    def test_three_lines_share_the_same_origin(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        assert len(fan.lines) == FAN_LINE_COUNT
        assert all(line.origin.index == 0 for line in fan.lines)
        assert all(line.origin.price == 100.0 for line in fan.lines)
        assert fan.origin.index == 0

    def test_each_line_is_flatter_than_the_previous(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        slopes = [line.slope for line in fan.lines]
        assert slopes == sorted(slopes, reverse=True)
        assert all(slope > 0 for slope in slopes)  # masih garis naik
        assert [line.anchor.index for line in fan.lines] == [10, 20, 32]
        assert [line.order for line in fan.lines] == [1, 2, 3]

    def test_each_anchor_comes_after_the_previous_break(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        for previous, current in zip(fan.lines, fan.lines[1:]):
            assert current.anchor.index > previous.break_index

    def test_lines_pass_through_origin_and_anchor(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        for line in fan.lines:
            assert line.value_at(line.origin.index) == pytest.approx(line.origin.price)
            assert line.value_at(line.anchor.index) == pytest.approx(line.anchor.price)

    def test_explicit_origin_is_respected(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx, origin_index=10)
        assert fan.origin.index == 10
        assert all(line.origin.index == 10 for line in fan.lines)


class TestConfirmationOnThirdLine:
    """Kunci teori: reversal hanya diakui saat garis KETIGA tertembus."""

    def test_third_break_confirms_reversal(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        assert fan.status == FAN_CONFIRMED
        assert fan.is_confirmed
        assert fan.lines_broken == 3
        assert fan.confirmation_index == fan.lines[2].break_index == 40

    def test_two_broken_lines_are_not_a_reversal(self):
        df, highs_idx, lows_idx = _bearish_fan()
        # Potong data sebelum penembusan ketiga (bar 40).
        fan = _detect(df, highs_idx, lows_idx, end_index=39)
        assert fan.status == FAN_FORMING
        assert not fan.is_confirmed
        assert fan.lines_broken == 2
        assert fan.confirmation_index is None
        assert "bisa sekadar koreksi" in fan.reason

    def test_one_broken_line_is_not_a_reversal(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx, end_index=25)
        assert fan.lines_broken == 1
        assert not fan.is_confirmed

    def test_third_line_untouched_stays_forming(self):
        df, highs_idx, lows_idx = _bearish_fan()
        closes = df["Close"].copy()
        closes.iloc[40:] = 200.0  # harga melesat naik, garis ketiga tidak tertembus
        df = df.assign(Close=closes, High=closes + 1)
        fan = _detect(df, highs_idx, lows_idx)
        assert len(fan.lines) == FAN_LINE_COUNT
        assert fan.lines[2].break_index is None
        assert fan.status == FAN_FORMING

    def test_tolerance_can_suppress_a_thin_third_break(self):
        df, highs_idx, lows_idx = _bearish_fan()
        strict = _detect(df, highs_idx, lows_idx, tol=0.0)
        loose = _detect(df, highs_idx, lows_idx, tol=0.10)  # 10%: tembusan tipis diabaikan
        assert strict.is_confirmed
        assert not loose.is_confirmed

    def test_negative_tolerance_is_rejected(self):
        df, highs_idx, lows_idx = _bearish_fan()
        with pytest.raises(ValueError):
            _detect(df, highs_idx, lows_idx, tol=-0.01)

    def test_unknown_direction_is_rejected(self):
        df, highs_idx, lows_idx = _bearish_fan()
        with pytest.raises(ValueError):
            _detect(df, highs_idx, lows_idx, direction="sideways")


class TestBothDirections:
    """Fan bullish adalah cermin fan bearish."""

    def test_bullish_fan_confirms_on_third_upward_break(self):
        df, highs_idx, lows_idx = _bullish_fan()
        fan = _detect(df, highs_idx, lows_idx, direction=FAN_BULLISH)
        assert fan.direction == FAN_BULLISH
        assert fan.origin.price == 200.0
        assert len(fan.lines) == FAN_LINE_COUNT
        assert fan.is_confirmed and fan.confirmation_index == 40

    def test_bullish_fan_lines_fall_and_flatten(self):
        df, highs_idx, lows_idx = _bullish_fan()
        fan = _detect(df, highs_idx, lows_idx, direction=FAN_BULLISH)
        slopes = [line.slope for line in fan.lines]
        assert all(slope < 0 for slope in slopes)
        assert slopes == sorted(slopes)  # makin landai: -4 < -2,5 < -1,72
        assert all(line.kind == DOWN_TRENDLINE for line in fan.lines)

    def test_bearish_fan_lines_are_up_trendlines(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        assert all(line.kind == UP_TRENDLINE for line in fan.lines)


class TestRoleReversal:
    """Garis yang sudah tertembus berganti fungsi jadi penghalang."""

    def test_broken_bearish_fan_line_becomes_resistance(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx)
        assert [line.role_after_break for line in fan.lines] == [ROLE_RESISTANCE] * 3

    def test_broken_bullish_fan_line_becomes_support(self):
        df, highs_idx, lows_idx = _bullish_fan()
        fan = _detect(df, highs_idx, lows_idx, direction=FAN_BULLISH)
        assert [line.role_after_break for line in fan.lines] == [ROLE_SUPPORT] * 3

    def test_unbroken_line_has_no_new_role(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx, end_index=39)
        assert fan.lines[-1].role_after_break is None

    def test_retest_of_a_broken_line_is_recorded(self):
        df, highs_idx, lows_idx = _retest_chart(touch_bar=14)
        fan = _detect(df, highs_idx, lows_idx)
        first = fan.lines[0]
        assert first.break_index == 12
        assert first.role_after_break == ROLE_RESISTANCE
        assert 14 in first.retest_indices

    def test_close_back_above_the_line_is_not_a_retest(self):
        # Bar yang sama, tapi Close merebut garis lagi: bukan uji ulang yang gagal.
        df, highs_idx, lows_idx = _retest_chart(touch_bar=14, close_above=True)
        fan = _detect(df, highs_idx, lows_idx)
        assert 14 not in fan.lines[0].retest_indices


class TestNoPattern:
    def test_no_swings_gives_no_fan(self):
        df, _, _ = _bearish_fan()
        assert _detect(df, np.array([]), np.array([])) is None

    def test_no_fan_when_no_line_has_been_broken(self):
        """Tanpa penembusan, yang ada baru sebuah trendline biasa - bukan kipas,
        jadi jangan dipaksakan jadi pola."""
        origin = (0, 100.0)
        anchors = [(10, 140.0)]
        df = _fan_chart(origin, anchors, breaks=[], n_bars=20)
        assert _detect(df, np.array([5, 15]), np.array([0, 10])) is None

    def test_first_break_starts_the_fan(self):
        df, highs_idx, lows_idx = _bearish_fan()
        fan = _detect(df, highs_idx, lows_idx, end_index=18)  # baru garis pertama tertembus
        assert fan is not None
        assert fan.lines_broken == 1
        assert fan.status == FAN_FORMING
