"""Unit test orkestrasi analisis chart (Lapis 2).

Fokusnya dua hal yang sebelumnya tidak bisa diuji karena logikanya ada di
app.py: rencana keluar tidak boleh memakai informasi masa depan, dan level
S/R punya jumlah sendiri supaya level lama tidak hilang.
"""

import numpy as np
import pandas as pd
import pytest

from screener.analysis import (
    DEFAULT_LEVELS_PER_SIDE,
    analyze_chart,
    build_exit_trendline,
)
from screener.extrema import get_extrema
from screener.fan import FAN_BEARISH, FAN_BULLISH
from screener.levels import RESISTANCE, SUPPORT
from screener.trend import DOWNTREND, UPTREND


def _zigzag_frame(peaks, troughs, bars_per_leg=6):
    """OHLC sintetis zigzag: dasar[0] -> puncak[0] -> dasar[1] -> ..."""
    anchors = [peaks[0]]
    for trough, peak in zip(troughs, peaks, strict=True):
        anchors += [trough, peak]
    anchors.append(troughs[-1])
    points = []
    for start, end in zip(anchors, anchors[1:], strict=False):
        points += list(np.linspace(start, end, bars_per_leg, endpoint=False))
    points.append(anchors[-1])
    close = pd.Series(points, dtype=float)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
        }
    )


def _uptrend_frame():
    # Tiap lembah harus di bawah puncak sebelumnya supaya zigzag-nya nyata,
    # sambil tetap higher high & higher low.
    return _zigzag_frame(peaks=[110, 125, 140, 155], troughs=[100, 108, 120, 135])


def _downtrend_frame():
    return _zigzag_frame(peaks=[155, 140, 125, 110], troughs=[145, 130, 115, 100])


class TestAnalyzeChart:
    def test_runs_the_whole_layer_two_pipeline(self):
        analysis = analyze_chart(_uptrend_frame(), distance=3)
        assert analysis.trend.trend == UPTREND
        assert analysis.trendline is not None
        assert analysis.break_check is not None
        assert analysis.levels, "level S/R harus terbentuk dari swing"
        assert len(analysis.highs_idx) and len(analysis.lows_idx)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError):
            analyze_chart(pd.DataFrame())

    def test_fan_direction_follows_the_running_trend(self):
        assert analyze_chart(_uptrend_frame(), distance=3).fan_direction == FAN_BEARISH
        assert analyze_chart(_downtrend_frame(), distance=3).fan_direction == FAN_BULLISH

    def test_no_fan_direction_when_trend_is_unclear(self):
        # Puncak naik, dasar turun -> tidak masuk definisi tren mana pun.
        frame = _zigzag_frame(peaks=[110, 125, 140], troughs=[100, 90, 80])
        analysis = analyze_chart(frame, distance=3)
        assert analysis.trend.trend not in (UPTREND, DOWNTREND)
        assert analysis.fan_direction is None and analysis.fan is None

    def test_sideways_chart_has_no_trendline_or_channel(self):
        frame = _zigzag_frame(peaks=[120, 121, 120], troughs=[100, 101, 100])
        analysis = analyze_chart(frame, distance=3)
        assert analysis.trendline is None
        assert analysis.channel is None and analysis.channel_break is None


class TestLevelCount:
    """Level lama tidak boleh hilang hanya karena lookback tren kecil."""

    def test_level_count_is_independent_of_trend_lookback(self):
        frame = _uptrend_frame()
        analysis = analyze_chart(frame, distance=3, lookback_swings=2, levels_per_side=4)
        assert len(analysis.trend.peaks) == 2  # klasifikasi tren tetap pakai 2
        assert len(analysis.levels) > 2  # level tetap banyak

    def test_more_levels_reach_further_back_in_time(self):
        frame = _uptrend_frame()
        few = analyze_chart(frame, distance=3, levels_per_side=1)
        many = analyze_chart(frame, distance=3, levels_per_side=4)
        assert len(many.levels) > len(few.levels)
        assert min(lv.origin_index for lv in many.levels) < min(
            lv.origin_index for lv in few.levels
        )

    def test_default_keeps_more_levels_than_trend_lookback(self):
        assert DEFAULT_LEVELS_PER_SIDE > 3


class TestNoLookahead:
    """Garis keluar hanya boleh dibangun dari swing sampai bar masuk."""

    def test_exit_trendline_uses_only_past_anchors(self):
        df = _uptrend_frame()
        highs_idx, lows_idx = get_extrema(df["Close"], distance=3)
        cutoff = int(lows_idx[1])
        line = build_exit_trendline(
            df, highs_idx, lows_idx, until_index=cutoff,
            tol=0.02, lookback_swings=3, confirmation_ratio=1.0,
        )
        if line is not None:
            assert all(p.index <= cutoff for p in line.points)

    def test_exit_line_is_extended_to_the_last_bar(self):
        df = _uptrend_frame()
        highs_idx, lows_idx = get_extrema(df["Close"], distance=3)
        line = build_exit_trendline(
            df, highs_idx, lows_idx, until_index=int(lows_idx[-1]),
            tol=0.02, lookback_swings=2, confirmation_ratio=1.0,
        )
        if line is not None:
            assert line.end_index == len(df) - 1

    def test_no_line_before_a_trend_exists(self):
        df = _uptrend_frame()
        highs_idx, lows_idx = get_extrema(df["Close"], distance=3)
        # Di bar-bar awal belum ada cukup swing untuk menyatakan uptrend.
        assert build_exit_trendline(
            df, highs_idx, lows_idx, until_index=3,
            tol=0.02, lookback_swings=3, confirmation_ratio=1.0,
        ) is None

    def test_plans_never_use_anchors_from_after_entry(self):
        analysis = analyze_chart(_uptrend_frame(), distance=3, levels_per_side=6)
        for plan in analysis.plans_with_entry:
            if plan.exit_trendline is None:
                continue
            entry_index = plan.signal.entry_index
            assert all(p.index <= entry_index for p in plan.exit_trendline.points), (
                "titik acuan garis keluar terbentuk setelah posisi dibuka"
            )


class TestResistancePlans:
    def test_only_resistance_levels_get_a_plan(self):
        analysis = analyze_chart(_uptrend_frame(), distance=3)
        assert all(p.level.initial_role == RESISTANCE for p in analysis.resistance_plans)
        assert any(lv.initial_role == SUPPORT for lv in analysis.levels)

    def test_latest_plan_is_the_most_recent_entry(self):
        analysis = analyze_chart(_uptrend_frame(), distance=3, levels_per_side=6)
        if analysis.plans_with_entry:
            latest = analysis.latest_plan
            assert latest.signal.entry_index == max(
                p.signal.entry_index for p in analysis.plans_with_entry
            )

    def test_plan_numbers_follow_the_book_rules(self):
        analysis = analyze_chart(_uptrend_frame(), distance=3, levels_per_side=6)
        for plan in analysis.plans_with_entry:
            resistance = plan.level.price
            assert plan.signal.threshold == pytest.approx(resistance * 1.015)
            assert plan.plan.cut_loss == pytest.approx(resistance * 0.985)
            assert plan.plan.support == resistance
            assert plan.plan.risk_per_share == pytest.approx(
                plan.plan.entry_price - plan.plan.cut_loss
            )
