"""Unit test keselarasan gambar: tiap penanda harus duduk tepat di bar-nya.

Sumbu x memakai posisi bar, jadi koordinat semua trace wajib berupa indeks
bar (bilangan bulat 0..n-1) - bukan tanggal. Ini yang menjamin marker, garis,
dan candle tetap sejajar untuk interval apa pun dan parameter apa pun.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import pytest

from screener.breakout import build_trading_plan, evaluate_breakout_position, find_breakout
from screener.channel import build_channel
from screener.extrema import get_extrema
from screener.fan import FAN_BEARISH, detect_fan
from screener.levels import levels_from_swings
from screener.plotting import (
    add_break_markers,
    add_breakout_plan,
    add_channel,
    add_fan,
    add_levels,
    add_trendline,
    bar_positions,
    plot_candlestick,
)
from screener.trend import build_trendline, check_trendline_break, classify_trend


def _chart(n_bars=120, freq="B"):
    """OHLC sintetis zigzag naik dengan tanggal nyata (ada celah akhir pekan)."""
    dates = pd.date_range("2025-01-06", periods=n_bars, freq=freq, tz="Asia/Jakarta")
    wave = np.sin(np.linspace(0, 6 * np.pi, n_bars)) * 40
    close = pd.Series(1000 + np.linspace(0, 300, n_bars) + wave, index=dates)
    return pd.DataFrame(
        {"Open": close.shift(1).fillna(close.iloc[0]), "High": close + 15,
         "Low": close - 15, "Close": close}
    )


def _all_x(fig: go.Figure) -> list:
    values = []
    for trace in fig.data:
        x = trace.x
        if x is not None:
            values.extend(list(x))
    return values


class TestBarPositionAxis:
    def test_candlestick_x_is_bar_index_not_date(self):
        df = _chart()
        fig = plot_candlestick(df, title="T")
        assert list(fig.data[0].x) == list(range(len(df)))
        assert list(bar_positions(df)) == list(range(len(df)))

    def test_dates_are_shown_as_tick_labels(self):
        df = _chart()
        fig = plot_candlestick(df)
        ticktext = fig.layout.xaxis.ticktext
        assert ticktext, "label tanggal harus tetap ada di sumbu"
        assert str(df.index[0].date()) == ticktext[0]
        assert all(isinstance(v, int) for v in fig.layout.xaxis.tickvals)

    def test_no_rangebreaks_needed(self):
        # Celah akhir pekan tidak perlu disembunyikan lagi; sumbunya sudah
        # berjarak seragam, dan lebar candle ikut benar.
        fig = plot_candlestick(_chart())
        assert not fig.layout.xaxis.rangebreaks

    def test_swing_markers_sit_on_their_own_bars(self):
        df = _chart()
        highs_idx, lows_idx = get_extrema(df["Close"], distance=5)
        fig = plot_candlestick(df, highs_idx=highs_idx, lows_idx=lows_idx)
        high_trace = next(t for t in fig.data if t.name == "Swing high")
        low_trace = next(t for t in fig.data if t.name == "Swing low")
        assert list(high_trace.x) == list(highs_idx)
        assert list(low_trace.x) == list(lows_idx)
        # Nilai y-nya memang High/Low bar tersebut.
        assert list(high_trace.y) == list(df["High"].iloc[highs_idx])
        assert list(low_trace.y) == list(df["Low"].iloc[lows_idx])


def _full_chart_figure(df, distance=5, tol=0.02):
    """Bangun chart selengkap mungkin: trendline, channel, level, breakout, fan."""
    highs_idx, lows_idx = get_extrema(df["Close"], distance=distance)
    trend = classify_trend(df["Close"], highs_idx, lows_idx)
    fig = plot_candlestick(df, highs_idx=highs_idx, lows_idx=lows_idx)
    trendline = build_trendline(trend, df["Low"], df["High"], len(df) - 1)
    if trendline is not None:
        add_trendline(fig, df, trendline)
        break_check = check_trendline_break(
            trendline, df["Close"], df["Low"], df["High"], df["Open"], tol=tol
        )
        add_break_markers(fig, df, trendline, break_check)
        channel = build_channel(trendline, df["High"], df["Low"], highs_idx, lows_idx)
        if channel is not None:
            add_channel(fig, df, channel)
    add_levels(fig, df, levels_from_swings(df, highs_idx, lows_idx, per_side=3))
    fan = detect_fan(FAN_BEARISH, df["Close"], df["High"], df["Low"], highs_idx, lows_idx)
    if fan is not None:
        add_fan(fig, df, fan)
    resistance = float(df["High"].iloc[: len(df) // 3].max())
    signal = find_breakout(resistance, df["Close"], df["Open"], start_index=1)
    if signal is not None and signal.entry_index is not None:
        plan = build_trading_plan(resistance, signal.entry_price)
        position = evaluate_breakout_position(plan, signal.entry_index, df["Close"])
        add_breakout_plan(fig, df, signal, plan, position)
    return fig


class TestEveryOverlayStaysOnTheGrid:
    """Apa pun parameternya, tiap koordinat x harus indeks bar yang sah."""

    @pytest.mark.parametrize("freq", ["B", "W-MON"])  # harian & mingguan
    @pytest.mark.parametrize("distance", [3, 5, 10])
    def test_all_trace_x_are_valid_bar_indices(self, freq, distance):
        df = _chart(freq=freq)
        fig = _full_chart_figure(df, distance=distance)
        xs = _all_x(fig)
        assert xs, "chart harus punya isi"
        for value in xs:
            assert isinstance(value, (int, np.integer)), f"{value!r} bukan indeks bar"
            assert 0 <= value <= len(df) - 1

    @pytest.mark.parametrize("tol", [0.0, 0.02, 0.05])
    def test_tolerance_changes_markers_but_keeps_alignment(self, tol):
        df = _chart()
        fig = _full_chart_figure(df, tol=tol)
        for value in _all_x(fig):
            assert 0 <= value <= len(df) - 1

    def test_weekly_and_daily_use_the_same_grid_width(self):
        # Sumbu posisi bar membuat jarak antar-bar selalu 1, jadi lebar candle
        # benar untuk kedua interval - ini yang dulu bikin candle mingguan melebar.
        for freq in ("B", "W-MON"):
            df = _chart(freq=freq)
            x = list(plot_candlestick(df).data[0].x)
            assert np.allclose(np.diff(x), 1)
