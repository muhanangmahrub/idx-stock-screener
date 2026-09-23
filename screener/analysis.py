"""Orkestrasi analisis chart Lapis 2 - satu pintu untuk semua detektor.

Sebelumnya urutan ini hidup di `app.py` sehingga tidak bisa diuji tanpa
menjalankan Streamlit. Di sini hasilnya berupa objek biasa, jadi UI tinggal
menggambar dan logikanya punya unit test sendiri.

Dua hal yang diperbaiki sekalian (audit 2026-09-23):

1. **Tanpa lookahead.** Trendline untuk menilai keluar posisi dibangun hanya
   dari swing yang terjadi SAMPAI bar masuk. Sebelumnya garis dibangun dari
   seluruh periode, sehingga keputusan keluar memakai titik acuan yang baru
   terbentuk setelah posisi dibuka - hasil simulasinya jadi terlalu bagus.
2. **Level lama ikut dinilai.** Jumlah level S/R punya parameter sendiri,
   terpisah dari `lookback_swings` milik klasifikasi tren. Buku menyatakan
   level yang lebih lama lebih kuat, jadi level lama justru tidak boleh
   hilang dari daftar.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from screener.breakout import (
    BREAKOUT_TOLERANCE,
    CUT_LOSS_TOLERANCE,
    BreakoutSignal,
    PositionStatus,
    TradingPlan,
    build_trading_plan,
    count_resistance_tests,
    evaluate_breakout_position,
    find_breakout,
)
from screener.channel import Channel, build_channel, check_channel_break
from screener.extrema import get_extrema
from screener.fan import FAN_BEARISH, FAN_BULLISH, FanPattern, detect_fan
from screener.levels import RESISTANCE, LevelStatus, levels_from_swings
from screener.trend import (
    DEFAULT_BREAK_TOLERANCE,
    DEFAULT_CONFIRMATION_RATIO,
    DEFAULT_LOOKBACK_SWINGS,
    DEFAULT_TOLERANCE,
    DOWNTREND,
    UPTREND,
    BreakCheck,
    Trendline,
    TrendResult,
    build_trendline,
    check_trendline_break,
    classify_trend,
)

# Level S/R default lebih banyak daripada lookback tren: klasifikasi tren hanya
# butuh beberapa swing terakhir, sedangkan level justru makin kuat bila lama.
DEFAULT_LEVELS_PER_SIDE = 6


@dataclass
class ResistancePlan:
    """Satu resistance beserta uji, breakout, rencana, dan hasilnya."""

    level: LevelStatus
    tests: int
    signal: BreakoutSignal | None = None
    plan: TradingPlan | None = None
    position: PositionStatus | None = None
    exit_trendline: Trendline | None = None  # garis yang dipakai menilai keluar


@dataclass
class ChartAnalysis:
    highs_idx: np.ndarray
    lows_idx: np.ndarray
    trend: TrendResult
    trendline: Trendline | None = None
    break_check: BreakCheck | None = None
    channel: Channel | None = None
    channel_break: object | None = None
    fan: FanPattern | None = None
    fan_direction: str | None = None
    levels: list[LevelStatus] = field(default_factory=list)
    resistance_plans: list[ResistancePlan] = field(default_factory=list)

    @property
    def plans_with_entry(self) -> list[ResistancePlan]:
        return [p for p in self.resistance_plans if p.plan is not None]

    @property
    def latest_plan(self) -> ResistancePlan | None:
        """Rencana dengan tanggal masuk terbaru - yang digambar di chart."""
        entered = self.plans_with_entry
        return max(entered, key=lambda p: p.signal.entry_index) if entered else None


def build_exit_trendline(
    df: pd.DataFrame,
    highs_idx: np.ndarray,
    lows_idx: np.ndarray,
    until_index: int,
    tol: float,
    lookback_swings: int,
    confirmation_ratio: float,
) -> Trendline | None:
    """Up-trendline dari data SAMPAI `until_index` saja.

    Dipakai untuk menilai kapan posisi keluar, supaya keputusannya hanya
    memakai informasi yang sudah tersedia saat itu. None bila sampai bar itu
    trennya belum naik atau garisnya belum terbentuk.
    """
    visible_highs = highs_idx[highs_idx <= until_index]
    visible_lows = lows_idx[lows_idx <= until_index]
    closes = df["Close"].iloc[: until_index + 1]
    trend = classify_trend(
        closes, visible_highs, visible_lows, tol=tol, lookback_swings=lookback_swings
    )
    if trend.trend != UPTREND:
        return None
    return build_trendline(
        trend,
        df["Low"].iloc[: until_index + 1],
        df["High"].iloc[: until_index + 1],
        end_index=len(df) - 1,  # garis diperpanjang, tapi dibangun dari masa lalu
        confirmation_ratio=confirmation_ratio,
    )


def analyze_chart(
    df: pd.DataFrame,
    distance: int = 5,
    prominence: float | None = None,
    trend_tol: float = DEFAULT_TOLERANCE,
    lookback_swings: int = DEFAULT_LOOKBACK_SWINGS,
    confirmation_ratio: float = DEFAULT_CONFIRMATION_RATIO,
    break_tol: float = DEFAULT_BREAK_TOLERANCE,
    pullback_tol: float = 0.0,
    breakout_tol: float = BREAKOUT_TOLERANCE,
    cut_loss_tol: float = CUT_LOSS_TOLERANCE,
    levels_per_side: int = DEFAULT_LEVELS_PER_SIDE,
) -> ChartAnalysis:
    """Jalankan seluruh analisis Lapis 2 untuk satu DataFrame OHLC."""
    if df.empty:
        raise ValueError("DataFrame harga kosong")

    highs_idx, lows_idx = get_extrema(df["Close"], distance=distance, prominence=prominence)
    trend = classify_trend(
        df["Close"], highs_idx, lows_idx, tol=trend_tol, lookback_swings=lookback_swings
    )
    analysis = ChartAnalysis(highs_idx=highs_idx, lows_idx=lows_idx, trend=trend)
    last_index = len(df) - 1

    analysis.trendline = build_trendline(
        trend, df["Low"], df["High"], end_index=last_index,
        confirmation_ratio=confirmation_ratio,
    )
    if analysis.trendline is not None:
        analysis.break_check = check_trendline_break(
            analysis.trendline, df["Close"], df["Low"], df["High"], df["Open"], tol=break_tol
        )
        analysis.channel = build_channel(
            analysis.trendline, df["High"], df["Low"], highs_idx, lows_idx
        )
        if analysis.channel is not None:
            analysis.channel_break = check_channel_break(
                analysis.channel, df["Close"], tol=break_tol
            )

    # Fan principle hanya bila ada tren yang sedang berlangsung.
    if trend.trend in (UPTREND, DOWNTREND):
        analysis.fan_direction = FAN_BEARISH if trend.trend == UPTREND else FAN_BULLISH
        analysis.fan = detect_fan(
            analysis.fan_direction, df["Close"], df["High"], df["Low"],
            highs_idx, lows_idx, tol=break_tol,
        )

    analysis.levels = levels_from_swings(
        df, highs_idx, lows_idx, per_side=levels_per_side, pullback_tol=pullback_tol
    )

    for level in analysis.levels:
        if level.initial_role != RESISTANCE:
            continue
        signal = find_breakout(
            level.price, df["Close"], df["Open"],
            start_index=level.origin_index + 1, tol=breakout_tol,
        )
        tests = count_resistance_tests(
            level.price, highs_idx, df["High"], df["Close"],
            until_index=signal.breakout_index if signal else None,
            breakout_tol=breakout_tol,
        )
        entry = ResistancePlan(level=level, tests=tests, signal=signal)
        if signal is not None and signal.entry_index is not None:
            entry.plan = build_trading_plan(
                level.price, signal.entry_price, cut_loss_tol=cut_loss_tol
            )
            entry.exit_trendline = build_exit_trendline(
                df, highs_idx, lows_idx, signal.entry_index,
                tol=trend_tol, lookback_swings=lookback_swings,
                confirmation_ratio=confirmation_ratio,
            )
            entry.position = evaluate_breakout_position(
                entry.plan, signal.entry_index, df["Close"],
                trendline=entry.exit_trendline, trendline_tol=break_tol,
            )
        analysis.resistance_plans.append(entry)

    return analysis
