"""Acuan kebenaran: contoh saham McD dari buku Edianto Ong.

Resistance 50,5 -> tembus bila Close > 51,25; diuji 3 kali = strong
resistance; masuk di Open hari berikutnya 51,8; cut-loss 49,8 (1,5% di
bawah support baru 50,5); risiko 2/lembar; keluar saat trendline patah di
58,3, untung 6,5/lembar. Buku membulatkan ke 0,05 - test memakai
toleransi pembulatan itu.
"""

import numpy as np
import pandas as pd
import pytest

from screener.breakout import (
    BREAKOUT_TOLERANCE,
    CUT_LOSS_TOLERANCE,
    EXIT_CUT_LOSS,
    EXIT_TAKE_PROFIT,
    POSITION_HOLD,
    SECOND_DAY_CONFIRMED,
    SECOND_DAY_PENDING,
    STRONG_RESISTANCE_MIN_TESTS,
    breakout_threshold,
    build_trading_plan,
    count_resistance_tests,
    evaluate_breakout_position,
    find_breakout,
    is_strong_resistance,
)
from screener.trend import UP_TRENDLINE, DOWN_TRENDLINE, AnchorPoint, Trendline

MCD_RESISTANCE = 50.5
BOOK_ROUNDING = 0.06  # buku membulatkan 51,26 -> 51,25 dan 49,74 -> 49,8


def _series(values):
    return pd.Series(values, dtype=float)


def _mcd_chart():
    """Chart sintetis McD (Close): tiga puncak gagal di 50,5, breakout, rally, patah.

    bar 0-11 : tiga kali menguji resistance 50,5 (swing high di bar 2, 6, 10)
    bar 12   : breakout, Close 51,5 > 51,25
    bar 13   : Open 51,8 = masuk (2nd day)
    bar 14-19: rally ke 60
    bar 20   : Close 58,3 menembus trendline -> keluar
    """
    closes = _series([
        48.0, 49.5, 50.4, 49.0, 48.5, 49.8, 50.3, 48.8, 49.2, 50.0, 50.5, 49.6,  # 0-11
        51.5,                                                                      # 12 breakout
        52.5, 53.5, 55.0, 56.5, 58.0, 59.5, 60.0,                                  # 13-19 rally
        58.3,                                                                      # 20 patah
    ])
    opens = closes.shift(1).fillna(closes.iloc[0])
    opens.iloc[13] = 51.8  # Open hari setelah breakout = harga masuk buku
    highs = closes + 0.2
    highs.iloc[[2, 6, 10]] = 50.5  # ketiga puncak tepat menyentuh resistance
    lows = closes - 0.2
    highs_idx = np.array([2, 6, 10])
    return pd.DataFrame({"Open": opens, "High": highs, "Low": lows, "Close": closes}), highs_idx


class TestBreakoutThreshold:
    def test_book_example_resistance_50_5_needs_close_above_51_25(self):
        assert breakout_threshold(50.5) == pytest.approx(51.25, abs=BOOK_ROUNDING)

    def test_default_tolerance_is_book_value(self):
        assert BREAKOUT_TOLERANCE == 0.015
        assert breakout_threshold(100.0, tol=0.02) == pytest.approx(102.0)


class TestStrongResistance:
    def test_mcd_resistance_tested_three_times_is_strong(self):
        df, highs_idx = _mcd_chart()
        tests = count_resistance_tests(MCD_RESISTANCE, highs_idx, df["High"], df["Close"])
        assert tests == 3
        assert is_strong_resistance(tests)
        assert STRONG_RESISTANCE_MIN_TESTS == 3

    def test_two_tests_is_not_strong(self):
        assert not is_strong_resistance(2)

    def test_peak_far_below_level_is_not_a_test(self):
        df, highs_idx = _mcd_chart()
        highs = df["High"].copy()
        highs.iloc[6] = 48.0  # 5% di bawah level: bukan uji resistance
        tests = count_resistance_tests(MCD_RESISTANCE, highs_idx, highs, df["Close"])
        assert tests == 2

    def test_peak_that_closes_past_threshold_is_a_breakout_not_a_test(self):
        df, highs_idx = _mcd_chart()
        closes = df["Close"].copy()
        closes.iloc[10] = 51.6  # Close lewat 51,25 -> itu breakout, bukan uji gagal
        tests = count_resistance_tests(MCD_RESISTANCE, highs_idx, df["High"], closes)
        assert tests == 2

    def test_tests_after_until_index_are_ignored(self):
        df, highs_idx = _mcd_chart()
        tests = count_resistance_tests(
            MCD_RESISTANCE, highs_idx, df["High"], df["Close"], until_index=7
        )
        assert tests == 2


class TestFindBreakout:
    def test_mcd_breakout_and_second_day_entry(self):
        df, _ = _mcd_chart()
        signal = find_breakout(MCD_RESISTANCE, df["Close"], df["Open"], start_index=0)
        assert signal.breakout_index == 12
        assert signal.second_day == SECOND_DAY_CONFIRMED
        assert signal.entry_index == 13
        assert signal.entry_price == pytest.approx(51.8)
        assert signal.threshold == pytest.approx(51.25, abs=BOOK_ROUNDING)

    def test_touching_or_thin_close_is_not_a_breakout(self):
        # Close 51,0 di atas resistance tapi belum lewat 51,25 -> bukan breakout.
        closes = _series([49.0, 50.5, 51.0, 50.8, 50.2])
        assert find_breakout(MCD_RESISTANCE, closes, closes, start_index=0) is None

    def test_next_open_gapping_back_cancels_and_scan_continues(self):
        closes = _series([49.0, 51.5, 50.0, 49.5, 51.6, 52.0, 52.5])
        opens = closes.shift(1).fillna(49.0)
        opens.iloc[2] = 50.9  # gap kembali ke bawah 51,25 setelah breakout bar 1
        opens.iloc[5] = 51.7  # breakout bar 4 dikonfirmasi
        signal = find_breakout(MCD_RESISTANCE, closes, opens, start_index=0)
        assert signal.gap_back_indices == [1]
        assert signal.breakout_index == 4
        assert signal.entry_index == 5 and signal.entry_price == pytest.approx(51.7)

    def test_breakout_on_last_bar_is_pending(self):
        closes = _series([49.0, 50.0, 51.5])
        signal = find_breakout(MCD_RESISTANCE, closes, closes, start_index=0)
        assert signal.second_day == SECOND_DAY_PENDING
        assert signal.entry_index is None and signal.entry_price is None


class TestTradingPlan:
    def test_mcd_plan_entry_cut_loss_and_risk(self):
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        assert plan.support == 50.5
        assert plan.cut_loss == pytest.approx(49.8, abs=BOOK_ROUNDING)
        assert plan.risk_per_share == pytest.approx(2.0, abs=BOOK_ROUNDING)
        assert CUT_LOSS_TOLERANCE == 0.015

    def test_risk_pct_is_relative_to_entry(self):
        plan = build_trading_plan(100.0, entry_price=102.0)
        assert plan.cut_loss == pytest.approx(98.5)
        assert plan.risk_pct == pytest.approx(3.5 / 102.0)


def _mcd_trendline():
    """Up-trendline sintetis yang berjalan di bawah rally: nilai 59,6 di bar 20,
    toleransi 2% -> batas 58,4; Close 58,3 di bawahnya = patah."""
    slope = 1.0
    intercept = 59.6 - slope * 20
    return Trendline(
        kind=UP_TRENDLINE,
        slope=slope,
        intercept=intercept,
        points=[AnchorPoint(13, intercept + 13 * slope), AnchorPoint(16, intercept + 16 * slope)],
        end_index=20,
    )


class TestEvaluatePosition:
    def test_mcd_hold_through_rally_then_exit_when_trendline_breaks(self):
        df, _ = _mcd_chart()
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        status = evaluate_breakout_position(plan, entry_index=13, closes=df["Close"], trendline=_mcd_trendline())
        assert status.status == EXIT_TAKE_PROFIT
        assert status.exit_index == 20
        assert status.exit_price == pytest.approx(58.3)
        assert status.pnl_per_share == pytest.approx(6.5)

    def test_hold_while_price_stays_above_trendline(self):
        df, _ = _mcd_chart()
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        status = evaluate_breakout_position(
            plan, entry_index=13, closes=df["Close"], trendline=_mcd_trendline(), end_index=19
        )
        assert status.status == POSITION_HOLD
        assert status.exit_index is None
        assert status.pnl_per_share == pytest.approx(60.0 - 51.8)  # mengambang

    def test_false_breakout_closing_below_cut_loss_is_exit(self):
        closes = _series([51.5, 51.8, 51.0, 50.6, 49.5, 48.0])
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        status = evaluate_breakout_position(plan, entry_index=1, closes=closes)
        assert status.status == EXIT_CUT_LOSS
        assert status.exit_index == 4  # 49,5 < 49,74; bar 3 (50,6) masih di atas
        assert status.pnl_per_share == pytest.approx(49.5 - 51.8)

    def test_dip_to_support_without_closing_below_cut_loss_is_hold(self):
        closes = _series([51.5, 51.8, 50.6, 50.0, 52.0])
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        assert evaluate_breakout_position(plan, entry_index=1, closes=closes).status == POSITION_HOLD

    def test_cut_loss_checked_before_trendline_on_same_bar(self):
        closes = _series([51.5, 51.8, 53.0, 54.0, 49.0])
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        line = Trendline(UP_TRENDLINE, 0.0, 52.0, [AnchorPoint(1, 52.0), AnchorPoint(2, 52.0)], 4)
        status = evaluate_breakout_position(plan, entry_index=1, closes=closes, trendline=line)
        assert status.status == EXIT_CUT_LOSS

    def test_down_trendline_is_rejected(self):
        plan = build_trading_plan(MCD_RESISTANCE, entry_price=51.8)
        line = Trendline(DOWN_TRENDLINE, 0.0, 60.0, [AnchorPoint(0, 60.0), AnchorPoint(1, 60.0)], 5)
        with pytest.raises(ValueError):
            evaluate_breakout_position(plan, entry_index=1, closes=_series([50.0] * 6), trendline=line)
