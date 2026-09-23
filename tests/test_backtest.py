"""Unit test mesin backtest walk-forward.

Yang dikunci di sini: tidak ada lookahead, aturan keluar dijalankan dengan
urutan yang benar, biaya ikut dihitung, dan gerbang Lapis 1 benar-benar
menahan sinyal. Semua data sintetis - tidak ada jaringan.
"""

import numpy as np
import pandas as pd
import pytest

from screener.backtest import (
    EXIT_FUNDAMENTAL,
    EXIT_STILL_OPEN,
    BacktestResult,
    Trade,
    run_backtest,
    summarize,
)
from screener.breakout import EXIT_CUT_LOSS, EXIT_TAKE_PROFIT


def _zigzag(peaks, troughs, bars_per_leg=6):
    anchors = [peaks[0]]
    for trough, peak in zip(troughs, peaks, strict=True):
        anchors += [trough, peak]
    anchors.append(troughs[-1])
    points = []
    for start, end in zip(anchors, anchors[1:], strict=False):
        points += list(np.linspace(start, end, bars_per_leg, endpoint=False))
    points.append(anchors[-1])
    close = pd.Series(points, dtype=float)
    dates = pd.date_range("2024-01-01", periods=len(close), freq="W-MON", tz="Asia/Jakarta")
    close.index = dates
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 10_000_000,
        }
    )


def _uptrend_frame():
    return _zigzag(peaks=[110, 125, 140, 155, 170], troughs=[100, 108, 120, 135, 150])


def _false_breakout_frame():
    """Tren naik yang breakout-nya gagal: harga jebol jauh di bawah resistance
    lama beberapa bar setelah masuk - kasus yang memicu cut-loss."""
    df = _uptrend_frame()
    tail = [178.0, 182.0, 150.0, 145.0, 140.0, 138.0]  # tembus lalu ambruk
    index = pd.date_range(
        df.index[-1] + pd.Timedelta(weeks=1), periods=len(tail), freq="W-MON", tz=df.index.tz
    )
    close = pd.Series(tail, index=index)
    extra = pd.DataFrame(
        {
            "Open": [df["Close"].iloc[-1]] + tail[:-1],
            "High": close + 2,
            "Low": close - 2,
            "Close": close,
            "Volume": 10_000_000,
        },
        index=index,
    )
    return pd.concat([df, extra])


class TestEngineBasics:
    def test_runs_and_reports_buy_hold(self):
        df = _uptrend_frame()
        result = run_backtest(df, ticker="TEST.JK", warmup=10)
        assert isinstance(result, BacktestResult)
        assert result.bars_tested == len(df) - 10
        expected = (df["Close"].iloc[-1] - df["Close"].iloc[10]) / df["Close"].iloc[10] * 100
        assert result.buy_hold_pct == pytest.approx(expected)

    def test_empty_frame_is_rejected(self):
        with pytest.raises(ValueError):
            run_backtest(pd.DataFrame())

    @pytest.mark.parametrize("warmup", [0, 999])
    def test_impossible_warmup_is_rejected(self, warmup):
        with pytest.raises(ValueError):
            run_backtest(_uptrend_frame(), warmup=warmup)

    def test_eligible_length_must_match(self):
        with pytest.raises(ValueError):
            run_backtest(_uptrend_frame(), eligible=[True, False], warmup=10)

    def test_only_one_position_at_a_time(self):
        result = run_backtest(_uptrend_frame(), warmup=10)
        for earlier, later in zip(result.trades, result.trades[1:], strict=False):
            assert earlier.exit_index is not None
            assert later.entry_index > earlier.exit_index

    def test_open_position_is_closed_at_the_last_bar(self):
        df = _uptrend_frame()
        result = run_backtest(df, warmup=10)
        if result.trades and result.trades[-1].reason == EXIT_STILL_OPEN:
            assert result.trades[-1].exit_index == len(df) - 1


class TestNoLookahead:
    def test_entry_only_on_the_bar_where_the_signal_forms(self):
        df = _uptrend_frame()
        result = run_backtest(df, warmup=10)
        for trade in result.trades:
            assert trade.entry_index >= 10
            # Harga masuk harus sama dengan Open bar itu (aturan 2nd day).
            assert trade.entry_price == pytest.approx(float(df["Open"].iloc[trade.entry_index]))

    def test_exit_never_happens_before_entry(self):
        result = run_backtest(_uptrend_frame(), warmup=10)
        for trade in result.trades:
            assert trade.exit_index > trade.entry_index
            assert trade.bars_held >= 1

    def test_result_does_not_change_when_future_bars_are_appended(self):
        """Uji paling tegas: menambah data SETELAH periode uji tidak boleh
        mengubah satu pun keputusan di dalam periode itu."""
        df = _uptrend_frame()
        base = run_backtest(df, warmup=10)

        extra = df.tail(20).copy() * 1.5
        extra.index = pd.date_range(
            df.index[-1] + pd.Timedelta(weeks=1), periods=len(extra), freq="W-MON",
            tz=df.index.tz,
        )
        extended = run_backtest(pd.concat([df, extra]), warmup=10)

        inside = [t for t in extended.trades if t.entry_index < len(df)]
        for original, later in zip(base.trades, inside, strict=False):
            assert (original.entry_index, original.entry_price) == (
                later.entry_index, later.entry_price
            )


class TestExitRules:
    def _entered(self, frame=None, **kwargs):
        df = frame if frame is not None else _uptrend_frame()
        result = run_backtest(df, warmup=10, **kwargs)
        return df, result

    def test_cut_loss_fires_when_close_drops_below_the_level(self):
        df, result = self._entered(_false_breakout_frame())
        stopped = [t for t in result.trades if t.reason == EXIT_CUT_LOSS]
        assert stopped, "breakout gagal harus memicu cut-loss"
        for trade in stopped:
            assert trade.exit_price < trade.cut_loss
            # Bar sebelumnya belum boleh menembus cut-loss.
            assert float(df["Close"].iloc[trade.exit_index - 1]) >= trade.cut_loss

    def test_cut_loss_can_be_switched_off(self):
        frame = _false_breakout_frame()
        _, with_stop = self._entered(frame)
        _, without_stop = self._entered(frame, use_cut_loss=False)
        assert any(t.reason == EXIT_CUT_LOSS for t in with_stop.trades)
        assert all(t.reason != EXIT_CUT_LOSS for t in without_stop.trades)

    def test_trend_break_exit_happens_in_a_clean_uptrend(self):
        _, result = self._entered()
        assert any(t.reason == EXIT_TAKE_PROFIT for t in result.trades)

    def test_exit_reasons_are_from_the_known_set(self):
        _, result = self._entered(_false_breakout_frame())
        reasons = {t.reason for t in result.trades}
        assert reasons <= {EXIT_CUT_LOSS, EXIT_TAKE_PROFIT, EXIT_STILL_OPEN, EXIT_FUNDAMENTAL}

    def test_fundamental_exit_closes_the_position(self):
        df = _uptrend_frame()
        eligible = [True] * len(df)
        eligible[25:] = [False] * (len(df) - 25)
        result = run_backtest(
            df, warmup=10, eligible=eligible, exit_on_fundamental=True, use_cut_loss=False
        )
        for trade in result.trades:
            if trade.entry_index < 25:
                assert trade.exit_index <= 25 or trade.reason == EXIT_FUNDAMENTAL


class TestFundamentalGate:
    def test_no_entry_while_ineligible(self):
        df = _uptrend_frame()
        result = run_backtest(df, warmup=10, eligible=[False] * len(df))
        assert result.trades == []
        assert result.eligible_bars == 0
        assert result.exposure_pct == 0

    def test_gate_reduces_trade_count(self):
        df = _uptrend_frame()
        open_gate = run_backtest(df, warmup=10)
        half = [i % 2 == 0 for i in range(len(df))]
        gated = run_backtest(df, warmup=10, eligible=half)
        assert len(gated.trades) <= len(open_gate.trades)

    def test_exposure_is_reported(self):
        df = _uptrend_frame()
        eligible = [i >= len(df) // 2 for i in range(len(df))]
        result = run_backtest(df, warmup=10, eligible=eligible)
        assert 0 < result.exposure_pct < 100


class TestFeesAndReturns:
    def _trade(self, entry=100.0, exit_price=110.0, **kwargs):
        trade = Trade(
            ticker="X", entry_index=1, entry_date="2024-01-01", entry_price=entry,
            resistance=entry, cut_loss=entry * 0.985, **kwargs
        )
        trade.exit_index, trade.exit_price, trade.exit_date = 5, exit_price, "2024-02-01"
        return trade

    def test_net_is_lower_than_gross_because_of_fees(self):
        trade = self._trade()
        assert trade.gross_pct == pytest.approx(10.0)
        assert trade.net_pct < trade.gross_pct
        assert trade.net_pct == pytest.approx(
            (110 * (1 - 0.0025) - 100 * 1.0015) / (100 * 1.0015) * 100
        )

    def test_open_trade_has_no_return_yet(self):
        trade = Trade(ticker="X", entry_index=1, entry_date="d", entry_price=100,
                      resistance=100, cut_loss=98)
        assert trade.net_pct is None and trade.gross_pct is None
        assert trade.bars_held is None and not trade.is_closed

    def test_zero_fees_make_net_equal_gross(self):
        trade = self._trade(fee_buy=0.0, fee_sell=0.0)
        assert trade.net_pct == pytest.approx(trade.gross_pct)

    def test_compound_chains_the_trades(self):
        result = BacktestResult(ticker="X")
        result.trades = [self._trade(100, 110, fee_buy=0.0, fee_sell=0.0),
                         self._trade(100, 90, fee_buy=0.0, fee_sell=0.0)]
        assert result.compound_pct == pytest.approx((1.10 * 0.90 - 1) * 100)


class TestSummary:
    def _trade(self, net_target, reason=EXIT_CUT_LOSS):
        trade = Trade(ticker="X", entry_index=0, entry_date="d", entry_price=100.0,
                      resistance=100.0, cut_loss=98.0, fee_buy=0.0, fee_sell=0.0)
        trade.exit_index, trade.exit_date = 4, "d2"
        trade.exit_price = 100.0 * (1 + net_target / 100)
        trade.reason = reason
        return trade

    def test_statistics_match_hand_calculation(self):
        trades = [self._trade(20), self._trade(-10), self._trade(-10), self._trade(-10)]
        summary = summarize(trades)
        assert summary.trades == 4 and summary.wins == 1
        assert summary.win_rate == pytest.approx(25.0)
        assert summary.expectancy_pct == pytest.approx(-2.5)
        assert summary.profit_factor == pytest.approx(20 / 30)
        assert summary.avg_win_pct == pytest.approx(20.0)
        assert summary.avg_loss_pct == pytest.approx(-10.0)

    def test_empty_input_is_safe(self):
        assert summarize([]).trades == 0

    def test_grouped_by_exit_reason(self):
        trades = [self._trade(20, EXIT_TAKE_PROFIT), self._trade(-10), self._trade(-10)]
        summary = summarize(trades)
        assert summary.by_reason[EXIT_TAKE_PROFIT] == (1, pytest.approx(20.0))
        assert summary.by_reason[EXIT_CUT_LOSS][0] == 2

    def test_accepts_results_as_well_as_trades(self):
        result = BacktestResult(ticker="X")
        result.trades = [self._trade(20), self._trade(-10)]
        assert summarize([result]).trades == 2

    def test_profit_factor_is_infinite_without_losses(self):
        assert summarize([self._trade(10), self._trade(5)]).profit_factor == float("inf")
