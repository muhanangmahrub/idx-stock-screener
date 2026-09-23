"""Backtest walk-forward untuk aturan yang sudah diimplementasi di repo ini.

Gunanya bukan mencari strategi terbaik, melainkan MENGUKUR DAMPAK: tiap kali
sebuah aturan diubah (toleransi breakout, cut-loss, syarat titik acuan), di
sini terlihat apa pengaruhnya terhadap ekspektasi dan profit factor - jadi
perubahan aturan tidak dinilai dari kesan, melainkan dari angka.

Prinsip yang dijaga:

- **Walk-forward, tanpa lookahead.** Tiap keputusan di bar `t` hanya memakai
  `df.iloc[:t+1]`, termasuk pembentukan level S/R dan trendline keluar.
- **Aturan yang diuji = aturan repo.** Mesin ini memanggil `analysis`,
  `breakout`, dan `trend`; tidak ada logika sinyal yang ditulis ulang di sini.
- **Hasil bukan janji.** Satu rezim pasar, universe pilihan sendiri, biaya
  diasumsikan. Angkanya untuk membandingkan antar-aturan, bukan untuk
  memperkirakan keuntungan masa depan.

Eksekusi yang diasumsikan: masuk di Open bar konfirmasi 2nd day (itu yang
dikembalikan `find_breakout`), keluar di Close bar saat syarat keluar
terpenuhi. Pada chart mingguan berarti cut-loss baru diperiksa di penutupan
minggu - kenyataannya harga bisa jatuh lebih dalam di tengah minggu.
"""

from dataclasses import dataclass, field
from statistics import fmean, median

import pandas as pd

from screener.analysis import DEFAULT_LEVELS_PER_SIDE, analyze_chart, build_exit_trendline
from screener.breakout import (
    BREAKOUT_TOLERANCE,
    CUT_LOSS_TOLERANCE,
    EXIT_CUT_LOSS,
    EXIT_TAKE_PROFIT,
)
from screener.extrema import get_extrema
from screener.trend import (
    DEFAULT_BREAK_TOLERANCE,
    DEFAULT_CONFIRMATION_RATIO,
    DEFAULT_LOOKBACK_SWINGS,
    DEFAULT_TOLERANCE,
    UP_TRENDLINE,
    break_threshold,
)

# ASUMSI biaya (bukan angka buku): komisi beli & komisi + pajak jual di IDX.
DEFAULT_FEE_BUY = 0.0015
DEFAULT_FEE_SELL = 0.0025

EXIT_FUNDAMENTAL = "fundamental tidak lagi lolos"
EXIT_STILL_OPEN = "masih terbuka di akhir periode"


@dataclass
class Trade:
    ticker: str
    entry_index: int
    entry_date: str
    entry_price: float
    resistance: float
    cut_loss: float
    exit_index: int | None = None
    exit_date: str | None = None
    exit_price: float | None = None
    reason: str = EXIT_STILL_OPEN
    fee_buy: float = DEFAULT_FEE_BUY
    fee_sell: float = DEFAULT_FEE_SELL

    @property
    def is_closed(self) -> bool:
        return self.exit_price is not None

    @property
    def gross_pct(self) -> float | None:
        if not self.is_closed:
            return None
        return (self.exit_price - self.entry_price) / self.entry_price * 100

    @property
    def net_pct(self) -> float | None:
        """Imbal hasil setelah biaya beli & jual."""
        if not self.is_closed:
            return None
        paid = self.entry_price * (1 + self.fee_buy)
        received = self.exit_price * (1 - self.fee_sell)
        return (received - paid) / paid * 100

    @property
    def bars_held(self) -> int | None:
        return None if self.exit_index is None else self.exit_index - self.entry_index


@dataclass
class BacktestResult:
    ticker: str
    trades: list[Trade] = field(default_factory=list)
    bars_tested: int = 0
    eligible_bars: int = 0
    buy_hold_pct: float = 0.0

    @property
    def closed_trades(self) -> list[Trade]:
        return [t for t in self.trades if t.is_closed]

    @property
    def compound_pct(self) -> float:
        """Hasil bila tiap trade dijalankan berurutan dengan seluruh modal."""
        total = 1.0
        for trade in self.closed_trades:
            total *= 1 + trade.net_pct / 100
        return (total - 1) * 100

    @property
    def exposure_pct(self) -> float:
        """Porsi waktu yang lolos Lapis 1 (100% bila tanpa filter)."""
        return 0.0 if not self.bars_tested else self.eligible_bars / self.bars_tested * 100


@dataclass
class Summary:
    trades: int = 0
    wins: int = 0
    win_rate: float = 0.0
    expectancy_pct: float = 0.0  # rata-rata imbal hasil bersih per trade
    profit_factor: float = 0.0
    avg_win_pct: float = 0.0
    avg_loss_pct: float = 0.0
    median_bars_held: float = 0.0
    by_reason: dict[str, tuple[int, float]] = field(default_factory=dict)


def summarize(results: list[BacktestResult] | list[Trade]) -> Summary:
    """Statistik gabungan; menerima daftar hasil per emiten atau daftar trade."""
    trades: list[Trade] = []
    for item in results:
        trades.extend(item.closed_trades if isinstance(item, BacktestResult) else [item])
    trades = [t for t in trades if t.is_closed]
    if not trades:
        return Summary()

    returns = [t.net_pct for t in trades]
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r <= 0]
    gross_loss = abs(sum(losses))

    by_reason: dict[str, tuple[int, float]] = {}
    for reason in {t.reason for t in trades}:
        subset = [t.net_pct for t in trades if t.reason == reason]
        by_reason[reason] = (len(subset), fmean(subset))

    return Summary(
        trades=len(trades),
        wins=len(wins),
        win_rate=len(wins) / len(trades) * 100,
        expectancy_pct=fmean(returns),
        profit_factor=(sum(wins) / gross_loss) if gross_loss else float("inf"),
        avg_win_pct=fmean(wins) if wins else 0.0,
        avg_loss_pct=fmean(losses) if losses else 0.0,
        median_bars_held=median([t.bars_held for t in trades]),
        by_reason=by_reason,
    )


def run_backtest(
    df: pd.DataFrame,
    ticker: str = "",
    eligible: list[bool] | None = None,
    warmup: int = 30,
    distance: int = 3,
    trend_tol: float = DEFAULT_TOLERANCE,
    lookback_swings: int = DEFAULT_LOOKBACK_SWINGS,
    confirmation_ratio: float = DEFAULT_CONFIRMATION_RATIO,
    break_tol: float = DEFAULT_BREAK_TOLERANCE,
    breakout_tol: float = BREAKOUT_TOLERANCE,
    cut_loss_tol: float = CUT_LOSS_TOLERANCE,
    levels_per_side: int = DEFAULT_LEVELS_PER_SIDE,
    use_cut_loss: bool = True,
    exit_on_fundamental: bool = False,
    fee_buy: float = DEFAULT_FEE_BUY,
    fee_sell: float = DEFAULT_FEE_SELL,
) -> BacktestResult:
    """Jalankan aturan repo bar demi bar pada satu emiten.

    `eligible[i]` (opsional) adalah hasil Lapis 1 pada bar i - dipakai sebagai
    gerbang masuk, dan bila `exit_on_fundamental` juga sebagai syarat keluar.
    Satu posisi terbuka pada satu waktu; posisi yang masih terbuka di akhir
    periode ditutup di bar terakhir supaya hasilnya bisa dihitung.
    """
    if df.empty:
        raise ValueError("DataFrame harga kosong")
    if eligible is not None and len(eligible) != len(df):
        raise ValueError("panjang `eligible` harus sama dengan jumlah bar")
    if warmup < 1 or warmup >= len(df):
        raise ValueError("warmup harus di antara 1 dan jumlah bar")

    closes = df["Close"].to_numpy()
    dates = [str(pd.Timestamp(d).date()) for d in df.index]
    result = BacktestResult(
        ticker=ticker,
        bars_tested=len(df) - warmup,
        eligible_bars=sum(eligible[warmup:]) if eligible else len(df) - warmup,
        buy_hold_pct=(closes[-1] - closes[warmup]) / closes[warmup] * 100,
    )
    open_trade: Trade | None = None

    def close_trade(trade: Trade, index: int, reason: str) -> None:
        trade.exit_index = index
        trade.exit_price = float(closes[index])
        trade.exit_date = dates[index]
        trade.reason = reason
        result.trades.append(trade)

    for t in range(warmup, len(df)):
        visible = df.iloc[: t + 1]
        close = float(closes[t])

        if open_trade is not None:
            if exit_on_fundamental and eligible is not None and not eligible[t]:
                close_trade(open_trade, t, EXIT_FUNDAMENTAL)
                open_trade = None
                continue
            # Cut your loss fast: kerugian diperiksa lebih dulu.
            if use_cut_loss and close < open_trade.cut_loss:
                close_trade(open_trade, t, EXIT_CUT_LOSS)
                open_trade = None
                continue
            # Let your profit run: baru keluar saat trendline naik tertembus.
            highs_idx, lows_idx = get_extrema(visible["Close"], distance=distance)
            line = build_exit_trendline(
                visible, highs_idx, lows_idx, until_index=t,
                tol=trend_tol, lookback_swings=lookback_swings,
                confirmation_ratio=confirmation_ratio,
            )
            if line is not None and t > line.points[-1].index:
                boundary = break_threshold(line.value_at(t), UP_TRENDLINE, break_tol)
                if close < boundary:
                    close_trade(open_trade, t, EXIT_TAKE_PROFIT)
                    open_trade = None
            continue

        if eligible is not None and not eligible[t]:
            continue

        try:
            analysis = analyze_chart(
                visible, distance=distance, trend_tol=trend_tol,
                lookback_swings=lookback_swings, confirmation_ratio=confirmation_ratio,
                break_tol=break_tol, breakout_tol=breakout_tol,
                cut_loss_tol=cut_loss_tol, levels_per_side=levels_per_side,
            )
        except ValueError:
            continue

        # Sinyal hanya diambil bila titik masuknya JATUH di bar ini; sinyal
        # lama yang sudah lewat tidak boleh dikejar.
        for plan in analysis.plans_with_entry:
            if plan.signal.entry_index != t:
                continue
            open_trade = Trade(
                ticker=ticker, entry_index=t, entry_date=dates[t],
                entry_price=plan.signal.entry_price, resistance=plan.level.price,
                cut_loss=plan.plan.cut_loss, fee_buy=fee_buy, fee_sell=fee_sell,
            )
            break

    if open_trade is not None:
        close_trade(open_trade, len(df) - 1, EXIT_STILL_OPEN)

    return result
