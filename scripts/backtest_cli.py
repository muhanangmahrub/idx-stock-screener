"""Jalankan backtest aturan repo ke beberapa emiten IDX.

Contoh:
    venv/bin/python scripts/backtest_cli.py                      # Lapis 2 saja
    venv/bin/python scripts/backtest_cli.py --layer1             # Lapis 1 + 2
    venv/bin/python scripts/backtest_cli.py --layer1 --growth    # + syarat pertumbuhan
    venv/bin/python scripts/backtest_cli.py --tickers BBCA,ICBP --no-cut-loss

Butuh jaringan (yfinance). Hasilnya untuk MEMBANDINGKAN antar-aturan, bukan
memperkirakan keuntungan masa depan - lihat batasan di screener/backtest.py.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from screener.backtest import run_backtest, summarize  # noqa: E402
from screener.data import get_annual_statements, get_price_history  # noqa: E402
from screener.fundamental_history import build_snapshots, eligibility_flags  # noqa: E402

DEFAULT_TICKERS = [
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "ICBP", "INDF", "UNVR", "KLBF",
    "SMGR", "INTP", "PTBA", "ADRO", "ANTM", "MDKA", "AMRT", "ACES", "JPFA", "CPIN",
]
DEFAULT_BLUE_CHIPS = {
    "BBCA", "BBRI", "BMRI", "BBNI", "TLKM", "ASII", "ICBP", "INDF", "UNVR", "KLBF",
    "SMGR", "INTP", "AMRT", "CPIN",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tickers", default=",".join(DEFAULT_TICKERS))
    parser.add_argument("--period", default="5y")
    parser.add_argument("--interval", default="1wk")
    parser.add_argument("--distance", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--layer1", action="store_true", help="pakai gerbang fundamental")
    parser.add_argument("--growth", action="store_true", help="syaratkan kualitas pertumbuhan")
    parser.add_argument("--no-cut-loss", action="store_true")
    parser.add_argument("--exit-on-fundamental", action="store_true")
    args = parser.parse_args()

    codes = [c.strip().upper() for c in args.tickers.split(",") if c.strip()]
    results = []
    print(f"{'Kode':<7}{'Trade':>6}{'Strategi':>11}{'Buy&Hold':>11}{'Selisih':>10}{'Eligible':>10}")
    print("-" * 55)
    for code in codes:
        df = get_price_history(f"{code}.JK", period=args.period, interval=args.interval)
        if df.empty or len(df) < args.warmup + 10:
            print(f"{code:<7}  data tidak cukup")
            continue

        eligible = None
        if args.layer1:
            income, balance, shares = get_annual_statements(f"{code}.JK")
            snapshots = build_snapshots(income, balance, shares)
            eligible = eligibility_flags(
                df, snapshots,
                is_blue_chip=code in DEFAULT_BLUE_CHIPS,
                require_growth=args.growth,
                bars_per_week=5 if args.interval == "1wk" else 1,
            )

        result = run_backtest(
            df, ticker=f"{code}.JK", eligible=eligible, warmup=args.warmup,
            distance=args.distance, use_cut_loss=not args.no_cut_loss,
            exit_on_fundamental=args.exit_on_fundamental,
        )
        results.append(result)
        print(f"{code:<7}{len(result.closed_trades):>6}{result.compound_pct:>11.1f}"
              f"{result.buy_hold_pct:>11.1f}{result.compound_pct - result.buy_hold_pct:>10.1f}"
              f"{result.exposure_pct:>9.0f}%")

    summary = summarize(results)
    if not summary.trades:
        print("\nTidak ada trade sama sekali.")
        return
    print(f"\n{summary.trades} trade | win {summary.win_rate:.1f}% | "
          f"ekspektasi {summary.expectancy_pct:+.2f}%/trade | PF {summary.profit_factor:.2f}")
    print(f"rata-rata menang {summary.avg_win_pct:+.1f}% | kalah {summary.avg_loss_pct:+.1f}% "
          f"| median {summary.median_bars_held:.0f} bar")
    for reason, (count, average) in sorted(summary.by_reason.items(), key=lambda kv: -kv[1][0]):
        print(f"  {reason:<42}{count:>4}x  rata2 {average:+6.1f}%")
    print("\nAngka ini membandingkan aturan, bukan memperkirakan hasil ke depan.")


if __name__ == "__main__":
    main()
