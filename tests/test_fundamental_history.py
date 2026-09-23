"""Unit test fundamental titik-waktu (gerbang Lapis 1 untuk backtest).

Inti yang dikunci: laporan hanya boleh dipakai SETELAH terbit, dan kriteria
Lapis 1 dijalankan dengan urutan yang sama seperti corong screening.
"""

import pandas as pd
import pytest

from screener.fundamental_history import (
    FAIL_ABSOLUTE,
    FAIL_GROWTH,
    FAIL_LIQUIDITY,
    FAIL_NO_REPORT,
    FAIL_ROE,
    FAIL_VALUATION,
    PASS,
    REPORTING_LAG_DAYS,
    FundamentalSnapshot,
    build_snapshots,
    eligibility_flags,
    latest_snapshot,
    passes_layer_one,
    previous_snapshot,
)

MILIAR = 1_000_000_000


def _statements(years=("2024-12-31", "2023-12-31"), net_income=(500, 400),
                equity=(2_000, 1_800), sales=(5_000, 4_500), operating=(700, 600),
                retained=(1_500, 1_200)):
    columns = [pd.Timestamp(y) for y in years]
    income = pd.DataFrame(
        {c: {"Net Income": n * MILIAR, "Total Revenue": s * MILIAR,
             "Operating Income": o * MILIAR}
         for c, n, s, o in zip(columns, net_income, sales, operating, strict=True)}
    )
    balance = pd.DataFrame(
        {c: {"Stockholders Equity": e * MILIAR, "Retained Earnings": r * MILIAR}
         for c, e, r in zip(columns, equity, retained, strict=True)}
    )
    return income, balance


def _snapshot(**kwargs):
    base = dict(
        available_from=pd.Timestamp("2025-03-31"), fiscal_year="2024-12-31",
        net_income=500 * MILIAR, equity=2_000 * MILIAR, retained_earnings=1_500 * MILIAR,
        shares=1_000_000_000,
    )
    return FundamentalSnapshot(**{**base, **kwargs})


class TestSnapshotAvailability:
    """Laporan tidak boleh dipakai sebelum terbit."""

    def test_annual_report_becomes_available_after_the_lag(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        latest = [s for s in snapshots if s.fiscal_year == "2024-12-31"][0]
        assert latest.available_from == pd.Timestamp("2024-12-31") + pd.Timedelta(
            days=REPORTING_LAG_DAYS
        )

    def test_report_is_invisible_before_publication(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        before = latest_snapshot(snapshots, "2025-02-01")
        assert before.fiscal_year == "2023-12-31"  # masih laporan tahun sebelumnya

    def test_report_is_used_once_published(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        assert latest_snapshot(snapshots, "2025-06-01").fiscal_year == "2024-12-31"

    def test_nothing_available_before_the_first_report(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        assert latest_snapshot(snapshots, "2020-01-01") is None

    def test_timezone_aware_dates_are_accepted(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        aware = pd.Timestamp("2025-06-01", tz="Asia/Jakarta")
        assert latest_snapshot(snapshots, aware).fiscal_year == "2024-12-31"

    def test_previous_snapshot_is_the_year_before(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        latest = latest_snapshot(snapshots, "2025-06-01")
        assert previous_snapshot(snapshots, latest).fiscal_year == "2023-12-31"

    def test_no_statements_gives_no_snapshots(self):
        assert build_snapshots(None, None, 1e9) == []
        assert build_snapshots(pd.DataFrame(), pd.DataFrame(), 1e9) == []


class TestDerivedRatios:
    def test_roe_bvps_and_eps(self):
        snapshot = _snapshot()
        assert snapshot.roe_pct == pytest.approx(25.0)
        assert snapshot.bvps == pytest.approx(2_000)
        assert snapshot.eps == pytest.approx(500)

    def test_missing_components_give_none(self):
        assert _snapshot(equity=None).roe_pct is None
        assert _snapshot(shares=None).eps is None
        assert _snapshot(equity=0).bvps is None


class TestLayerOneCriteria:
    """Urutan sama seperti corong: tolak mutlak, likuiditas, ROE, valuasi."""

    def test_cheap_and_profitable_passes(self):
        ok, reason = passes_layer_one(_snapshot(), price=3_000, daily_transaction_value=10e9)
        assert ok and reason == PASS  # PER 6x, ROE 25%

    def test_missing_report_fails(self):
        assert passes_layer_one(None, price=1_000) == (False, FAIL_NO_REPORT)

    def test_negative_equity_is_rejected_first(self):
        ok, reason = passes_layer_one(
            _snapshot(equity=-1.0), price=3_000, daily_transaction_value=10e9
        )
        assert not ok and reason == FAIL_ABSOLUTE

    def test_illiquid_stock_fails(self):
        ok, reason = passes_layer_one(_snapshot(), price=3_000, daily_transaction_value=1e9)
        assert not ok and reason == FAIL_LIQUIDITY

    def test_low_roe_fails(self):
        ok, reason = passes_layer_one(
            _snapshot(net_income=50 * MILIAR), price=200, daily_transaction_value=10e9
        )
        assert not ok and reason == FAIL_ROE  # ROE 2,5%

    def test_expensive_valuation_fails(self):
        ok, reason = passes_layer_one(_snapshot(), price=9_000, daily_transaction_value=10e9)
        assert not ok and reason == FAIL_VALUATION  # PER 18x, PBV 4,5x

    def test_blue_chip_gets_the_wider_per_limit(self):
        price = 5_000  # PER 10x: mahal untuk small cap, murah untuk blue chip
        assert passes_layer_one(_snapshot(), price, 10e9, is_blue_chip=False)[0] is False
        assert passes_layer_one(_snapshot(), price, 10e9, is_blue_chip=True)[0] is True

    def test_liquidity_is_skipped_when_unknown(self):
        ok, _ = passes_layer_one(_snapshot(), price=3_000, daily_transaction_value=None)
        assert ok


class TestGrowthRequirement:
    def _pair(self, sales=(5_000, 4_500), operating=(700, 600), net=(500, 400)):
        income, balance = _statements(net_income=net, sales=sales, operating=operating)
        snapshots = build_snapshots(income, balance, shares_outstanding=1e9)
        latest = latest_snapshot(snapshots, "2025-06-01")
        return latest, previous_snapshot(snapshots, latest)

    def test_ideal_growth_passes(self):
        # sales +11,1% < operating +16,7% < net +25%
        latest, previous = self._pair()
        ok, reason = passes_layer_one(
            latest, price=3_000, daily_transaction_value=10e9,
            require_growth=True, previous=previous,
        )
        assert ok and reason == PASS

    def test_out_of_order_growth_fails(self):
        latest, previous = self._pair(sales=(6_000, 4_500), net=(410, 400))
        ok, reason = passes_layer_one(
            latest, price=3_000, daily_transaction_value=10e9,
            require_growth=True, previous=previous,
        )
        assert not ok and reason == FAIL_GROWTH

    def test_growth_check_needs_a_previous_report(self):
        latest, _ = self._pair()
        ok, reason = passes_layer_one(
            latest, price=3_000, daily_transaction_value=10e9,
            require_growth=True, previous=None,
        )
        assert not ok and reason == FAIL_GROWTH

    def test_growth_is_not_required_by_default(self):
        latest, previous = self._pair(sales=(6_000, 4_500), net=(410, 400))
        assert passes_layer_one(latest, 3_000, 10e9, previous=previous)[0] is True


class TestEligibilityFlags:
    def _prices(self, price, bars=8, volume=1e9):
        index = pd.date_range("2025-01-06", periods=bars, freq="W-MON", tz="Asia/Jakarta")
        return pd.DataFrame(
            {"Close": [price] * bars, "Volume": [volume] * bars}, index=index
        )

    def test_flags_switch_on_after_the_report_is_published(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        index = pd.date_range("2025-01-06", periods=20, freq="W-MON", tz="Asia/Jakarta")
        df = pd.DataFrame({"Close": [3_000.0] * 20, "Volume": [1e9] * 20}, index=index)
        flags = eligibility_flags(df, snapshots)
        assert len(flags) == len(df)
        assert any(flags), "setelah laporan terbit harus ada bar yang lolos"

    def test_expensive_price_makes_every_bar_fail(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        df = self._prices(50_000, bars=20)
        df.index = pd.date_range("2025-06-01", periods=20, freq="W-MON", tz="Asia/Jakarta")
        assert not any(eligibility_flags(df, snapshots))

    def test_weekly_volume_is_divided_into_daily_value(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        df = self._prices(3_000, bars=4, volume=2_000_000)  # 6 M/minggu -> 1,2 M/hari
        df.index = pd.date_range("2025-06-01", periods=4, freq="W-MON", tz="Asia/Jakarta")
        assert not any(eligibility_flags(df, snapshots, bars_per_week=5))
        # Kalau dianggap satu bar = satu hari, nilainya 6 M -> lolos likuiditas.
        assert any(eligibility_flags(df, snapshots, bars_per_week=1))

    def test_missing_volume_column_is_tolerated(self):
        snapshots = build_snapshots(*_statements(), shares_outstanding=1e9)
        df = pd.DataFrame(
            {"Close": [3_000.0] * 4},
            index=pd.date_range("2025-06-01", periods=4, freq="W-MON", tz="Asia/Jakarta"),
        )
        assert all(eligibility_flags(df, snapshots))
