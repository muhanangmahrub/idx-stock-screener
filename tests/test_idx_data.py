import pandas as pd
import pytest

from screener.idx_data import (
    compute_public_float_pct,
    lookup_daily_transaction_value,
    to_idx_code,
)

# Potongan nyata dari profil BBCA di idx.co.id (Sep 2026), disederhanakan.
BBCA_SHAREHOLDERS = [
    {"Nama": "PT Dwimuria Investama Andalan", "Kategori": "Lebih dari 5%", "Persentase": 54.942},
    {"Nama": "Saham Treasury", "Kategori": "Saham Treasury", "Persentase": 0.351},
    {"Nama": "Masyarakat Warkat", "Kategori": "Masyarakat Warkat", "Persentase": 2.508},
    {"Nama": "Masyarakat Non Warkat", "Kategori": "Masyarakat Non Warkat", "Persentase": 42.134},
    {"Nama": "Jahja Setiaatmadja", "Kategori": "Komisaris", "Persentase": 0.03},
]


class TestToIdxCode:
    def test_strips_jk_suffix(self):
        assert to_idx_code("BBCA.JK") == "BBCA"

    def test_handles_lowercase_and_no_suffix(self):
        assert to_idx_code("bbca.jk") == "BBCA"
        assert to_idx_code("BBCA") == "BBCA"


class TestComputePublicFloatPct:
    def test_sums_only_masyarakat_categories(self):
        assert compute_public_float_pct(BBCA_SHAREHOLDERS) == pytest.approx(44.642)

    def test_ignores_controlling_treasury_and_insiders(self):
        no_public = [row for row in BBCA_SHAREHOLDERS if "Masyarakat" not in row["Kategori"]]
        assert compute_public_float_pct(no_public) == 0.0

    def test_empty_returns_none(self):
        assert compute_public_float_pct([]) is None

    def test_tolerates_missing_percentage(self):
        rows = [{"Kategori": "Masyarakat Non Warkat", "Persentase": None}]
        assert compute_public_float_pct(rows) == 0.0


class TestLookupDailyTransactionValue:
    def _summary(self):
        return pd.DataFrame(
            [
                {"StockCode": "BBCA", "Date": "2026-09-17T00:00:00", "Value": 393_453_237_500.0},
                {"StockCode": "TLKM", "Date": "2026-09-17T00:00:00", "Value": 150_000_000_000.0},
            ]
        )

    def test_returns_value_and_date_for_code(self):
        value, date = lookup_daily_transaction_value(self._summary(), "BBCA")
        assert value == pytest.approx(393_453_237_500.0)
        assert date == "2026-09-17"

    def test_unknown_code_returns_none(self):
        assert lookup_daily_transaction_value(self._summary(), "ZZZZ") is None

    def test_empty_summary_returns_none(self):
        assert lookup_daily_transaction_value(pd.DataFrame(), "BBCA") is None
