"""Unit test perbandingan PER/PBV terhadap sektor (jalur opsional e-book lama)."""

import pytest

from screener.sector import (
    CHEAPER_THAN_SECTOR,
    PRICIER_THAN_SECTOR,
    SAME_AS_SECTOR,
    compare_to_sector,
    sector_averages,
)

ROWS = [
    {"sector": "Bank", "per": 10.0, "pbv": 1.0},
    {"sector": "Bank", "per": 14.0, "pbv": 2.0},
    {"sector": "Bank", "per": 30.0, "pbv": 3.0},
    {"sector": "Consumer", "per": 20.0, "pbv": 5.0},
]


class TestSectorAverages:
    def test_median_and_mean_per_sector(self):
        stats = sector_averages(ROWS)
        assert stats["Bank"].per_median == pytest.approx(14.0)
        assert stats["Bank"].per_mean == pytest.approx(18.0)
        assert stats["Bank"].pbv_median == pytest.approx(2.0)
        assert stats["Bank"].count == 3
        assert stats["Consumer"].per_median == pytest.approx(20.0)

    def test_median_resists_one_extreme_outlier(self):
        rows = ROWS + [{"sector": "Bank", "per": 900.0, "pbv": 1.5}]
        stats = sector_averages(rows)
        assert stats["Bank"].per_median == pytest.approx(22.0)  # 14 & 30 -> 22
        assert stats["Bank"].per_mean > 200  # rata-rata tertarik jauh

    def test_negative_and_missing_ratios_are_excluded(self):
        rows = [
            {"sector": "Bank", "per": 10.0, "pbv": None},
            {"sector": "Bank", "per": -5.0, "pbv": 2.0},
            {"sector": "Bank", "per": None, "pbv": 4.0},
        ]
        stats = sector_averages(rows)
        assert stats["Bank"].per_median == pytest.approx(10.0)
        assert stats["Bank"].pbv_median == pytest.approx(3.0)

    def test_sector_without_valid_ratios_is_reported_as_empty(self):
        stats = sector_averages([{"sector": "Bank", "per": None, "pbv": None}])
        assert stats["Bank"].count == 0
        assert stats["Bank"].per_median is None

    def test_rows_without_sector_are_skipped(self):
        assert sector_averages([{"per": 10.0, "pbv": 1.0}]) == {}


class TestCompareToSector:
    def test_cheaper_equal_and_pricier(self):
        assert compare_to_sector(10.0, 20.0) == CHEAPER_THAN_SECTOR
        assert compare_to_sector(20.0, 20.0) == SAME_AS_SECTOR
        assert compare_to_sector(30.0, 20.0) == PRICIER_THAN_SECTOR

    def test_small_difference_stays_inside_the_tolerance_band(self):
        assert compare_to_sector(20.5, 20.0) == SAME_AS_SECTOR
        assert compare_to_sector(20.5, 20.0, tolerance=0.01) == PRICIER_THAN_SECTOR

    def test_missing_values_give_no_verdict(self):
        assert compare_to_sector(None, 20.0) is None
        assert compare_to_sector(10.0, None) is None
        assert compare_to_sector(10.0, 0.0) is None
