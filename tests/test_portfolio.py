import pytest

from screener.portfolio import Position, review_portfolio


def _ideal_portfolio():
    """5 saham beda sektor, masing-masing 16% (= 80% saham), cash 20%."""
    sectors = ["bank", "telko", "konsumer", "energi", "properti"]
    return [Position(f"S{i}.JK", sector, 16.0) for i, sector in enumerate(sectors)], 20.0


class TestReviewPortfolio:
    def test_empty_portfolio_has_no_findings(self):
        review = review_portfolio([], cash=0)
        assert review.total_value == 0
        assert review.warnings == [] and review.notes == []

    def test_weights_and_cash_share_the_same_base(self):
        positions, cash = _ideal_portfolio()
        review = review_portfolio(positions, cash)
        assert review.total_value == pytest.approx(100.0)
        assert review.cash_pct == pytest.approx(20.0)
        assert all(w == pytest.approx(16.0) for w in review.weights_pct.values())

    def test_ideal_count_sectors_and_cash_produce_no_warnings(self):
        positions, cash = _ideal_portfolio()
        review = review_portfolio(positions, cash)
        assert review.warnings == []
        # 16% per saham memang di luar target 25-30% -> catatan, bukan pelanggaran
        assert all("di luar target" in note for note in review.notes)

    def test_same_sector_twice_is_a_warning(self):
        positions = [
            Position("BBCA.JK", "Bank", 40.0),
            Position("BBRI.JK", "bank", 40.0),
        ]
        review = review_portfolio(positions, cash=20.0)
        assert any("Sektor 'bank'" in w and "BBCA.JK, BBRI.JK" in w for w in review.warnings)

    def test_weight_above_35_is_warning_above_50_is_extreme(self):
        review = review_portfolio([Position("A.JK", "x", 40.0), Position("B.JK", "y", 40.0)], cash=20.0)
        assert any("A.JK 40.0% melebihi maksimum 35%" in w for w in review.warnings)

        review = review_portfolio([Position("A.JK", "x", 60.0)], cash=40.0)
        assert any("melebihi batas ekstrem 50%" in w for w in review.warnings)

    def test_weight_below_5_is_warning(self):
        review = review_portfolio([Position("A.JK", "x", 2.0), Position("B.JK", "y", 78.0)], cash=20.0)
        assert any("A.JK 2.0% di bawah minimum 5%" in w for w in review.warnings)

    def test_cash_below_10_is_warning_below_20_is_note(self):
        review = review_portfolio([Position("A.JK", "x", 95.0)], cash=5.0)
        assert any("Cash 5.0% di bawah minimum 10%" in w for w in review.warnings)

        review = review_portfolio([Position("A.JK", "x", 85.0)], cash=15.0)
        assert any("Cash 15.0% di bawah ideal 20%" in n for n in review.notes)
        assert not any("Cash" in w for w in review.warnings)

    def test_stock_count_thresholds(self):
        def portfolio_of(n):
            return [Position(f"S{i}.JK", f"sektor{i}", 1.0) for i in range(n)]

        assert any("di luar target 5-6" in n for n in review_portfolio(portfolio_of(3), 1.0).notes)
        assert any("di luar target 5-6" in n for n in review_portfolio(portfolio_of(8), 1.0).notes)
        assert any("melebihi maksimum 10" in w for w in review_portfolio(portfolio_of(11), 1.0).warnings)
        assert any("melebihi batas ekstrem 12" in w for w in review_portfolio(portfolio_of(13), 1.0).warnings)
