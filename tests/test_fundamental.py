import pandas as pd
import pytest

from screener.fundamental import (
    annualize_roe_pct,
    compute_free_float_pct,
    compute_position_limit,
    compute_valuation,
    estimate_daily_transaction_value,
    is_cheap_valuation,
    passes_initial_screening,
)


class TestPassesInitialScreening:
    def test_passes_when_all_criteria_met(self):
        result = passes_initial_screening(
            ticker="BBCA.JK",
            daily_transaction_value=6_000_000_000,
            free_float_pct=20.0,
            roe_annualized_pct=15.0,
        )
        assert result.passed is True
        assert result.reasons == []

    def test_fails_on_low_liquidity(self):
        result = passes_initial_screening(
            ticker="XYZ.JK",
            daily_transaction_value=1_000_000_000,
            free_float_pct=20.0,
            roe_annualized_pct=15.0,
        )
        assert result.passed is False
        assert any("Likuiditas" in reason for reason in result.reasons)

    def test_fails_on_low_free_float(self):
        result = passes_initial_screening(
            ticker="XYZ.JK",
            daily_transaction_value=6_000_000_000,
            free_float_pct=5.0,
            roe_annualized_pct=15.0,
        )
        assert result.passed is False
        assert any("Free float" in reason for reason in result.reasons)

    def test_fails_on_low_roe(self):
        result = passes_initial_screening(
            ticker="XYZ.JK",
            daily_transaction_value=6_000_000_000,
            free_float_pct=20.0,
            roe_annualized_pct=5.0,
        )
        assert result.passed is False
        assert any("ROE" in reason for reason in result.reasons)

    def test_fails_when_blacklisted(self):
        result = passes_initial_screening(
            ticker="XYZ.JK",
            daily_transaction_value=6_000_000_000,
            free_float_pct=20.0,
            roe_annualized_pct=15.0,
            is_blacklisted=True,
        )
        assert result.passed is False
        assert any("blacklist" in reason for reason in result.reasons)

    def test_reports_all_failed_reasons_at_once(self):
        result = passes_initial_screening(
            ticker="XYZ.JK",
            daily_transaction_value=1_000_000_000,
            free_float_pct=5.0,
            roe_annualized_pct=5.0,
        )
        assert len(result.reasons) == 3


class TestIsCheapValuation:
    def test_blue_chip_cheap_by_per(self):
        assert is_cheap_valuation(per=10, pbv=2.0, is_blue_chip=True) is True

    def test_blue_chip_cheap_by_pbv(self):
        assert is_cheap_valuation(per=20, pbv=0.5, is_blue_chip=True) is True

    def test_blue_chip_not_cheap(self):
        assert is_cheap_valuation(per=20, pbv=2.0, is_blue_chip=True) is False

    def test_small_cap_uses_lower_per_threshold(self):
        assert is_cheap_valuation(per=10, pbv=2.0, is_blue_chip=False) is False
        assert is_cheap_valuation(per=7, pbv=2.0, is_blue_chip=False) is True

    def test_small_cap_cheap_by_pbv(self):
        assert is_cheap_valuation(per=20, pbv=0.7, is_blue_chip=False) is True


class TestComputeValuation:
    def test_pbv_base_follows_spec_example(self):
        """Contoh di spec: ROE 51% -> PBV dasar 5,1x."""
        valuation = compute_valuation(roe_annualized_pct=51.0, bvps=1000.0)
        assert valuation.pbv_base == pytest.approx(5.1)
        assert valuation.fair_price_base == pytest.approx(5100.0)

    def test_full_chain_with_default_mos_and_no_risk(self):
        valuation = compute_valuation(roe_annualized_pct=20.0, bvps=1000.0)
        assert valuation.fair_price_adj == pytest.approx(2000.0)
        assert valuation.best_buy == pytest.approx(2000.0 * 0.65)
        assert valuation.max_buy == pytest.approx(2000.0 * 0.65 * 1.15)

    def test_each_risk_factor_discounts_ten_percent(self):
        valuation = compute_valuation(
            roe_annualized_pct=20.0, bvps=1000.0, risk_factor_count=2
        )
        assert valuation.fair_price_adj == pytest.approx(2000.0 * 0.8)

    def test_custom_margin_of_safety(self):
        valuation = compute_valuation(
            roe_annualized_pct=20.0, bvps=1000.0, margin_of_safety=0.50
        )
        assert valuation.best_buy == pytest.approx(1000.0)

    def test_rejects_margin_of_safety_outside_spec_range(self):
        with pytest.raises(ValueError):
            compute_valuation(roe_annualized_pct=20.0, bvps=1000.0, margin_of_safety=0.2)
        with pytest.raises(ValueError):
            compute_valuation(roe_annualized_pct=20.0, bvps=1000.0, margin_of_safety=0.6)

    def test_rejects_negative_risk_factor_count(self):
        with pytest.raises(ValueError):
            compute_valuation(roe_annualized_pct=20.0, bvps=1000.0, risk_factor_count=-1)


class TestComputePositionLimit:
    def test_liquidity_limit_binds_when_stock_is_thin(self):
        limit = compute_position_limit(
            daily_transaction_value=10_000_000_000, liquid_net_worth=100_000_000_000
        )
        assert limit.by_liquidity == pytest.approx(3_000_000_000)
        assert limit.by_net_worth == pytest.approx(20_000_000_000)
        assert limit.max_position == pytest.approx(3_000_000_000)

    def test_net_worth_limit_binds_for_small_investor(self):
        limit = compute_position_limit(
            daily_transaction_value=500_000_000_000, liquid_net_worth=100_000_000
        )
        assert limit.max_position == pytest.approx(20_000_000)


class TestDerivedInputs:
    def test_estimate_daily_transaction_value_averages_close_times_volume(self):
        price_df = pd.DataFrame(
            {"Close": [100.0, 200.0, 300.0], "Volume": [10, 10, 10]}
        )
        assert estimate_daily_transaction_value(price_df) == pytest.approx(2000.0)

    def test_estimate_daily_transaction_value_respects_window(self):
        price_df = pd.DataFrame(
            {"Close": [1.0, 100.0, 100.0], "Volume": [1, 1, 1]}
        )
        assert estimate_daily_transaction_value(price_df, window_days=2) == 100.0

    def test_estimate_daily_transaction_value_handles_empty(self):
        assert estimate_daily_transaction_value(pd.DataFrame()) is None

    def test_free_float_pct(self):
        assert compute_free_float_pct(30, 100) == pytest.approx(30.0)
        assert compute_free_float_pct(None, 100) is None
        assert compute_free_float_pct(30, 0) is None

    def test_annualize_roe_multiplies_latest_quarter_by_four(self):
        assert annualize_roe_pct(25.0, 1000.0) == pytest.approx(10.0)
        assert annualize_roe_pct(None, 1000.0) is None
        assert annualize_roe_pct(25.0, 0) is None


class TestScreeningNotes:
    def test_roe_between_min_and_ideal_adds_note_but_passes(self):
        result = passes_initial_screening("X.JK", 6e9, 20.0, roe_annualized_pct=15.0)
        assert result.passed is True
        assert any("band ideal 20-30%" in note for note in result.notes)

    def test_roe_in_ideal_band_has_no_note(self):
        result = passes_initial_screening("X.JK", 6e9, 20.0, roe_annualized_pct=25.0)
        assert result.notes == []

    def test_failing_roe_does_not_duplicate_as_note(self):
        result = passes_initial_screening("X.JK", 6e9, 20.0, roe_annualized_pct=5.0)
        assert result.notes == []
