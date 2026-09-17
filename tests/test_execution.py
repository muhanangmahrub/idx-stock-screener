import pytest

from screener.execution import (
    HORIZON_BEYOND_PLAN,
    HORIZON_LONG,
    HORIZON_MEDIUM,
    HORIZON_SHORT,
    classify_horizon,
    evaluate_position,
)


class TestClassifyHorizon:
    def test_boundaries_follow_spec(self):
        assert classify_horizon(0) == HORIZON_SHORT
        assert classify_horizon(2.9) == HORIZON_SHORT
        assert classify_horizon(3) == HORIZON_MEDIUM
        assert classify_horizon(11.9) == HORIZON_MEDIUM
        assert classify_horizon(12) == HORIZON_LONG
        assert classify_horizon(24) == HORIZON_LONG
        assert classify_horizon(25) == HORIZON_BEYOND_PLAN

    def test_rejects_negative(self):
        with pytest.raises(ValueError):
            classify_horizon(-1)


def _status(**overrides):
    params = dict(
        current_price=1000,
        max_buy=1200,
        fair_price=2000,
        is_cheap=True,
        holding_months=0,
        fundamentals_growing=True,
        growth_on_track=True,
    )
    params.update(overrides)
    return evaluate_position(**params)


def _met(status, label_start):
    return next(c for c in status.conditions if c.label.startswith(label_start)).met


class TestEvaluatePosition:
    def test_accumulation_requires_price_below_max_buy_and_cheap(self):
        assert _met(_status(), "Zona akumulasi") is True
        assert _met(_status(current_price=1300), "Zona akumulasi") is False
        assert _met(_status(is_cheap=False), "Zona akumulasi") is False

    def test_profit_taking_when_price_reaches_fair_price(self):
        assert _met(_status(current_price=2000), "Zona pertimbangkan") is True
        assert _met(_status(current_price=1999), "Zona pertimbangkan") is False

    def test_profit_taking_ignored_without_fair_price(self):
        assert _met(_status(fair_price=0), "Zona pertimbangkan") is False

    def test_hold_mirrors_manual_growth_judgement(self):
        assert _met(_status(fundamentals_growing=False), "Kondisi hold") is False

    def test_rebalance_review_needs_12_months_and_off_track(self):
        assert _met(_status(holding_months=12, growth_on_track=False), "Tinjau") is True
        assert _met(_status(holding_months=11, growth_on_track=False), "Tinjau") is False
        assert _met(_status(holding_months=12, growth_on_track=True), "Tinjau") is False

    def test_active_lists_only_met_conditions(self):
        status = _status(current_price=2500, is_cheap=False)
        assert [c.label for c in status.active] == ["Zona pertimbangkan profit taking", "Kondisi hold"]
