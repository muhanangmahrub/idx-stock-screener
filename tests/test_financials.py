"""Unit test rasio dari komponen LK mentah (e-book Metode Analisis Fundamental).

Acuan kebenaran: contoh UNVR 1H10 di dokumen - EDR 48,6%; EER 94,6%;
EAR 32,7%; PBV 43,3x; PER 37x; market cap Rp130.855 miliar; ROE disetahunkan
117,3%. Dokumen membulatkan angkanya, jadi perbandingan memakai toleransi
pembulatan itu, bukan kesamaan persis.

Dokumen sedikit tidak konsisten soal ekuitas (Rp3.019 miliar dipakai untuk
ROE/PBV, sementara tabel menulis Rp3.192 miliar). Test memakai angka yang
mereproduksi hasil akhir dokumen.
"""

import pytest

from screener.financials import (
    ANNUALIZATION_FACTORS,
    DEFAULT_MIN_EER_FOR_HEALTHY_EQUITY,
    PERIOD_FULL_YEAR,
    PERIOD_HALF_YEAR,
    PERIOD_Q1,
    PERIOD_Q3,
    REJECT_NEGATIVE_EQUITY,
    REJECT_NEGATIVE_NET_INCOME,
    REJECT_NEGATIVE_PBV,
    REJECT_NEGATIVE_PER,
    REJECT_NEGATIVE_RETAINED_EARNINGS,
    absolute_rejects,
    annualization_factor,
    annualize_profit,
    assess_balance_quality,
    assess_growth_quality,
    compute_ratios,
)

MILIAR = 1_000_000_000

# Komponen UNVR 1H10 (semester I 2010) sesuai dokumen.
UNVR = dict(
    net_income=1_770.6 * MILIAR,  # laba bersih semester I
    period=PERIOD_HALF_YEAR,
    equity=3_019 * MILIAR,
    total_assets=9_231 * MILIAR,
    total_liabilities=6_212 * MILIAR,
    retained_earnings=2_856 * MILIAR,
    shares_outstanding=7_630_000_000,
    price=17_150.0,
)
PCT = 0.002  # toleransi pembulatan dokumen untuk rasio dalam persen


class TestUnileverGroundTruth:
    """Angka contoh UNVR 1H10 harus tereproduksi dari komponen mentah."""

    def test_market_cap_matches_document(self):
        ratios = compute_ratios(**UNVR)
        assert ratios.market_cap == pytest.approx(130_855 * MILIAR, rel=0.001)

    def test_annualized_roe_is_117_3_percent(self):
        ratios = compute_ratios(**UNVR)
        assert ratios.roe_pct == pytest.approx(117.3, abs=0.2)

    def test_pbv_is_43_3x(self):
        assert compute_ratios(**UNVR).pbv == pytest.approx(43.3, abs=0.1)

    def test_per_is_37x(self):
        assert compute_ratios(**UNVR).per == pytest.approx(37, abs=0.2)

    def test_balance_ratios_edr_eer_ear(self):
        ratios = compute_ratios(**UNVR)
        assert ratios.edr == pytest.approx(0.486, abs=PCT)
        assert ratios.eer == pytest.approx(0.946, abs=PCT)
        assert ratios.ear == pytest.approx(0.327, abs=PCT)

    def test_roa_uses_annualized_profit_over_assets(self):
        ratios = compute_ratios(**UNVR)
        assert ratios.roa == pytest.approx(1_770.6 * 2 / 9_231, abs=0.001)

    def test_growth_quality_of_unvr_is_ideal(self):
        # Dokumen: sales +10,8% < operating +13,9% < net +18,4%.
        quality = assess_growth_quality(
            sales=110.8, sales_previous=100.0,
            operating_profit=113.9, operating_profit_previous=100.0,
            net_income=118.4, net_income_previous=100.0,
        )
        assert quality.sales_growth == pytest.approx(0.108, abs=0.001)
        assert quality.operating_growth == pytest.approx(0.139, abs=0.001)
        assert quality.net_growth == pytest.approx(0.184, abs=0.001)
        assert quality.is_ideal
        assert quality.warnings == []


class TestAnnualization:
    """Rasio laba hanya valid kalau labanya disetahunkan sesuai periode LK."""

    def test_factors_follow_the_document(self):
        assert ANNUALIZATION_FACTORS[PERIOD_Q1] == 4
        assert ANNUALIZATION_FACTORS[PERIOD_HALF_YEAR] == 2
        assert ANNUALIZATION_FACTORS[PERIOD_Q3] == pytest.approx(4 / 3)
        assert ANNUALIZATION_FACTORS[PERIOD_FULL_YEAR] == 1

    @pytest.mark.parametrize(
        "period,expected", [(PERIOD_Q1, 400), (PERIOD_HALF_YEAR, 200), (PERIOD_FULL_YEAR, 100)]
    )
    def test_annualize_profit(self, period, expected):
        assert annualize_profit(100, period) == pytest.approx(expected)

    def test_q3_multiplies_by_four_thirds(self):
        assert annualize_profit(90, PERIOD_Q3) == pytest.approx(120)

    def test_unknown_period_is_rejected(self):
        with pytest.raises(ValueError):
            annualization_factor("Q2")

    def test_same_profit_gives_different_roe_per_period(self):
        base = dict(net_income=100.0, equity=1_000.0)
        assert compute_ratios(**base, period=PERIOD_FULL_YEAR).roe == pytest.approx(0.10)
        assert compute_ratios(**base, period=PERIOD_HALF_YEAR).roe == pytest.approx(0.20)
        assert compute_ratios(**base, period=PERIOD_Q1).roe == pytest.approx(0.40)


class TestRatioMechanics:
    def test_liabilities_derived_from_assets_minus_equity(self):
        ratios = compute_ratios(
            net_income=100.0, equity=400.0, total_assets=1_000.0, retained_earnings=200.0
        )
        assert ratios.edr == pytest.approx(400 / 600)
        assert ratios.ear == pytest.approx(0.4)

    def test_diluted_eps_is_used_when_given(self):
        # EPS terdilusi lebih kecil daripada laba/jumlah saham dasar -> PER lebih tinggi.
        shared = dict(net_income=1_000.0, shares_outstanding=100, price=50.0)
        basic = compute_ratios(**shared)
        diluted = compute_ratios(**shared, eps_diluted=8.0)
        assert basic.eps_annualized == pytest.approx(10.0)
        assert diluted.eps_annualized == pytest.approx(8.0)
        assert diluted.per > basic.per

    def test_diluted_eps_is_annualized_too(self):
        ratios = compute_ratios(
            net_income=1_000.0, period=PERIOD_HALF_YEAR, eps_diluted=5.0, price=100.0
        )
        assert ratios.eps_annualized == pytest.approx(10.0)
        assert ratios.per == pytest.approx(10.0)

    def test_missing_components_give_none_not_zero(self):
        ratios = compute_ratios(net_income=100.0)
        assert ratios.roe is None and ratios.pbv is None and ratios.per is None
        assert ratios.edr is None and ratios.eer is None and ratios.ear is None
        assert ratios.market_cap is None

    def test_zero_equity_does_not_raise(self):
        assert compute_ratios(net_income=100.0, equity=0.0).roe is None

    def test_foreign_currency_is_converted_before_valuation_ratios(self):
        # LK dalam USD, harga saham dalam Rupiah: ekuitas & EPS dikali kurs.
        usd = dict(net_income=10.0, equity=100.0, shares_outstanding=1_000, price=1_500.0)
        converted = compute_ratios(**usd, fx_rate=15_000.0)
        assert converted.pbv == pytest.approx(1_500 * 1_000 / (100 * 15_000))
        assert converted.eps_annualized == pytest.approx(10 * 15_000 / 1_000)
        assert converted.per == pytest.approx(1_500 / 150)
        # Rasio yang kedua sisinya dari LK tidak berubah oleh kurs.
        assert converted.roe == pytest.approx(compute_ratios(**usd).roe)

    def test_non_positive_fx_rate_is_rejected(self):
        with pytest.raises(ValueError):
            compute_ratios(net_income=100.0, fx_rate=0)


class TestAbsoluteRejects:
    """Tolak mutlak: jangan beli tanpa toleransi."""

    @pytest.mark.parametrize(
        "kwargs,expected",
        [
            (dict(retained_earnings=-1.0), REJECT_NEGATIVE_RETAINED_EARNINGS),
            (dict(equity=-1.0), REJECT_NEGATIVE_EQUITY),
            (dict(net_income=-1.0), REJECT_NEGATIVE_NET_INCOME),
            (dict(per=-1.0), REJECT_NEGATIVE_PER),
            (dict(pbv=-1.0), REJECT_NEGATIVE_PBV),
        ],
    )
    def test_each_negative_condition_is_rejected(self, kwargs, expected):
        assert absolute_rejects(**kwargs) == [expected]

    def test_healthy_numbers_are_not_rejected(self):
        ratios = compute_ratios(**UNVR)
        assert absolute_rejects(
            retained_earnings=UNVR["retained_earnings"],
            equity=UNVR["equity"],
            net_income=UNVR["net_income"],
            per=ratios.per,
            pbv=ratios.pbv,
        ) == []

    def test_all_conditions_reported_at_once(self):
        assert len(absolute_rejects(retained_earnings=-1, equity=-1, net_income=-1)) == 3

    def test_unknown_values_are_not_treated_as_negative(self):
        assert absolute_rejects() == []
        assert absolute_rejects(equity=None, per=None) == []

    def test_zero_is_not_negative(self):
        assert absolute_rejects(retained_earnings=0.0, equity=0.0) == []


class TestGrowthQualityWarnings:
    def test_net_income_up_while_sales_down_is_warned(self):
        quality = assess_growth_quality(
            sales=90.0, sales_previous=100.0,
            operating_profit=110.0, operating_profit_previous=100.0,
            net_income=120.0, net_income_previous=100.0,
        )
        assert not quality.is_ideal
        assert any("non-operasional" in w for w in quality.warnings)
        assert any("penjualan" in w for w in quality.warnings)

    def test_net_income_up_while_sales_and_operating_down_names_both(self):
        quality = assess_growth_quality(
            sales=90.0, sales_previous=100.0,
            operating_profit=80.0, operating_profit_previous=100.0,
            net_income=120.0, net_income_previous=100.0,
        )
        assert any("penjualan dan laba usaha" in w for w in quality.warnings)

    def test_all_three_down_is_the_harsher_warning(self):
        quality = assess_growth_quality(
            sales=90.0, sales_previous=100.0,
            operating_profit=85.0, operating_profit_previous=100.0,
            net_income=80.0, net_income_previous=100.0,
        )
        assert any("melemah menyeluruh" in w for w in quality.warnings)
        assert not quality.is_ideal

    def test_all_up_but_out_of_order_is_not_ideal(self):
        quality = assess_growth_quality(
            sales=120.0, sales_previous=100.0,
            operating_profit=110.0, operating_profit_previous=100.0,
            net_income=105.0, net_income_previous=100.0,
        )
        assert not quality.is_ideal
        assert any("tidak berurutan" in w for w in quality.warnings)
        assert quality.warnings and "non-operasional" not in quality.warnings[0]

    def test_incomplete_data_is_reported_not_guessed(self):
        quality = assess_growth_quality(None, None, None, None, None, None)
        assert quality.warnings == ["Data pertumbuhan tidak lengkap - tidak bisa dinilai"]
        assert not quality.is_ideal


class TestBalanceQualityWarnings:
    """Warning untuk ditinjau manual, bukan penggugur."""

    def test_healthy_balance_has_no_warnings(self):
        assert assess_balance_quality(
            total_assets=1_000.0, current_assets=600.0, non_current_assets=400.0,
            cash=100.0, goodwill_and_other_assets=50.0, total_liabilities=400.0,
            interest_bearing_debt=100.0, equity=600.0, eer=0.8,
        ) == []

    def test_large_goodwill_is_warned(self):
        warnings = assess_balance_quality(
            total_assets=1_000.0, goodwill_and_other_assets=300.0
        )
        assert any("Goodwill" in w for w in warnings)

    def test_interest_bearing_debt_dominating_liabilities_is_warned(self):
        warnings = assess_balance_quality(
            total_liabilities=1_000.0, interest_bearing_debt=700.0
        )
        assert any("Utang berbunga" in w for w in warnings)

    def test_thin_cash_is_warned(self):
        warnings = assess_balance_quality(total_assets=1_000.0, cash=10.0)
        assert any("Kas" in w for w in warnings)

    def test_current_assets_below_non_current_is_warned(self):
        warnings = assess_balance_quality(current_assets=300.0, non_current_assets=700.0)
        assert any("Aset lancar" in w for w in warnings)

    def test_low_eer_flags_equity_from_right_issue(self):
        # Kasus UNSP di dokumen: ekuitas besar tapi bukan dari akumulasi laba.
        warnings = assess_balance_quality(equity=5_000.0, eer=0.05)
        assert any("EER" in w and "right issue" in w for w in warnings)
        assert DEFAULT_MIN_EER_FOR_HEALTHY_EQUITY == 0.50

    def test_thresholds_are_parameters(self):
        assert assess_balance_quality(
            total_assets=1_000.0, goodwill_and_other_assets=300.0, max_goodwill_pct=0.40
        ) == []

    def test_unvr_balance_passes_its_own_checks(self):
        ratios = compute_ratios(**UNVR)
        warnings = assess_balance_quality(equity=UNVR["equity"], eer=ratios.eer)
        assert warnings == []
