import numpy as np
import pandas as pd
import pytest

from screener.extrema import get_extrema
from screener.trend import (
    FALSE_BREAK,
    LINE_INTACT,
    VALID_BREAK,
    AnchorPoint,
    check_trendline_break,
    CONFIRM_FULL_BREAK,
    CONFIRM_HALF_WAY,
    DOWN_TRENDLINE,
    DOWNTREND,
    SIDEWAYS,
    STEP_DOWN,
    STEP_FLAT,
    STEP_UP,
    UNDEFINED,
    UP_TRENDLINE,
    UPTREND,
    SwingPoint,
    Trendline,
    TrendResult,
    build_trendline,
    classify_steps,
    classify_trend,
    fit_line,
    select_anchor_points,
)


def _zigzag(peaks: list[float], troughs: list[float], bars_per_leg: int = 5) -> pd.Series:
    """Deret harga sintetis zigzag: dasar[0] -> puncak[0] -> dasar[1] -> puncak[1] ...

    Tiap kaki diinterpolasi linear `bars_per_leg` bar supaya `get_extrema`
    (distance kecil) menemukan puncak/dasar tepat di nilai yang diberikan.
    `find_peaks` tidak menganggap ujung deret sebagai swing, jadi deret
    dibuka dari puncak[0] (turun ke dasar[0]) dan ditutup turun ke dasar
    terakhir, supaya semua dasar/puncak yang diminta ada di tengah.
    """
    anchors = [peaks[0]]
    for trough, peak in zip(troughs, peaks):
        anchors += [trough, peak]
    anchors.append(troughs[-1])
    points = []
    for start, end in zip(anchors, anchors[1:]):
        points += list(np.linspace(start, end, bars_per_leg, endpoint=False))
    points.append(anchors[-1])
    return pd.Series(points, dtype=float)


def _classify_series(prices: pd.Series, **kwargs):
    highs, lows = get_extrema(prices, distance=2)
    return classify_trend(prices, highs, lows, **kwargs)


class TestClassifySteps:
    def test_labels_up_down_flat_with_tolerance(self):
        assert classify_steps([100, 110, 100, 101], tol=0.02) == [STEP_UP, STEP_DOWN, STEP_FLAT]

    def test_change_exactly_at_tolerance_counts_as_flat(self):
        assert classify_steps([100, 102], tol=0.02) == [STEP_FLAT]
        assert classify_steps([100, 98], tol=0.02) == [STEP_FLAT]

    def test_single_value_has_no_steps(self):
        assert classify_steps([100], tol=0.02) == []


class TestClassifyTrendFromSyntheticCharts:
    def test_higher_peaks_and_higher_troughs_is_uptrend(self):
        prices = _zigzag(peaks=[110, 120, 130], troughs=[100, 105, 115])
        result = _classify_series(prices)
        assert result.trend == UPTREND
        assert result.peak_steps == [STEP_UP, STEP_UP]
        assert result.trough_steps == [STEP_UP, STEP_UP]
        assert [p.price for p in result.peaks] == [110, 120, 130]
        assert [t.price for t in result.troughs] == [100, 105, 115]

    def test_lower_peaks_and_lower_troughs_is_downtrend(self):
        prices = _zigzag(peaks=[130, 120, 110], troughs=[125, 112, 100])
        result = _classify_series(prices)
        assert result.trend == DOWNTREND
        assert result.peak_steps == [STEP_DOWN, STEP_DOWN]
        assert result.trough_steps == [STEP_DOWN, STEP_DOWN]

    def test_flat_peaks_and_flat_troughs_is_sideways(self):
        # Puncak 120/121/119 dan dasar 100/101/100 masih di dalam toleransi 2%.
        prices = _zigzag(peaks=[120, 121, 119], troughs=[100, 101, 100])
        result = _classify_series(prices)
        assert result.trend == SIDEWAYS
        assert result.peak_steps == [STEP_FLAT, STEP_FLAT]
        assert result.trough_steps == [STEP_FLAT, STEP_FLAT]

    def test_higher_peaks_but_lower_troughs_is_undefined_not_forced(self):
        """Puncak naik, dasar turun (melebar) - bukan uptrend, bukan sideways."""
        prices = _zigzag(peaks=[110, 120, 130], troughs=[100, 95, 90])
        result = _classify_series(prices)
        assert result.trend == UNDEFINED
        assert result.peak_steps == [STEP_UP, STEP_UP]
        assert result.trough_steps == [STEP_DOWN, STEP_DOWN]
        assert "tidak masuk definisi" in result.reason

    def test_one_flat_step_breaks_an_uptrend(self):
        prices = _zigzag(peaks=[110, 120, 121], troughs=[100, 105, 115])
        result = _classify_series(prices)
        assert result.trend == UNDEFINED
        assert result.peak_steps == [STEP_UP, STEP_FLAT]

    def test_tolerance_changes_the_verdict(self):
        """Selisih 3% dianggap 'sama' pada tol 5%, tapi 'naik' pada tol 2%."""
        prices = _zigzag(peaks=[100, 103, 106], troughs=[90, 92.7, 95.5])
        assert _classify_series(prices, tol=0.05).trend == SIDEWAYS
        assert _classify_series(prices, tol=0.02).trend == UPTREND


class TestClassifyTrendGuards:
    def test_only_last_lookback_swings_are_used(self):
        # 5 swing: awalnya turun, 3 terakhir naik -> dengan lookback 3 = uptrend.
        prices = _zigzag(peaks=[150, 140, 110, 120, 130], troughs=[145, 135, 100, 105, 115])
        highs, lows = get_extrema(prices, distance=2)
        assert classify_trend(prices, highs, lows, lookback_swings=3).trend == UPTREND
        assert classify_trend(prices, highs, lows, lookback_swings=5).trend == UNDEFINED

    def test_too_few_swings_is_undefined_with_reason(self):
        prices = pd.Series([100, 110, 100], dtype=float)
        highs, lows = get_extrema(prices, distance=1)
        result = classify_trend(prices, highs, lows)
        assert result.trend == UNDEFINED
        assert "Butuh minimal 2 puncak" in result.reason

    def test_flat_line_has_no_swings_and_is_undefined(self):
        prices = pd.Series([100.0] * 30)
        result = _classify_series(prices)
        assert result.trend == UNDEFINED
        assert result.peaks == [] and result.troughs == []

    def test_lookback_below_minimum_is_rejected(self):
        prices = pd.Series([100, 110, 100, 110], dtype=float)
        with pytest.raises(ValueError):
            classify_trend(prices, np.array([1, 3]), np.array([0, 2]), lookback_swings=1)


class TestFitLine:
    def test_two_points_gives_exact_line(self):
        slope, intercept = fit_line([10, 20], [100.0, 120.0])
        assert slope == pytest.approx(2.0)
        assert intercept == pytest.approx(80.0)

    def test_three_collinear_points_gives_exact_line(self):
        slope, intercept = fit_line([0, 5, 10], [50.0, 60.0, 70.0])
        assert (slope, intercept) == (pytest.approx(2.0), pytest.approx(50.0))

    def test_non_collinear_points_use_least_squares(self):
        # Titik tengah 1 di atas garis 100->120; least squares menggeser garis naik 1/3.
        slope, intercept = fit_line([0, 10, 20], [100.0, 111.0, 120.0])
        assert slope == pytest.approx(1.0)
        assert intercept == pytest.approx(100 + 1 / 3)

    def test_needs_two_points(self):
        with pytest.raises(ValueError):
            fit_line([3], [100.0])


def _ohlc(close: pd.Series, low_offset: float = 2.0, high_offset: float = 3.0) -> pd.DataFrame:
    """OHLC sintetis: Low & High digeser dari Close supaya bisa dicek trendline
    memakai kolom yang benar (Low untuk dasar, High untuk puncak), bukan Close."""
    return pd.DataFrame(
        {"Open": close, "High": close + high_offset, "Low": close - low_offset, "Close": close}
    )


def _uptrend_result(trough_indices: list[int]) -> TrendResult:
    """TrendResult uptrend buatan tangan; puncak tidak dipakai oleh konfirmasi
    dasar (resistance diambil dari High mentah), jadi cukup dummy."""
    return TrendResult(
        trend=UPTREND,
        peaks=[SwingPoint(0, 0.0), SwingPoint(1, 0.0)],
        troughs=[SwingPoint(i, 0.0) for i in trough_indices],
    )


class TestSelectAnchorPoints:
    """Aturan buku: A2 siap setelah resistance (puncak terakhir sebelum A2)
    dilewati - atau 50% jaraknya, tergantung mazhab. Pakai High/Low, bukan Close."""

    # bar:      0    1    2    3    4    5    6
    # Dasar di bar 0 dan 3. Resistance A2 = High tertinggi antara keduanya = 113.
    # Low A2 = 103 -> jarak 10: syarat penuh 113, syarat 50% = 108.
    # Setelah A2, High tertinggi hanya 109: lolos 50%, gagal 100%.
    LOWS = pd.Series([100, 104, 108, 103, 106, 104, 102], dtype=float)
    HIGHS = pd.Series([102, 108, 113, 105, 109, 106, 104], dtype=float)

    def test_first_anchor_is_trend_origin_without_confirmation(self):
        anchors = select_anchor_points(_uptrend_result([0, 3]), self.LOWS, self.HIGHS)
        assert anchors[0].index == 0 and anchors[0].price == 100
        assert anchors[0].confirmed is True
        assert anchors[0].threshold is None

    def test_full_break_rule_rejects_trough_when_resistance_not_exceeded(self):
        anchors = select_anchor_points(
            _uptrend_result([0, 3]), self.LOWS, self.HIGHS, CONFIRM_FULL_BREAK
        )
        a2 = anchors[1]
        assert a2.price == 103  # Low, bukan Close
        assert a2.threshold == pytest.approx(113)
        assert a2.confirmed is False
        assert a2.confirmed_index is None

    def test_half_way_rule_accepts_the_same_trough(self):
        anchors = select_anchor_points(
            _uptrend_result([0, 3]), self.LOWS, self.HIGHS, CONFIRM_HALF_WAY
        )
        a2 = anchors[1]
        assert a2.threshold == pytest.approx(108)
        assert a2.confirmed is True
        assert a2.confirmed_index == 4  # bar pertama dengan High > 108

    def test_reaching_threshold_exactly_is_not_a_break(self):
        highs = self.HIGHS.copy()
        highs.iloc[4] = 113  # tepat menyentuh resistance (double top), belum melewati
        anchors = select_anchor_points(_uptrend_result([0, 3]), self.LOWS, highs)
        assert anchors[1].confirmed is False

    def test_uses_high_of_bars_after_the_trough_not_close(self):
        # Close tidak dilibatkan sama sekali: hanya Low (titik) dan High (tembus).
        highs = self.HIGHS.copy()
        highs.iloc[5] = 114
        anchors = select_anchor_points(_uptrend_result([0, 3]), self.LOWS, highs)
        assert anchors[1].confirmed is True
        assert anchors[1].confirmed_index == 5

    def test_downtrend_mirrors_with_support_and_lows(self):
        # Puncak di bar 0 dan 3. Support B2 = Low terendah di antaranya = 87.
        # High B2 = 97 -> jarak 10: syarat penuh 87, syarat 50% = 92.
        # Setelah B2, Low terendah 91: lolos 50%, gagal 100%.
        lows = pd.Series([98, 92, 87, 95, 91, 94, 96], dtype=float)
        highs = pd.Series([100, 96, 92, 97, 94, 96, 98], dtype=float)
        trend = TrendResult(
            trend=DOWNTREND,
            peaks=[SwingPoint(0, 0.0), SwingPoint(3, 0.0)],
            troughs=[SwingPoint(1, 0.0), SwingPoint(2, 0.0)],
        )
        full = select_anchor_points(trend, lows, highs, CONFIRM_FULL_BREAK)
        half = select_anchor_points(trend, lows, highs, CONFIRM_HALF_WAY)
        assert full[0].price == 100 and full[1].price == 97  # High, bukan Close
        assert full[1].threshold == pytest.approx(87) and full[1].confirmed is False
        assert half[1].threshold == pytest.approx(92) and half[1].confirmed is True
        assert half[1].confirmed_index == 4

    def test_resistance_is_the_peak_between_the_two_troughs_only(self):
        # Puncak yang lebih tinggi SEBELUM dasar pertama (bar 0) tidak dihitung.
        lows = pd.Series([120, 100, 104, 108, 103, 106, 104], dtype=float)
        highs = pd.Series([140, 102, 108, 113, 105, 114, 106], dtype=float)
        anchors = select_anchor_points(_uptrend_result([1, 4]), lows, highs)
        assert anchors[1].threshold == pytest.approx(113)
        assert anchors[1].confirmed is True

    @pytest.mark.parametrize("trend_name", [SIDEWAYS, UNDEFINED])
    def test_empty_for_sideways_or_undefined(self, trend_name):
        trend = TrendResult(trend=trend_name, troughs=[SwingPoint(0, 0.0), SwingPoint(3, 0.0)])
        assert select_anchor_points(trend, self.LOWS, self.HIGHS) == []

    @pytest.mark.parametrize("ratio", [0.0, 1.5, -0.5])
    def test_ratio_outside_range_is_rejected(self, ratio):
        with pytest.raises(ValueError):
            select_anchor_points(_uptrend_result([0, 3]), self.LOWS, self.HIGHS, ratio)


class TestBuildTrendline:
    def test_uptrend_connects_lows_of_confirmed_troughs(self):
        # Tiap dasar diikuti puncak yang melewati puncak sebelumnya -> semua siap.
        df = _ohlc(_zigzag(peaks=[110, 120, 130], troughs=[100, 105, 115]))
        trend = _classify_series(df["Close"])
        assert trend.trend == UPTREND
        line = build_trendline(trend, df["Low"], df["High"], end_index=len(df) - 1)
        assert line.kind == UP_TRENDLINE
        assert [p.price for p in line.points] == [98, 103, 113]  # Low, bukan Close
        assert [p.index for p in line.points] == [t.index for t in trend.troughs]
        assert [p.confirmed_index for p in line.points[1:]] == [17, 27]
        assert line.pending_points == []
        assert line.slope > 0
        assert line.end_index == len(df) - 1
        # Garis melewati (kurang lebih) titik acuan pertama & terakhir.
        assert line.value_at(line.points[0].index) == pytest.approx(98, abs=2)
        assert line.value_at(line.points[-1].index) == pytest.approx(113, abs=2)

    def test_last_trough_is_pending_until_its_resistance_is_broken(self):
        df = _ohlc(_zigzag(peaks=[110, 120, 130], troughs=[100, 105, 115]))
        trend = _classify_series(df["Close"])
        last_trough = trend.troughs[-1].index
        # Resistance dasar ke-3 = High puncak 120 = 123; Low dasar ke-3 = 113.
        # Batasi High setelahnya ke 120: lewat 50% (118) tapi tidak lewat penuh (123).
        highs = df["High"].copy()
        highs.iloc[last_trough + 1 :] = highs.iloc[last_trough + 1 :].clip(upper=120)

        strict = build_trendline(trend, df["Low"], highs, len(df) - 1, CONFIRM_FULL_BREAK)
        assert [p.price for p in strict.points] == [98, 103]
        assert [p.price for p in strict.pending_points] == [113]
        assert strict.pending_points[0].threshold == pytest.approx(123)

        lenient = build_trendline(trend, df["Low"], highs, len(df) - 1, CONFIRM_HALF_WAY)
        assert [p.price for p in lenient.points] == [98, 103, 113]
        assert lenient.pending_points == []

    def test_downtrend_connects_highs_of_confirmed_peaks(self):
        df = _ohlc(_zigzag(peaks=[130, 120, 110], troughs=[125, 112, 100]))
        trend = _classify_series(df["Close"])
        assert trend.trend == DOWNTREND
        line = build_trendline(trend, df["Low"], df["High"], end_index=len(df) - 1)
        assert line.kind == DOWN_TRENDLINE
        # Puncak ke-3 (High 113): support = Low dasar 100 = 98; deret berakhir
        # tepat di 98 tanpa menembus -> belum siap, tidak dihubungkan.
        assert [p.price for p in line.points] == [133, 123]  # High, bukan Close
        assert [p.index for p in line.points] == [p.index for p in trend.peaks[:2]]
        assert [p.price for p in line.pending_points] == [113]
        assert line.pending_points[0].threshold == pytest.approx(98)
        assert line.slope < 0

    def test_two_swings_give_exact_line_through_both(self):
        df = _ohlc(_zigzag(peaks=[110, 120], troughs=[100, 105]))
        trend = _classify_series(df["Close"])
        assert trend.trend == UPTREND
        line = build_trendline(trend, df["Low"], df["High"], end_index=len(df) - 1)
        assert line.value_at(line.points[0].index) == pytest.approx(98)
        assert line.value_at(line.points[1].index) == pytest.approx(103)

    def test_no_line_when_fewer_than_two_anchors_confirmed(self):
        df = _ohlc(_zigzag(peaks=[110, 120], troughs=[100, 105]))
        trend = _classify_series(df["Close"])
        second_trough = trend.troughs[1].index
        # High setelah dasar ke-2 hanya menyentuh resistance (113), tidak melewati.
        highs = df["High"].copy()
        highs.iloc[second_trough + 1 :] = highs.iloc[second_trough + 1 :].clip(upper=113)
        assert build_trendline(trend, df["Low"], highs, end_index=len(df) - 1) is None

    @pytest.mark.parametrize("trend_name", [SIDEWAYS, UNDEFINED])
    def test_no_trendline_for_sideways_or_undefined(self, trend_name):
        trend = TrendResult(
            trend=trend_name,
            peaks=[SwingPoint(5, 110.0), SwingPoint(15, 110.0)],
            troughs=[SwingPoint(10, 100.0), SwingPoint(20, 100.0)],
        )
        prices = pd.Series([100.0] * 25)
        assert build_trendline(trend, prices, prices, end_index=24) is None


def _flat_up_trendline(level: float = 100.0, last_anchor: int = 1, end_index: int = 9) -> Trendline:
    """Up-trendline datar di `level` supaya angka test mudah dibaca."""
    return Trendline(
        kind=UP_TRENDLINE,
        slope=0.0,
        intercept=level,
        points=[AnchorPoint(0, level), AnchorPoint(last_anchor, level)],
        end_index=end_index,
    )


class TestCheckTrendlineBreak:
    """Aturan utama buku: valid break hanya bila Close di luar garis; tembusan
    intraday (Low/High) yang Close-nya kembali ke dalam = false break/whipsaw."""

    def test_intact_when_price_stays_inside(self):
        closes = pd.Series([100, 101, 103, 102, 104, 105, 103, 104, 106, 107], dtype=float)
        lows = closes - 1
        highs = closes + 1
        result = check_trendline_break(_flat_up_trendline(), closes, lows, highs)
        assert result.status == LINE_INTACT
        assert result.checked_from == 2
        assert result.valid_break_index is None and result.whipsaw_indices == []

    def test_intraday_pierce_with_close_inside_is_whipsaw(self):
        closes = pd.Series([100, 101, 103, 102, 104, 105, 103, 104, 106, 107], dtype=float)
        lows = closes - 1
        lows.iloc[4] = 98  # Low menembus ke bawah 100, Close 104 tetap di atas
        lows.iloc[7] = 99
        result = check_trendline_break(_flat_up_trendline(), closes, lows, closes + 1)
        assert result.status == FALSE_BREAK
        assert result.whipsaw_indices == [4, 7]
        assert result.valid_break_index is None

    def test_close_outside_is_valid_break_at_first_such_bar(self):
        closes = pd.Series([100, 101, 103, 102, 104, 99, 98, 104, 106, 107], dtype=float)
        lows = closes - 1
        lows.iloc[3] = 97  # whipsaw sebelum valid break tetap dicatat
        result = check_trendline_break(_flat_up_trendline(), closes, lows, closes + 1)
        assert result.status == VALID_BREAK
        assert result.valid_break_index == 5
        assert result.whipsaw_indices == [3]

    def test_close_exactly_on_line_is_not_outside(self):
        closes = pd.Series([100, 101, 103, 100, 104, 105, 103, 104, 106, 107], dtype=float)
        result = check_trendline_break(_flat_up_trendline(), closes, closes, closes)
        assert result.status == LINE_INTACT

    def test_bars_up_to_last_anchor_are_not_checked(self):
        # Close di bar 1 (= titik acuan terakhir) di bawah garis: artefak regresi, bukan tembusan.
        closes = pd.Series([100, 95, 103, 102, 104, 105, 103, 104, 106, 107], dtype=float)
        result = check_trendline_break(_flat_up_trendline(last_anchor=1), closes, closes, closes)
        assert result.status == LINE_INTACT

    def test_down_trendline_mirrors_with_highs_and_close_above(self):
        line = Trendline(
            kind=DOWN_TRENDLINE,
            slope=0.0,
            intercept=100.0,
            points=[AnchorPoint(0, 100.0), AnchorPoint(1, 100.0)],
            end_index=9,
        )
        closes = pd.Series([100, 99, 97, 98, 96, 95, 97, 96, 94, 93], dtype=float)
        highs = closes + 1
        highs.iloc[3] = 101  # tembus intraday saja
        assert check_trendline_break(line, closes, closes - 1, highs).status == FALSE_BREAK
        closes.iloc[6] = 102  # Close di atas garis
        result = check_trendline_break(line, closes, closes - 1, highs)
        assert result.status == VALID_BREAK
        assert result.valid_break_index == 6
        assert result.whipsaw_indices == [3]

    def test_sloped_line_is_evaluated_per_bar(self):
        # Garis naik 1/bar dari 100: di bar 8 nilainya 108. Close 107 = di bawah garis.
        line = Trendline(
            kind=UP_TRENDLINE,
            slope=1.0,
            intercept=100.0,
            points=[AnchorPoint(0, 100.0), AnchorPoint(2, 102.0)],
            end_index=9,
        )
        closes = pd.Series([100, 102, 102, 105, 106, 107, 108, 109, 107, 111], dtype=float)
        result = check_trendline_break(line, closes, closes, closes)
        assert result.status == VALID_BREAK
        assert result.valid_break_index == 8
