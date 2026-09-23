import pandas as pd

from screener.universe import (
    STATUS_DATA_MISSING,
    STATUS_FAILED,
    STATUS_PASSED,
    STATUS_REJECTED,
    UniverseCandidate,
    apply_sector_comparison,
    candidates_to_dataframe,
    filter_liquid_stocks,
    screen_candidate,
    screen_universe,
)


def _summary() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"StockCode": "BBCA", "StockName": "Bank Central Asia", "Value": 393e9},
            {"StockCode": "SEPI", "StockName": "Emiten Sepi", "Value": 4.9e9},
            {"StockCode": "TLKM", "StockName": "Telkom", "Value": 150e9},
            {"StockCode": "ABBA", "StockName": "Mahaka Media", "Value": 0.0},
        ]
    )


def _candidate(code: str = "BBCA", value: float = 100e9) -> UniverseCandidate:
    return UniverseCandidate(stock_code=code, stock_name=code, daily_transaction_value=value)


class TestFilterLiquidStocks:
    def test_keeps_only_value_at_or_above_threshold_sorted_desc(self):
        codes = [c.stock_code for c in filter_liquid_stocks(_summary())]
        assert codes == ["BBCA", "TLKM"]

    def test_threshold_is_inclusive(self):
        summary = pd.DataFrame([{"StockCode": "X", "StockName": "X", "Value": 5e9}])
        assert len(filter_liquid_stocks(summary)) == 1

    def test_empty_or_malformed_summary_returns_empty(self):
        assert filter_liquid_stocks(pd.DataFrame()) == []
        assert filter_liquid_stocks(pd.DataFrame([{"StockCode": "X"}])) == []


class TestScreenCandidateFunnel:
    def test_passes_when_all_stages_ok(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 22.0, "per": 7.0, "pbv": 1.5},
        )
        assert result.status == STATUS_PASSED
        assert result.reasons == []
        assert (result.free_float_pct, result.roe_annualized_pct, result.per, result.pbv) == (
            40.0, 22.0, 7.0, 1.5,
        )

    def test_blacklist_stops_before_any_fetch(self):
        calls = []
        result = screen_candidate(
            _candidate("GORE"),
            fetch_free_float=lambda code: calls.append(code) or 50.0,
            fetch_fundamental=lambda ticker: calls.append(ticker) or {},
            blacklist={"GORE"},
        )
        assert result.status == STATUS_FAILED
        assert "blacklist" in result.reasons[0]
        assert calls == []

    def test_low_free_float_stops_before_yfinance(self):
        calls = []
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 10.0,
            fetch_fundamental=lambda ticker: calls.append(ticker) or {},
        )
        assert result.status == STATUS_FAILED
        assert "Free float 10.0%" in result.reasons[0]
        assert result.roe_annualized_pct is None
        assert calls == []

    def test_missing_free_float_is_data_missing_not_failed(self):
        result = screen_candidate(
            _candidate(), fetch_free_float=lambda code: None, fetch_fundamental=lambda t: {}
        )
        assert result.status == STATUS_DATA_MISSING

    def test_missing_roe_is_data_missing(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"per": 10.0},
        )
        assert result.status == STATUS_DATA_MISSING
        assert result.per == 10.0

    def test_low_roe_fails_via_initial_screening_with_note_band(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 8.0},
        )
        assert result.status == STATUS_FAILED
        assert "ROE disetahunkan 8.0%" in result.reasons[0]

    def test_not_cheap_fails_even_if_initial_screening_passes(self):
        """Kasus screenshot: ROE lolos tapi PER 32,7 / PBV tinggi -> harus gugur."""
        result = screen_candidate(
            _candidate("AMMN"),
            fetch_free_float=lambda code: 25.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 11.7, "per": 32.7, "pbv": 3.7},
        )
        assert result.status == STATUS_FAILED
        assert result.reasons == ["Valuasi tidak murah: PER 32.7 > 8 dan PBV 3.70 > 0.7"]
        # Catatan band ROE dari bagian A tetap ikut.
        assert any("band ideal" in note for note in result.notes)

    def test_cheap_by_pbv_alone_passes(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 15.0, "per": 20.0, "pbv": 0.6},
        )
        assert result.status == STATUS_PASSED

    def test_blue_chip_range_per_fails_with_manual_check_note(self):
        """Semua dinilai small cap; PER 8-12 gugur tapi diberi catatan cek manual."""
        result = screen_candidate(
            _candidate("BBCA"),
            fetch_free_float=lambda code: 44.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 22.0, "per": 11.9, "pbv": 2.7},
        )
        assert result.status == STATUS_FAILED
        assert "Valuasi tidak murah" in result.reasons[0]
        assert any("batas blue chip" in note for note in result.notes)

    def test_per_above_blue_chip_limit_gets_no_manual_check_note(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 44.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 22.0, "per": 14.4, "pbv": 2.1},
        )
        assert result.status == STATUS_FAILED
        assert not any("batas blue chip" in note for note in result.notes)

    def test_missing_pbv_still_judged_by_per(self):
        """PBV dikosongkan (laporan USD) -> PER sendirian yang menentukan."""
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 15.0, "per": 6.0, "pbv": None},
        )
        assert result.status == STATUS_PASSED

        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 15.0, "per": 20.0, "pbv": None},
        )
        assert result.status == STATUS_FAILED
        assert "PBV n/a" in result.reasons[0]

    def test_missing_per_and_pbv_is_data_missing(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 15.0},
        )
        assert result.status == STATUS_DATA_MISSING
        assert "PER dan PBV" in result.reasons[0]

    def test_data_notes_from_fetcher_are_shown_as_notes(self):
        result = screen_candidate(
            _candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {
                "roe_annualized_pct": 22.0,
                "per": 6.0,
                "pbv": None,
                "data_notes": ["Laporan keuangan dalam USD; PBV/BVPS yfinance tidak valid"],
            },
        )
        assert result.status == STATUS_PASSED
        assert result.notes == ["Laporan keuangan dalam USD; PBV/BVPS yfinance tidak valid"]

    def test_fetch_fundamental_uses_jk_ticker(self):
        seen = []
        screen_candidate(
            _candidate("TLKM"),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: seen.append(ticker) or {"roe_annualized_pct": 15.0},
        )
        assert seen == ["TLKM.JK"]


class TestScreenUniverse:
    def test_reports_progress_and_returns_all_candidates(self):
        progress = []
        candidates = [_candidate("A"), _candidate("B")]
        results = screen_universe(
            candidates,
            fetch_free_float=lambda code: 30.0,
            fetch_fundamental=lambda ticker: {"roe_annualized_pct": 12.0, "per": 6.0, "pbv": 1.0},
            on_progress=lambda done, total, code: progress.append((done, total, code)),
        )
        assert [r.status for r in results] == [STATUS_PASSED, STATUS_PASSED]
        assert progress == [(1, 2, "A"), (2, 2, "B")]


def test_candidates_to_dataframe_joins_reasons_and_notes():
    candidate = _candidate()
    candidate.status = STATUS_FAILED
    candidate.reasons = ["alasan"]
    candidate.notes = ["catatan"]
    df = candidates_to_dataframe([candidate])
    assert list(df["Kode"]) == ["BBCA"]
    assert df.loc[0, "Alasan / catatan"] == "alasan; catatan"
    assert df.loc[0, "Nilai transaksi"] == "Rp100.000.000.000"


class TestAbsoluteRejectsInFunnel:
    """Tolak mutlak e-book: gugur tanpa toleransi, status terpisah."""

    def _candidate(self):
        return UniverseCandidate(
            stock_code="XXXX", stock_name="PT Rugi", daily_transaction_value=10e9
        )

    def test_negative_equity_rejects_before_threshold_check(self):
        candidate = screen_candidate(
            self._candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {
                "equity": -1_000.0, "roe_annualized_pct": 50.0, "per": 5.0, "pbv": 0.5
            },
        )
        assert candidate.status == STATUS_REJECTED
        assert candidate.reasons == ["Ekuitas negatif"]

    def test_negative_per_is_rejected_not_called_cheap(self):
        candidate = screen_candidate(
            self._candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {"per": -3.0, "roe_annualized_pct": 20.0},
        )
        assert candidate.status == STATUS_REJECTED
        assert "PER negatif" in candidate.reasons

    def test_healthy_candidate_still_passes(self):
        candidate = screen_candidate(
            self._candidate(),
            fetch_free_float=lambda code: 40.0,
            fetch_fundamental=lambda ticker: {
                "equity": 1_000.0, "net_income": 100.0, "retained_earnings": 500.0,
                "roe_annualized_pct": 20.0, "per": 8.0, "pbv": 0.6, "sector": "Bank",
            },
        )
        assert candidate.status == STATUS_PASSED
        assert candidate.sector == "Bank"


class TestSectorComparison:
    """Jalur opsional: catatan pembanding sektor, tidak mengubah status."""

    def _passed(self, code, per, pbv, sector):
        candidate = UniverseCandidate(
            stock_code=code, stock_name=code, daily_transaction_value=10e9,
            per=per, pbv=pbv, sector=sector, status=STATUS_PASSED,
        )
        return candidate

    def test_adds_note_without_changing_status(self):
        candidates = [
            self._passed("AAAA", 10.0, 1.0, "Bank"),
            self._passed("BBBB", 14.0, 2.0, "Bank"),
            self._passed("CCCC", 30.0, 3.0, "Bank"),
        ]
        stats = apply_sector_comparison(candidates)
        assert stats["Bank"].per_median == 14.0
        assert all(c.status == STATUS_PASSED for c in candidates)
        assert any("PER lebih murah" in note for note in candidates[0].notes)
        assert any("PER lebih mahal" in note for note in candidates[2].notes)

    def test_candidate_without_sector_gets_no_note(self):
        candidates = [self._passed("AAAA", 10.0, 1.0, None)]
        apply_sector_comparison(candidates)
        assert not candidates[0].notes
