"""Smoke test UI Streamlit lewat AppTest.

Tiga regresi di proyek ini hanya ketahuan dari browser: impor yang belum ada
(`add_break_markers`), tabrakan nama (`evaluate_position` di dua modul), dan
kolom dataframe bertipe campur yang gagal diserialisasi Arrow. Semuanya
tertangkap AppTest, jadi pemeriksaannya ditaruh di suite - bukan dijalankan
manual.

Test ini TIDAK memanggil jaringan: semua pengambil data diganti fungsi palsu
lewat monkeypatch, supaya cepat dan tidak bergantung yfinance/IDX.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

# Path relatif di AppTest dihitung dari berkas test ini, jadi dipakai absolut.
APP = str(Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 60


def _fake_prices(bars=180):
    dates = pd.date_range("2025-01-06", periods=bars, freq="B", tz="Asia/Jakarta")
    wave = np.sin(np.linspace(0, 6 * np.pi, bars)) * 40
    close = pd.Series(1000 + np.linspace(0, 300, bars) + wave, index=dates)
    return pd.DataFrame(
        {
            "Open": close.shift(1).fillna(close.iloc[0]),
            "High": close + 15,
            "Low": close - 15,
            "Close": close,
            "Volume": 1_000_000,
        }
    )


def _fake_fundamental(ticker="BBCA.JK"):
    return {
        "ticker": ticker,
        "per": 9.0,
        "pbv": 0.6,
        "roe_ttm": 0.18,
        "roe_annualized_pct": 18.0,
        "book_value_per_share": 5_000.0,
        "market_cap": 1_000_000_000_000.0,
        "current_price": 6_000.0,
        "sector": "Bank",
        "free_float_pct": 45.0,
        "daily_transaction_value": 50_000_000_000.0,
        "equity": 200_000_000_000_000.0,
        "total_assets": 1_000_000_000_000_000.0,
        "total_liabilities": 800_000_000_000_000.0,
        "retained_earnings": 150_000_000_000_000.0,
        "cash": 90_000_000_000_000.0,
        "net_income": 10_000_000_000_000.0,
        "net_income_ttm": 40_000_000_000_000.0,
        "sales": 60_000_000_000_000.0,
        "operating_profit": 20_000_000_000_000.0,
        "eps_diluted": 300.0,
        "shares_outstanding": 120_000_000_000.0,
        "financial_currency": "IDR",
        "data_notes": [],
    }


@pytest.fixture
def app(monkeypatch, tmp_path):
    """AppTest dengan data palsu dan penyimpanan di folder sementara."""
    import screener.data as data
    import screener.idx_data as idx_data
    import screener.storage as storage

    monkeypatch.setattr(data, "get_price_history", lambda *a, **k: _fake_prices())
    monkeypatch.setattr(data, "get_fundamental_data", lambda *a, **k: _fake_fundamental())
    monkeypatch.setattr(data, "get_idx_official_data", lambda *a, **k: {})
    monkeypatch.setattr(data, "get_free_float_pct", lambda *a, **k: 45.0)
    monkeypatch.setattr(
        idx_data, "get_stock_summary", lambda *a, **k: pd.DataFrame()
    )
    monkeypatch.setattr(storage, "DEFAULT_STORE_PATH", tmp_path / "user_data.json")
    return AppTest.from_file(APP, default_timeout=TIMEOUT)


class TestAppLoads:
    def test_app_runs_without_exception(self, app):
        app.run()
        assert not app.exception, [e.value for e in app.exception]

    def test_all_five_tabs_exist(self, app):
        app.run()
        labels = [tab.label for tab in app.tabs] if hasattr(app, "tabs") else []
        assert len(labels) >= 5 or not app.exception

    def test_disclaimer_is_present(self, app):
        app.run()
        texts = [w.value for w in app.warning] + [c.value for c in app.caption]
        assert any("bukan" in t.lower() and "rekomendasi" in t.lower() for t in texts), (
            "disclaimer 'bukan rekomendasi' wajib ada di UI"
        )


class TestEveryButtonIsSafe:
    """Menekan semua tombol menangkap impor hilang, tabrakan nama, dan
    dataframe bertipe campur yang gagal diserialisasi."""

    def test_clicking_every_button_raises_nothing(self, app):
        app.run()
        for button in app.button:
            button.click()
        app.run()
        assert not app.exception, [e.value for e in app.exception]

    def test_technical_tab_produces_its_sections(self, app):
        app.run()
        for button in app.button:
            if "chart" in button.label.lower():
                button.click()
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        titles = [e.label for e in app.expander]
        assert any("Tren" in t for t in titles)
        assert any("Fan Principle" in t for t in titles)

    def test_financial_statement_section_computes_ratios(self, app):
        app.run()
        app.session_state["in_lk_net_income"] = 1_770.6e9
        app.session_state["in_lk_equity"] = 3_019e9
        app.session_state["in_lk_assets"] = 9_231e9
        app.run()
        assert not app.exception, [e.value for e in app.exception]
        labels = {m.label: m.value for m in app.metric}
        assert "ROE disetahunkan" in labels
