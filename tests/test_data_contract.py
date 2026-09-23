"""Penjaga kontrak sumber data eksternal (yfinance & IDX).

Modul fundamental membaca baris laporan keuangan berdasarkan NAMA, mis.
"Stockholders Equity" dan "Diluted EPS". Kalau yfinance mengganti namanya,
kode tidak error - nilainya diam-diam jadi None dan rasio ikut kosong. Test
ini memastikan kegagalan seperti itu terlihat.

Ditandai `network` dan dilewati secara default supaya suite tetap cepat dan
bisa jalan offline. Jalankan sewaktu menaikkan versi yfinance:

    pytest -m network
"""

import pytest

pytestmark = pytest.mark.network

TICKER = "BBCA.JK"  # bank: skema laporannya paling berbeda
NON_BANK_TICKER = "UNVR.JK"

# Nama baris yang dibaca screener/data.py dan harus ada di semua emiten.
INCOME_ROWS = ["Net Income", "Total Revenue", "Diluted EPS"]
# Ada di emiten non-bank saja; bank tidak melaporkan "Operating Income" di
# skema yfinance, sehingga laba usaha kosong dan aturan kualitas pertumbuhan
# hanya bisa memakai penjualan & laba bersih.
SECTOR_DEPENDENT_INCOME_ROWS = ["Operating Income"]
BALANCE_ROWS = [
    "Stockholders Equity",
    "Total Assets",
    "Total Liabilities Net Minority Interest",
    "Retained Earnings",
    "Cash And Cash Equivalents",
]
INFO_FIELDS = [
    "trailingPE",
    "priceToBook",
    "returnOnEquity",
    "bookValue",
    "marketCap",
    "currentPrice",
    "sector",
    "sharesOutstanding",
    "financialCurrency",
]


@pytest.fixture(scope="module")
def stock():
    yf = pytest.importorskip("yfinance")
    return yf.Ticker(TICKER)


def test_quarterly_income_rows_still_exist(stock):
    missing = [row for row in INCOME_ROWS if row not in stock.quarterly_income_stmt.index]
    assert not missing, f"nama baris laba/rugi berubah di yfinance: {missing}"


def test_operating_income_exists_for_non_banks():
    yf = pytest.importorskip("yfinance")
    rows = yf.Ticker(NON_BANK_TICKER).quarterly_income_stmt.index
    missing = [row for row in SECTOR_DEPENDENT_INCOME_ROWS if row not in rows]
    assert not missing, f"nama baris laba usaha berubah di yfinance: {missing}"


def test_banks_have_no_operating_income_row(stock):
    """Bukan kegagalan, tapi perlu tercatat: laba usaha kosong untuk bank."""
    assert "Operating Income" not in stock.quarterly_income_stmt.index


def test_quarterly_balance_rows_still_exist(stock):
    missing = [row for row in BALANCE_ROWS if row not in stock.quarterly_balance_sheet.index]
    assert not missing, f"nama baris neraca berubah di yfinance: {missing}"


def test_info_fields_still_exist(stock):
    info = stock.info
    missing = [field for field in INFO_FIELDS if field not in info]
    assert not missing, f"field info berubah di yfinance: {missing}"


def test_price_history_is_nominal_not_adjusted(stock):
    from screener.data import get_price_history

    df = get_price_history(TICKER, period="1mo", interval="1d")
    assert not df.empty
    # Harga IDX bergerak per tick bulat; harga tersesuaikan dividen tidak bulat.
    assert all(float(c).is_integer() for c in df["Close"].tail(5)), (
        "harga tidak lagi nominal - cek argumen auto_adjust"
    )


def test_fundamental_fetch_fills_the_fields_screener_needs():
    from screener.data import get_fundamental_data

    data = get_fundamental_data(TICKER)
    assert data, "fetch fundamental gagal total"
    for field in ("equity", "net_income_ttm", "roe_annualized_pct", "shares_outstanding"):
        assert data.get(field) is not None, f"{field} kosong - kontrak yfinance berubah?"
