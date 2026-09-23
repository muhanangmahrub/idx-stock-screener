"""Pengambilan data harga & fundamental saham IDX via yfinance.

Ticker IDX di yfinance memakai format "XXXX.JK" (mis. "BBCA.JK").
"""

import pandas as pd
import yfinance as yf

from screener.financials import sum_trailing_quarters
from screener.fundamental import (
    compute_free_float_pct,
    estimate_daily_transaction_value,
    roe_ttm_pct,
)
from screener.idx_data import (
    compute_public_float_pct,
    get_shareholders,
    get_stock_summary,
    lookup_daily_transaction_value,
    to_idx_code,
)


def get_price_history(
    ticker: str, period: str = "1y", interval: str = "1d", adjusted: bool = False
) -> pd.DataFrame:
    """Ambil data OHLCV historis untuk satu ticker.

    `adjusted=False` (default) memberi harga NOMINAL apa adanya. Itu yang
    benar untuk analisis teknikal: support/resistance, batas breakout, dan
    cut-loss harus memakai harga yang juga dilihat pelaku pasar lain dan
    sesuai tick IDX. Harga yang disesuaikan dividen menggeser seluruh sejarah
    dan memunculkan level berkoma seperti 6474,9035.

    `adjusted=True` untuk perbandingan imbal hasil jangka panjang, di mana
    dividen memang perlu ikut diperhitungkan.

    Mengembalikan DataFrame kosong (bukan raise) kalau ticker tidak
    ditemukan atau tidak ada data, supaya caller (mis. app.py) bisa
    menampilkan pesan yang ramah tanpa try/except di banyak tempat.
    """
    try:
        return yf.Ticker(ticker).history(
            period=period, interval=interval, auto_adjust=adjusted
        )
    except Exception:
        return pd.DataFrame()


def is_last_bar_provisional(
    df: pd.DataFrame, interval: str, now: pd.Timestamp | None = None
) -> bool:
    """True bila bar terakhir masih berjalan, jadi Close-nya belum final.

    Semua aturan Edianto Ong bertumpu pada harga PENUTUPAN. Kalau bar terakhir
    adalah hari (atau minggu) yang belum selesai, status valid break/cut-loss
    yang menyala sekarang masih bisa batal sebelum penutupan - itu perlu
    diberitahukan, bukan disembunyikan.
    """
    if df.empty:
        return False
    now = now or pd.Timestamp.now(tz=df.index.tz)
    last = pd.Timestamp(df.index[-1])
    if interval == "1wk":
        # Bar mingguan selesai setelah akhir pekan yang memuatnya lewat.
        return (now - last).days < 7
    return last.date() >= now.date()


def _row_values(statement: pd.DataFrame, row_name: str) -> list[float]:
    """Semua nilai satu baris laporan keuangan, urut dari periode terbaru."""
    if statement is None or statement.empty or row_name not in statement.index:
        return []
    return [float(v) for v in statement.loc[row_name].dropna()]


def _latest_row_value(statement: pd.DataFrame, row_name: str) -> float | None:
    """Ambil nilai kolom terbaru dari satu baris laporan keuangan yfinance."""
    values = _row_values(statement, row_name)
    return values[0] if values else None


def _book_value_currency_matches_price(info: dict) -> bool:
    """True bila mata uang laporan keuangan sama dengan mata uang harga saham.

    Kalau salah satu tidak diketahui, dianggap cocok supaya tidak membuang
    data tanpa alasan.
    """
    price_currency = info.get("currency")
    statement_currency = info.get("financialCurrency")
    if not price_currency or not statement_currency:
        return True
    return price_currency == statement_currency


def get_fundamental_data(ticker: str) -> dict:
    """Ambil ringkasan data fundamental dari yfinance.

    Mengembalikan dict kosong kalau ticker tidak dikenal atau gagal diambil.

    Selain field mentah dari `Ticker.info`, dihitung juga tiga input metode
    Teguh Hidayat yang tidak disediakan langsung oleh yfinance: free float
    (floatShares / sharesOutstanding), ROE disetahunkan (laporan kuartalan),
    dan estimasi nilai transaksi harian (Close x Volume). Semuanya estimasi -
    angka resmi ada di IDX dan laporan keuangan emiten.
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        if not info or info.get("currentPrice") is None:
            return {}

        quarterly_income = stock.quarterly_income_stmt
        quarterly_balance = stock.quarterly_balance_sheet
        recent_prices = stock.history(period="3mo")
    except Exception:
        return {}

    # Emiten yang laporan keuangannya dalam USD (mis. AMMN, BRPT) diberi
    # `bookValue` dalam USD oleh yfinance, sementara harga dalam IDR - hasilnya
    # priceToBook puluhan ribu. PER aman karena EPS sudah dikonversi ke IDR.
    # Daripada menampilkan angka salah, PBV & BVPS dikosongkan dan diberi catatan.
    pbv_valid = _book_value_currency_matches_price(info)
    data_notes = []
    if not pbv_valid:
        data_notes.append(
            f"Laporan keuangan dalam {info.get('financialCurrency')}; PBV/BVPS "
            "yfinance tidak valid, isi manual dari laporan keuangan"
        )

    # Komponen LK mentah untuk jalur hitung sendiri (financials.py): dipakai
    # saat rasio jadi dari yfinance bolong, dan untuk aturan tolak mutlak.
    quarterly_net_income = _row_values(quarterly_income, "Net Income")
    net_income_ttm = sum_trailing_quarters(quarterly_net_income)

    raw_components = {
        "net_income_ttm": net_income_ttm,
        "equity": _latest_row_value(quarterly_balance, "Stockholders Equity"),
        "total_assets": _latest_row_value(quarterly_balance, "Total Assets"),
        "total_liabilities": _latest_row_value(
            quarterly_balance, "Total Liabilities Net Minority Interest"
        ),
        "retained_earnings": _latest_row_value(quarterly_balance, "Retained Earnings"),
        "cash": _latest_row_value(quarterly_balance, "Cash And Cash Equivalents"),
        "net_income": _latest_row_value(quarterly_income, "Net Income"),
        "sales": _latest_row_value(quarterly_income, "Total Revenue"),
        "operating_profit": _latest_row_value(quarterly_income, "Operating Income"),
        "eps_diluted": _latest_row_value(quarterly_income, "Diluted EPS"),
        "shares_outstanding": info.get("sharesOutstanding"),
        "financial_currency": info.get("financialCurrency"),
    }

    return {
        "ticker": ticker,
        **raw_components,
        "per": info.get("trailingPE"),
        "pbv": info.get("priceToBook") if pbv_valid else None,
        "roe_ttm": info.get("returnOnEquity"),
        "book_value_per_share": info.get("bookValue") if pbv_valid else None,
        "data_notes": data_notes,
        "market_cap": info.get("marketCap"),
        "current_price": info.get("currentPrice"),
        "sector": info.get("sector"),
        "free_float_pct": compute_free_float_pct(
            info.get("floatShares"), info.get("sharesOutstanding")
        ),
        # ROE dari laba TTM (keputusan pemilik 2026-09-23): satu kuartal x4
        # terlalu musiman untuk dipakai sebagai filter sekaligus dasar harga wajar.
        "roe_annualized_pct": roe_ttm_pct(
            net_income_ttm, _latest_row_value(quarterly_balance, "Stockholders Equity")
        ),
        "daily_transaction_value": estimate_daily_transaction_value(recent_prices),
    }


def get_annual_statements(ticker: str) -> tuple:
    """Laporan keuangan TAHUNAN + jumlah saham, untuk backtest titik-waktu.

    Dipisahkan dari `get_fundamental_data` karena backtest butuh seluruh
    kolom tahunan (beberapa tahun ke belakang), bukan hanya periode terbaru.
    Mengembalikan `(income_stmt, balance_sheet, shares_outstanding)`; ketiganya
    None/kosong bila gagal diambil.
    """
    try:
        stock = yf.Ticker(ticker)
        return stock.income_stmt, stock.balance_sheet, stock.info.get("sharesOutstanding")
    except Exception:
        return None, None, None


def get_idx_official_data(
    ticker: str, stock_summary: pd.DataFrame | None = None
) -> dict:
    """Free float & nilai transaksi harian resmi dari idx.co.id.

    `stock_summary` bisa disuplai caller (mis. hasil cache Streamlit) karena
    satu request-nya sudah memuat semua emiten. Mengembalikan dict kosong
    bila kedua angka tidak didapat.
    """
    stock_code = to_idx_code(ticker)

    free_float_pct = compute_public_float_pct(get_shareholders(stock_code))

    if stock_summary is None:
        stock_summary = get_stock_summary()
    transaction = lookup_daily_transaction_value(stock_summary, stock_code)

    if free_float_pct is None and transaction is None:
        return {}

    return {
        "ticker": ticker,
        "free_float_pct": free_float_pct,
        "daily_transaction_value": transaction[0] if transaction else None,
        "date": transaction[1] if transaction else "",
    }


def get_free_float_pct(stock_code: str) -> float | None:
    """Free float resmi IDX untuk satu kode emiten (format 'BBCA', bukan 'BBCA.JK').

    Dipakai screening massal sebagai fetcher tahap free float.
    """
    return compute_public_float_pct(get_shareholders(stock_code))
