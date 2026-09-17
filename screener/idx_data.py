"""Data resmi dari situs IDX (idx.co.id) sebagai pelengkap yfinance.

Dipakai untuk dua angka yang yfinance hanya bisa mengestimasi: free float
(dari komposisi pemegang saham) dan nilai transaksi harian (dari Ringkasan
Saham). Teknik akses (endpoint `/primary/...` + curl_cffi impersonasi
browser) dipinjam dari proyek open-source idx-bei; proyek itu sendiri tidak
dipakai karena membawa dependency berat (Neo4j, PostgreSQL, arelle).

Ini endpoint tidak resmi untuk publik (dipakai frontend idx.co.id), jadi
bisa berubah sewaktu-waktu. Semua fungsi jaringan mengembalikan None/kosong
saat gagal, bukan raise, supaya UI tetap jalan dengan data yfinance.
"""

import pandas as pd
from curl_cffi import requests

IDX_BASE_URL = "https://www.idx.co.id/primary"
STOCK_SUMMARY_REFERER = (
    "https://www.idx.co.id/id/data-pasar/ringkasan-perdagangan/ringkasan-saham/"
)
COMPANY_PROFILE_REFERER = "https://www.idx.co.id/id/perusahaan-tercatat/profil-perusahaan/"

# Cloudflare di idx.co.id menolak sebagian profil impersonasi (per Sep 2026:
# "chrome" dan "firefox" dapat 403, "chrome131" dan "safari" lolos). Dicoba
# berurutan sampai ada yang berhasil.
IMPERSONATE_PROFILES = ("chrome131", "safari")
REQUEST_TIMEOUT_SECONDS = 30

PUBLIC_SHAREHOLDER_CATEGORY_PREFIX = "Masyarakat"


def to_idx_code(ticker: str) -> str:
    """'BBCA.JK' (format yfinance) -> 'BBCA' (format IDX)."""
    return ticker.upper().removesuffix(".JK")


def _get_json(path: str, referer: str) -> dict | None:
    headers = {"accept": "application/json, text/plain, */*", "Referer": referer}
    for profile in IMPERSONATE_PROFILES:
        try:
            response = requests.get(
                f"{IDX_BASE_URL}/{path}",
                headers=headers,
                impersonate=profile,
                timeout=REQUEST_TIMEOUT_SECONDS,
            )
            if response.status_code == 200:
                return response.json()
        except Exception:
            continue
    return None


def get_stock_summary() -> pd.DataFrame:
    """Ringkasan Saham IDX hari bursa terakhir, semua emiten (satu request).

    Kolom penting: StockCode, Date, Close, Volume, Value (nilai transaksi
    pasar reguler, Rp), NonRegularValue, ListedShares.
    """
    payload = _get_json("TradingSummary/GetStockSummary?length=9999&start=0", STOCK_SUMMARY_REFERER)
    if not payload or not payload.get("data"):
        return pd.DataFrame()
    return pd.DataFrame(payload["data"])


def get_shareholders(stock_code: str) -> list[dict]:
    """Komposisi pemegang saham dari profil emiten IDX."""
    payload = _get_json(
        f"ListedCompany/GetCompanyProfilesDetail?KodeEmiten={stock_code}&language=id-id",
        COMPANY_PROFILE_REFERER,
    )
    if not payload:
        return []
    return payload.get("PemegangSaham", [])


# --- Fungsi murni (tanpa jaringan) supaya bisa di-unit-test ---


def compute_public_float_pct(shareholders: list[dict]) -> float | None:
    """Free float = total persentase kategori 'Masyarakat ...'.

    IDX sudah memisahkan pemegang >= 5%, pengendali, treasury, direksi, dan
    komisaris ke kategori masing-masing, jadi sisa yang berlabel Masyarakat
    (warkat + non warkat) adalah saham yang benar-benar beredar di publik.
    """
    if not shareholders:
        return None
    public_pct = sum(
        float(row.get("Persentase") or 0)
        for row in shareholders
        if str(row.get("Kategori", "")).startswith(PUBLIC_SHAREHOLDER_CATEGORY_PREFIX)
    )
    return public_pct


def lookup_daily_transaction_value(
    summary: pd.DataFrame, stock_code: str
) -> tuple[float, str] | None:
    """Nilai transaksi pasar reguler (Rp) dan tanggalnya untuk satu emiten.

    Pasar negosiasi (NonRegularValue) sengaja tidak dihitung: transaksi
    blok/tutup sendiri tidak mencerminkan likuiditas yang bisa diakses
    investor ritel.
    """
    if summary.empty or "StockCode" not in summary:
        return None
    rows = summary[summary["StockCode"] == stock_code]
    if rows.empty:
        return None
    row = rows.iloc[0]
    return float(row["Value"]), str(row.get("Date", ""))[:10]
