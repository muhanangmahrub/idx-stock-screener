"""Fundamental titik-waktu dari laporan tahunan - untuk backtest Lapis 1.

Backtest hanya jujur kalau keputusan di suatu tanggal memakai laporan yang
benar-benar sudah terbit saat itu. Emiten IDX wajib menyampaikan laporan
tahunan audited paling lambat akhir Maret, jadi laporan tutup buku 31
Desember dianggap tersedia ~90 hari sesudahnya (`REPORTING_LAG_DAYS`).

Modul ini tidak mengambil data sendiri: laporan diberikan pemanggil (dari
`data.get_annual_statements`) supaya bisa diuji tanpa jaringan.

BATASAN yang harus diingat saat membaca hasil backtest:
- yfinance tidak memberi jumlah saham historis, jadi jumlah saham sekarang
  dipakai untuk semua tahun. Kalau pernah ada right issue/buyback, PER & PBV
  masa lalu sedikit meleset.
- Free float tidak punya historis di IDX, jadi kriteria itu dilewati.
- Laporan tahunan hanya memperbarui fundamental setahun sekali; pemakaian
  nyata memakai laporan kuartalan yang lebih cepat bereaksi.
"""

from dataclasses import dataclass
from datetime import timedelta

import pandas as pd

from screener.financials import absolute_rejects, assess_growth_quality
from screener.fundamental import (
    MIN_DAILY_TRANSACTION_VALUE,
    MIN_ROE_ANNUALIZED_PCT,
    is_cheap_valuation,
)

REPORTING_LAG_DAYS = 90

# Alasan gugur - dipakai untuk menjelaskan mengapa sebuah bar tidak layak.
PASS = "lolos"
FAIL_NO_REPORT = "belum ada laporan"
FAIL_ABSOLUTE = "tolak mutlak"
FAIL_LIQUIDITY = "likuiditas"
FAIL_ROE = "ROE di bawah batas"
FAIL_VALUATION = "valuasi tidak murah"
FAIL_GROWTH = "kualitas pertumbuhan"


@dataclass
class FundamentalSnapshot:
    """Satu laporan tahunan beserta kapan ia bisa dipakai."""

    available_from: pd.Timestamp
    fiscal_year: str
    net_income: float | None = None
    equity: float | None = None
    retained_earnings: float | None = None
    shares: float | None = None
    sales: float | None = None
    operating_profit: float | None = None

    @property
    def roe_pct(self) -> float | None:
        if self.net_income is None or not self.equity:
            return None
        return self.net_income / self.equity * 100

    @property
    def bvps(self) -> float | None:
        # Ekuitas nol diperlakukan sama seperti tidak diketahui (konsisten
        # dengan roe_pct): nilai buku nol bukan angka yang bisa dipakai.
        if not self.equity or not self.shares:
            return None
        return self.equity / self.shares

    @property
    def eps(self) -> float | None:
        if self.net_income is None or not self.shares:
            return None
        return self.net_income / self.shares


def _value(frame: pd.DataFrame | None, row: str, column) -> float | None:
    if frame is None or frame.empty or row not in frame.index or column not in frame.columns:
        return None
    value = frame.loc[row, column]
    return None if pd.isna(value) else float(value)


def build_snapshots(
    income_stmt: pd.DataFrame | None,
    balance_sheet: pd.DataFrame | None,
    shares_outstanding: float | None,
    lag_days: int = REPORTING_LAG_DAYS,
) -> list[FundamentalSnapshot]:
    """Daftar laporan tahunan, urut dari yang paling dulu tersedia."""
    if income_stmt is None or income_stmt.empty:
        return []

    snapshots = [
        FundamentalSnapshot(
            available_from=pd.Timestamp(column) + timedelta(days=lag_days),
            fiscal_year=str(pd.Timestamp(column).date()),
            net_income=_value(income_stmt, "Net Income", column),
            sales=_value(income_stmt, "Total Revenue", column),
            operating_profit=_value(income_stmt, "Operating Income", column),
            equity=_value(balance_sheet, "Stockholders Equity", column),
            retained_earnings=_value(balance_sheet, "Retained Earnings", column),
            shares=shares_outstanding,
        )
        for column in income_stmt.columns
    ]
    return sorted(snapshots, key=lambda s: s.available_from)


def latest_snapshot(
    snapshots: list[FundamentalSnapshot], when
) -> FundamentalSnapshot | None:
    """Laporan terbaru yang SUDAH terbit pada tanggal `when`."""
    moment = pd.Timestamp(when)
    if moment.tzinfo is not None:
        moment = moment.tz_localize(None)
    published = [s for s in snapshots if s.available_from <= moment]
    return published[-1] if published else None


def previous_snapshot(
    snapshots: list[FundamentalSnapshot], snapshot: FundamentalSnapshot
) -> FundamentalSnapshot | None:
    """Laporan setahun sebelumnya, untuk menilai pertumbuhan."""
    earlier = [s for s in snapshots if s.available_from < snapshot.available_from]
    return earlier[-1] if earlier else None


def passes_layer_one(
    snapshot: FundamentalSnapshot | None,
    price: float | None,
    daily_transaction_value: float | None = None,
    is_blue_chip: bool = False,
    require_growth: bool = False,
    previous: FundamentalSnapshot | None = None,
    min_roe_pct: float = MIN_ROE_ANNUALIZED_PCT,
    min_daily_value: float = MIN_DAILY_TRANSACTION_VALUE,
) -> tuple[bool, str]:
    """Kriteria Lapis 1 yang bisa direkonstruksi titik-waktu.

    Urutannya mengikuti corong di `universe.screen_candidate`: tolak mutlak
    dulu, lalu likuiditas, ROE, dan terakhir valuasi murah. `require_growth`
    menambahkan syarat kualitas pertumbuhan (penjualan < laba usaha < laba
    bersih) yang butuh laporan tahun sebelumnya.
    """
    if snapshot is None or price is None:
        return False, FAIL_NO_REPORT
    if absolute_rejects(
        retained_earnings=snapshot.retained_earnings,
        equity=snapshot.equity,
        net_income=snapshot.net_income,
    ):
        return False, FAIL_ABSOLUTE
    if daily_transaction_value is not None and daily_transaction_value < min_daily_value:
        return False, FAIL_LIQUIDITY

    roe = snapshot.roe_pct
    if roe is None or roe < min_roe_pct:
        return False, FAIL_ROE

    eps, bvps = snapshot.eps, snapshot.bvps
    per = price / eps if eps and eps > 0 else None
    pbv = price / bvps if bvps and bvps > 0 else None
    if not is_cheap_valuation(per=per, pbv=pbv, is_blue_chip=is_blue_chip):
        return False, FAIL_VALUATION

    if require_growth:
        if previous is None:
            return False, FAIL_GROWTH
        quality = assess_growth_quality(
            sales=snapshot.sales, sales_previous=previous.sales,
            operating_profit=snapshot.operating_profit,
            operating_profit_previous=previous.operating_profit,
            net_income=snapshot.net_income, net_income_previous=previous.net_income,
        )
        if not quality.is_ideal:
            return False, FAIL_GROWTH

    return True, PASS


def eligibility_flags(
    df: pd.DataFrame,
    snapshots: list[FundamentalSnapshot],
    is_blue_chip: bool = False,
    require_growth: bool = False,
    bars_per_week: int = 5,
) -> list[bool]:
    """Lolos/tidak Lapis 1 untuk tiap bar harga.

    Nilai transaksi harian diperkirakan dari Close x Volume bar itu; untuk
    bar mingguan dibagi `bars_per_week` supaya sebanding dengan batas harian
    Rp5 miliar.
    """
    has_volume = "Volume" in df.columns
    flags = []
    for position, when in enumerate(df.index):
        snapshot = latest_snapshot(snapshots, when)
        daily_value = None
        if has_volume:
            bar_value = float(df["Close"].iloc[position]) * float(df["Volume"].iloc[position])
            daily_value = bar_value / bars_per_week
        passed, _ = passes_layer_one(
            snapshot,
            float(df["Close"].iloc[position]),
            daily_value,
            is_blue_chip=is_blue_chip,
            require_growth=require_growth,
            previous=previous_snapshot(snapshots, snapshot) if snapshot else None,
        )
        flags.append(passed)
    return flags
