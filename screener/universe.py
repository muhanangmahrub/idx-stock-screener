"""Screening massal universe IDX (CLAUDE.md bagian A + C, dijalankan ke semua emiten).

Ringkasan Saham IDX memuat nilai transaksi semua emiten dalam satu request,
tapi free float (idx.co.id) dan ROE/PER/PBV (yfinance) harus diambil satu per
satu. Karena itu screening disusun sebagai corong: tahap tanpa jaringan dulu
(likuiditas, blacklist), lalu tahap satu request (free float), dan tahap
paling mahal (yfinance, ~4 request per ticker) hanya untuk yang masih
bertahan. Emiten yang gugur di satu tahap tidak diambil datanya lebih lanjut,
jadi kolom tahap berikutnya sengaja kosong (None) - bukan bug.

Tahap terakhir adalah aturan valuasi murah (bagian C). Di screening massal
tidak ada cara membedakan blue chip vs small cap per emiten (CLAUDE.md tidak
memberi batasnya), jadi SEMUA emiten dinilai dengan batas small cap yang
lebih ketat (PER <= 8). Blue chip dengan PER 8-12 akan tidak lolos di sini
dan diberi catatan supaya dicek manual di tab analisis tunggal.

Fungsi pengambil data disuntikkan sebagai parameter supaya modul ini bisa
di-unit-test dengan data palsu tanpa menyentuh jaringan.
"""

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

from screener.financials import absolute_rejects
from screener.formatting import format_rupiah
from screener.fundamental import (
    BLUE_CHIP_MAX_PER,
    MAX_PBV_CHEAP,
    MIN_DAILY_TRANSACTION_VALUE,
    MIN_FREE_FLOAT_PCT,
    SMALL_CAP_MAX_PER,
    ScreeningResult,
    is_cheap_valuation,
    passes_initial_screening,
)
from screener.sector import SectorStats, compare_to_sector, sector_averages

STATUS_PASSED = "lolos"
STATUS_FAILED = "tidak lolos"
STATUS_DATA_MISSING = "data kurang"  # tidak bisa dinilai, bukan gagal
# Kondisi tolak mutlak e-book (saldo laba/ekuitas/laba bersih/PER/PBV negatif):
# gugur tanpa toleransi, dibedakan dari gagal threshold biasa supaya alasannya
# terbaca jelas di hasil screening.
STATUS_REJECTED = "ditolak mutlak"

FetchFreeFloat = Callable[[str], float | None]
FetchFundamental = Callable[[str], dict]
ProgressCallback = Callable[[int, int, str], None]


@dataclass
class UniverseCandidate:
    stock_code: str
    stock_name: str
    daily_transaction_value: float
    free_float_pct: float | None = None
    roe_annualized_pct: float | None = None
    per: float | None = None
    pbv: float | None = None
    sector: str | None = None
    status: str = STATUS_DATA_MISSING
    reasons: list[str] | None = None
    notes: list[str] | None = None

    @property
    def ticker(self) -> str:
        return f"{self.stock_code}.JK"


def filter_liquid_stocks(
    summary: pd.DataFrame, min_value: float = MIN_DAILY_TRANSACTION_VALUE
) -> list[UniverseCandidate]:
    """Tahap 1: saring Ringkasan Saham berdasarkan nilai transaksi pasar reguler.

    Hanya satu hari bursa (yang tersedia di endpoint), jadi emiten yang
    kebetulan sepi/ramai di hari itu bisa salah masuk atau salah keluar.
    """
    if summary.empty or "StockCode" not in summary or "Value" not in summary:
        return []
    liquid = summary[summary["Value"] >= min_value].sort_values("Value", ascending=False)
    return [
        UniverseCandidate(
            stock_code=str(row["StockCode"]),
            stock_name=str(row.get("StockName", "")),
            daily_transaction_value=float(row["Value"]),
        )
        for _, row in liquid.iterrows()
    ]


def _fail(candidate: UniverseCandidate, reason: str) -> None:
    candidate.status = STATUS_FAILED
    candidate.reasons = [reason]


def _reject(candidate: UniverseCandidate, reasons: list[str]) -> None:
    candidate.status = STATUS_REJECTED
    candidate.reasons = reasons


def _missing(candidate: UniverseCandidate, reason: str) -> None:
    candidate.status = STATUS_DATA_MISSING
    candidate.reasons = [reason]


def _apply_result(candidate: UniverseCandidate, result: ScreeningResult) -> None:
    candidate.status = STATUS_PASSED if result.passed else STATUS_FAILED
    candidate.reasons = result.reasons
    candidate.notes = result.notes


def _ratio_text(value: float | None, decimals: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{decimals}f}"


def _apply_cheap_valuation(candidate: UniverseCandidate, is_blue_chip: bool) -> None:
    """Tahap 5: aturan valuasi murah (bagian C).

    Blue chip memakai batas PER 12, sisanya batas small cap PER 8. Daftar blue
    chip ditentukan pemilik secara manual (keputusan 2026-09-23) - bukan
    ditebak dari market cap, karena buku tidak memberi ambangnya.

    Dipanggil hanya untuk kandidat yang sudah lolos bagian A, jadi status
    awalnya STATUS_PASSED dan hanya perlu diubah kalau gugur.
    """
    per, pbv = candidate.per, candidate.pbv
    if per is None and pbv is None:
        _missing(candidate, "PER dan PBV tidak didapat dari yfinance")
        return

    max_per = BLUE_CHIP_MAX_PER if is_blue_chip else SMALL_CAP_MAX_PER
    if not is_cheap_valuation(per=per, pbv=pbv, is_blue_chip=is_blue_chip):
        _fail(
            candidate,
            f"Valuasi tidak murah: PER {_ratio_text(per)} > {max_per} "
            f"dan PBV {_ratio_text(pbv, 2)} > {MAX_PBV_CHEAP}",
        )
        # Emiten di luar daftar blue chip dinilai dengan batas small cap; bila
        # PER-nya masih di bawah batas blue chip, itu layak dicek ulang manual.
        if not is_blue_chip and per is not None and SMALL_CAP_MAX_PER < per <= BLUE_CHIP_MAX_PER:
            candidate.notes = (candidate.notes or []) + [
                f"PER {per:.1f} masih <= {BLUE_CHIP_MAX_PER} (batas blue chip); "
                "bila emiten ini blue chip, tambahkan ke daftar blue chip"
            ]


def screen_candidate(
    candidate: UniverseCandidate,
    fetch_free_float: FetchFreeFloat,
    fetch_fundamental: FetchFundamental,
    blacklist: set[str] = frozenset(),
    blue_chips: set[str] = frozenset(),
) -> UniverseCandidate:
    """Jalankan tahap 2-5 untuk satu emiten; berhenti di tahap pertama yang gugur.

    Pesan alasan di tahap awal ditulis di sini (bukan lewat
    `passes_initial_screening`) karena fungsi itu butuh semua angka sekaligus,
    sedangkan corong sengaja tidak mengambil angka yang tidak diperlukan lagi.
    """
    if candidate.stock_code in blacklist:
        _fail(candidate, "Ticker ada di blacklist manual")
        return candidate

    free_float_pct = fetch_free_float(candidate.stock_code)
    candidate.free_float_pct = free_float_pct
    if free_float_pct is None:
        _missing(candidate, "Free float tidak didapat dari IDX")
        return candidate
    if free_float_pct < MIN_FREE_FLOAT_PCT:
        _fail(candidate, f"Free float {free_float_pct:.1f}% < {MIN_FREE_FLOAT_PCT:.1f}%")
        return candidate

    fundamental = fetch_fundamental(candidate.ticker) or {}
    candidate.roe_annualized_pct = fundamental.get("roe_annualized_pct")
    candidate.per = fundamental.get("per")
    candidate.pbv = fundamental.get("pbv")
    candidate.sector = fundamental.get("sector")

    # Tolak mutlak diperiksa sebelum threshold: kalau ekuitas/laba/valuasi
    # negatif, angka rasio lain tidak perlu dinilai lagi.
    # Laba TTM, bukan satu kuartal (keputusan pemilik 2026-09-23): rugi satu
    # kuartal musiman bukan maksud "laba bersih negatif" di buku.
    rejects = absolute_rejects(
        retained_earnings=fundamental.get("retained_earnings"),
        equity=fundamental.get("equity"),
        net_income=fundamental.get("net_income_ttm"),
        per=candidate.per,
        pbv=candidate.pbv,
    )
    if rejects:
        _reject(candidate, rejects)
        return candidate

    if candidate.roe_annualized_pct is None:
        _missing(candidate, "ROE disetahunkan tidak didapat dari yfinance")
        return candidate

    _apply_result(
        candidate,
        passes_initial_screening(
            ticker=candidate.ticker,
            daily_transaction_value=candidate.daily_transaction_value,
            free_float_pct=free_float_pct,
            roe_annualized_pct=candidate.roe_annualized_pct,
        ),
    )
    if candidate.status == STATUS_PASSED:
        _apply_cheap_valuation(candidate, is_blue_chip=candidate.stock_code in blue_chips)

    # Catatan kualitas data dari fetcher (mis. PBV dikosongkan karena laporan
    # keuangan dalam USD) ikut ditampilkan supaya sel kosong bisa dijelaskan.
    data_notes = fundamental.get("data_notes") or []
    if data_notes:
        candidate.notes = (candidate.notes or []) + list(data_notes)
    return candidate


def screen_universe(
    candidates: list[UniverseCandidate],
    fetch_free_float: FetchFreeFloat,
    fetch_fundamental: FetchFundamental,
    blacklist: set[str] = frozenset(),
    blue_chips: set[str] = frozenset(),
    on_progress: ProgressCallback | None = None,
) -> list[UniverseCandidate]:
    """Jalankan `screen_candidate` ke daftar hasil `filter_liquid_stocks`.

    `on_progress(selesai, total, kode)` dipanggil setelah tiap emiten supaya
    UI bisa menampilkan progress; proses ini bisa memakan beberapa menit.
    """
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        screen_candidate(
            candidate, fetch_free_float, fetch_fundamental, blacklist, blue_chips
        )
        if on_progress is not None:
            on_progress(index, total, candidate.stock_code)
    return candidates


def apply_sector_comparison(candidates: list[UniverseCandidate]) -> dict[str, SectorStats]:
    """Bandingkan PER/PBV tiap kandidat dengan median sektornya (jalur OPSIONAL).

    Default proyek tetap jalur harga absolut/story; ini pembanding tambahan
    dari e-book lama, ditulis sebagai catatan - tidak mengubah status lolos.
    """
    stats = sector_averages(
        [{"sector": c.sector, "per": c.per, "pbv": c.pbv} for c in candidates]
    )
    for candidate in candidates:
        sector_stats = stats.get(candidate.sector or "")
        if sector_stats is None:
            continue
        notes = []
        per_verdict = compare_to_sector(candidate.per, sector_stats.per_median)
        pbv_verdict = compare_to_sector(candidate.pbv, sector_stats.pbv_median)
        if per_verdict:
            notes.append(f"PER {per_verdict} ({sector_stats.per_median:.1f}x)")
        if pbv_verdict:
            notes.append(f"PBV {pbv_verdict} ({sector_stats.pbv_median:.2f}x)")
        if notes:
            candidate.notes = (candidate.notes or []) + notes
    return stats


def candidates_to_dataframe(candidates: list[UniverseCandidate]) -> pd.DataFrame:
    """Tabel ringkas untuk ditampilkan di UI; None tampil sebagai sel kosong."""
    return pd.DataFrame(
        [
            {
                "Kode": c.stock_code,
                "Nama": c.stock_name,
                "Nilai transaksi": format_rupiah(c.daily_transaction_value),
                "Free float (%)": c.free_float_pct,
                "ROE disetahunkan (%)": c.roe_annualized_pct,
                "PER": c.per,
                "PBV": c.pbv,
                "Sektor": c.sector,
                "Status": c.status,
                "Alasan / catatan": "; ".join((c.reasons or []) + (c.notes or [])),
            }
            for c in candidates
        ]
    )
