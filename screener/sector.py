"""Rata-rata PER/PBV per sektor - jalur valuasi OPSIONAL (e-book lama).

Ketegangan sumber yang disengaja (lihat CLAUDE.md): e-book "Metode Analisis
Fundamental" menyarankan membandingkan PER/PBV sebuah saham dengan rata-rata
sektor sejenis, sedangkan rekap Investment Planning yang lebih baru menyatakan
tidak perlu membandingkan antar-saham karena tiap perusahaan punya "story"
sendiri. Kemungkinan evolusi pemikiran; default proyek mengikuti yang lebih
baru (jalur harga absolut di `fundamental.compute_valuation`). Modul ini
disediakan sebagai pembanding tambahan, bukan penentu.

Median dipakai, bukan rata-rata aritmetik, karena satu emiten dengan PER
ekstrem (mis. laba nyaris nol) bisa menarik rata-rata sektor terlalu jauh.
Rata-rata aritmetik tetap dilaporkan untuk pembanding.
"""

from dataclasses import dataclass
from statistics import fmean, median

# Rasio negatif berarti laba/ekuitas negatif - itu kondisi tolak mutlak,
# bukan valuasi murah, jadi tidak boleh ikut menurunkan rata-rata sektor.
MIN_VALID_RATIO = 0.0

CHEAPER_THAN_SECTOR = "lebih murah daripada median sektor"
PRICIER_THAN_SECTOR = "lebih mahal daripada median sektor"
SAME_AS_SECTOR = "setara median sektor"


@dataclass
class SectorStats:
    sector: str
    count: int  # jumlah emiten yang rasionya terpakai
    per_median: float | None = None
    per_mean: float | None = None
    pbv_median: float | None = None
    pbv_mean: float | None = None


def _valid(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None and v > MIN_VALID_RATIO]


def sector_averages(rows: list[dict]) -> dict[str, SectorStats]:
    """Median & rata-rata PER/PBV per sektor dari daftar {sector, per, pbv}.

    Emiten tanpa sektor dilewati; rasio negatif/kosong tidak ikut dihitung.
    Sektor yang tidak punya satu pun rasio sah tetap dilaporkan dengan
    nilai None supaya terlihat bahwa sektornya ada tetapi datanya kosong.
    """
    by_sector: dict[str, list[dict]] = {}
    for row in rows:
        sector = row.get("sector")
        if sector:
            by_sector.setdefault(sector, []).append(row)

    stats = {}
    for sector, members in by_sector.items():
        pers = _valid([m.get("per") for m in members])
        pbvs = _valid([m.get("pbv") for m in members])
        stats[sector] = SectorStats(
            sector=sector,
            count=max(len(pers), len(pbvs)),
            per_median=median(pers) if pers else None,
            per_mean=fmean(pers) if pers else None,
            pbv_median=median(pbvs) if pbvs else None,
            pbv_mean=fmean(pbvs) if pbvs else None,
        )
    return stats


def compare_to_sector(
    value: float | None, sector_value: float | None, tolerance: float = 0.05
) -> str | None:
    """Bandingkan satu rasio dengan angka sektornya.

    `tolerance` (default 5%, ASUMSI - bukan angka dokumen) adalah lebar zona
    "setara" supaya selisih receh tidak disebut lebih murah/mahal. None bila
    salah satu angkanya tidak ada.
    """
    if value is None or not sector_value:
        return None
    ratio = value / sector_value
    if ratio < 1 - tolerance:
        return CHEAPER_THAN_SECTOR
    if ratio > 1 + tolerance:
        return PRICIER_THAN_SECTOR
    return SAME_AS_SECTOR
