"""Klasifikasi tren (uptrend / downtrend / sideways), mengacu pada Edianto Ong.

Aturan dari buku (CLAUDE.md "Spesifikasi metode Edianto Ong", diberikan
pemilik 2026-09-20):
- Uptrend: puncak maupun dasar yang terbentuk semakin lama semakin tinggi.
- Downtrend: puncak dan dasar yang terbentuk semakin lama semakin rendah.
- Sideways: puncak ke puncak dan dasar ke dasar (hampir) sama.

Diterjemahkan literal: ambil beberapa swing high (puncak) dan swing low
(dasar) terakhir dari `get_extrema`, lalu bandingkan tiap pasangan yang
berurutan. Semua pasangan naik -> uptrend, semua turun -> downtrend, semua
(hampir) sama -> sideways. Kombinasi lain (mis. puncak naik tapi dasar turun)
tidak masuk ketiga definisi itu, jadi dilaporkan "tidak jelas" - bukan
dipaksa ke salah satu.

Dua angka di modul ini BUKAN dari buku dan wajib dikalibrasi ke contoh chart
di buku: DEFAULT_TOLERANCE (batas "hampir sama") dan DEFAULT_LOOKBACK_SWINGS
(berapa puncak/dasar terakhir yang dibandingkan).

Trendline (aturan buku, diberikan 2026-09-20):
- Up-trendline: menghubungkan titik harga terendah pada lembah-lembah (dasar)
  yang terbentuk pada chart uptrend; level tempat uptrend diuji.
- Down-trendline: menghubungkan titik harga pada puncak-puncak yang terbentuk
  pada chart downtrend; level tempat downtrend diuji.
Buku tidak merinci garis mana yang dipakai bila 3+ titik tidak segaris.
ASUMSI: garis regresi kuadrat terkecil melalui semua titik (2 titik -> persis
melewati keduanya). Alternatif yang mungkin: garis dari titik pertama ke
terakhir. Kalibrasi ke contoh chart di buku.

Konfirmasi titik acuan (aturan buku, diberikan 2026-09-20):
- Uptrend: dasar A2 baru "siap" dipakai menggaris up-trendline setelah level
  puncak terakhir sebelum A2 (resistance) dilewati harga. Sebagian
  technicalist cukup mensyaratkan 50% jarak vertikal A2 -> resistance
  terlampaui.
- Downtrend: cermin - puncak B2 siap setelah dasar terakhir sebelum B2
  (support) ditembus, atau 50% jaraknya.
- Dipakai harga keseluruhan: High untuk puncak/resistance dan pengecekan
  "dilewati", Low untuk dasar/support dan pengecekan "ditembus".
Kedua mazhab (100% / 50%) diwakili `confirmation_ratio`. ASUMSI (bukan
buku): default memakai yang ketat (100%) sampai pemilik menentukan; titik
acuan pertama dianggap titik awal tren dan tidak perlu konfirmasi, karena
aturan buku berbicara tentang A2 dan seterusnya.

Penembusan trendline (aturan utama buku, diberikan 2026-09-20):
- Valid break: harga PENUTUPAN berada di luar garis (di bawah up-trendline /
  di atas down-trendline). Close jauh lebih signifikan daripada pergerakan
  sementara intraday.
- Tembusan sementara oleh High/Low intraday yang Close-nya kembali ke dalam
  garis = false break / whipsaw, bukan penembusan.
ASUMSI (bukan buku): pengecekan dimulai dari bar setelah titik acuan
terakhir, karena di sekitar titik acuan garis regresi bisa menyilang titik
acuannya sendiri - itu artefak garis, bukan penembusan. Close tepat di garis
belum dihitung di luar.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

UPTREND = "uptrend"
DOWNTREND = "downtrend"
SIDEWAYS = "sideways"
UNDEFINED = "tidak jelas"

STEP_UP = "naik"
STEP_DOWN = "turun"
STEP_FLAT = "sama"

# ASUMSI (bukan angka Edianto Ong): selisih <= 2% dianggap "hampir sama".
DEFAULT_TOLERANCE = 0.02
# ASUMSI (bukan angka Edianto Ong): 3 puncak + 3 dasar terakhir = minimal
# dua perbandingan per sisi, supaya satu swing tidak menentukan tren sendirian.
DEFAULT_LOOKBACK_SWINGS = 3
MIN_SWINGS_PER_SIDE = 2  # butuh minimal satu pasangan untuk dibandingkan


@dataclass
class SwingPoint:
    index: int  # posisi bar pada deret harga (bukan tanggal)
    price: float


@dataclass
class TrendResult:
    trend: str
    peaks: list[SwingPoint] = field(default_factory=list)
    troughs: list[SwingPoint] = field(default_factory=list)
    peak_steps: list[str] = field(default_factory=list)  # naik/turun/sama antar puncak
    trough_steps: list[str] = field(default_factory=list)  # naik/turun/sama antar dasar
    reason: str = ""  # penjelasan singkat untuk UI, terutama saat "tidak jelas"


UP_TRENDLINE = "up-trendline"
DOWN_TRENDLINE = "down-trendline"

# Dua mazhab syarat "titik siap": tembus level penuh vs 50% jarak vertikal.
CONFIRM_FULL_BREAK = 1.0
CONFIRM_HALF_WAY = 0.5
# ASUMSI (bukan pilihan buku): pakai yang ketat sampai pemilik menentukan.
DEFAULT_CONFIRMATION_RATIO = CONFIRM_FULL_BREAK


@dataclass
class AnchorPoint(SwingPoint):
    """Calon titik acuan trendline beserta status konfirmasinya."""

    confirmed: bool = True
    threshold: float | None = None  # level yang harus dilewati; None untuk titik awal
    confirmed_index: int | None = None  # bar pertama yang melewati threshold


@dataclass
class Trendline:
    kind: str  # UP_TRENDLINE | DOWN_TRENDLINE
    slope: float  # perubahan harga per bar
    intercept: float  # nilai garis di bar ke-0
    points: list[AnchorPoint]  # titik acuan terkonfirmasi (Low di dasar / High di puncak)
    end_index: int  # bar terakhir tempat garis dipanjangkan (biasanya bar terkini)
    pending_points: list[AnchorPoint] = field(default_factory=list)  # belum siap dipakai

    @property
    def start_index(self) -> int:
        return self.points[0].index

    def value_at(self, index: int) -> float:
        """Nilai garis pada bar ke-`index`; di luar titik acuan = ekstrapolasi."""
        return self.slope * index + self.intercept


def fit_line(indices: list[int], prices: list[float]) -> tuple[float, float]:
    """Garis lurus y = slope * x + intercept melalui titik-titik (indeks bar, harga).

    Dipakai regresi kuadrat terkecil (polyfit derajat 1): untuk 2 titik hasilnya
    garis yang persis melewati keduanya; untuk 3+ titik yang tidak segaris,
    garis dengan jumlah kuadrat jarak vertikal paling kecil. Indeks bar
    (bukan tanggal) dipakai sebagai sumbu x supaya libur bursa tidak membuat
    kemiringan garis melompat.
    """
    if len(indices) < 2:
        raise ValueError("Trendline butuh minimal 2 titik")
    slope, intercept = np.polyfit(indices, prices, deg=1)
    return float(slope), float(intercept)


def select_anchor_points(
    trend: TrendResult,
    lows: pd.Series,
    highs: pd.Series,
    confirmation_ratio: float = DEFAULT_CONFIRMATION_RATIO,
) -> list[AnchorPoint]:
    """Calon titik acuan trendline (dasar untuk uptrend, puncak untuk downtrend)
    beserta status "siap"-nya menurut aturan konfirmasi di docstring modul.

    Untuk uptrend: resistance sebuah dasar = High tertinggi di antara dasar
    sebelumnya dan dasar ini (bukan hasil deteksi puncak, supaya tidak
    bergantung pada parameter find_peaks). Dasar siap bila ada bar setelahnya
    yang High-nya melewati `Low + ratio * (resistance - Low)`. Downtrend
    cermin: support = Low terendah di antara dua puncak, puncak siap bila ada
    bar setelahnya yang Low-nya menembus `High - ratio * (High - support)`.
    Kosong untuk sideways / tidak jelas.
    """
    if not 0 < confirmation_ratio <= 1:
        raise ValueError("confirmation_ratio harus di antara 0 (eksklusif) dan 1")

    if trend.trend == UPTREND:
        swings, anchor_source, break_source = trend.troughs, lows, highs
    elif trend.trend == DOWNTREND:
        swings, anchor_source, break_source = trend.peaks, highs, lows
    else:
        return []

    anchor_values = anchor_source.to_numpy()
    break_values = break_source.to_numpy()
    anchors: list[AnchorPoint] = []
    for previous, current in zip([None] + swings[:-1], swings):
        price = float(anchor_values[current.index])
        if previous is None:
            # Titik awal tren: tidak ada level sebelumnya yang harus dilewati.
            anchors.append(AnchorPoint(current.index, price))
            continue

        between = break_values[previous.index + 1 : current.index]
        after = break_values[current.index + 1 :]
        if trend.trend == UPTREND:
            resistance = float(between.max())
            threshold = price + confirmation_ratio * (resistance - price)
            crossed = after > threshold
        else:
            support = float(between.min())
            threshold = price - confirmation_ratio * (price - support)
            crossed = after < threshold

        confirmed = bool(crossed.any())
        confirmed_index = current.index + 1 + int(np.argmax(crossed)) if confirmed else None
        anchors.append(
            AnchorPoint(
                current.index,
                price,
                confirmed=confirmed,
                threshold=threshold,
                confirmed_index=confirmed_index,
            )
        )
    return anchors


def build_trendline(
    trend: TrendResult,
    lows: pd.Series,
    highs: pd.Series,
    end_index: int,
    confirmation_ratio: float = DEFAULT_CONFIRMATION_RATIO,
) -> Trendline | None:
    """Gambar trendline sesuai jenis tren; None untuk sideways / tidak jelas
    atau bila titik acuan yang sudah terkonfirmasi kurang dari 2.

    `lows`/`highs` adalah kolom Low/High dari data OHLC yang sama dengan yang
    dipakai `classify_trend`, karena buku menghubungkan harga terendah lembah
    (bukan Close) dan harga puncak. Titik acuan = dasar/puncak yang dipakai
    saat klasifikasi, supaya garis dan vonis tren konsisten; hanya yang lolos
    `select_anchor_points` yang dihubungkan, sisanya dilaporkan di
    `pending_points`.
    """
    anchors = select_anchor_points(trend, lows, highs, confirmation_ratio)
    if not anchors:
        return None
    kind = UP_TRENDLINE if trend.trend == UPTREND else DOWN_TRENDLINE

    points = [a for a in anchors if a.confirmed]
    pending = [a for a in anchors if not a.confirmed]
    if len(points) < 2:
        return None

    slope, intercept = fit_line([p.index for p in points], [p.price for p in points])
    return Trendline(
        kind=kind,
        slope=slope,
        intercept=intercept,
        points=points,
        end_index=end_index,
        pending_points=pending,
    )


def classify_steps(values: list[float], tol: float) -> list[str]:
    """Label tiap pasangan nilai berurutan: naik, turun, atau (hampir) sama.

    Perubahan relatif terhadap nilai sebelumnya dibandingkan dengan `tol`;
    di dalam +-tol dianggap "sama". Toleransi dipakai untuk ketiga label
    supaya batas naik/sama/turun konsisten.
    """
    steps = []
    for previous, current in zip(values, values[1:]):
        change = (current - previous) / previous
        if change > tol:
            steps.append(STEP_UP)
        elif change < -tol:
            steps.append(STEP_DOWN)
        else:
            steps.append(STEP_FLAT)
    return steps


def _trend_from_steps(peak_steps: list[str], trough_steps: list[str]) -> str:
    all_steps = peak_steps + trough_steps
    if all(step == STEP_UP for step in all_steps):
        return UPTREND
    if all(step == STEP_DOWN for step in all_steps):
        return DOWNTREND
    if all(step == STEP_FLAT for step in all_steps):
        return SIDEWAYS
    return UNDEFINED


def classify_trend(
    prices: pd.Series,
    highs_idx: np.ndarray,
    lows_idx: np.ndarray,
    tol: float = DEFAULT_TOLERANCE,
    lookback_swings: int = DEFAULT_LOOKBACK_SWINGS,
) -> TrendResult:
    """Tentukan tren dari `lookback_swings` puncak & dasar terakhir.

    `highs_idx`/`lows_idx` adalah keluaran `get_extrema` pada `prices` yang
    sama. Kalau salah satu sisi punya kurang dari MIN_SWINGS_PER_SIDE swing,
    tren tidak bisa dinilai (bukan sideways).
    """
    if lookback_swings < MIN_SWINGS_PER_SIDE:
        raise ValueError(f"lookback_swings minimal {MIN_SWINGS_PER_SIDE}")

    values = prices.to_numpy()
    peaks = [SwingPoint(int(i), float(values[i])) for i in highs_idx[-lookback_swings:]]
    troughs = [SwingPoint(int(i), float(values[i])) for i in lows_idx[-lookback_swings:]]

    if len(peaks) < MIN_SWINGS_PER_SIDE or len(troughs) < MIN_SWINGS_PER_SIDE:
        return TrendResult(
            trend=UNDEFINED,
            peaks=peaks,
            troughs=troughs,
            reason=(
                f"Butuh minimal {MIN_SWINGS_PER_SIDE} puncak dan {MIN_SWINGS_PER_SIDE} "
                f"dasar; didapat {len(peaks)} puncak, {len(troughs)} dasar. "
                "Coba periode lebih panjang atau jarak swing lebih kecil."
            ),
        )

    peak_steps = classify_steps([p.price for p in peaks], tol)
    trough_steps = classify_steps([t.price for t in troughs], tol)
    trend = _trend_from_steps(peak_steps, trough_steps)

    reason = f"puncak: {', '.join(peak_steps)}; dasar: {', '.join(trough_steps)}"
    if trend == UNDEFINED:
        reason = (
            "Puncak dan dasar tidak seragam naik/turun/sama, jadi tidak masuk "
            f"definisi uptrend, downtrend, maupun sideways ({reason})"
        )

    return TrendResult(
        trend=trend,
        peaks=peaks,
        troughs=troughs,
        peak_steps=peak_steps,
        trough_steps=trough_steps,
        reason=reason,
    )


LINE_INTACT = "belum tembus"
FALSE_BREAK = "false break"
VALID_BREAK = "valid break"


@dataclass
class BreakCheck:
    status: str  # LINE_INTACT | FALSE_BREAK | VALID_BREAK
    checked_from: int  # bar pertama yang diperiksa
    valid_break_index: int | None = None  # bar pertama dengan Close di luar garis
    whipsaw_indices: list[int] = field(default_factory=list)  # tembus intraday saja


def check_trendline_break(
    trendline: Trendline, closes: pd.Series, lows: pd.Series, highs: pd.Series
) -> BreakCheck:
    """Periksa bar demi bar apakah trendline sudah ditembus secara sah.

    Up-trendline: "di luar" = di bawah garis, jadi valid break bila Close <
    garis; Low < garis tapi Close masih >= garis = whipsaw. Down-trendline
    cermin dengan High dan Close > garis. Whipsaw hanya dicatat sampai valid
    break pertama - setelah itu garisnya sudah dianggap tembus.
    """
    close_values = closes.to_numpy()
    start = trendline.points[-1].index + 1
    end = min(trendline.end_index, len(close_values) - 1)

    outside_is_below = trendline.kind == UP_TRENDLINE
    pierce_values = lows.to_numpy() if outside_is_below else highs.to_numpy()

    def is_outside(price: float, line: float) -> bool:
        return price < line if outside_is_below else price > line

    whipsaws: list[int] = []
    for i in range(start, end + 1):
        line = trendline.value_at(i)
        if is_outside(close_values[i], line):
            return BreakCheck(VALID_BREAK, start, valid_break_index=i, whipsaw_indices=whipsaws)
        if is_outside(pierce_values[i], line):
            whipsaws.append(i)

    status = FALSE_BREAK if whipsaws else LINE_INTACT
    return BreakCheck(status, start, whipsaw_indices=whipsaws)
