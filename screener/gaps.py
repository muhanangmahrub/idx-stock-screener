"""Gap (celah kosong) pada chart, mengacu pada Edianto Ong.

Aturan buku (diberikan pemilik 2026-10-01):
- Gap adalah celah kosong di dalam chart yang timbul akibat lonjakan harga
  karena TIDAK ADA transaksi pada level harga tersebut.
- **Gap up**: harga pembukaan sesi berikutnya melonjak sehingga ada
  kesenjangan dengan harga TERTINGGI sesi sebelumnya.
- Bila celah itu **tetap tidak tertutupi** oleh pergerakan harga di sesi baru
  tersebut, barulah ia meninggalkan gap pada chart.
- **Gap down**: pembukaan melonjak turun meninggalkan harga TERENDAH sesi
  sebelumnya.
- Gap bisa dianalisis di berbagai timeframe, tapi yang paling umum adalah
  daily chart.
- **Arti**: gap up menunjukkan dorongan/minat beli yang tinggi, gap down
  menandakan tekanan jual yang kuat. Karena itu gap sering ditandai volume
  transaksi yang meningkat drastis.
- **Common gap** (klasifikasi pertama): gap yang paling sering terjadi dan
  kurang penting. Cirinya: (a) biasanya TIDAK didukung peningkatan volume
  yang signifikan, dan (b) celahnya sering tertutup kembali dalam tempo
  relatif cepat - biasanya kurang dari 1 minggu.

Dua tahap itu dibedakan di sini, persis seperti kalimat bukunya:

1. `opened` - pembukaan melompati High/Low sesi sebelumnya.
2. `remains` - setelah sesi itu selesai, celahnya masih kosong (untuk gap up:
   Low sesi baru tetap di atas High sesi sebelumnya). Kalau pergerakan sesi
   itu turun menutup celahnya, gap tidak tertinggal di chart.

Jenis gap lain (breakaway, runaway, exhaustion) BELUM diberikan pemilik, jadi
gap yang bukan common tidak ditebak jenisnya - statusnya "belum
terklasifikasi". Gap yang masih muda dan belum tertutup juga belum bisa
dinilai: ciri common baru terbukti setelah lewat tenggat satu minggu.

ASUMSI (bukan aturan buku, wajib dikalibrasi):
- Ukuran minimum agar sebuah lompatan disebut gap: default 0 (sekecil apa
  pun dihitung). Beri `min_size_pct` bila buku memberi batas.
- `filled_index` mencatat kapan harga KEMUDIAN kembali masuk ke zona celah.
  Itu pengamatan faktual; ARTI penutupan gap (apakah jadi sinyal) belum
  diberikan buku, jadi tidak disimpulkan di sini.
"""

from dataclasses import dataclass

import pandas as pd

from screener.volume import DEFAULT_HIGH_RATIO, DEFAULT_WINDOW, relative_volume

GAP_UP = "gap up"
GAP_DOWN = "gap down"

# Buku menyebut daily chart paling umum dipakai untuk analisis gap.
PREFERRED_INTERVAL = "1d"

# Arti gap menurut buku.
PRESSURE_BUYING = "dorongan/minat beli tinggi"
PRESSURE_SELLING = "tekanan jual kuat"

# Klasifikasi jenis gap. Baru satu jenis yang diberikan buku.
GAP_COMMON = "common gap"
GAP_UNCLASSIFIED = "belum terklasifikasi"  # menunggu aturan jenis gap lain
GAP_UNDETERMINED = "belum bisa dinilai"  # masih muda & belum tertutup

# Angka buku: common gap biasanya tertutup dalam tempo kurang dari 1 minggu.
COMMON_GAP_MAX_DAYS = 7


@dataclass
class Gap:
    """Satu celah kosong beserta nasibnya.

    `lower`/`upper` adalah lompatan saat PEMBUKAAN (dari High/Low sesi
    sebelumnya ke harga pembukaan). Yang benar-benar tertinggal di chart
    setelah sesi itu selesai ada di `visible_zone` - bisa lebih sempit,
    atau tidak ada sama sekali kalau pergerakan sesi itu menutupnya.
    """

    index: int  # bar yang membukanya (sesi baru)
    kind: str  # GAP_UP | GAP_DOWN
    previous_extreme: float  # High (gap up) / Low (gap down) sesi sebelumnya
    open_price: float
    session_low: float
    session_high: float
    lower: float  # batas bawah lompatan pembukaan
    upper: float  # batas atas lompatan pembukaan
    remains: bool  # celah masih kosong setelah sesi itu selesai
    filled_index: int | None = None  # bar saat harga kembali masuk zona celah
    volume_ratio: float | None = None  # volume bar gap / rata-ratanya
    days_to_fill: float | None = None  # lama sampai tertutup (hari kalender)
    days_open: float | None = None  # umur celah bila belum tertutup
    kind_label: str = GAP_UNCLASSIFIED  # GAP_COMMON | GAP_UNCLASSIFIED | GAP_UNDETERMINED

    @property
    def size(self) -> float:
        return self.upper - self.lower

    @property
    def size_pct(self) -> float:
        """Lebar lompatan relatif terhadap harga acuan sesi sebelumnya."""
        return 0.0 if not self.previous_extreme else self.size / self.previous_extreme * 100

    @property
    def is_up(self) -> bool:
        return self.kind == GAP_UP

    @property
    def is_filled(self) -> bool:
        return self.filled_index is not None

    @property
    def pressure(self) -> str:
        """Arti gap menurut buku: minat beli tinggi vs tekanan jual kuat."""
        return PRESSURE_BUYING if self.is_up else PRESSURE_SELLING

    @property
    def has_volume_surge(self) -> bool | None:
        """Apakah gap ini disertai lonjakan volume; None bila tidak diketahui."""
        if self.volume_ratio is None:
            return None
        return self.volume_ratio >= DEFAULT_HIGH_RATIO

    @property
    def is_common(self) -> bool:
        return self.kind_label == GAP_COMMON

    @property
    def visible_zone(self) -> tuple[float, float] | None:
        """Celah yang benar-benar tersisa di chart, atau None bila tertutup."""
        if not self.remains:
            return None
        if self.is_up:
            return (self.previous_extreme, self.session_low)
        return (self.session_high, self.previous_extreme)


def classify_gap(
    gap: Gap,
    high_volume_ratio: float = DEFAULT_HIGH_RATIO,
    max_common_days: int = COMMON_GAP_MAX_DAYS,
) -> str:
    """Tentukan jenis gap sejauh aturan yang sudah diberikan buku.

    Common gap: tidak didukung peningkatan volume yang signifikan DAN
    celahnya tertutup kembali dalam tempo kurang dari `max_common_days`
    (buku: 1 minggu).

    Gap yang volumenya melonjak bukan common - tapi jenisnya tidak ditebak,
    karena breakaway/runaway/exhaustion belum diberikan. Gap yang belum
    tertutup dan umurnya belum melewati tenggat juga belum bisa dinilai:
    ia masih mungkin menjadi common bila segera tertutup.
    """
    if gap.volume_ratio is not None and gap.volume_ratio >= high_volume_ratio:
        return GAP_UNCLASSIFIED  # ada lonjakan volume -> bukan common gap

    if gap.days_to_fill is not None:
        return GAP_COMMON if gap.days_to_fill < max_common_days else GAP_UNCLASSIFIED

    # Belum tertutup: masih mungkin jadi common selama tenggat belum lewat.
    if gap.days_open is not None and gap.days_open < max_common_days:
        return GAP_UNDETERMINED
    return GAP_UNCLASSIFIED


def _fill_index(df: pd.DataFrame, gap: Gap) -> int | None:
    """Bar pertama SETELAH gap yang harganya kembali masuk ke zona celah.

    Gap up tertutup bila ada Low yang turun sampai batas bawah zona (High
    sesi sebelum gap); gap down cermin dengan High.
    """
    lows = df["Low"].to_numpy()
    highs = df["High"].to_numpy()
    for i in range(gap.index + 1, len(df)):
        if gap.is_up and lows[i] <= gap.lower:
            return i
        if not gap.is_up and highs[i] >= gap.upper:
            return i
    return None


def detect_gaps(
    df: pd.DataFrame,
    min_size_pct: float = 0.0,
    only_remaining: bool = False,
    high_volume_ratio: float = DEFAULT_HIGH_RATIO,
    max_common_days: int = COMMON_GAP_MAX_DAYS,
    volume_window: int = DEFAULT_WINDOW,
) -> list[Gap]:
    """Cari gap up & gap down pada satu DataFrame OHLC(V).

    `only_remaining=True` hanya mengembalikan celah yang benar-benar
    tertinggal di chart (tidak tertutup oleh pergerakan sesi pembukanya) -
    itulah yang disebut buku "meninggalkan gap pada chart".

    Bila ada kolom Volume, tiap gap juga dibandingkan dengan kebiasaan
    volumenya dan diklasifikasikan lewat `classify_gap`.
    """
    if min_size_pct < 0:
        raise ValueError("min_size_pct tidak boleh negatif")
    if df.empty or not {"Open", "High", "Low"} <= set(df.columns):
        return []

    volume_ratios = (
        relative_volume(df["Volume"], window=volume_window)
        if "Volume" in df.columns
        else None
    )

    opens = df["Open"].to_numpy()
    highs = df["High"].to_numpy()
    lows = df["Low"].to_numpy()

    gaps: list[Gap] = []
    for i in range(1, len(df)):
        previous_high, previous_low = float(highs[i - 1]), float(lows[i - 1])
        open_price = float(opens[i])

        session_low, session_high = float(lows[i]), float(highs[i])

        if open_price > previous_high:
            # Lompatan pembukaan: dari High sesi lalu ke harga pembukaan.
            # Celah hanya tertinggal bila sesi ini tidak turun menutupinya.
            gap = Gap(
                index=i, kind=GAP_UP, previous_extreme=previous_high,
                open_price=open_price, session_low=session_low, session_high=session_high,
                lower=previous_high, upper=open_price,
                remains=session_low > previous_high,
            )
        elif open_price < previous_low:
            gap = Gap(
                index=i, kind=GAP_DOWN, previous_extreme=previous_low,
                open_price=open_price, session_low=session_low, session_high=session_high,
                lower=open_price, upper=previous_low,
                remains=session_high < previous_low,
            )
        else:
            continue
        if min_size_pct and gap.size_pct < min_size_pct:
            continue
        if only_remaining and not gap.remains:
            continue
        gap.filled_index = _fill_index(df, gap)
        gap.volume_ratio = None if volume_ratios is None else _ratio_at(volume_ratios, i)
        gap.days_to_fill, gap.days_open = _gap_timing(df, gap)
        gap.kind_label = classify_gap(
            gap, high_volume_ratio=high_volume_ratio, max_common_days=max_common_days
        )
        gaps.append(gap)

    return gaps


def _ratio_at(ratios: pd.Series, position: int) -> float | None:
    value = ratios.iloc[position]
    return None if pd.isna(value) else float(value)


def _gap_timing(df: pd.DataFrame, gap: Gap) -> tuple[float | None, float | None]:
    """(hari sampai tertutup, umur celah bila belum tertutup).

    Dihitung dalam hari kalender dari tanggal indeks, bukan jumlah bar,
    supaya tenggat "kurang dari 1 minggu" berlaku sama untuk interval apa
    pun. Tanpa indeks tanggal, keduanya None.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        return None, None
    opened_at = df.index[gap.index]
    if gap.filled_index is not None:
        return (df.index[gap.filled_index] - opened_at).days, None
    return None, (df.index[-1] - opened_at).days


def is_preferred_interval(interval: str) -> bool:
    """Buku: gap paling umum dianalisis pada daily chart."""
    return interval == PREFERRED_INTERVAL
