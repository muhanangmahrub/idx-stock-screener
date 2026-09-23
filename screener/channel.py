"""Channeling: koridor dua garis sejajar di sekitar tren, mengacu pada Edianto Ong.

Aturan buku (diberikan pemilik 2026-09-22):
- Harga kadang bergerak rapi dalam koridor tertentu (channeling): dua garis
  paralel seperti "saluran"/"terowongan".
- Uptrend channeling: garis BAWAH = basic trendline, garis ATAS = channel
  line. Downtrend channeling: garis ATAS = basic trendline, garis BAWAH =
  channel line.
- Cara menggambar: tentukan basic trendline dulu, lalu gambar garis yang
  merupakan proyeksi SEJAJAR dari basic trendline, tempat harga bergerak di
  dalam channel yang terbentuk.
- Arti penembusan pada uptrend channeling: basic trendline tertembus =
  kemungkinan awal perubahan tren yang sedang berlangsung (sinyal bearish);
  channel line tertembus = akselerasi tren yang sedang berlangsung (sinyal
  bullish). Downtrend channeling berlaku cermin.

Terjemahan ke kode: basic trendline = `trend.Trendline` yang sudah ada
(up-trendline dari Low lembah, down-trendline dari High puncak). Channel line
memakai SLOPE yang sama (itulah arti "proyeksi sejajar"), hanya intercept-nya
digeser sampai menyentuh sisi seberang.

ASUMSI (bukan buku, wajib dikalibrasi):
- Titik acuan channel line = swing di sisi seberang yang PALING JAUH dari
  basic trendline dalam rentang channel (uptrend: swing high tertinggi
  relatif garis; downtrend: swing low terendah). Buku tidak merinci swing
  mana yang dipakai bila ada beberapa.
- Rentang channel = dari titik acuan pertama basic trendline sampai bar
  terkini.
- Penembusan dinilai dengan aturan yang sama seperti trendline (Close di
  luar garis ± toleransi, lihat trend.check_trendline_break); di sini
  dilaporkan sebagai status kondisi - bukan sinyal beli/jual.
- Buku tidak memberi angka "cukup rapi untuk disebut channeling", tapi
  channel line tetap sebuah GARIS: butuh minimal 2 titik sentuh supaya
  koridornya terbukti, sama seperti trendline yang butuh 2 titik acuan.
  `min_touches` (default 2) menjaga agar channel tidak dipaksakan pada chart
  yang harganya tidak benar-benar bergerak dalam saluran; `touches`
  dilaporkan supaya kerapian koridor tetap bisa dinilai manual.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.trend import (
    DEFAULT_BREAK_TOLERANCE,
    DOWN_TRENDLINE,
    UP_TRENDLINE,
    SwingPoint,
    Trendline,
    break_threshold,
)

# Arti penembusan menurut buku.
BASIC_BREAK_TREND_CHANGE = "basic trendline tertembus: kemungkinan awal perubahan tren"
CHANNEL_BREAK_ACCELERATION = "channel line tertembus: akselerasi tren"
CHANNEL_INTACT = "harga masih di dalam channel"

BIAS_BEARISH = "bearish"
BIAS_BULLISH = "bullish"

# ASUMSI (bukan buku): seberapa dekat sebuah swing dihitung "menyentuh" channel line.
DEFAULT_TOUCH_TOLERANCE = 0.02
# ASUMSI (bukan angka buku): channel line butuh >= 2 titik sentuh supaya
# koridornya terbukti - sebuah garis butuh dua titik.
DEFAULT_MIN_TOUCHES = 2


@dataclass
class Channel:
    """Koridor dua garis sejajar: basic trendline + channel line."""

    basic: Trendline
    channel_intercept: float  # slope-nya sama dengan basic (proyeksi sejajar)
    anchor: SwingPoint  # swing sisi seberang yang dilewati channel line
    touches: int  # berapa swing sisi seberang menyentuh channel line (kerapian koridor)

    @property
    def slope(self) -> float:
        return self.basic.slope

    @property
    def is_uptrend(self) -> bool:
        return self.basic.kind == UP_TRENDLINE

    @property
    def basic_position(self) -> str:
        """Posisi basic trendline dalam koridor (buku: bawah saat uptrend)."""
        return "bawah" if self.is_uptrend else "atas"

    def channel_value_at(self, index: int) -> float:
        return self.slope * index + self.channel_intercept

    def basic_value_at(self, index: int) -> float:
        return self.basic.value_at(index)

    def width_at(self, index: int) -> float:
        """Lebar koridor di satu bar; konstan karena kedua garis sejajar."""
        return abs(self.channel_value_at(index) - self.basic_value_at(index))


def build_channel(
    trendline: Trendline,
    highs: pd.Series,
    lows: pd.Series,
    highs_idx: np.ndarray,
    lows_idx: np.ndarray,
    touch_tol: float = DEFAULT_TOUCH_TOLERANCE,
    min_touches: int = DEFAULT_MIN_TOUCHES,
) -> Channel | None:
    """Proyeksi sejajar dari `trendline` ke sisi seberang koridor.

    Uptrend: channel line di ATAS, ditarik lewat swing high yang paling jauh
    di atas basic trendline. Downtrend: channel line di BAWAH, lewat swing
    low yang paling jauh di bawahnya.

    None bila tidak ada swing di sisi seberang, atau bila channel line hanya
    disentuh kurang dari `min_touches` swing - artinya harga tidak terbukti
    bergerak dalam saluran, jadi channel tidak dipaksakan.
    """
    if trendline.kind == UP_TRENDLINE:
        opposite_idx, opposite_prices = highs_idx, highs.to_numpy()
    else:
        opposite_idx, opposite_prices = lows_idx, lows.to_numpy()

    start, end = trendline.start_index, trendline.end_index
    candidates = [int(i) for i in opposite_idx if start <= i <= end]
    if not candidates:
        return None

    # Jarak bertanda dari basic trendline; diambil yang paling jauh ke sisi
    # seberang supaya seluruh pergerakan harga terkurung di dalam koridor.
    def distance(i: int) -> float:
        gap = opposite_prices[i] - trendline.value_at(i)
        return gap if trendline.kind == UP_TRENDLINE else -gap

    anchor_index = max(candidates, key=distance)
    anchor = SwingPoint(anchor_index, float(opposite_prices[anchor_index]))
    channel_intercept = anchor.price - trendline.slope * anchor_index

    channel = Channel(
        basic=trendline, channel_intercept=channel_intercept, anchor=anchor, touches=0
    )
    channel.touches = sum(
        abs(opposite_prices[i] - channel.channel_value_at(i))
        <= abs(channel.channel_value_at(i)) * touch_tol
        for i in candidates
    )
    if channel.touches < min_touches:
        return None
    return channel


@dataclass
class ChannelBreak:
    status: str  # CHANNEL_INTACT | BASIC_BREAK_TREND_CHANGE | CHANNEL_BREAK_ACCELERATION
    bias: str | None  # BIAS_BEARISH | BIAS_BULLISH saat ada penembusan
    index: int | None  # bar penembusan
    price: float | None  # Close di bar itu


def check_channel_break(
    channel: Channel,
    closes: pd.Series,
    tol: float = DEFAULT_BREAK_TOLERANCE,
    start_index: int | None = None,
    end_index: int | None = None,
) -> ChannelBreak:
    """Cari penembusan pertama keluar koridor, memakai aturan Close ± toleransi.

    Uptrend: Close di bawah basic trendline = sinyal bearish (awal perubahan
    tren); Close di atas channel line = sinyal bullish (akselerasi).
    Downtrend cermin: Close di atas basic = bullish, Close di bawah channel
    line = bearish (akselerasi tren turun). Diperiksa sejak bar setelah titik
    acuan terakhir basic trendline (alasan sama seperti
    `trend.check_trendline_break`).
    """
    close_values = closes.to_numpy()
    if start_index is None:
        start_index = channel.basic.points[-1].index + 1
    if end_index is None:
        end_index = min(channel.basic.end_index, len(close_values) - 1)

    uptrend = channel.is_uptrend
    # Kedua sisi koridor + arah "keluar"-nya, sesuai jenis garis.
    basic_kind = channel.basic.kind
    channel_kind = DOWN_TRENDLINE if uptrend else UP_TRENDLINE

    for i in range(start_index, end_index + 1):
        close = float(close_values[i])
        basic_boundary = break_threshold(channel.basic_value_at(i), basic_kind, tol)
        channel_boundary = break_threshold(channel.channel_value_at(i), channel_kind, tol)

        basic_broken = close < basic_boundary if uptrend else close > basic_boundary
        channel_broken = close > channel_boundary if uptrend else close < channel_boundary

        if basic_broken:
            return ChannelBreak(
                BASIC_BREAK_TREND_CHANGE,
                BIAS_BEARISH if uptrend else BIAS_BULLISH,
                i,
                close,
            )
        if channel_broken:
            return ChannelBreak(
                CHANNEL_BREAK_ACCELERATION,
                BIAS_BULLISH if uptrend else BIAS_BEARISH,
                i,
                close,
            )

    return ChannelBreak(CHANNEL_INTACT, None, None, None)
