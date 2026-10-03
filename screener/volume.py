"""Volume perdagangan, mengacu pada Edianto Ong.

Aturan buku (diberikan pemilik 2026-10-01):
- Volume dicatat dalam bentuk batang (volume bar), biasanya ditampilkan di
  bagian bawah chart harga.
- Volume bar yang TINGGI melambangkan jumlah perdagangan yang besar dari
  para pelaku pasar.
- Volume bar yang PENDEK mencerminkan aktivitas minim atau minat yang kurang
  dari pelaku pasar.

Yang BELUM diberikan buku, jadi tidak dikarang di sini: berapa besar sebuah
bar baru disebut "tinggi", dan bagaimana volume dipakai untuk mengonfirmasi
pola/breakout. Modul ini hanya menyediakan ukuran deskriptif - rata-rata
volume dan volume relatif terhadap rata-rata - supaya "tinggi/pendek" bisa
dibaca sebagai angka, bukan kesan. Ambangnya parameter dengan default yang
ditandai ASUMSI dan wajib dikalibrasi ke contoh chart di buku.

Catatan data IDX (dari CLAUDE.md, bukan dari buku): bar dengan volume 0
muncul pada sesi pre-opening; bar seperti itu bukan "minat kurang" melainkan
ketiadaan perdagangan, jadi dipisahkan lewat `is_tradeless`.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

# ASUMSI (bukan angka buku): batas relatif terhadap rata-rata untuk menyebut
# sebuah bar tinggi atau pendek. Kalibrasi ke contoh chart di buku.
DEFAULT_HIGH_RATIO = 1.5
DEFAULT_LOW_RATIO = 0.5
# ASUMSI (bukan angka buku): jendela rata-rata volume.
DEFAULT_WINDOW = 20

VOLUME_HIGH = "tinggi"
VOLUME_NORMAL = "normal"
VOLUME_LOW = "pendek"
VOLUME_NO_TRADE = "tanpa perdagangan"


@dataclass
class VolumeProfile:
    """Ringkasan volume satu bar terhadap kebiasaannya sendiri."""

    index: int
    volume: float
    average: float | None  # rata-rata volume jendela sebelum bar ini
    ratio: float | None  # volume / rata-rata
    label: str  # VOLUME_HIGH | VOLUME_NORMAL | VOLUME_LOW | VOLUME_NO_TRADE

    @property
    def is_high(self) -> bool:
        return self.label == VOLUME_HIGH

    @property
    def is_low(self) -> bool:
        return self.label == VOLUME_LOW


def is_tradeless(volume: float | None) -> bool:
    """Bar tanpa perdagangan sama sekali (mis. pre-opening di data intraday).

    Dibedakan dari volume kecil: nol berarti tidak ada transaksi, bukan minat
    yang rendah.
    """
    return volume is None or float(volume) <= 0


def average_volume(
    volumes: pd.Series, window: int = DEFAULT_WINDOW, exclude_current: bool = True
) -> pd.Series:
    """Rata-rata volume bergerak.

    `exclude_current=True` (default) membuat rata-rata hanya memakai bar-bar
    SEBELUM bar itu, sehingga sebuah bar dibandingkan dengan kebiasaan yang
    sudah terbentuk - bukan dengan rata-rata yang ikut terangkat oleh bar itu
    sendiri. Bar tanpa perdagangan tidak ikut menurunkan rata-rata.
    """
    if window < 1:
        raise ValueError("window minimal 1")
    counted = volumes.where(volumes > 0)
    shifted = counted.shift(1) if exclude_current else counted
    return shifted.rolling(window, min_periods=1).mean()


def relative_volume(
    volumes: pd.Series, window: int = DEFAULT_WINDOW, exclude_current: bool = True
) -> pd.Series:
    """Volume dibagi rata-ratanya: 1,0 berarti sebesar kebiasaan."""
    averages = average_volume(volumes, window=window, exclude_current=exclude_current)
    return volumes / averages.replace(0, np.nan)


def classify_volume(
    volume: float | None,
    average: float | None,
    high_ratio: float = DEFAULT_HIGH_RATIO,
    low_ratio: float = DEFAULT_LOW_RATIO,
) -> str:
    """Label satu bar: tinggi, pendek, atau normal - relatif terhadap rata-rata."""
    if high_ratio <= low_ratio:
        raise ValueError("high_ratio harus lebih besar daripada low_ratio")
    if is_tradeless(volume):
        return VOLUME_NO_TRADE
    if not average or average <= 0 or pd.isna(average):
        return VOLUME_NORMAL  # belum ada pembanding; jangan menebak
    ratio = volume / average
    if ratio >= high_ratio:
        return VOLUME_HIGH
    if ratio <= low_ratio:
        return VOLUME_LOW
    return VOLUME_NORMAL


def volume_profiles(
    df: pd.DataFrame,
    window: int = DEFAULT_WINDOW,
    high_ratio: float = DEFAULT_HIGH_RATIO,
    low_ratio: float = DEFAULT_LOW_RATIO,
) -> list[VolumeProfile]:
    """Profil volume tiap bar pada satu DataFrame OHLCV."""
    if "Volume" not in df.columns:
        return []
    volumes = df["Volume"]
    averages = average_volume(volumes, window=window)
    profiles = []
    for position in range(len(df)):
        volume = float(volumes.iloc[position])
        average = averages.iloc[position]
        average = None if pd.isna(average) else float(average)
        ratio = volume / average if average else None
        profiles.append(
            VolumeProfile(
                index=position,
                volume=volume,
                average=average,
                ratio=ratio,
                label=classify_volume(volume, average, high_ratio, low_ratio),
            )
        )
    return profiles


def drop_tradeless_bars(df: pd.DataFrame) -> pd.DataFrame:
    """Buang bar tanpa perdagangan (volume 0).

    Perlu untuk data intraday IDX: bar 09:00 sering bervolume 0 karena masih
    pre-opening, dan bar semacam itu mengacaukan deteksi pola maupun
    rata-rata volume.
    """
    if "Volume" not in df.columns:
        return df
    return df[df["Volume"] > 0]
