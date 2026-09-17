"""Deteksi swing high/low - fondasi untuk semua detektor pola teknikal."""

import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def get_extrema(
    prices: pd.Series,
    distance: int = 5,
    prominence: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Cari indeks swing high dan swing low pada deret harga.

    `distance` dan `prominence` diteruskan langsung ke `find_peaks` supaya
    sensitivitas deteksi bisa dikalibrasi terhadap contoh nyata di buku,
    bukan di-hardcode.

    Returns:
        (highs_idx, lows_idx): indeks posisi (bukan tanggal) dari swing
        high dan swing low pada `prices`.
    """
    values = prices.to_numpy()

    highs_idx, _ = find_peaks(values, distance=distance, prominence=prominence)
    lows_idx, _ = find_peaks(-values, distance=distance, prominence=prominence)

    return highs_idx, lows_idx
