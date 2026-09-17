"""Detektor pola chart teknikal (Lapis 2), mengacu pada Edianto Ong.

Belum ada detektor terdaftar di sini: aturan konfirmasi tiap pola (mis.
double top, head & shoulders) harus diambil dari buku pemilik dulu sebelum
diterjemahkan ke kode - lihat CLAUDE.md prinsip 1 & 4. Jangan menambah
detektor dengan aturan yang ditebak.

Kontrak setiap detektor: `detect_xxx(prices, highs, lows, tol) -> dict`
dengan bentuk seragam:
    {"type": str, "bias": "bullish" | "bearish", "points": ..., \
     "neckline": ..., "confirmed": bool}

Detektor baru didaftarkan di DETECTORS, dan scan() tidak perlu diubah.
"""

from typing import Callable

import numpy as np
import pandas as pd

DETECTORS: dict[str, Callable[..., dict]] = {}


def scan(
    prices: pd.Series,
    highs: np.ndarray,
    lows: np.ndarray,
    tol: float = 0.02,
) -> list[dict]:
    """Jalankan semua detektor terdaftar dan kumpulkan hasil yang confirmed."""
    results = []
    for detector in DETECTORS.values():
        result = detector(prices, highs, lows, tol)
        if result is not None:
            results.append(result)
    return results
