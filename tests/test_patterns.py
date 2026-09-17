import numpy as np
import pandas as pd

from screener.patterns import DETECTORS, scan


def test_detectors_registry_starts_empty():
    """Belum ada detektor pola: aturan Edianto Ong belum diterjemahkan ke kode."""
    assert DETECTORS == {}


def test_scan_returns_empty_list_with_no_detectors():
    prices = pd.Series([1.0, 2.0, 1.5, 3.0, 2.0])
    highs = np.array([1, 3])
    lows = np.array([0, 2])

    assert scan(prices, highs, lows) == []
