"""Validasi breakout resistance & trading plan, mengacu pada Edianto Ong.

Masalah yang dipecahkan: harga yang menembus resistance bisa palsu (false
breakout) - naik sebentar lalu balik turun dan menjebak pembeli. Modul ini
membantu membedakan breakout yang layak ditindaklanjuti dan menyiapkan
rencana masuk-keluar SEBELUM bertindak. Output = status kondisi + angka
rencana untuk dinilai manual, bukan perintah beli/jual.

Aturan buku (diberikan pemilik 2026-09-22, contoh saham McD):
1. Breakout sah hanya bila Close melewati resistance + toleransi 1,5%.
   Menyentuh saja tidak cukup. Contoh: resistance 50,5 -> tembus bila Close
   di atas 51,25.
2. Makin sering sebuah level gagal ditembus, makin penting level itu.
   Resistance yang sudah diuji 3 kali = "strong resistance".
3. Jangan masuk di hari breakout; tunggu konfirmasi lalu masuk di
   PEMBUKAAN hari berikutnya (aturan 2nd day).
4. Rencana keluar disiapkan sebelum masuk: resistance lama menjadi support
   baru, cut-loss 1,5% di bawahnya. Contoh: masuk 51,8, cut-loss 49,8,
   risiko 2/lembar.
5. Bila setelah masuk harga jatuh menembus cut-loss = false breakout ->
   keluar sesuai rencana.
6. Biarkan profit berjalan: tahan selama harga di atas up-trendline; keluar
   saat harga menembus trendline ke bawah. Contoh: keluar 58,3, untung
   6,5/lembar.

Prinsip keseluruhan: "Cut your loss fast, let your profit run." Modul ini
menghasilkan sinyal & rencana untuk ditinjau dan dieksekusi manual oleh
pemilik - bukan robot yang menaruh order sendiri.

Catatan angka contoh: buku membulatkan (50,5 x 1,015 = 51,26 ditulis 51,25;
50,5 x 0,985 = 49,74 ditulis 49,8). Kode memakai hitungan persis; test
membandingkan dengan toleransi pembulatan itu.

ASUMSI (bukan buku, wajib dikalibrasi):
- "Menguji" resistance = swing high yang High-nya sampai dalam
  `test_tol` di bawah level (default 1,5%, meniru angka toleransi buku)
  tanpa Close melewati batas breakout.
- Cut-loss dan patah trendline dinilai dari Close (prinsip Close lebih
  signifikan), dan harga keluar = Close bar itu. Buku tidak merinci apakah
  keluar memakai 2nd day juga.
- Toleransi patah trendline memakai `trend.DEFAULT_BREAK_TOLERANCE` (2%,
  medium term).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from screener.trend import (
    DEFAULT_BREAK_TOLERANCE,
    SECOND_DAY_CONFIRMED,
    SECOND_DAY_PENDING,
    UP_TRENDLINE,
    Trendline,
    break_threshold,
)

# Angka buku.
BREAKOUT_TOLERANCE = 0.015  # Close harus > resistance * 1,015
CUT_LOSS_TOLERANCE = 0.015  # cut-loss = support baru * 0,985
STRONG_RESISTANCE_MIN_TESTS = 3  # diuji >= 3 kali = strong resistance
# ASUMSI (bukan buku): seberapa dekat swing high ke level supaya dihitung "menguji".
DEFAULT_TEST_TOLERANCE = 0.015

# Label 2nd day sama dengan yang dipakai trendline (aturan bukunya memang sama).
SECOND_DAY_GAP_BACK = "gap kembali"

POSITION_HOLD = "hold"
EXIT_CUT_LOSS = "cut loss (false breakout)"
# Buku menyebutnya ambil untung; labelnya netral karena P/L bisa saja negatif.
EXIT_TAKE_PROFIT = "keluar: trendline patah"


def breakout_threshold(resistance: float, tol: float = BREAKOUT_TOLERANCE) -> float:
    """Batas breakout sah: resistance + toleransi (50,5 -> 51,26; buku: 51,25)."""
    return resistance * (1 + tol)


def count_resistance_tests(
    resistance: float,
    highs_idx: np.ndarray,
    highs: pd.Series,
    closes: pd.Series,
    until_index: int | None = None,
    test_tol: float = DEFAULT_TEST_TOLERANCE,
    breakout_tol: float = BREAKOUT_TOLERANCE,
) -> int:
    """Berapa kali resistance diuji: swing high yang High-nya mencapai
    [resistance * (1 - test_tol), ...) tapi Close-nya tidak melewati batas
    breakout. Swing high setelah `until_index` (mis. setelah breakout) tidak
    dihitung. Puncak yang membentuk level itu sendiri ikut dihitung.
    """
    high_values = highs.to_numpy()
    close_values = closes.to_numpy()
    threshold = breakout_threshold(resistance, breakout_tol)
    tests = 0
    for i in highs_idx:
        if until_index is not None and i > until_index:
            continue
        reached = high_values[i] >= resistance * (1 - test_tol)
        failed = close_values[i] <= threshold
        if reached and failed:
            tests += 1
    return tests


def is_strong_resistance(tests: int, min_tests: int = STRONG_RESISTANCE_MIN_TESTS) -> bool:
    return tests >= min_tests


@dataclass
class BreakoutSignal:
    resistance: float
    threshold: float  # batas Close untuk breakout sah
    breakout_index: int  # bar dengan Close > threshold
    second_day: str  # SECOND_DAY_CONFIRMED | SECOND_DAY_PENDING
    entry_index: int | None = None  # bar masuk = bar setelah breakout (Open)
    entry_price: float | None = None
    gap_back_indices: list[int] | None = None  # breakout yang Open berikutnya gap kembali


def find_breakout(
    resistance: float,
    closes: pd.Series,
    opens: pd.Series,
    start_index: int,
    end_index: int | None = None,
    tol: float = BREAKOUT_TOLERANCE,
) -> BreakoutSignal | None:
    """Cari breakout sah pertama sejak `start_index`: Close > resistance +
    tol, lalu 2nd day: Open bar berikutnya harus masih di atas batas ->
    itulah bar & harga masuk. Open yang gap kembali ke bawah batas
    membatalkan breakout itu dan pencarian lanjut. None bila tidak ada.
    """
    close_values = closes.to_numpy()
    open_values = opens.to_numpy()
    if end_index is None:
        end_index = len(close_values) - 1
    threshold = breakout_threshold(resistance, tol)

    gap_backs: list[int] = []
    for i in range(start_index, end_index + 1):
        if close_values[i] <= threshold:
            continue
        next_bar = i + 1
        if next_bar > end_index:
            return BreakoutSignal(
                resistance, threshold, i, SECOND_DAY_PENDING, gap_back_indices=gap_backs
            )
        if open_values[next_bar] > threshold:
            return BreakoutSignal(
                resistance,
                threshold,
                i,
                SECOND_DAY_CONFIRMED,
                entry_index=next_bar,
                entry_price=float(open_values[next_bar]),
                gap_back_indices=gap_backs,
            )
        gap_backs.append(i)
    return None


@dataclass
class TradingPlan:
    entry_price: float
    support: float  # resistance lama yang jadi lantai support baru
    cut_loss: float  # support * (1 - tol)
    risk_per_share: float  # entry - cut_loss

    @property
    def risk_pct(self) -> float:
        return self.risk_per_share / self.entry_price


def build_trading_plan(
    resistance: float, entry_price: float, cut_loss_tol: float = CUT_LOSS_TOLERANCE
) -> TradingPlan:
    """Rencana keluar SEBELUM masuk: support baru = resistance lama,
    cut-loss 1,5% di bawahnya, risiko per lembar = entry - cut-loss
    (buku: masuk 51,8, cut-loss 49,8, risiko 2)."""
    cut_loss = resistance * (1 - cut_loss_tol)
    return TradingPlan(
        entry_price=entry_price,
        support=resistance,
        cut_loss=cut_loss,
        risk_per_share=entry_price - cut_loss,
    )


@dataclass
class PositionStatus:
    status: str  # POSITION_HOLD | EXIT_CUT_LOSS | EXIT_TAKE_PROFIT
    exit_index: int | None
    exit_price: float | None
    pnl_per_share: float  # realisasi bila keluar; mengambang (Close terakhir) bila hold


def evaluate_breakout_position(
    plan: TradingPlan,
    entry_index: int,
    closes: pd.Series,
    trendline: Trendline | None = None,
    trendline_tol: float = DEFAULT_BREAK_TOLERANCE,
    end_index: int | None = None,
) -> PositionStatus:
    """Jalankan rencana bar demi bar setelah masuk.

    Urutan tiap bar: (a) Close < cut-loss -> keluar, false breakout;
    (b) ada up-trendline dan Close < garis - toleransi (setelah titik acuan
    terakhir) -> keluar, take profit. Kalau tidak ada -> tahan. Harga keluar
    = Close bar itu (ASUMSI).
    """
    close_values = closes.to_numpy()
    if end_index is None:
        end_index = len(close_values) - 1
    if trendline is not None and trendline.kind != UP_TRENDLINE:
        raise ValueError("Rencana keluar memakai up-trendline (tahan selama harga di atasnya)")

    for i in range(entry_index + 1, end_index + 1):
        close = float(close_values[i])
        if close < plan.cut_loss:
            return PositionStatus(EXIT_CUT_LOSS, i, close, close - plan.entry_price)
        if trendline is not None and i > trendline.points[-1].index:
            boundary = break_threshold(trendline.value_at(i), UP_TRENDLINE, trendline_tol)
            if close < boundary:
                return PositionStatus(EXIT_TAKE_PROFIT, i, close, close - plan.entry_price)

    last_close = float(close_values[end_index])
    return PositionStatus(POSITION_HOLD, None, None, last_close - plan.entry_price)
