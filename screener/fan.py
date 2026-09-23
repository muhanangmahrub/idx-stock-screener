"""The Fan Principle (Edianto Ong Bab 14): konfirmasi pembalikan tren.

Masalah yang dipecahkan: satu trendline yang tertembus sering menipu - harga
bisa cuma pullback lalu lanjut tren lama. Fan principle menuntut konfirmasi
bertingkat: reversal baru diakui setelah harga menembus TIGA trendline
berturut-turut yang memancar dari satu titik pangkal.

Aturan buku (diberikan pemilik 2026-09-22):
1. Tiga trendline berbagi satu titik pangkal. Pada tren naik yang melemah:
   harga menembus trendline naik pertama, membentuk lembah baru, lalu dari
   titik pangkal yang SAMA ditarik trendline kedua yang lebih landai,
   begitu seterusnya sampai trendline ketiga. Ketiganya tampak seperti kipas
   yang makin melebar.
2. Reversal dikonfirmasi HANYA saat trendline KETIGA tertembus. Tembusnya
   garis pertama atau kedua belum berarti apa-apa (bisa sekadar koreksi).
   Tren naik melemah -> tembus garis ketiga ke bawah = konfirmasi tren turun.
   Tren turun menguat -> tembus garis ketiga ke atas = konfirmasi tren naik.
3. Berlaku dua arah (bearish & bullish), logikanya cermin.
4. Tiap kali harga pullback menguji garis yang sudah ditembus, garis itu
   berganti fungsi jadi penghalang: resistance saat tren melemah, support
   saat tren menguat - konsisten dengan pembalikan peran di levels.py.

Prinsip: menyaring sinyal reversal palsu dengan konfirmasi bertingkat.
Keluarannya sinyal untuk ditinjau manual oleh pemilik, bukan eksekusi
otomatis.

ASUMSI (bukan buku, wajib dikalibrasi):
- Titik pangkal (origin) = swing low terendah (fan bearish) / swing high
  tertinggi (fan bullish) sebelum kipas mulai terbentuk, kecuali pemilik
  menentukan `origin_index` sendiri.
- Titik acuan tiap garis = swing berikutnya setelah garis sebelumnya
  tertembus; garis harus tetap searah tren lama (naik untuk fan bearish)
  dan makin landai, sesuai gambaran "kipas yang melebar".
- Penembusan memakai aturan yang sama seperti trendline: Close di luar garis
  +- toleransi (`trend.DEFAULT_BREAK_TOLERANCE`, 2% medium term).
- Uji ulang (retest) = bar yang menyentuh garis lama dalam `touch_tol` (2%)
  tanpa Close kembali ke sisi lama.

Yang TIDAK dipaksakan: kipas baru dianggap mulai terbentuk setelah garis
pertama tertembus (`min_broken_lines`, default 1). Tanpa itu yang ada cuma
sebuah trendline biasa, bukan kipas - `detect_fan` mengembalikan None.
Pemanggil juga harus memastikan ada tren yang sedang berlangsung: reversal
mensyaratkan tren lama, jadi chart yang trennya "tidak jelas" atau sideways
tidak diuji dengan fan principle.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from screener.trend import (
    DEFAULT_BREAK_TOLERANCE,
    DOWN_TRENDLINE,
    UP_TRENDLINE,
    SwingPoint,
    break_threshold,
    fit_line,
)

# Arah kipas = arah pembalikan yang sedang diuji.
FAN_BEARISH = "bearish"  # tren naik melemah -> kemungkinan berbalik turun
FAN_BULLISH = "bullish"  # tren turun menguat -> kemungkinan berbalik naik

FAN_FORMING = "kipas terbentuk sebagian"
FAN_CONFIRMED = "reversal terkonfirmasi (garis ketiga tertembus)"

ROLE_RESISTANCE = "resistance"
ROLE_SUPPORT = "support"

FAN_LINE_COUNT = 3  # aturan buku: konfirmasi pada garis ketiga
DEFAULT_TOUCH_TOLERANCE = 0.02  # ASUMSI (bukan buku): batas "menyentuh" saat retest
# Kipas baru ada setelah garis pertama tertembus; sebelum itu cuma trendline.
DEFAULT_MIN_BROKEN_LINES = 1


@dataclass
class FanLine:
    """Satu garis kipas: dari titik pangkal ke satu swing acuan."""

    order: int  # 1, 2, 3
    slope: float
    intercept: float
    origin: SwingPoint
    anchor: SwingPoint
    kind: str  # UP_TRENDLINE (fan bearish) | DOWN_TRENDLINE (fan bullish)
    break_index: int | None = None  # bar dengan Close di luar garis + toleransi
    retest_indices: list[int] = field(default_factory=list)  # uji ulang setelah tertembus

    def value_at(self, index: int) -> float:
        return self.slope * index + self.intercept

    @property
    def is_broken(self) -> bool:
        return self.break_index is not None

    @property
    def role_after_break(self) -> str | None:
        """Peran garis setelah tertembus: penghalang di sisi seberang."""
        if not self.is_broken:
            return None
        return ROLE_RESISTANCE if self.kind == UP_TRENDLINE else ROLE_SUPPORT


@dataclass
class FanPattern:
    direction: str  # FAN_BEARISH | FAN_BULLISH
    origin: SwingPoint
    lines: list[FanLine]
    status: str  # FAN_FORMING | FAN_CONFIRMED
    confirmation_index: int | None = None  # bar penembusan garis ketiga
    tolerance: float = DEFAULT_BREAK_TOLERANCE

    @property
    def lines_broken(self) -> int:
        return sum(line.is_broken for line in self.lines)

    @property
    def is_confirmed(self) -> bool:
        return self.status == FAN_CONFIRMED

    @property
    def reason(self) -> str:
        if self.is_confirmed:
            return (
                f"{self.lines_broken} dari {FAN_LINE_COUNT} garis kipas tertembus; "
                "garis ketiga tertembus = reversal terkonfirmasi."
            )
        return (
            f"{self.lines_broken} dari {FAN_LINE_COUNT} garis kipas tertembus "
            f"({len(self.lines)} garis terbentuk); belum cukup untuk menyatakan "
            "reversal - tembusnya garis pertama/kedua bisa sekadar koreksi."
        )


def _first_break(
    line: FanLine, closes: np.ndarray, start: int, end: int, tol: float
) -> int | None:
    """Bar pertama setelah `start` dengan Close di luar garis + toleransi."""
    for i in range(start, end + 1):
        boundary = break_threshold(line.value_at(i), line.kind, tol)
        if line.kind == UP_TRENDLINE and closes[i] < boundary:
            return i
        if line.kind == DOWN_TRENDLINE and closes[i] > boundary:
            return i
    return None


def _collect_retests(
    line: FanLine,
    closes: np.ndarray,
    pierce_values: np.ndarray,
    start: int,
    end: int,
    touch_tol: float,
) -> list[int]:
    """Bar yang kembali menguji garis yang sudah tertembus tanpa merebutnya:
    fan bearish -> High menyentuh garis tapi Close tetap di bawah (garis jadi
    resistance); fan bullish cermin."""
    retests = []
    for i in range(start, end + 1):
        level = line.value_at(i)
        near = abs(pierce_values[i] - level) <= abs(level) * touch_tol
        if line.kind == UP_TRENDLINE:
            reached = pierce_values[i] >= level * (1 - touch_tol)
            held = closes[i] < level
        else:
            reached = pierce_values[i] <= level * (1 + touch_tol)
            held = closes[i] > level
        if (near or reached) and held:
            retests.append(i)
    return retests


def detect_fan(
    direction: str,
    closes: pd.Series,
    highs: pd.Series,
    lows: pd.Series,
    highs_idx: np.ndarray,
    lows_idx: np.ndarray,
    origin_index: int | None = None,
    tol: float = DEFAULT_BREAK_TOLERANCE,
    touch_tol: float = DEFAULT_TOUCH_TOLERANCE,
    end_index: int | None = None,
    min_broken_lines: int = DEFAULT_MIN_BROKEN_LINES,
) -> FanPattern | None:
    """Cari pola kipas tiga trendline dari satu titik pangkal.

    Fan bearish (tren naik melemah): garis ditarik dari swing low pangkal ke
    swing low berikutnya (garis naik = support). Tiap kali sebuah garis
    tertembus ke bawah, garis berikutnya ditarik dari pangkal yang SAMA ke
    lembah baru sesudah penembusan itu - hasilnya makin landai (kipas
    melebar). Fan bullish cermin: dari swing high pangkal ke swing high
    berikutnya (garis turun = resistance), tembus ke atas.

    None bila titik pangkal/garis pertama tidak terbentuk, atau bila garis yang
    tertembus kurang dari `min_broken_lines` - tanpa penembusan, yang ada baru
    sebuah trendline biasa, bukan kipas.
    """
    if direction not in (FAN_BEARISH, FAN_BULLISH):
        raise ValueError(f"direction harus {FAN_BEARISH} atau {FAN_BULLISH}")
    if tol < 0 or touch_tol < 0:
        raise ValueError("toleransi tidak boleh negatif")

    bearish = direction == FAN_BEARISH
    kind = UP_TRENDLINE if bearish else DOWN_TRENDLINE
    close_values = closes.to_numpy()
    # Fan bearish bertumpu pada lembah (Low); fan bullish pada puncak (High).
    anchor_values = lows.to_numpy() if bearish else highs.to_numpy()
    anchor_idx = [int(i) for i in (lows_idx if bearish else highs_idx)]
    pierce_values = highs.to_numpy() if bearish else lows.to_numpy()
    if end_index is None:
        end_index = len(close_values) - 1

    if origin_index is None:
        candidates = [i for i in anchor_idx if i <= end_index]
        if not candidates:
            return None
        # Pangkal kipas = awal tren lama: lembah terendah (bearish) / puncak
        # tertinggi (bullish) di antara swing yang ada.
        origin_index = (
            min(candidates, key=lambda i: anchor_values[i])
            if bearish
            else max(candidates, key=lambda i: anchor_values[i])
        )
    origin = SwingPoint(origin_index, float(anchor_values[origin_index]))

    lines: list[FanLine] = []
    search_from = origin_index + 1
    previous: FanLine | None = None

    while len(lines) < FAN_LINE_COUNT:
        anchor_index = None
        for i in anchor_idx:
            if i < search_from or i > end_index:
                continue
            slope = (anchor_values[i] - origin.price) / (i - origin.index)
            # Garis harus tetap searah tren lama, dan makin landai dari garis
            # sebelumnya - itulah bentuk kipas yang melebar.
            if bearish and slope <= 0:
                continue
            if not bearish and slope >= 0:
                continue
            if previous is not None:
                if bearish and slope >= previous.slope:
                    continue
                if not bearish and slope <= previous.slope:
                    continue
            anchor_index = i
            break
        if anchor_index is None:
            break

        anchor = SwingPoint(anchor_index, float(anchor_values[anchor_index]))
        slope, intercept = fit_line(
            [origin.index, anchor.index], [origin.price, anchor.price]
        )
        line = FanLine(
            order=len(lines) + 1,
            slope=slope,
            intercept=intercept,
            origin=origin,
            anchor=anchor,
            kind=kind,
        )
        line.break_index = _first_break(
            line, close_values, anchor.index + 1, end_index, tol
        )
        lines.append(line)
        previous = line

        if line.break_index is None:
            break  # garis terakhir belum tertembus: kipas masih terbentuk
        search_from = line.break_index + 1

    if sum(line.is_broken for line in lines) < min_broken_lines:
        return None

    # Uji ulang tiap garis yang sudah tertembus, sampai bar terakhir.
    for line in lines:
        if line.is_broken:
            line.retest_indices = _collect_retests(
                line, close_values, pierce_values, line.break_index + 1, end_index, touch_tol
            )

    third = lines[FAN_LINE_COUNT - 1] if len(lines) == FAN_LINE_COUNT else None
    confirmed = third is not None and third.is_broken
    return FanPattern(
        direction=direction,
        origin=origin,
        lines=lines,
        status=FAN_CONFIRMED if confirmed else FAN_FORMING,
        confirmation_index=third.break_index if confirmed else None,
        tolerance=tol,
    )
