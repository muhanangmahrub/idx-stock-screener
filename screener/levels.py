"""Pelacakan level support & resistance horizontal, mengacu pada Edianto Ong.

Aturan buku (diberikan pemilik 2026-09-20):
- Sebuah level dinyatakan tertembus dengan prinsip yang sama seperti
  trendline: harga PENUTUPAN berada di luar level (Close < support, atau
  Close > resistance). Tembusan intraday (Low/High) yang Close-nya kembali
  = false break / whipsaw.
- Bila support berhasil ditembus, garis itu berubah menjadi resistance;
  bila resistance ditembus, berubah menjadi support. Semakin kuat level itu
  sebelumnya, semakin kuat pula perannya yang baru.
- Faktor waktu ikut menentukan kekuatan: support/resistance lima bulan lebih
  kuat daripada lima hari.
- Garis support dibentuk dengan menarik garis mendatar dari titik terendah
  pada lembah yang sudah terjadi; garis resistance dari titik tertinggi pada
  puncak yang sudah terjadi.
- Pullback: ketika harga kembali menguji suatu level support maupun
  resistance yang sudah dilewati (diberikan 2026-09-22).

Terjemahan ke kode:
- `track_level` berjalan bar demi bar sejak level terbentuk, membalik peran
  pada tiap valid break, dan mencatat whipsaw. Usia level dihitung sejak
  terbentuk dan TIDAK direset saat peran berbalik - ini terjemahan "menjadi
  resistance yang sama kuatnya".
- Buku tidak memberi skala kekuatan, jadi modul ini hanya melaporkan usia
  (bar) untuk dinilai manual - tidak ada skor "kuat/lemah" otomatis.
- Pullback hanya mungkin SETELAH level pernah ditembus sah. Sesudah itu,
  bar yang harganya kembali menyentuh level (Low <= support / High >=
  resistance) dicatat sebagai PULLBACK bila Close tetap di dalam (level
  bertahan). Bila Close menembus lagi, itu tercatat sebagai VALID_BREAK
  berikutnya = pullback yang gagal (peran berbalik lagi).
- ASUMSI (bukan buku): "menguji" = menyentuh level. `pullback_tol`
  (default 0 = harus menyentuh) melonggarkan ke "mendekati sekian %".

- `levels_from_swings` menerapkan definisi buku: swing low -> support di
  harga Low bar itu, swing high -> resistance di harga High bar itu.
  Catatan: swing dideteksi dari Close (`get_extrema`), jadi "titik terendah
  lembah" diambil dari Low pada bar swing tersebut - bar tetangga bisa punya
  Low sedikit lebih rendah. Kalau perlu lebih presisi, ambil Low terendah di
  sekitar bar swing.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

SUPPORT = "support"
RESISTANCE = "resistance"

# Label penembusan dipakai bersama trendline supaya istilah di UI tidak
# bercabang: mengubahnya di satu tempat ikut berlaku di semua tampilan.
from screener.trend import FALSE_BREAK, VALID_BREAK  # noqa: E402  (re-export)

PULLBACK = "pullback"  # harga kembali menguji level yang sudah dilewati, dan level bertahan


@dataclass
class LevelEvent:
    index: int  # bar tempat kejadian
    kind: str  # VALID_BREAK | FALSE_BREAK | PULLBACK
    role_before: str  # peran level saat kejadian (SUPPORT | RESISTANCE)


@dataclass
class LevelStatus:
    price: float
    origin_index: int  # bar tempat level terbentuk
    initial_role: str
    role: str  # peran saat ini, setelah semua pembalikan
    end_index: int
    events: list[LevelEvent] = field(default_factory=list)

    @property
    def age_bars(self) -> int:
        """Usia level sejak terbentuk; tidak direset saat peran berbalik."""
        return self.end_index - self.origin_index

    @property
    def valid_breaks(self) -> list[LevelEvent]:
        return [e for e in self.events if e.kind == VALID_BREAK]

    @property
    def false_breaks(self) -> list[LevelEvent]:
        return [e for e in self.events if e.kind == FALSE_BREAK]

    @property
    def pullbacks_held(self) -> list[LevelEvent]:
        """Harga kembali menguji level yang sudah dilewati, Close tetap di dalam."""
        return [e for e in self.events if e.kind == PULLBACK]

    @property
    def pullbacks_failed(self) -> list[LevelEvent]:
        """Uji ulang yang gagal: Close menembus lagi. Tembusan pertama bukan
        pullback (belum ada level yang 'sudah dilewati' saat itu)."""
        return self.valid_breaks[1:]


def _opposite(role: str) -> str:
    return RESISTANCE if role == SUPPORT else SUPPORT


def track_level(
    price: float,
    initial_role: str,
    origin_index: int,
    closes: pd.Series,
    lows: pd.Series,
    highs: pd.Series,
    end_index: int | None = None,
    pullback_tol: float = 0.0,
) -> LevelStatus:
    """Ikuti satu level horizontal dari `origin_index` sampai `end_index`.

    Support: Close < level = valid break (peran jadi resistance); hanya Low <
    level = whipsaw. Resistance cermin dengan Close/High > level. Close tepat
    di level belum dihitung di luar. Setelah berbalik, pengecekan bar
    berikutnya memakai peran yang baru, dan sentuhan ke level (dalam
    `pullback_tol`) yang Close-nya bertahan dicatat sebagai PULLBACK.
    """
    if initial_role not in (SUPPORT, RESISTANCE):
        raise ValueError(f"initial_role harus {SUPPORT} atau {RESISTANCE}")
    if pullback_tol < 0:
        raise ValueError("pullback_tol tidak boleh negatif")

    close_values = closes.to_numpy()
    low_values = lows.to_numpy()
    high_values = highs.to_numpy()
    if end_index is None:
        end_index = len(close_values) - 1

    role = initial_role
    broken_before = False  # sudah pernah dilewati? syarat sebuah uji ulang disebut pullback
    events: list[LevelEvent] = []
    for i in range(origin_index + 1, end_index + 1):
        if role == SUPPORT:
            closed_outside = close_values[i] < price
            pierced = low_values[i] < price
            touched = low_values[i] <= price * (1 + pullback_tol)
        else:
            closed_outside = close_values[i] > price
            pierced = high_values[i] > price
            touched = high_values[i] >= price * (1 - pullback_tol)

        if closed_outside:
            events.append(LevelEvent(i, VALID_BREAK, role))
            role = _opposite(role)
            broken_before = True
        elif broken_before and touched:
            events.append(LevelEvent(i, PULLBACK, role))
        elif pierced:
            events.append(LevelEvent(i, FALSE_BREAK, role))

    return LevelStatus(
        price=price,
        origin_index=origin_index,
        initial_role=initial_role,
        role=role,
        end_index=end_index,
        events=events,
    )


def levels_from_swings(
    df: pd.DataFrame,
    highs_idx: np.ndarray,
    lows_idx: np.ndarray,
    per_side: int,
    pullback_tol: float = 0.0,
) -> list[LevelStatus]:
    """Aturan buku: support = garis mendatar dari titik terendah lembah (Low
    di swing low), resistance = dari titik tertinggi puncak (High di swing
    high). Diambil `per_side` swing terakhir tiap sisi, dilacak sampai bar
    terkini, diurutkan dari yang paling tua (faktor waktu: level yang lebih
    lama lebih kuat).
    """
    end_index = len(df) - 1
    levels = []
    for i in lows_idx[-per_side:]:
        levels.append(
            track_level(
                float(df["Low"].iloc[i]), SUPPORT, int(i),
                df["Close"], df["Low"], df["High"], end_index, pullback_tol,
            )
        )
    for i in highs_idx[-per_side:]:
        levels.append(
            track_level(
                float(df["High"].iloc[i]), RESISTANCE, int(i),
                df["Close"], df["Low"], df["High"], end_index, pullback_tol,
            )
        )
    return sorted(levels, key=lambda level: level.origin_index)
