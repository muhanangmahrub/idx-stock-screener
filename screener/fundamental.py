"""Filter fundamental (Lapis 1), mengacu pada metode Teguh Hidayat.

Angka-angka di modul ini diambil dari CLAUDE.md bagian "Spesifikasi metode
Teguh Hidayat (lapis 1)", bukan tebakan. Modul ini mencakup bagian A
(screening awal), B (valuasi harga absolut), C (valuasi murah), dan D
(manajemen risiko posisi). Bagian E ada di portfolio.py, bagian F di
execution.py, checklist kualitatif di checklist.py.
"""

from dataclasses import dataclass, field

import pandas as pd

from screener.formatting import format_rupiah

MIN_DAILY_TRANSACTION_VALUE = 5_000_000_000  # Rp5 miliar
MIN_FREE_FLOAT_PCT = 15.0
MIN_ROE_ANNUALIZED_PCT = 10.0
IDEAL_ROE_MIN_PCT = 20.0  # band ideal 20-30% tergantung sektor; bukan syarat lolos
IDEAL_ROE_MAX_PCT = 30.0

BLUE_CHIP_MAX_PER = 12
SMALL_CAP_MAX_PER = 8
MAX_PBV_CHEAP = 0.7

# Bagian B: valuasi & harga best buy.
ROE_TO_PBV_DIVISOR = 10  # ROE 51% -> PBV dasar 5,1x
RISK_DISCOUNT_PER_FACTOR = 0.10
DEFAULT_MARGIN_OF_SAFETY = 0.35
MIN_MARGIN_OF_SAFETY = 0.35
MAX_MARGIN_OF_SAFETY = 0.50
MAX_BUY_TOLERANCE = 1.15  # boleh beli sampai +15% di atas best buy

# Bagian D: manajemen risiko posisi.
MAX_POSITION_PCT_OF_DAILY_VALUE = 0.30
MAX_POSITION_PCT_OF_NET_WORTH = 0.20

# ASUMSI (bukan angka Teguh Hidayat): jendela rata-rata untuk estimasi
# nilai transaksi harian dari data harga. Wajib dikalibrasi.
DEFAULT_TRANSACTION_VALUE_WINDOW_DAYS = 20


@dataclass
class ScreeningResult:
    ticker: str
    passed: bool
    reasons: list[str] = field(default_factory=list)  # penyebab tidak lolos
    notes: list[str] = field(default_factory=list)  # catatan, tidak menggagalkan


def passes_initial_screening(
    ticker: str,
    daily_transaction_value: float,
    free_float_pct: float,
    roe_annualized_pct: float,
    is_blacklisted: bool = False,
) -> ScreeningResult:
    """Filter kuantitatif awal (CLAUDE.md bagian A).

    Saham gorengan/bandar tidak dideteksi otomatis - `is_blacklisted`
    diisi dari daftar manual pemilik, bukan heuristik.
    """
    reasons = []

    if daily_transaction_value < MIN_DAILY_TRANSACTION_VALUE:
        reasons.append(
            f"Likuiditas harian {format_rupiah(daily_transaction_value)} "
            f"< batas {format_rupiah(MIN_DAILY_TRANSACTION_VALUE)}"
        )
    if free_float_pct < MIN_FREE_FLOAT_PCT:
        reasons.append(f"Free float {free_float_pct:.1f}% < {MIN_FREE_FLOAT_PCT:.1f}%")
    if roe_annualized_pct < MIN_ROE_ANNUALIZED_PCT:
        reasons.append(
            f"ROE disetahunkan {roe_annualized_pct:.1f}% < {MIN_ROE_ANNUALIZED_PCT:.1f}%"
        )
    if is_blacklisted:
        reasons.append("Ticker ada di blacklist manual")

    notes = []
    if MIN_ROE_ANNUALIZED_PCT <= roe_annualized_pct < IDEAL_ROE_MIN_PCT:
        notes.append(
            f"ROE {roe_annualized_pct:.1f}% lolos batas minimum, tapi di bawah band "
            f"ideal {IDEAL_ROE_MIN_PCT:.0f}-{IDEAL_ROE_MAX_PCT:.0f}% (tergantung sektor)"
        )

    return ScreeningResult(ticker=ticker, passed=not reasons, reasons=reasons, notes=notes)


def is_cheap_valuation(per: float | None, pbv: float | None, is_blue_chip: bool) -> bool:
    """Aturan valuasi murah (CLAUDE.md bagian C).

    Blue chip murah bila PER <= 12 ATAU PBV <= 0.7.
    Small cap murah bila PER <= 8 ATAU PBV <= 0.7.

    Rasio yang tidak tersedia (None) tidak bisa membuktikan "murah", jadi
    dianggap tidak memenuhi sisi itu; sisi lain tetap dinilai.
    """
    max_per = BLUE_CHIP_MAX_PER if is_blue_chip else SMALL_CAP_MAX_PER
    cheap_by_per = per is not None and per <= max_per
    cheap_by_pbv = pbv is not None and pbv <= MAX_PBV_CHEAP
    return cheap_by_per or cheap_by_pbv


@dataclass
class Valuation:
    pbv_base: float
    fair_price_base: float
    fair_price_adj: float
    best_buy: float
    max_buy: float


def compute_valuation(
    roe_annualized_pct: float,
    bvps: float,
    risk_factor_count: int = 0,
    margin_of_safety: float = DEFAULT_MARGIN_OF_SAFETY,
) -> Valuation:
    """Jalur harga absolut (CLAUDE.md bagian B), urutan hitung sesuai pseudo-code.

    Ide dasarnya: ROE tinggi "membenarkan" PBV yang lebih tinggi. Harga wajar
    diturunkan dari BVPS, lalu didiskon per faktor risiko makro/sektoral, lalu
    dipotong margin of safety untuk dapat harga best buy.
    """
    if not MIN_MARGIN_OF_SAFETY <= margin_of_safety <= MAX_MARGIN_OF_SAFETY:
        raise ValueError(
            f"Margin of safety harus {MIN_MARGIN_OF_SAFETY:.0%}-"
            f"{MAX_MARGIN_OF_SAFETY:.0%}, dapat {margin_of_safety:.0%}"
        )
    if risk_factor_count < 0:
        raise ValueError("Jumlah faktor risiko tidak boleh negatif")

    pbv_base = roe_annualized_pct / ROE_TO_PBV_DIVISOR
    fair_price_base = bvps * pbv_base
    fair_price_adj = fair_price_base * (1 - RISK_DISCOUNT_PER_FACTOR * risk_factor_count)
    best_buy = fair_price_adj * (1 - margin_of_safety)
    max_buy = best_buy * MAX_BUY_TOLERANCE

    return Valuation(
        pbv_base=pbv_base,
        fair_price_base=fair_price_base,
        fair_price_adj=fair_price_adj,
        best_buy=best_buy,
        max_buy=max_buy,
    )


@dataclass
class PositionLimit:
    by_liquidity: float
    by_net_worth: float

    @property
    def max_position(self) -> float:
        return min(self.by_liquidity, self.by_net_worth)


def compute_position_limit(
    daily_transaction_value: float, liquid_net_worth: float
) -> PositionLimit:
    """Batas nilai beli per emiten (CLAUDE.md bagian D).

    Dua batas dihitung terpisah dan yang lebih kecil yang mengikat: batas
    likuiditas (supaya posisi bisa dijual tanpa menggerakkan harga) dan
    batas konsentrasi terhadap kekayaan bersih likuid investor.
    """
    return PositionLimit(
        by_liquidity=MAX_POSITION_PCT_OF_DAILY_VALUE * daily_transaction_value,
        by_net_worth=MAX_POSITION_PCT_OF_NET_WORTH * liquid_net_worth,
    )


# --- Helper turunan data: menghitung input screening dari data mentah yfinance ---


def estimate_daily_transaction_value(
    price_df: pd.DataFrame, window_days: int = DEFAULT_TRANSACTION_VALUE_WINDOW_DAYS
) -> float | None:
    """Estimasi nilai transaksi harian = rata-rata Close x Volume.

    Ini aproksimasi: nilai transaksi sebenarnya adalah jumlah (harga x lot)
    tiap transaksi, bukan Close x Volume. Cukup untuk screening kasar;
    angka resmi ada di ringkasan perdagangan IDX.
    """
    if price_df.empty or "Close" not in price_df or "Volume" not in price_df:
        return None
    daily_value = (price_df["Close"] * price_df["Volume"]).tail(window_days)
    return float(daily_value.mean())


def compute_free_float_pct(
    float_shares: float | None, shares_outstanding: float | None
) -> float | None:
    if not float_shares or not shares_outstanding:
        return None
    return float_shares / shares_outstanding * 100


def annualize_roe_pct(
    latest_quarter_net_income: float | None, latest_equity: float | None
) -> float | None:
    """ROE disetahunkan dari satu laporan kuartalan.

    ASUMSI metode: laba bersih kuartal terakhir x 4 dibagi ekuitas terakhir.
    Sumber hanya menyebut "ROE disetahunkan" tanpa merinci cara anualisasi;
    kalau pemilik memakai cara lain (mis. YTD x 12/bulan), ubah di sini.
    """
    if latest_quarter_net_income is None or not latest_equity:
        return None
    return latest_quarter_net_income * 4 / latest_equity * 100
