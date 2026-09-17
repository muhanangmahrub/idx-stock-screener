"""Pemeriksaan alokasi portofolio & cash (CLAUDE.md bagian E, Teguh Hidayat).

Modul ini tidak menyarankan saham apa yang dibeli/dijual. Ia hanya
membandingkan komposisi portofolio yang dimasukkan pemilik dengan aturan
alokasi, lalu mengembalikan daftar penyimpangan untuk ditinjau manual.
"""

from dataclasses import dataclass, field

TARGET_STOCK_COUNT_MIN = 5
TARGET_STOCK_COUNT_MAX = 6
MAX_STOCK_COUNT = 10
EXTREME_MAX_STOCK_COUNT = 12

MIN_WEIGHT_PCT = 5.0
TARGET_WEIGHT_PCT_MIN = 25.0
TARGET_WEIGHT_PCT_MAX = 30.0
MAX_WEIGHT_PCT = 35.0
EXTREME_MAX_WEIGHT_PCT = 50.0  # hanya untuk keyakinan sangat tinggi

MIN_CASH_PCT = 10.0
IDEAL_CASH_PCT = 20.0


@dataclass
class Position:
    ticker: str
    sector: str
    value: float


@dataclass
class PortfolioReview:
    total_value: float
    cash_pct: float
    weights_pct: dict[str, float]
    warnings: list[str] = field(default_factory=list)  # melanggar batas keras
    notes: list[str] = field(default_factory=list)  # di luar target, belum melanggar


def review_portfolio(positions: list[Position], cash: float) -> PortfolioReview:
    """Bandingkan portofolio dengan aturan alokasi; kembalikan penyimpangannya.

    Total nilai = saham + cash, jadi bobot dan porsi cash dihitung dari basis
    yang sama.
    """
    stock_value = sum(position.value for position in positions)
    total_value = stock_value + cash
    if total_value <= 0:
        return PortfolioReview(total_value=0.0, cash_pct=0.0, weights_pct={})

    cash_pct = cash / total_value * 100
    weights_pct = {
        position.ticker: position.value / total_value * 100 for position in positions
    }
    review = PortfolioReview(
        total_value=total_value, cash_pct=cash_pct, weights_pct=weights_pct
    )

    _check_stock_count(positions, review)
    _check_sector_overlap(positions, review)
    _check_weights(review)
    _check_cash(review)
    return review


def _check_stock_count(positions: list[Position], review: PortfolioReview) -> None:
    count = len(positions)
    if count > EXTREME_MAX_STOCK_COUNT:
        review.warnings.append(
            f"{count} saham melebihi batas ekstrem {EXTREME_MAX_STOCK_COUNT}"
        )
    elif count > MAX_STOCK_COUNT:
        review.warnings.append(
            f"{count} saham melebihi maksimum {MAX_STOCK_COUNT} "
            f"(ekstrem {EXTREME_MAX_STOCK_COUNT})"
        )
    elif count > TARGET_STOCK_COUNT_MAX or 0 < count < TARGET_STOCK_COUNT_MIN:
        review.notes.append(
            f"{count} saham, di luar target {TARGET_STOCK_COUNT_MIN}-"
            f"{TARGET_STOCK_COUNT_MAX} saham"
        )


def _check_sector_overlap(positions: list[Position], review: PortfolioReview) -> None:
    tickers_by_sector: dict[str, list[str]] = {}
    for position in positions:
        sector = position.sector.strip().lower() or "(tanpa sektor)"
        tickers_by_sector.setdefault(sector, []).append(position.ticker)

    for sector, tickers in tickers_by_sector.items():
        if len(tickers) > 1:
            review.warnings.append(
                f"Sektor '{sector}' terisi lebih dari satu saham: {', '.join(tickers)}"
            )


def _check_weights(review: PortfolioReview) -> None:
    for ticker, weight in review.weights_pct.items():
        if weight > EXTREME_MAX_WEIGHT_PCT:
            review.warnings.append(
                f"{ticker} {weight:.1f}% melebihi batas ekstrem {EXTREME_MAX_WEIGHT_PCT:.0f}%"
            )
        elif weight > MAX_WEIGHT_PCT:
            review.warnings.append(
                f"{ticker} {weight:.1f}% melebihi maksimum {MAX_WEIGHT_PCT:.0f}% "
                f"(ekstrem {EXTREME_MAX_WEIGHT_PCT:.0f}% hanya bila keyakinan sangat tinggi)"
            )
        elif weight < MIN_WEIGHT_PCT:
            review.warnings.append(
                f"{ticker} {weight:.1f}% di bawah minimum {MIN_WEIGHT_PCT:.0f}%"
            )
        elif not TARGET_WEIGHT_PCT_MIN <= weight <= TARGET_WEIGHT_PCT_MAX:
            review.notes.append(
                f"{ticker} {weight:.1f}% di luar target "
                f"{TARGET_WEIGHT_PCT_MIN:.0f}-{TARGET_WEIGHT_PCT_MAX:.0f}%"
            )


def _check_cash(review: PortfolioReview) -> None:
    if review.cash_pct < MIN_CASH_PCT:
        review.warnings.append(
            f"Cash {review.cash_pct:.1f}% di bawah minimum {MIN_CASH_PCT:.0f}%"
        )
    elif review.cash_pct < IDEAL_CASH_PCT:
        review.notes.append(
            f"Cash {review.cash_pct:.1f}% di bawah ideal {IDEAL_CASH_PCT:.0f}%"
        )
