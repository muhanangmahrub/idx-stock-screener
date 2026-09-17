"""Status eksekusi & horizon (CLAUDE.md bagian F, Teguh Hidayat).

Spesifikasi sumber memakai label BUY / HOLD / EXIT. Di sini label itu
diterjemahkan menjadi *kondisi yang terpenuhi atau tidak*, supaya screener
tetap alat verifikasi dan bukan pemberi sinyal: pemilik melihat kondisi mana
yang aktif, lalu memutuskan sendiri.
"""

from dataclasses import dataclass

HORIZON_SHORT_MAX_MONTHS = 3
HORIZON_MEDIUM_MAX_MONTHS = 12
HORIZON_LONG_MAX_MONTHS = 24

REBALANCE_REVIEW_MIN_MONTHS = 12

HORIZON_SHORT = "pendek (< 3 bulan)"
HORIZON_MEDIUM = "menengah (3-12 bulan)"
HORIZON_LONG = "panjang (1-2 tahun)"
HORIZON_BEYOND_PLAN = "melebihi horizon panjang (> 2 tahun)"


def classify_horizon(holding_months: float) -> str:
    if holding_months < 0:
        raise ValueError("Lama holding tidak boleh negatif")
    if holding_months < HORIZON_SHORT_MAX_MONTHS:
        return HORIZON_SHORT
    if holding_months < HORIZON_MEDIUM_MAX_MONTHS:
        return HORIZON_MEDIUM
    if holding_months <= HORIZON_LONG_MAX_MONTHS:
        return HORIZON_LONG
    return HORIZON_BEYOND_PLAN


@dataclass
class Condition:
    label: str
    met: bool
    detail: str


@dataclass
class PositionStatus:
    horizon: str
    conditions: list[Condition]

    @property
    def active(self) -> list[Condition]:
        return [condition for condition in self.conditions if condition.met]


def evaluate_position(
    current_price: float,
    max_buy: float,
    fair_price: float,
    is_cheap: bool,
    holding_months: float,
    fundamentals_growing: bool,
    growth_on_track: bool,
) -> PositionStatus:
    """Periksa empat kondisi bagian F terhadap satu posisi/kandidat.

    Beberapa kondisi bisa aktif sekaligus (mis. "hold" dan "tinjau
    rebalancing") - itu memang informasi untuk pemilik, bukan kontradiksi.
    `fundamentals_growing` dan `growth_on_track` adalah penilaian manual dari
    laporan keuangan terakhir; screener tidak menghitungnya.
    """
    accumulation = current_price <= max_buy and is_cheap
    profit_taking = fair_price > 0 and current_price >= fair_price
    rebalance_review = holding_months >= REBALANCE_REVIEW_MIN_MONTHS and not growth_on_track

    return PositionStatus(
        horizon=classify_horizon(holding_months),
        conditions=[
            Condition(
                label="Zona akumulasi (spec: BUY / average down)",
                met=accumulation,
                detail="harga ≤ max buy DAN valuasi murah",
            ),
            Condition(
                label="Zona pertimbangkan profit taking",
                met=profit_taking,
                detail="harga ≥ harga wajar",
            ),
            Condition(
                label="Kondisi hold",
                met=fundamentals_growing,
                detail="fundamental masih bertumbuh (penilaian manual)",
            ),
            Condition(
                label="Tinjau rebalancing / exit",
                met=rebalance_review,
                detail=f"holding ≥ {REBALANCE_REVIEW_MIN_MONTHS} bulan DAN growth tidak on-track",
            ),
        ],
    )
