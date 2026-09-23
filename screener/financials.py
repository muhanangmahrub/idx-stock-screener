"""Hitung rasio fundamental langsung dari komponen laporan keuangan mentah.

Mengacu pada e-book "Metode Analisis Fundamental" (Teguh Hidayat). Modul ini
PELENGKAP `fundamental.py`: di sini rasio dihitung sendiri dari komponen LK,
lalu hasilnya dipakai oleh threshold & valuasi yang sudah ada di sana.
Alasannya praktis: rasio jadi dari yfinance untuk saham IDX sering bolong.

Rumus (dokumen):
- ROE = laba bersih / ekuitas
- ROA = laba bersih / aset
- PER = harga / EPS       (EPS terdilusi, bukan basic)
- PBV = market cap / ekuitas
- market cap = harga x jumlah saham
- EDR = ekuitas / kewajiban
- EER = saldo laba / ekuitas
- EAR = ekuitas / aset

Anualisasi laba dari LK kuartalan (dokumen): rasio laba hanya valid kalau
labanya disetahunkan sesuai periode laporan - Q1 x4, half-year x2, Q3 x4/3,
full year x1.

Acuan kebenaran (UNVR 1H10 di dokumen): EDR 48,6%; EER 94,6%; EAR 32,7%;
PBV 43,3x; PER 37x; market cap Rp130.855 miliar; ROE disetahunkan 117,3%.

Catatan sumber: dokumen sedikit tidak konsisten soal ekuitas (memakai
Rp3.019 miliar untuk ROE/PBV, bukan Rp3.192 miliar di tabel). Kode & test
memakai angka yang mereproduksi hasil akhir dokumen.

Batas otomasi: aturan kualitas laba & neraca di sini menghasilkan WARNING
untuk ditinjau pemilik, bukan vonis. Hanya kondisi tolak mutlak yang
bersifat menggugurkan. Ini alat bantu, bukan rekomendasi jual/beli.
"""

from dataclasses import dataclass, field

# Faktor anualisasi laba per periode laporan (angka dokumen).
PERIOD_Q1 = "Q1"
PERIOD_HALF_YEAR = "1H"
PERIOD_Q3 = "Q3"
PERIOD_FULL_YEAR = "FY"
ANNUALIZATION_FACTORS = {
    PERIOD_Q1: 4.0,
    PERIOD_HALF_YEAR: 2.0,
    PERIOD_Q3: 4 / 3,
    PERIOD_FULL_YEAR: 1.0,
}

# Kondisi tolak mutlak (dokumen): "jangan beli tanpa toleransi".
REJECT_NEGATIVE_RETAINED_EARNINGS = "Saldo laba negatif"
REJECT_NEGATIVE_EQUITY = "Ekuitas negatif"
REJECT_NEGATIVE_NET_INCOME = "Laba bersih negatif"
REJECT_NEGATIVE_PER = "PER negatif"
REJECT_NEGATIVE_PBV = "PBV negatif"

# ASUMSI (bukan angka dokumen) untuk aturan kualitas neraca - wajib dikalibrasi.
DEFAULT_MAX_GOODWILL_PCT_OF_ASSETS = 0.20
DEFAULT_MAX_INTEREST_BEARING_PCT_OF_LIABILITIES = 0.50
DEFAULT_MIN_CASH_PCT_OF_ASSETS = 0.05
DEFAULT_MIN_EER_FOR_HEALTHY_EQUITY = 0.50


@dataclass
class Ratios:
    """Rasio hasil hitung; None bila komponennya tidak tersedia/nol."""

    net_income_annualized: float
    market_cap: float | None = None
    eps_annualized: float | None = None
    roe: float | None = None  # laba bersih disetahunkan / ekuitas
    roa: float | None = None
    per: float | None = None
    pbv: float | None = None
    edr: float | None = None  # ekuitas / kewajiban
    eer: float | None = None  # saldo laba / ekuitas
    ear: float | None = None  # ekuitas / aset

    @property
    def roe_pct(self) -> float | None:
        return None if self.roe is None else self.roe * 100


def annualization_factor(period: str) -> float:
    """Faktor penyetahunan laba sesuai periode laporan keuangan."""
    try:
        return ANNUALIZATION_FACTORS[period]
    except KeyError:
        raise ValueError(
            f"period harus salah satu dari {sorted(ANNUALIZATION_FACTORS)}"
        ) from None


def annualize_profit(profit: float, period: str) -> float:
    """Setahunkan laba satu periode laporan (Q1 x4, 1H x2, Q3 x4/3, FY x1)."""
    return profit * annualization_factor(period)


def _safe_div(numerator: float | None, denominator: float | None) -> float | None:
    """Bagi yang mengembalikan None kalau penyebutnya nol/None - supaya rasio
    yang tidak bisa dihitung tampil kosong, bukan mengarang angka."""
    if numerator is None or not denominator:
        return None
    return numerator / denominator


def compute_ratios(
    net_income: float,
    period: str = PERIOD_FULL_YEAR,
    equity: float | None = None,
    total_assets: float | None = None,
    total_liabilities: float | None = None,
    retained_earnings: float | None = None,
    shares_outstanding: float | None = None,
    price: float | None = None,
    eps_diluted: float | None = None,
    fx_rate: float = 1.0,
) -> Ratios:
    """Rasio Teguh Hidayat dari komponen LK mentah.

    `net_income` adalah laba periode laporan (belum disetahunkan); periode
    menentukan faktor penyetahunan. `eps_diluted` juga laba per saham periode
    itu - kalau tidak diberikan, dihitung dari laba bersih / jumlah saham.

    `fx_rate` untuk LK berdenominasi asing (mis. USD): ekuitas, aset,
    kewajiban, saldo laba, laba bersih dan EPS dikalikan kurs supaya sebanding
    dengan harga saham dalam Rupiah. Rasio yang kedua sisinya sama-sama dari
    LK (EDR/EER/EAR/ROE/ROA) tidak terpengaruh kurs, tapi tetap dikonversi
    demi konsistensi angka yang ditampilkan.
    """
    if fx_rate <= 0:
        raise ValueError("fx_rate harus positif")

    def to_rupiah(value: float | None) -> float | None:
        return None if value is None else value * fx_rate

    equity = to_rupiah(equity)
    total_assets = to_rupiah(total_assets)
    total_liabilities = to_rupiah(total_liabilities)
    retained_earnings = to_rupiah(retained_earnings)
    net_income_annualized = annualize_profit(to_rupiah(net_income), period)

    # Kewajiban boleh diturunkan dari aset - ekuitas bila tidak diberikan.
    if total_liabilities is None and total_assets is not None and equity is not None:
        total_liabilities = total_assets - equity

    market_cap = price * shares_outstanding if price is not None and shares_outstanding else None

    if eps_diluted is not None:
        eps_annualized = annualize_profit(to_rupiah(eps_diluted), period)
    elif shares_outstanding:
        eps_annualized = net_income_annualized / shares_outstanding
    else:
        eps_annualized = None

    return Ratios(
        net_income_annualized=net_income_annualized,
        market_cap=market_cap,
        eps_annualized=eps_annualized,
        roe=_safe_div(net_income_annualized, equity),
        roa=_safe_div(net_income_annualized, total_assets),
        per=_safe_div(price, eps_annualized),
        pbv=_safe_div(market_cap, equity),
        edr=_safe_div(equity, total_liabilities),
        eer=_safe_div(retained_earnings, equity),
        ear=_safe_div(equity, total_assets),
    )


def absolute_rejects(
    retained_earnings: float | None = None,
    equity: float | None = None,
    net_income: float | None = None,
    per: float | None = None,
    pbv: float | None = None,
) -> list[str]:
    """Kondisi tolak mutlak dokumen: "jangan beli tanpa toleransi".

    Daftar kosong berarti tidak ada penolakan - bukan berarti layak beli.
    Nilai None dilewati: tidak diketahui bukan berarti negatif.
    """
    rejects = []
    if retained_earnings is not None and retained_earnings < 0:
        rejects.append(REJECT_NEGATIVE_RETAINED_EARNINGS)
    if equity is not None and equity < 0:
        rejects.append(REJECT_NEGATIVE_EQUITY)
    if net_income is not None and net_income < 0:
        rejects.append(REJECT_NEGATIVE_NET_INCOME)
    if per is not None and per < 0:
        rejects.append(REJECT_NEGATIVE_PER)
    if pbv is not None and pbv < 0:
        rejects.append(REJECT_NEGATIVE_PBV)
    return rejects


@dataclass
class GrowthQuality:
    """Penilaian kualitas pertumbuhan laba; `warnings` untuk ditinjau manual."""

    sales_growth: float | None
    operating_growth: float | None
    net_growth: float | None
    is_ideal: bool = False  # ketiganya naik & berurutan sales < operating < net
    warnings: list[str] = field(default_factory=list)


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or not previous:
        return None
    return (current - previous) / abs(previous)


def assess_growth_quality(
    sales: float | None,
    sales_previous: float | None,
    operating_profit: float | None,
    operating_profit_previous: float | None,
    net_income: float | None,
    net_income_previous: float | None,
) -> GrowthQuality:
    """Aturan kualitas laba dokumen.

    Ideal: penjualan, laba usaha, dan laba bersih sama-sama naik, lebih baik
    berurutan dari kecil ke besar (UNVR: sales +10,8% < operating +13,9% <
    net +18,4%) - artinya efisiensi ikut membaik, bukan sekadar jualan lebih
    banyak. Warning bila laba bersih naik sementara penjualan/laba usaha
    turun (labanya dari pos non-operasional, tidak berkelanjutan), dan
    warning lebih keras bila ketiganya turun.
    """
    sales_growth = _growth(sales, sales_previous)
    operating_growth = _growth(operating_profit, operating_profit_previous)
    net_growth = _growth(net_income, net_income_previous)
    known = [g for g in (sales_growth, operating_growth, net_growth) if g is not None]

    quality = GrowthQuality(sales_growth, operating_growth, net_growth)
    if not known:
        quality.warnings.append("Data pertumbuhan tidak lengkap - tidak bisa dinilai")
        return quality

    all_known = len(known) == 3
    all_up = all(g > 0 for g in known)
    all_down = all(g < 0 for g in known)

    if all_known and all_down:
        quality.warnings.append(
            "Penjualan, laba usaha, dan laba bersih semuanya turun - "
            "kinerja melemah menyeluruh"
        )
        return quality

    if net_growth is not None and net_growth > 0:
        weak = [
            name
            for name, growth in (
                ("penjualan", sales_growth),
                ("laba usaha", operating_growth),
            )
            if growth is not None and growth < 0
        ]
        if weak:
            quality.warnings.append(
                f"Laba bersih naik tapi {' dan '.join(weak)} turun - kemungkinan "
                "dari pos non-operasional, periksa apakah berkelanjutan"
            )

    if all_known and all_up:
        if sales_growth < operating_growth < net_growth:
            quality.is_ideal = True
        else:
            quality.warnings.append(
                "Ketiganya naik tapi tidak berurutan sales < operating < net - "
                "efisiensi belum tentu ikut membaik"
            )
    return quality


def assess_balance_quality(
    total_assets: float | None = None,
    current_assets: float | None = None,
    non_current_assets: float | None = None,
    cash: float | None = None,
    goodwill_and_other_assets: float | None = None,
    total_liabilities: float | None = None,
    interest_bearing_debt: float | None = None,
    equity: float | None = None,
    eer: float | None = None,
    max_goodwill_pct: float = DEFAULT_MAX_GOODWILL_PCT_OF_ASSETS,
    max_interest_bearing_pct: float = DEFAULT_MAX_INTEREST_BEARING_PCT_OF_LIABILITIES,
    min_cash_pct: float = DEFAULT_MIN_CASH_PCT_OF_ASSETS,
    min_eer: float = DEFAULT_MIN_EER_FOR_HEALTHY_EQUITY,
) -> list[str]:
    """Aturan kualitas neraca dokumen - WARNING, bukan penggugur.

    Yang diperiksa: porsi goodwill/"aset lain-lain" terlalu besar; utang
    berbunga (bank, obligasi, senior notes) mendominasi kewajiban - lebih
    berbahaya daripada utang operasional karena ada bunga yang harus dibayar
    apa pun keadaannya; kas terlalu kecil; aset lancar lebih kecil daripada
    aset tak lancar; ekuitas besar tapi EER kecil (ekuitasnya dari right
    issue, bukan akumulasi laba - contoh UNSP di dokumen).

    Ambang persentasenya ASUMSI (tidak ada angkanya di dokumen) dan bisa
    diatur lewat parameter.
    """
    warnings = []
    goodwill_pct = _safe_div(goodwill_and_other_assets, total_assets)
    if goodwill_pct is not None and goodwill_pct > max_goodwill_pct:
        warnings.append(
            f"Goodwill & aset lain-lain {goodwill_pct:.1%} dari aset "
            f"(> {max_goodwill_pct:.0%}) - aset yang sulit diverifikasi nilainya"
        )

    debt_pct = _safe_div(interest_bearing_debt, total_liabilities)
    if debt_pct is not None and debt_pct > max_interest_bearing_pct:
        warnings.append(
            f"Utang berbunga {debt_pct:.1%} dari kewajiban "
            f"(> {max_interest_bearing_pct:.0%}) - lebih berat daripada utang "
            "operasional karena bunganya jalan terus"
        )

    cash_pct = _safe_div(cash, total_assets)
    if cash_pct is not None and cash_pct < min_cash_pct:
        warnings.append(
            f"Kas {cash_pct:.1%} dari aset (< {min_cash_pct:.0%}) - ruang napas "
            "untuk kewajiban jangka pendek tipis"
        )

    if current_assets is not None and non_current_assets is not None:
        if current_assets < non_current_assets:
            warnings.append(
                "Aset lancar lebih kecil daripada aset tak lancar - periksa "
                "kemampuan memenuhi kewajiban jangka pendek"
            )

    if eer is not None and equity is not None and eer < min_eer:
        warnings.append(
            f"EER {eer:.1%} (< {min_eer:.0%}) - ekuitas lebih banyak dari setoran "
            "modal/right issue daripada akumulasi laba"
        )
    return warnings
