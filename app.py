"""Entry point Streamlit — hanya UI, logika ada di modul screener/."""

import pandas as pd
import streamlit as st

from screener.checklist import ANSWER_OPTIONS, QUALITATIVE_CHECKLIST, summarize_checklist
from screener.data import (
    get_free_float_pct,
    get_fundamental_data,
    get_idx_official_data,
    get_price_history,
)
from screener.execution import evaluate_position
from screener.extrema import get_extrema
from screener.formatting import format_rupiah, format_rupiah_compact
from screener.financials import (
    ANNUALIZATION_FACTORS,
    PERIOD_FULL_YEAR,
    PERIOD_HALF_YEAR,
    PERIOD_Q1,
    PERIOD_Q3,
    absolute_rejects,
    assess_balance_quality,
    assess_growth_quality,
    compute_ratios,
)
from screener.fundamental import (
    DEFAULT_MARGIN_OF_SAFETY,
    MAX_MARGIN_OF_SAFETY,
    MIN_MARGIN_OF_SAFETY,
    compute_position_limit,
    compute_valuation,
    is_cheap_valuation,
    passes_initial_screening,
)
from screener.idx_data import get_stock_summary
from screener.patterns import scan
from screener.fan import (
    FAN_BEARISH,
    FAN_BULLISH,
    FAN_LINE_COUNT,
    detect_fan,
)
from screener.channel import (
    CHANNEL_INTACT,
    build_channel,
    check_channel_break,
)
from screener.breakout import (
    BREAKOUT_TOLERANCE,
    CUT_LOSS_TOLERANCE,
    DEFAULT_TEST_TOLERANCE,
    STRONG_RESISTANCE_MIN_TESTS,
    build_trading_plan,
    count_resistance_tests,
    evaluate_breakout_position,
    find_breakout,
    is_strong_resistance,
)
from screener.levels import RESISTANCE, levels_from_swings
from screener.plotting import (
    add_break_markers,
    add_breakout_plan,
    add_channel,
    add_fan,
    add_levels,
    add_trendline,
    plot_candlestick,
)
from screener.portfolio import Position, review_portfolio
from screener.trend import (
    CONFIRM_FULL_BREAK,
    DOWNTREND,
    CONFIRM_HALF_WAY,
    DEFAULT_BREAK_TOLERANCE,
    DEFAULT_LOOKBACK_SWINGS,
    DEFAULT_TOLERANCE,
    MIN_SWINGS_PER_SIDE,
    UNDEFINED,
    SECOND_DAY_CONFIRMED,
    UPTREND,
    VALID_BREAK,
    break_threshold,
    build_trendline,
    check_trendline_break,
    classify_trend,
    select_anchor_points,
)
from screener.universe import (
    STATUS_PASSED,
    STATUS_REJECTED,
    apply_sector_comparison,
    candidates_to_dataframe,
    filter_liquid_stocks,
    screen_universe,
)

st.set_page_config(page_title="IDX Stock Screener", layout="wide")

# Ringkasan Saham IDX memuat semua emiten dalam satu request; cukup diambil
# sekali per jam, bukan tiap klik.
load_stock_summary = st.cache_data(ttl=3600, show_spinner=False)(get_stock_summary)

# Fetcher per emiten untuk screening massal. Di-cache supaya menjalankan ulang
# (mis. ganti blacklist) tidak mengulang ratusan request.
load_free_float_pct = st.cache_data(ttl=86400, show_spinner=False)(get_free_float_pct)
load_fundamental_data = st.cache_data(ttl=3600, show_spinner=False)(get_fundamental_data)

# Field input yang bisa terisi otomatis dari sumber data. Widget ber-key
# mengabaikan perubahan `value=`, jadi prefill ditulis ke session_state.
PREFILL_KEYS = {
    "daily_transaction_value": "in_daily_transaction_value",
    "free_float_pct": "in_free_float_pct",
    "roe_annualized_pct": "in_roe_annualized_pct",
    "book_value_per_share": "in_bvps",
    "current_price": "in_current_price",
    "per": "in_per",
    "pbv": "in_pbv",
    # Komponen LK mentah untuk bagian Analisis Laporan Keuangan.
    "equity": "in_lk_equity",
    "total_assets": "in_lk_assets",
    "total_liabilities": "in_lk_liabilities",
    "retained_earnings": "in_lk_retained",
    "net_income": "in_lk_net_income",
    "sales": "in_lk_sales",
    "operating_profit": "in_lk_operating",
}


def ratio_text(value: float | None, suffix: str = "") -> str:
    """Angka rasio untuk st.metric; tanda strip bila tidak bisa dihitung."""
    return "—" if value is None else f"{value:,.2f}{suffix}"


def apply_prefill(data: dict) -> None:
    for field, widget_key in PREFILL_KEYS.items():
        if data.get(field) is not None:
            st.session_state[widget_key] = float(data[field])


def rupiah_input(label: str, key: str, step: float, help: str | None = None) -> float:
    """number_input Rupiah dengan nilai terformat di label.

    number_input tidak bisa menampilkan pemisah ribuan (format-nya sprintf.js),
    jadi angka terformat disisipkan ke label; label ikut berubah tiap rerun.
    """
    st.session_state.setdefault(key, 0.0)
    return st.number_input(
        f"{label} · {format_rupiah(st.session_state[key])}",
        key=key,
        min_value=0.0,
        step=step,
        format="%.0f",
        help=help,
    )


def big_rupiah_metric(container, label: str, value: float) -> None:
    """Metric untuk nominal besar: tampil ringkas, angka penuh di tooltip."""
    container.metric(label, format_rupiah_compact(value), help=format_rupiah(value))


def ratio_input(label: str, key: str, help: str | None = None, **kwargs) -> float:
    st.session_state.setdefault(key, 0.0)
    return st.number_input(label, key=key, format="%.2f", help=help, **kwargs)


# --- Sidebar: ticker & sumber data ---
with st.sidebar:
    st.title("IDX Stock Screener")
    ticker = st.text_input("Kode saham", value="BBCA.JK", help="Format yfinance: XXXX.JK")

    st.subheader("Sumber Data")
    if st.button("Ambil dari yfinance", key="fetch_fundamental", width="stretch"):
        with st.spinner("Mengambil data yfinance..."):
            data = get_fundamental_data(ticker)
        if data:
            st.session_state["fundamental_data"] = data
            apply_prefill(data)
        else:
            st.session_state.pop("fundamental_data", None)
            st.error(f"Data {ticker} tidak ditemukan di yfinance.")

    if st.button(
        "Ambil data resmi IDX",
        key="fetch_idx",
        width="stretch",
        help="Free float (komposisi pemegang saham) & nilai transaksi (Ringkasan "
        "Saham) dari idx.co.id. Menimpa angka yfinance untuk dua field ini.",
    ):
        with st.spinner("Mengambil data idx.co.id..."):
            idx_data = get_idx_official_data(ticker, load_stock_summary())
        if idx_data:
            st.session_state["idx_data"] = idx_data
            apply_prefill(idx_data)
        else:
            st.session_state.pop("idx_data", None)
            st.error("Data IDX tidak didapat (emiten tidak ditemukan / akses ditolak).")

    fetched = st.session_state.get("fundamental_data") or {}
    idx_official = st.session_state.get("idx_data") or {}
    if fetched:
        st.caption(f"yfinance: {fetched['ticker']} terisi.")
        for note in fetched.get("data_notes") or []:
            st.warning(note)
    if idx_official:
        st.caption(f"IDX resmi: {idx_official['ticker']} per {idx_official['date']}.")

    st.divider()
    st.caption(
        "Alat bantu screening, **bukan** rekomendasi jual/beli. Semua angka "
        "otomatis adalah estimasi yang wajib diverifikasi manual."
    )

st.warning(
    "Alat bantu screening, **bukan** robot trading atau rekomendasi jual/beli. "
    "Setiap kandidat yang muncul di sini wajib diverifikasi manual sebelum "
    "keputusan investasi."
)

tab_fundamental, tab_massal, tab_teknikal, tab_portofolio, tab_checklist = st.tabs(
    [
        "Lapis 1 — Fundamental",
        "Screening Massal",
        "Lapis 2 — Teknikal",
        "Portofolio",
        "Checklist & Eksekusi",
    ]
)

# --- Lapis 1: Fundamental (value investing, metode Teguh Hidayat) ---
with tab_fundamental:
    if fetched or idx_official:
        with st.container(border=True):
            st.markdown(
                f"**{ticker}** · {fetched.get('sector') or 'sektor tidak diketahui'}"
            )
            m = st.columns(5)
            m[0].metric("Harga", format_rupiah(fetched.get("current_price") or 0))
            m[1].metric("BVPS", format_rupiah(fetched.get("book_value_per_share") or 0))
            big_rupiah_metric(m[2], "Market cap", fetched.get("market_cap") or 0)
            m[3].metric("ROE TTM", f"{(fetched.get('roe_ttm') or 0) * 100:.1f}%")
            free_float_source = "IDX" if idx_official.get("free_float_pct") is not None else "yfinance"
            free_float_value = idx_official.get("free_float_pct") or fetched.get("free_float_pct") or 0
            m[4].metric(f"Free float ({free_float_source})", f"{free_float_value:.1f}%")
    else:
        st.info("Ambil data lewat sidebar, atau isi angka di bawah secara manual.")

    with st.expander("Analisis Laporan Keuangan (hitung rasio dari komponen mentah)"):
        st.caption(
            "Rasio dihitung sendiri dari komponen LK — berguna saat angka jadi dari "
            "yfinance bolong. Rumus e-book Metode Analisis Fundamental: ROE = laba "
            "bersih/ekuitas, ROA = laba bersih/aset, PER = harga/EPS terdilusi, "
            "PBV = market cap/ekuitas, EDR = ekuitas/kewajiban, EER = saldo "
            "laba/ekuitas, EAR = ekuitas/aset. Laba disetahunkan sesuai periode LK "
            "(Q1 ×4, 1H ×2, Q3 ×4/3, FY ×1). Angka terisi otomatis dari yfinance bila "
            "data sudah diambil; boleh ditimpa manual dari LK asli."
        )

        c = st.columns(3)
        with c[0]:
            lk_period = st.selectbox(
                "Periode laporan",
                list(ANNUALIZATION_FACTORS),
                index=list(ANNUALIZATION_FACTORS).index(PERIOD_Q1),
                help="Menentukan faktor penyetahunan laba sebelum rasio dihitung.",
            )
        with c[1]:
            lk_currency = st.selectbox(
                "Mata uang LK",
                ["IDR", "USD"],
                index=0 if (fetched.get("financial_currency") or "IDR") == "IDR" else 1,
            )
        with c[2]:
            fx_rate = st.number_input(
                "Kurs ke Rupiah",
                min_value=1.0,
                value=1.0 if lk_currency == "IDR" else 16_000.0,
                step=100.0,
                disabled=lk_currency == "IDR",
                help="Dipakai untuk mengonversi ekuitas & EPS sebelum menghitung "
                "PER/PBV saat LK berdenominasi asing.",
            )

        step = 1_000_000_000.0
        c = st.columns(4)
        with c[0]:
            lk_net_income = rupiah_input(
                "Laba bersih (periode LK)", key="in_lk_net_income", step=step
            )
        with c[1]:
            lk_equity = rupiah_input("Ekuitas", key="in_lk_equity", step=step)
        with c[2]:
            lk_assets = rupiah_input("Total aset", key="in_lk_assets", step=step)
        with c[3]:
            lk_liabilities = rupiah_input(
                "Total kewajiban", key="in_lk_liabilities", step=step
            )
        c = st.columns(4)
        with c[0]:
            lk_retained = rupiah_input("Saldo laba", key="in_lk_retained", step=step)
        with c[1]:
            lk_sales = rupiah_input("Penjualan", key="in_lk_sales", step=step)
        with c[2]:
            lk_operating = rupiah_input("Laba usaha", key="in_lk_operating", step=step)
        with c[3]:
            lk_shares = st.number_input(
                "Jumlah saham (modal disetor penuh)",
                min_value=0.0,
                value=float(fetched.get("shares_outstanding") or 0),
                step=1_000_000.0,
                format="%.0f",
            )
        c = st.columns(2)
        with c[0]:
            lk_price = st.number_input(
                "Harga saham",
                min_value=0.0,
                value=float(fetched.get("current_price") or 0),
                step=25.0,
            )
        with c[1]:
            lk_eps = st.number_input(
                "EPS terdilusi (periode LK, 0 = hitung dari laba)",
                min_value=0.0,
                value=float(fetched.get("eps_diluted") or 0),
                step=1.0,
                help="Dokumen memakai EPS terdilusi, bukan basic.",
            )

        if lk_net_income or lk_equity:
            ratios = compute_ratios(
                net_income=lk_net_income,
                period=lk_period,
                equity=lk_equity or None,
                total_assets=lk_assets or None,
                total_liabilities=lk_liabilities or None,
                retained_earnings=lk_retained or None,
                shares_outstanding=lk_shares or None,
                price=lk_price or None,
                eps_diluted=lk_eps or None,
                fx_rate=fx_rate if lk_currency != "IDR" else 1.0,
            )
            m = st.columns(4)
            m[0].metric("ROE disetahunkan", ratio_text(ratios.roe_pct, "%"))
            m[1].metric("ROA", ratio_text(None if ratios.roa is None else ratios.roa * 100, "%"))
            m[2].metric("PER", ratio_text(ratios.per, "x"))
            m[3].metric("PBV", ratio_text(ratios.pbv, "x"))
            m = st.columns(4)
            m[0].metric("EDR (ekuitas/kewajiban)", ratio_text(None if ratios.edr is None else ratios.edr * 100, "%"))
            m[1].metric("EER (saldo laba/ekuitas)", ratio_text(None if ratios.eer is None else ratios.eer * 100, "%"))
            m[2].metric("EAR (ekuitas/aset)", ratio_text(None if ratios.ear is None else ratios.ear * 100, "%"))
            big_rupiah_metric(m[3], "Market cap", ratios.market_cap or 0)

            with st.container(border=True):
                st.caption(
                    "Kualitas pertumbuhan (opsional): isi angka periode yang sama "
                    "tahun lalu. Ideal menurut e-book: penjualan, laba usaha, dan "
                    "laba bersih sama-sama naik dan berurutan dari kecil ke besar "
                    "(UNVR: +10,8% < +13,9% < +18,4%)."
                )
                c = st.columns(3)
                with c[0]:
                    prev_sales = rupiah_input(
                        "Penjualan tahun lalu", key="in_lk_sales_prev", step=step
                    )
                with c[1]:
                    prev_operating = rupiah_input(
                        "Laba usaha tahun lalu", key="in_lk_operating_prev", step=step
                    )
                with c[2]:
                    prev_net = rupiah_input(
                        "Laba bersih tahun lalu", key="in_lk_net_prev", step=step
                    )
                if prev_sales or prev_operating or prev_net:
                    growth = assess_growth_quality(
                        sales=lk_sales or None, sales_previous=prev_sales or None,
                        operating_profit=lk_operating or None,
                        operating_profit_previous=prev_operating or None,
                        net_income=lk_net_income or None, net_income_previous=prev_net or None,
                    )
                    g = st.columns(3)
                    g[0].metric(
                        "Penjualan",
                        ratio_text(None if growth.sales_growth is None else growth.sales_growth * 100, "%"),
                    )
                    g[1].metric(
                        "Laba usaha",
                        ratio_text(None if growth.operating_growth is None else growth.operating_growth * 100, "%"),
                    )
                    g[2].metric(
                        "Laba bersih",
                        ratio_text(None if growth.net_growth is None else growth.net_growth * 100, "%"),
                    )
                    if growth.is_ideal:
                        st.success(
                            "Pola pertumbuhan ideal: penjualan < laba usaha < laba bersih "
                            "— efisiensi ikut membaik. Tetap tinjau manual."
                        )
                    for warning in growth.warnings:
                        st.warning(warning)

            rejects = absolute_rejects(
                retained_earnings=lk_retained or None,
                equity=lk_equity or None,
                net_income=lk_net_income or None,
                per=ratios.per,
                pbv=ratios.pbv,
            )
            if rejects:
                st.error(
                    "**Ditolak mutlak** (e-book: jangan beli tanpa toleransi): "
                    + "; ".join(rejects)
                )

            balance_warnings = assess_balance_quality(
                total_assets=lk_assets or None,
                cash=float(fetched.get("cash") or 0) or None,
                total_liabilities=lk_liabilities or None,
                equity=lk_equity or None,
                eer=ratios.eer,
            )
            for warning in balance_warnings:
                st.warning(warning)
            if not rejects and not balance_warnings:
                st.caption(
                    "Tidak ada kondisi tolak mutlak maupun peringatan neraca dari "
                    "angka yang diisi — bukan berarti layak beli, tetap tinjau manual."
                )

    with st.container(border=True):
        st.markdown("**Screening Awal**")
        c = st.columns(3)
        with c[0]:
            daily_transaction_value = rupiah_input(
                "Nilai transaksi harian",
                key="in_daily_transaction_value",
                step=1_000_000_000.0,
                help="IDX: nilai pasar reguler hari bursa terakhir. yfinance: "
                "estimasi rata-rata Close x Volume 20 hari.",
            )
        with c[1]:
            free_float_pct = ratio_input(
                "Free float (%)",
                key="in_free_float_pct",
                min_value=0.0,
                max_value=100.0,
                help="IDX: total kategori Masyarakat di komposisi pemegang saham. "
                "yfinance: estimasi floatShares / sharesOutstanding.",
            )
        with c[2]:
            roe_annualized_pct = ratio_input(
                "ROE disetahunkan (%)",
                key="in_roe_annualized_pct",
                help="Laba bersih kuartal terakhir x 4 / ekuitas terakhir. Cara "
                "anualisasi ini asumsi — bandingkan dengan ROE TTM.",
            )
        is_blacklisted = st.checkbox("Masuk blacklist manual (saham bandar/gorengan)")

    with st.container(border=True):
        st.markdown("**Valuasi**")
        c = st.columns(3)
        with c[0]:
            per = ratio_input("PER", key="in_per")
            bvps = rupiah_input("BVPS", key="in_bvps", step=1.0)
        with c[1]:
            pbv = ratio_input("PBV", key="in_pbv")
            current_price = rupiah_input("Harga saat ini", key="in_current_price", step=1.0)
        with c[2]:
            risk_factor_count = st.number_input(
                "Faktor risiko makro/sektoral",
                min_value=0,
                value=0,
                help=(
                    "Ancaman dari **luar** perusahaan yang tidak tertangkap angka "
                    "keuangan — mis. harga komoditas turun, suku bunga naik, "
                    "regulasi berubah. Tiap faktor yang menurutmu relevan "
                    "memotong harga wajar **10%**:\n\n"
                    "`harga wajar adj = harga wajar dasar × (1 − 0,10 × jumlah faktor)`\n\n"
                    "Faktor mana yang dihitung adalah judgment manual — tidak ada "
                    "daftar baku di sumber."
                ),
            )
            margin_of_safety = st.slider(
                "Margin of safety",
                min_value=MIN_MARGIN_OF_SAFETY,
                max_value=MAX_MARGIN_OF_SAFETY,
                value=DEFAULT_MARGIN_OF_SAFETY,
                step=0.05,
                format="%.2f",
                help=(
                    "Jarak aman di bawah harga wajar untuk menampung **kesalahan "
                    "analisis** yang tidak kamu sadari (asumsi ROE meleset, laporan "
                    "direvisi, dsb.). Semakin tidak yakin, semakin besar:\n\n"
                    "`best buy = harga wajar adj × (1 − MoS)`\n\n"
                    "`max buy = best buy × 1,15`\n\n"
                    "MoS 0,35 berarti best buy = 65% dari harga wajar. Rentang "
                    "35–50% sesuai spesifikasi."
                ),
            )
        is_blue_chip = st.checkbox(
            "Klasifikasikan sebagai blue chip",
            help="Menentukan batas PER (12 vs 8). Tidak ada aturan otomatis di "
            "sumber — keputusan manual.",
        )

    with st.container(border=True):
        st.markdown("**Batas Posisi**")
        liquid_net_worth = rupiah_input(
            "Liquid net worth investor",
            key="in_liquid_net_worth",
            step=10_000_000.0,
            help="Batas alokasi per saham = maks 20% dari nilai ini.",
        )

    if st.button("Jalankan screening fundamental", type="primary", width="stretch"):
        # Disimpan ke session_state supaya hasil tetap tampil saat widget lain
        # (mis. input di tab Checklist & Eksekusi) memicu rerun.
        st.session_state["screening"] = {
            "ticker": ticker,
            "result": passes_initial_screening(
                ticker=ticker,
                daily_transaction_value=daily_transaction_value,
                free_float_pct=free_float_pct,
                roe_annualized_pct=roe_annualized_pct,
                is_blacklisted=is_blacklisted,
            ),
            "cheap": is_cheap_valuation(per=per, pbv=pbv, is_blue_chip=is_blue_chip),
            "valuation": (
                compute_valuation(
                    roe_annualized_pct=roe_annualized_pct,
                    bvps=bvps,
                    risk_factor_count=int(risk_factor_count),
                    margin_of_safety=margin_of_safety,
                )
                if bvps > 0
                else None
            ),
            "limit": (
                compute_position_limit(
                    daily_transaction_value=daily_transaction_value,
                    liquid_net_worth=liquid_net_worth,
                )
                if liquid_net_worth > 0
                else None
            ),
            "current_price": current_price,
        }

    screening = st.session_state.get("screening")
    if screening:
        result = screening["result"]
        cheap = screening["cheap"]
        valuation = screening["valuation"]
        limit = screening["limit"]

        st.markdown(f"**Hasil Screening {screening['ticker']}**")
        res = st.columns(2)
        with res[0]:
            if result.passed:
                st.success("Lolos screening awal (likuiditas, free float, ROE).")
            else:
                st.error("Tidak lolos screening awal:\n\n" + "\n".join(f"- {r}" for r in result.reasons))
            for note in result.notes:
                st.info(note)
        with res[1]:
            if cheap:
                st.success("Valuasi murah menurut aturan PER/PBV.")
            else:
                st.warning("Valuasi belum murah menurut aturan PER/PBV.")

        with st.container(border=True):
            st.markdown("**Harga Wajar & Best Buy (Jalur Harga Absolut)**")
            if valuation is None:
                st.info("Isi BVPS untuk menghitung harga wajar.")
            else:
                m = st.columns(4)
                m[0].metric("PBV dasar", f"{valuation.pbv_base:.2f}x")
                m[1].metric("Harga wajar (adj)", format_rupiah(valuation.fair_price_adj))
                m[2].metric("Best buy", format_rupiah(valuation.best_buy))
                m[3].metric("Max buy", format_rupiah(valuation.max_buy))

                price_at_run = screening["current_price"]
                if price_at_run > 0:
                    if price_at_run <= valuation.max_buy:
                        st.success(
                            f"Harga saat ini {format_rupiah(price_at_run)} di bawah max buy "
                            f"{format_rupiah(valuation.max_buy)} — masuk zona untuk dianalisis "
                            "lebih lanjut (bukan rekomendasi beli)."
                        )
                    else:
                        st.warning(
                            f"Harga saat ini {format_rupiah(price_at_run)} di atas max buy "
                            f"{format_rupiah(valuation.max_buy)}."
                        )

        with st.container(border=True):
            st.markdown("**Batas Posisi per Emiten**")
            if limit is None:
                st.info("Isi liquid net worth untuk menghitung batas posisi.")
            else:
                m = st.columns(3)
                big_rupiah_metric(m[0], "30% nilai transaksi harian", limit.by_liquidity)
                big_rupiah_metric(m[1], "20% liquid net worth", limit.by_net_worth)
                big_rupiah_metric(m[2], "Batas yang mengikat", limit.max_position)

# --- Screening massal: bagian A + C dijalankan ke seluruh universe IDX ---
with tab_massal:
    st.caption(
        "Screening awal (likuiditas, free float, ROE) lalu aturan valuasi murah "
        "(PER/PBV) untuk semua emiten IDX, bertahap: likuiditas dari Ringkasan "
        "Saham (instan), free float dari idx.co.id, lalu ROE/PER/PBV dari yfinance "
        "hanya untuk yang masih bertahan. Semua emiten dinilai dengan batas "
        "small cap (PER ≤ 8 atau PBV ≤ 0,7) karena blue chip tidak bisa "
        "dibedakan otomatis; blue chip dengan PER 8–12 diberi catatan untuk "
        "dicek manual. Hasilnya daftar kandidat untuk dianalisis manual — "
        "bukan rekomendasi."
    )

    with st.container(border=True):
        c = st.columns([2, 1])
        with c[0]:
            blacklist_text = st.text_input(
                "Blacklist manual (kode dipisah koma)",
                placeholder="mis. ABCD, EFGH",
                help="Saham bandar/gorengan menurut penilaianmu; tidak dideteksi otomatis.",
            )
        with c[1]:
            max_candidates = st.number_input(
                "Batasi jumlah emiten (0 = semua)",
                min_value=0,
                value=0,
                help="Urut dari nilai transaksi terbesar. Berguna untuk uji coba "
                "cepat sebelum menjalankan ke ratusan emiten.",
            )
        blacklist = {
            code.strip().upper() for code in blacklist_text.split(",") if code.strip()
        }

    if st.button("Jalankan screening massal", type="primary", width="stretch"):
        with st.spinner("Mengambil Ringkasan Saham IDX..."):
            summary = load_stock_summary()
        liquid = filter_liquid_stocks(summary)
        if not liquid:
            st.error("Ringkasan Saham IDX tidak didapat (akses ditolak / endpoint berubah).")
        else:
            total_universe = len(summary)
            if max_candidates > 0:
                liquid = liquid[: int(max_candidates)]

            progress_bar = st.progress(0.0, text="Memulai...")

            def show_progress(done: int, total: int, code: str) -> None:
                progress_bar.progress(done / total, text=f"{done}/{total} · {code}")

            screened = screen_universe(
                liquid,
                fetch_free_float=load_free_float_pct,
                fetch_fundamental=load_fundamental_data,
                blacklist=blacklist,
                on_progress=show_progress,
            )
            progress_bar.empty()
            st.session_state["universe"] = {
                "total_universe": total_universe,
                "liquid_count": len(liquid),
                "date": str(summary["Date"].iloc[0])[:10],
                "results": screened,
            }

    universe = st.session_state.get("universe")
    if universe:
        results = universe["results"]
        passed = [c for c in results if c.status == STATUS_PASSED]

        m = st.columns(3)
        m[0].metric("Emiten di Ringkasan Saham", universe["total_universe"])
        m[1].metric("Lolos likuiditas (diproses)", universe["liquid_count"])
        m[2].metric("Lolos screening + valuasi murah", len(passed))
        st.caption(
            f"Data Ringkasan Saham per {universe['date']} (satu hari bursa). "
            "Angka free float/ROE/PER/PBV adalah estimasi otomatis — verifikasi "
            "manual sebelum masuk watchlist. PBV kosong berarti yfinance tidak "
            "memberi angka yang valid (lihat kolom catatan)."
        )

        rejected = [c for c in results if c.status == STATUS_REJECTED]
        if rejected:
            with st.expander(f"Ditolak Mutlak ({len(rejected)} emiten)"):
                st.caption(
                    "Kondisi tolak mutlak e-book Metode Analisis Fundamental: saldo "
                    "laba, ekuitas, laba bersih, PER, atau PBV negatif — 'jangan beli "
                    "tanpa toleransi'. Dipisahkan dari yang gagal threshold biasa."
                )
                st.dataframe(
                    candidates_to_dataframe(rejected), width="stretch", hide_index=True
                )

        c = st.columns(2)
        with c[0]:
            show_only_passed = st.checkbox("Tampilkan hanya yang lolos", value=True)
        with c[1]:
            use_sector_comparison = st.checkbox(
                "Bandingkan PER/PBV dengan median sektor",
                value=False,
                help="Jalur OPSIONAL dari e-book lama. Default proyek memakai jalur "
                "harga absolut/story (materi Teguh Hidayat yang lebih baru), karena "
                "tiap perusahaan punya cerita sendiri. Hasilnya hanya catatan "
                "tambahan, tidak mengubah status lolos.",
            )

        if use_sector_comparison:
            sector_stats = apply_sector_comparison(results)
            usable = {s: v for s, v in sector_stats.items() if v.per_median or v.pbv_median}
            if usable:
                st.dataframe(
                    pd.DataFrame(
                        [
                            {
                                "Sektor": v.sector,
                                "Emiten": v.count,
                                "PER Median": v.per_median,
                                "PER Rata-rata": v.per_mean,
                                "PBV Median": v.pbv_median,
                                "PBV Rata-rata": v.pbv_mean,
                            }
                            for v in sorted(usable.values(), key=lambda v: v.sector)
                        ]
                    ),
                    width="stretch",
                    hide_index=True,
                )
                st.caption(
                    "Median dipakai sebagai pembanding utama karena satu emiten "
                    "dengan PER ekstrem bisa menarik rata-rata sektor terlalu jauh."
                )
            else:
                st.info(
                    "Belum ada rasio sektor yang bisa dihitung dari hasil screening ini."
                )

        table = candidates_to_dataframe(passed if show_only_passed else results)
        if table.empty:
            st.info("Tidak ada emiten yang lolos screening awal + valuasi murah.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)

# --- Lapis 2: Teknikal (chart pattern, Edianto Ong) ---
with tab_teknikal:
    st.caption(
        "Deteksi swing high/low, lalu klasifikasi tren (uptrend / downtrend / "
        "sideways) dari urutan puncak & dasar sesuai definisi Edianto Ong. "
        "Detektor pola (double top, head & shoulders, dst.) belum "
        "diimplementasikan — menunggu spesifikasi aturan dari buku pemilik. "
        "Alur medium term: weekly 3y (makro) lalu daily 1y. Hasilnya bahan "
        "analisis manual, bukan sinyal jual/beli."
    )

    with st.container(border=True):
        c = st.columns(4)
        with c[0]:
            period = st.selectbox("Periode data", ["3mo", "6mo", "1y", "2y", "3y", "5y"], index=2)
        with c[1]:
            interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
        with c[2]:
            distance = st.slider(
                "Jarak minimum antar swing (bar)",
                min_value=1,
                max_value=30,
                value=5,
                help="Diteruskan ke scipy.signal.find_peaks — kalibrasi sesuai contoh di buku.",
            )
        with c[3]:
            use_prominence = st.checkbox("Gunakan prominence")
            prominence = (
                st.number_input("Prominence", min_value=0.0, value=0.0)
                if use_prominence
                else None
            )
        c = st.columns(2)
        with c[0]:
            trend_tolerance_pct = st.number_input(
                "Toleransi 'hampir sama' (%)",
                min_value=0.0,
                max_value=20.0,
                value=DEFAULT_TOLERANCE * 100,
                step=0.5,
                help="Selisih puncak-ke-puncak / dasar-ke-dasar di bawah ini dianggap "
                "sama (sideways). ASUMSI, bukan angka dari buku — kalibrasi ke contoh chart.",
            )
        with c[1]:
            lookback_swings = st.number_input(
                "Jumlah puncak & dasar terakhir yang dibandingkan",
                min_value=MIN_SWINGS_PER_SIDE,
                max_value=10,
                value=DEFAULT_LOOKBACK_SWINGS,
                help="ASUMSI, bukan angka dari buku. Makin besar makin ketat.",
            )
        confirmation_options = {
            "Tembus penuh: puncak/dasar terakhir harus dilewati (100%)": CONFIRM_FULL_BREAK,
            "Setengah jalan: 50% jarak vertikal ke puncak/dasar terakhir terlampaui": CONFIRM_HALF_WAY,
        }
        confirmation_label = st.radio(
            "Syarat titik acuan trendline 'siap' dipakai",
            list(confirmation_options),
            help="Dua mazhab dari buku: dasar A2 (uptrend) baru dipakai menggaris "
            "up-trendline setelah resistance (puncak terakhir sebelum A2) dilewati "
            "harga, atau cukup 50% jaraknya. Downtrend cermin dengan support. "
            "Dicek pakai High/Low, bukan Close. Default 100% adalah ASUMSI.",
        )
        confirmation_ratio = confirmation_options[confirmation_label]
        break_tol_pct = st.number_input(
            "Toleransi penembusan trendline (%)",
            min_value=0.0,
            max_value=10.0,
            value=DEFAULT_BREAK_TOLERANCE * 100,
            step=0.5,
            help="Buku: valid break butuh Close melewati garis ± toleransi ini "
            "(meredam whipsaw). Panduan buku: short term 0,5–1,5%, medium term 2–3%, "
            "long term 3,5–5%. Default 2% = medium term. Contoh buku: garis 125, "
            "toleransi 2% → batas 122,5.",
        )
        c = st.columns(3)
        with c[0]:
            breakout_tol_pct = st.number_input(
                "Toleransi breakout resistance (%)",
                min_value=0.0, max_value=5.0, value=BREAKOUT_TOLERANCE * 100, step=0.25,
                help="Buku: breakout sah bila Close melewati resistance + 1,5% "
                "(contoh: 50,5 → 51,25). Menyentuh saja tidak cukup.",
            )
        with c[1]:
            cut_loss_tol_pct = st.number_input(
                "Cut-loss di bawah support baru (%)",
                min_value=0.0, max_value=5.0, value=CUT_LOSS_TOLERANCE * 100, step=0.25,
                help="Buku: resistance lama jadi support baru, cut-loss 1,5% di bawahnya "
                "(contoh: 50,5 → 49,8).",
            )
        with c[2]:
            min_tests = st.number_input(
                "Uji minimal untuk strong resistance",
                min_value=2, max_value=10, value=STRONG_RESISTANCE_MIN_TESTS,
                help="Buku: resistance yang sudah diuji 3 kali = strong resistance. "
                f"Swing high dalam {DEFAULT_TEST_TOLERANCE:.1%} di bawah level dihitung "
                "'menguji' (ASUMSI, bukan angka buku).",
            )
        pullback_tol_pct = st.number_input(
            "Toleransi sentuh pullback (%)",
            min_value=0.0,
            max_value=5.0,
            value=0.0,
            step=0.25,
            help="Pullback = harga kembali menguji level S/R yang sudah dilewati. "
            "0% = Low/High harus menyentuh level (ASUMSI default, bukan angka buku); "
            "lebih besar = 'mendekati' sekian persen sudah dihitung menguji.",
        )

    if st.button("Ambil & tampilkan chart", type="primary", width="stretch"):
        prices_df = get_price_history(ticker, period=period, interval=interval)

        if prices_df.empty:
            st.error(f"Data harga untuk {ticker} tidak ditemukan.")
        else:
            highs_idx, lows_idx = get_extrema(
                prices_df["Close"], distance=distance, prominence=prominence
            )
            trend = classify_trend(
                prices_df["Close"],
                highs_idx,
                lows_idx,
                tol=trend_tolerance_pct / 100,
                lookback_swings=int(lookback_swings),
            )
            anchors = select_anchor_points(
                trend, prices_df["Low"], prices_df["High"], confirmation_ratio
            )
            trendline = build_trendline(
                trend,
                prices_df["Low"],
                prices_df["High"],
                end_index=len(prices_df) - 1,
                confirmation_ratio=confirmation_ratio,
            )

            fig = plot_candlestick(
                prices_df, title=ticker, highs_idx=highs_idx, lows_idx=lows_idx
            )
            break_check = None
            channel = channel_break = None
            if trendline is not None:
                add_trendline(fig, prices_df, trendline)
                break_check = check_trendline_break(
                    trendline,
                    prices_df["Close"],
                    prices_df["Low"],
                    prices_df["High"],
                    prices_df["Open"],
                    tol=break_tol_pct / 100,
                )
                add_break_markers(fig, prices_df, trendline, break_check)
                channel = build_channel(
                    trendline, prices_df["High"], prices_df["Low"], highs_idx, lows_idx
                )
                if channel is not None:
                    add_channel(fig, prices_df, channel)
                    channel_break = check_channel_break(
                        channel, prices_df["Close"], tol=break_tol_pct / 100
                    )
            # The Fan Principle menguji PEMBALIKAN tren, jadi hanya masuk akal
            # bila ada tren yang sedang berlangsung. Chart sideways / tidak jelas
            # tidak dipaksakan: arah kipasnya pun tidak ada.
            fan = None
            fan_direction = None
            if trend.trend in (UPTREND, DOWNTREND):
                fan_direction = FAN_BEARISH if trend.trend == UPTREND else FAN_BULLISH
                fan = detect_fan(
                    fan_direction,
                    prices_df["Close"],
                    prices_df["High"],
                    prices_df["Low"],
                    highs_idx,
                    lows_idx,
                    tol=break_tol_pct / 100,
                )
            if fan is not None:
                add_fan(fig, prices_df, fan)

            sr_levels = levels_from_swings(
                prices_df,
                highs_idx,
                lows_idx,
                per_side=int(lookback_swings),
                pullback_tol=pullback_tol_pct / 100,
            )
            add_levels(fig, prices_df, sr_levels)

            # Validasi breakout per resistance: uji berulang, breakout sah,
            # 2nd day, rencana keluar, status posisi. Hanya up-trendline yang
            # dipakai sebagai patokan "tahan selama tren naik".
            exit_trendline = trendline if trendline is not None and trend.trend == UPTREND else None
            breakout_rows = []
            for level in sr_levels:
                if level.initial_role != RESISTANCE:
                    continue
                signal = find_breakout(
                    level.price,
                    prices_df["Close"],
                    prices_df["Open"],
                    start_index=level.origin_index + 1,
                    tol=breakout_tol_pct / 100,
                )
                tests = count_resistance_tests(
                    level.price,
                    highs_idx,
                    prices_df["High"],
                    prices_df["Close"],
                    until_index=signal.breakout_index if signal else None,
                    breakout_tol=breakout_tol_pct / 100,
                )
                plan = position = None
                if signal is not None and signal.entry_index is not None:
                    plan = build_trading_plan(
                        level.price, signal.entry_price, cut_loss_tol=cut_loss_tol_pct / 100
                    )
                    position = evaluate_breakout_position(
                        plan,
                        signal.entry_index,
                        prices_df["Close"],
                        trendline=exit_trendline,
                        trendline_tol=break_tol_pct / 100,
                    )
                breakout_rows.append((level, tests, signal, plan, position))

            # Resistance yang berdekatan sering memicu rencana yang sama; supaya
            # chart tidak penuh, gambar hanya rencana dengan tanggal masuk terbaru.
            plans_with_entry = [r for r in breakout_rows if r[3] is not None]
            if plans_with_entry:
                _, _, signal, plan, position = max(
                    plans_with_entry, key=lambda r: r[2].entry_index
                )
                add_breakout_plan(fig, prices_df, signal, plan, position)

            st.plotly_chart(fig, width="stretch")

            # Hasil ditaruh di expander supaya halaman bisa dilipat per bagian;
            # judulnya memuat ringkasan agar tetap informatif saat dilipat.
            trend_title = f"Tren ({interval}, {period}): {trend.trend.upper()}"
            if trendline is not None:
                trend_title += f" · {trendline.kind} · {break_check.status}"
            with st.expander(trend_title, expanded=True):
                st.caption(trend.reason)
                if trendline is not None:
                    last_close = float(prices_df["Close"].iloc[-1])
                    line_now = trendline.value_at(trendline.end_index)
                    threshold_now = break_threshold(
                        line_now, trendline.kind, break_tol_pct / 100
                    )
                    gap_pct = (last_close - line_now) / line_now * 100
                    st.markdown(
                        f"**{trendline.kind}** dari {len(trendline.points)} titik "
                        f"({'Low dasar' if trendline.kind == 'up-trendline' else 'High puncak'}). "
                        f"Level garis di bar terkini: {format_rupiah(line_now)} · "
                        f"batas valid break (toleransi {break_tol_pct:g}%): "
                        f"{format_rupiah(threshold_now)} · "
                        f"Close terakhir: {format_rupiah(last_close)} "
                        f"({gap_pct:+.1f}% dari garis)"
                    )
                    dates = prices_df.index
                    if break_check.status == VALID_BREAK:
                        break_date = dates[break_check.valid_break_index].strftime("%Y-%m-%d")
                        if break_check.second_day == SECOND_DAY_CONFIRMED:
                            next_open_date = dates[break_check.valid_break_index + 1].strftime(
                                "%Y-%m-%d"
                            )
                            second_day_text = (
                                f"2nd day: TERKONFIRMASI — Open {next_open_date} tetap di "
                                "luar garis."
                            )
                        else:
                            second_day_text = (
                                "2nd day: MENUNGGU Open sesi berikutnya (bisa gap kembali "
                                "ke dalam garis — konfirmasi akhir belum ada)."
                            )
                        st.markdown(
                            f"**Status Garis: VALID BREAK** — Close di luar garis pada "
                            f"{break_date}. {second_day_text}"
                        )
                    else:
                        st.markdown(f"**Status Garis: {break_check.status.upper()}**")
                    if break_check.gap_back_indices:
                        gap_dates = ", ".join(
                            dates[i].strftime("%Y-%m-%d") for i in break_check.gap_back_indices
                        )
                        st.caption(
                            f"Close pernah di luar garis tapi Open sesi berikutnya gap "
                            f"kembali ke dalam (2nd day gagal): {gap_dates}."
                        )
                    if break_check.whipsaw_indices:
                        whipsaw_dates = ", ".join(
                            dates[i].strftime("%Y-%m-%d") for i in break_check.whipsaw_indices
                        )
                        st.caption(
                            f"Tembusan intraday saja (Close kembali ke dalam garis) = "
                            f"false break / whipsaw: {whipsaw_dates}."
                        )
                    if channel is None:
                        st.caption(
                            "Channeling tidak terbentuk: harga belum terbukti bergerak "
                            "dalam koridor (channel line perlu disentuh minimal dua swing). "
                            "Tidak dipaksakan — chart ini cukup dibaca dari trendline-nya."
                        )
                    if channel is not None:
                        width_now = channel.width_at(channel.basic.end_index)
                        width_pct = width_now / channel.basic_value_at(channel.basic.end_index)
                        st.markdown(
                            f"**Channeling:** Basic trendline di {channel.basic_position}, "
                            f"channel line sejajar lewat swing "
                            f"{dates[channel.anchor.index].strftime('%Y-%m-%d')} "
                            f"({format_rupiah(channel.anchor.price)}); lebar koridor "
                            f"{format_rupiah(width_now)} ({width_pct:.1%}), "
                            f"{channel.touches} swing menyentuh channel line."
                        )
                        if channel_break.status == CHANNEL_INTACT:
                            st.caption(f"{channel_break.status} (belum ada Close di luar koridor).")
                        else:
                            st.caption(
                                f"{channel_break.status} — kecenderungan {channel_break.bias} "
                                f"pada {dates[channel_break.index].strftime('%Y-%m-%d')} "
                                f"(Close {format_rupiah(channel_break.price)}). Buku: pada "
                                "uptrend channeling, basic trendline tertembus menandakan "
                                "kemungkinan awal perubahan tren; channel line tertembus "
                                "menandakan akselerasi tren. Dinilai manual."
                            )
                    st.caption(
                        "Aturan buku: penembusan sah hanya bila harga PENUTUPAN di luar "
                        "garis; tembusan sementara intraday tidak dihitung. Aturan 2nd "
                        "day: Open sesi berikutnya jadi konfirmasi akhir (weekly: Close "
                        "Jumat validasi, Open Senin konfirmasi). Diperiksa sejak bar "
                        "setelah titik acuan terakhir. Tetap dinilai manual — screener "
                        "tidak memberi sinyal jual/beli."
                    )
                elif anchors:
                    st.warning(
                        "Trendline belum digambar: titik acuan yang sudah 'siap' "
                        "kurang dari 2. Lihat tabel titik acuan di bawah."
                    )
                if anchors:
                    dates = prices_df.index
                    anchor_label = "Dasar (Low)" if trend.trend == UPTREND else "Puncak (High)"
                    level_label = "Resistance" if trend.trend == UPTREND else "Support"
                    st.markdown(f"**Titik Acuan {trendline.kind if trendline else 'trendline'}**")
                    st.dataframe(
                        pd.DataFrame(
                            {
                                anchor_label: [dates[a.index].strftime("%Y-%m-%d") for a in anchors],
                                "Harga": [a.price for a in anchors],
                                # None (bukan teks) untuk titik awal supaya kolom tetap numerik.
                                f"Level {level_label.lower()} yang harus dilewati": [
                                    a.threshold for a in anchors
                                ],
                                "Status": [
                                    "siap" if a.confirmed else "belum siap" for a in anchors
                                ],
                                "Dilewati pada": [
                                    dates[a.confirmed_index].strftime("%Y-%m-%d")
                                    if a.confirmed_index is not None
                                    else "-"
                                    for a in anchors
                                ],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                if trend.peaks and trend.troughs:
                    dates = prices_df.index
                    c = st.columns(2)
                    c[0].dataframe(
                        pd.DataFrame(
                            {
                                "Puncak": [dates[p.index].strftime("%Y-%m-%d") for p in trend.peaks],
                                "Harga": [p.price for p in trend.peaks],
                                "vs sebelumnya": ["-"] + trend.peak_steps,
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                    c[1].dataframe(
                        pd.DataFrame(
                            {
                                "Dasar": [dates[t.index].strftime("%Y-%m-%d") for t in trend.troughs],
                                "Harga": [t.price for t in trend.troughs],
                                "vs sebelumnya": ["-"] + trend.trough_steps,
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )
                if trend.trend == UNDEFINED:
                    st.info(
                        "Tren tidak masuk salah satu definisi — ini bukan kesalahan, "
                        "cukup berarti chart ini perlu dilihat manual atau ubah "
                        "parameter swing/toleransi."
                    )

            if fan is not None:
                fan_title = (
                    f"The Fan Principle ({fan.direction}): {fan.lines_broken}/"
                    f"{FAN_LINE_COUNT} garis tertembus"
                    + (" — REVERSAL TERKONFIRMASI" if fan.is_confirmed else "")
                )
                with st.expander(fan_title, expanded=fan.is_confirmed):
                    dates = prices_df.index
                    st.caption(
                        "Aturan buku (Bab 14): tiga trendline ditarik dari satu titik "
                        "pangkal; tiap kali satu garis tertembus, garis berikutnya "
                        "ditarik dari pangkal yang sama ke swing baru sehingga makin "
                        "landai seperti kipas. Reversal diakui HANYA saat garis ketiga "
                        "tertembus — tembusnya garis pertama/kedua bisa sekadar koreksi. "
                        "Garis yang sudah tertembus berganti peran jadi penghalang. "
                        "Sinyal untuk ditinjau manual, bukan rekomendasi jual/beli."
                    )
                    st.markdown(
                        f"**Pangkal Kipas:** {dates[fan.origin.index].strftime('%Y-%m-%d')} "
                        f"({format_rupiah(fan.origin.price)}). {fan.reason}"
                    )
                    if fan.is_confirmed:
                        arah = "turun" if fan.direction == FAN_BEARISH else "naik"
                        st.markdown(
                            f"**Konfirmasi Reversal ke Tren {arah}** pada "
                            f"{dates[fan.confirmation_index].strftime('%Y-%m-%d')} "
                            f"(Close {format_rupiah(float(prices_df['Close'].iloc[fan.confirmation_index]))})."
                        )
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Garis": [line.order for line in fan.lines],
                                "Acuan": [
                                    dates[line.anchor.index].strftime("%Y-%m-%d")
                                    for line in fan.lines
                                ],
                                "Harga Acuan": [line.anchor.price for line in fan.lines],
                                "Kemiringan/bar": [round(line.slope, 2) for line in fan.lines],
                                "Tertembus": [
                                    dates[line.break_index].strftime("%Y-%m-%d")
                                    if line.is_broken
                                    else "belum"
                                    for line in fan.lines
                                ],
                                "Peran setelah Tembus": [
                                    line.role_after_break or "-" for line in fan.lines
                                ],
                                "Uji Ulang (x)": [
                                    len(line.retest_indices) for line in fan.lines
                                ],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            if fan is None:
                with st.expander("The Fan Principle: tidak ada pola kipas", expanded=False):
                    if fan_direction is None:
                        st.caption(
                            f"Kipas menguji pembalikan tren, sedangkan tren chart ini "
                            f"'{trend.trend}'. Tanpa tren yang sedang berlangsung tidak ada "
                            "yang bisa berbalik, jadi fan principle tidak dipaksakan di sini."
                        )
                    else:
                        st.caption(
                            "Belum ada garis kipas yang tertembus, jadi kipas belum mulai "
                            "terbentuk — yang ada baru trendline biasa. Tidak dipaksakan "
                            "jadi pola."
                        )

            if sr_levels:
                n_support = sum(level.role == "support" for level in sr_levels)
                with st.expander(
                    f"Level Support & Resistance Horizontal ({len(sr_levels)} level: "
                    f"{n_support} support, {len(sr_levels) - n_support} resistance)",
                    expanded=False,
                ):
                    st.caption(
                        "Sesuai buku: support = garis mendatar dari titik terendah "
                        "lembah (Low swing low), resistance = dari titik tertinggi "
                        f"puncak (High swing high); {int(lookback_swings)} terakhir tiap "
                        "sisi. Tembus sah hanya bila Close di luar level; setelah "
                        "tembus, peran berbalik (support jadi resistance, sebaliknya) "
                        "dengan kekuatan yang sama; level yang lebih lama lebih kuat. "
                        "Pullback = harga kembali menguji level yang sudah dilewati: "
                        "'bertahan' bila Close tetap di dalam (lingkaran di chart), "
                        "'gagal' bila Close menembus lagi. Buku tidak memberi skala "
                        "kekuatan, jadi hanya usia yang ditampilkan — dinilai manual."
                    )
                    dates = prices_df.index
                    days_per_bar = 7 if interval == "1wk" else 1
                    st.dataframe(
                        pd.DataFrame(
                            {
                                "Level": [level.price for level in sr_levels],
                                "Peran Awal": [level.initial_role for level in sr_levels],
                                "Peran Sekarang": [level.role for level in sr_levels],
                                "Terbentuk": [
                                    dates[level.origin_index].strftime("%Y-%m-%d")
                                    for level in sr_levels
                                ],
                                "Usia (Bar)": [level.age_bars for level in sr_levels],
                                "Usia (~Hari)": [
                                    level.age_bars * days_per_bar for level in sr_levels
                                ],
                                "Valid Break": [len(level.valid_breaks) for level in sr_levels],
                                "False Break": [len(level.false_breaks) for level in sr_levels],
                                "Pullback Bertahan": [
                                    len(level.pullbacks_held) for level in sr_levels
                                ],
                                "Pullback Gagal": [
                                    len(level.pullbacks_failed) for level in sr_levels
                                ],
                                "Pullback Terakhir": [
                                    dates[level.pullbacks_held[-1].index].strftime("%Y-%m-%d")
                                    if level.pullbacks_held
                                    else "-"
                                    for level in sr_levels
                                ],
                                "Break Terakhir": [
                                    dates[level.valid_breaks[-1].index].strftime("%Y-%m-%d")
                                    if level.valid_breaks
                                    else "-"
                                    for level in sr_levels
                                ],
                            }
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            if breakout_rows:
                n_breakout = sum(r[2] is not None for r in breakout_rows)
                n_strong = sum(is_strong_resistance(r[1], int(min_tests)) for r in breakout_rows)
                with st.expander(
                    f"Validasi Breakout & Rencana Keluar ({len(breakout_rows)} resistance, "
                    f"{n_strong} strong, {n_breakout} breakout sah)",
                    expanded=True,
                ):
                    st.caption(
                        f"Aturan buku: breakout sah bila Close > resistance + {breakout_tol_pct:g}% "
                        "(menyentuh saja tidak cukup); resistance yang diuji "
                        f"≥ {int(min_tests)} kali = strong resistance; masuk di Open sesi "
                        "berikutnya (2nd day); rencana keluar disiapkan sebelum masuk: "
                        f"cut-loss {cut_loss_tol_pct:g}% di bawah support baru (resistance lama). "
                        "Close di bawah cut-loss = false breakout → keluar sesuai rencana; "
                        "selama di atas up-trendline tahan, keluar saat trendline patah. "
                        "Prinsip: cut your loss fast, let your profit run. Ini simulasi "
                        "rencana untuk Anda tinjau dan eksekusi manual — bukan rekomendasi, "
                        "bukan robot yang menaruh order."
                    )
                    dates = prices_df.index
                    rows = []
                    for level, tests, signal, plan, position in breakout_rows:
                        rows.append(
                            {
                                "Resistance": level.price,
                                "Terbentuk": dates[level.origin_index].strftime("%Y-%m-%d"),
                                "Diuji (x)": tests,
                                "Strong": "ya" if is_strong_resistance(tests, int(min_tests)) else "-",
                                "Batas Breakout": signal.threshold if signal else level.price * (1 + breakout_tol_pct / 100),
                                "Breakout": dates[signal.breakout_index].strftime("%Y-%m-%d") if signal else "-",
                                "2nd day": signal.second_day if signal else "-",
                                "Masuk (Open)": signal.entry_price if signal and signal.entry_price else None,
                                "Cut-loss": plan.cut_loss if plan else None,
                                "Risiko/Lembar": plan.risk_per_share if plan else None,
                                "Status": position.status if position else "-",
                                "Keluar": dates[position.exit_index].strftime("%Y-%m-%d") if position and position.exit_index is not None else "-",
                                "P/L per Lembar": position.pnl_per_share if position else None,
                            }
                        )
                    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

            patterns_found = scan(prices_df["Close"], highs_idx, lows_idx)
            if patterns_found:
                for pattern in patterns_found:
                    st.write(pattern)
            else:
                st.info(
                    "Belum ada pola terdeteksi — registry detektor pola masih "
                    "kosong sampai aturan dari buku Edianto Ong ditambahkan."
                )

# --- Portofolio: alokasi & cash (CLAUDE.md bagian E) ---
with tab_portofolio:
    st.caption(
        "Bandingkan komposisi portofolio dengan aturan alokasi (5–6 saham beda "
        "sektor, bobot 25–30% per saham, cash cadangan 20%). Hasilnya daftar "
        "penyimpangan untuk ditinjau, bukan instruksi jual/beli."
    )

    c = st.columns([3, 1])
    with c[0]:
        positions_df = st.data_editor(
            pd.DataFrame(
                {"Ticker": ["BBCA.JK"], "Sektor": ["Bank"], "Nilai Posisi (Rp)": [0.0]}
            ),
            num_rows="dynamic",
            width="stretch",
            key="portfolio_editor",
            column_config={
                "Nilai Posisi (Rp)": st.column_config.NumberColumn(
                    min_value=0.0, step=1, format="localized"
                )
            },
        )
    with c[1]:
        cash = rupiah_input("Cash cadangan", key="in_cash", step=1_000_000.0)

    if st.button("Periksa alokasi", type="primary", width="stretch"):
        positions = [
            Position(
                ticker=str(row["Ticker"]).strip(),
                sector=str(row["Sektor"]),
                value=float(row["Nilai Posisi (Rp)"] or 0),
            )
            for _, row in positions_df.iterrows()
            if str(row["Ticker"]).strip() and str(row["Ticker"]) != "nan"
        ]
        review = review_portfolio(positions, cash)

        if review.total_value <= 0:
            st.info("Isi nilai posisi atau cash untuk memeriksa alokasi.")
        else:
            c = st.columns([1, 2])
            with c[0]:
                big_rupiah_metric(st, "Total (saham + cash)", review.total_value)
                st.metric("Porsi cash", f"{review.cash_pct:.1f}%")
            with c[1]:
                st.dataframe(
                    pd.DataFrame(
                        {
                            "Ticker": list(review.weights_pct),
                            "Bobot (%)": list(review.weights_pct.values()),
                        }
                    ).round(1),
                    width="stretch",
                    hide_index=True,
                )

            for warning in review.warnings:
                st.error(warning)
            for note in review.notes:
                st.info(note)
            if not review.warnings and not review.notes:
                st.success("Komposisi sesuai aturan alokasi.")

# --- Checklist kualitatif & status eksekusi (CLAUDE.md "Batas otomasi" & bagian F) ---
with tab_checklist:
    with st.container(border=True):
        st.markdown("**Checklist Kualitatif (Dijawab Manual)**")
        st.caption(
            "Kriteria ini sengaja tidak dinilai otomatis — tidak ada skor, tidak ada "
            "lolos/gagal. Rangkumannya hanya mengingatkan mana yang belum kamu cek."
        )
        answers = {}
        for item_key, question in QUALITATIVE_CHECKLIST.items():
            answers[item_key] = st.radio(
                question, ANSWER_OPTIONS, horizontal=True, key=f"chk_{item_key}"
            )

        summary = summarize_checklist(answers)
        c = st.columns(3)
        c[0].metric("Ya", len(summary.yes))
        c[1].metric("Tidak", len(summary.no))
        c[2].metric("Belum dicek", len(summary.unchecked))
        if summary.no:
            st.warning("Dijawab **Tidak**:\n\n" + "\n".join(f"- {q}" for q in summary.no))
        if summary.unchecked:
            st.info("Belum dicek:\n\n" + "\n".join(f"- {q}" for q in summary.unchecked))
        if summary.is_complete and not summary.no:
            st.success("Semua item kualitatif terjawab Ya.")

    with st.container(border=True):
        st.markdown("**Status Eksekusi & Horizon**")
        st.caption(
            "Spesifikasi sumber memakai label BUY/HOLD/EXIT; di sini ditampilkan "
            "sebagai kondisi yang aktif/tidak, keputusannya tetap milikmu. "
            "Evaluasi ulang tiap rilis laporan keuangan kuartalan."
        )

        screening = st.session_state.get("screening")
        if not screening or screening["valuation"] is None:
            st.info(
                "Jalankan screening fundamental (dengan BVPS terisi) di tab Lapis 1 "
                "dulu — status ini memakai harga wajar & max buy dari sana."
            )
        else:
            c = st.columns(3)
            with c[0]:
                holding_months = st.number_input(
                    "Lama holding (bulan)", min_value=0.0, value=0.0, step=1.0,
                    help="0 untuk kandidat yang belum dibeli.",
                )
            with c[1]:
                fundamentals_growing = st.checkbox(
                    "Fundamental masih bertumbuh",
                    value=True,
                    help="Penilaian manual dari laporan keuangan terakhir.",
                )
            with c[2]:
                growth_on_track = st.checkbox(
                    "Growth sesuai rencana awal",
                    value=True,
                    help="Penilaian manual: apakah pertumbuhan sesuai tesis saat beli.",
                )

            valuation = screening["valuation"]
            status = evaluate_position(
                current_price=screening["current_price"],
                max_buy=valuation.max_buy,
                fair_price=valuation.fair_price_adj,
                is_cheap=screening["cheap"],
                holding_months=holding_months,
                fundamentals_growing=fundamentals_growing,
                growth_on_track=growth_on_track,
            )

            st.markdown(
                f"{screening['ticker']} · harga {format_rupiah(screening['current_price'])} · "
                f"max buy {format_rupiah(valuation.max_buy)} · harga wajar "
                f"{format_rupiah(valuation.fair_price_adj)} · horizon **{status.horizon}**"
            )
            st.dataframe(
                pd.DataFrame(
                    {
                        "Kondisi": [cond.label for cond in status.conditions],
                        "Aktif": ["Ya" if cond.met else "-" for cond in status.conditions],
                        "Syarat": [cond.detail for cond in status.conditions],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
