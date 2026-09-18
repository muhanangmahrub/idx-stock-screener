"""Entry point Streamlit - hanya UI, logika ada di modul screener/."""

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
from screener.plotting import plot_candlestick
from screener.portfolio import Position, review_portfolio
from screener.universe import (
    STATUS_PASSED,
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
}


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

    st.subheader("Sumber data")
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
        "Lapis 1 - Fundamental",
        "Screening massal",
        "Lapis 2 - Teknikal",
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

    with st.container(border=True):
        st.markdown("**A. Screening awal**")
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
                "anualisasi ini asumsi - bandingkan dengan ROE TTM.",
            )
        is_blacklisted = st.checkbox("Masuk blacklist manual (saham bandar/gorengan)")

    with st.container(border=True):
        st.markdown("**B & C. Valuasi**")
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
                    "keuangan - mis. harga komoditas turun, suku bunga naik, "
                    "regulasi berubah. Tiap faktor yang menurutmu relevan "
                    "memotong harga wajar **10%**:\n\n"
                    "`harga wajar adj = harga wajar dasar × (1 − 0,10 × jumlah faktor)`\n\n"
                    "Faktor mana yang dihitung adalah judgment manual - tidak ada "
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
                    "35-50% sesuai spesifikasi."
                ),
            )
        is_blue_chip = st.checkbox(
            "Klasifikasikan sebagai blue chip",
            help="Menentukan batas PER (12 vs 8). Tidak ada aturan otomatis di "
            "sumber - keputusan manual.",
        )

    with st.container(border=True):
        st.markdown("**D. Batas posisi**")
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

        st.markdown(f"**Hasil screening {screening['ticker']}**")
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
            st.markdown("**Harga wajar & best buy (jalur harga absolut)**")
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
                            f"{format_rupiah(valuation.max_buy)} - masuk zona untuk dianalisis "
                            "lebih lanjut (bukan rekomendasi beli)."
                        )
                    else:
                        st.warning(
                            f"Harga saat ini {format_rupiah(price_at_run)} di atas max buy "
                            f"{format_rupiah(valuation.max_buy)}."
                        )

        with st.container(border=True):
            st.markdown("**Batas posisi per emiten**")
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
        "dibedakan otomatis; blue chip dengan PER 8-12 diberi catatan untuk "
        "dicek manual. Hasilnya daftar kandidat untuk dianalisis manual - "
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
            "Angka free float/ROE/PER/PBV adalah estimasi otomatis - verifikasi "
            "manual sebelum masuk watchlist. PBV kosong berarti yfinance tidak "
            "memberi angka yang valid (lihat kolom catatan)."
        )

        show_only_passed = st.checkbox("Tampilkan hanya yang lolos", value=True)
        table = candidates_to_dataframe(passed if show_only_passed else results)
        if table.empty:
            st.info("Tidak ada emiten yang lolos screening awal + valuasi murah.")
        else:
            st.dataframe(table, width="stretch", hide_index=True)

# --- Lapis 2: Teknikal (chart pattern, Edianto Ong) ---
with tab_teknikal:
    st.caption(
        "Deteksi swing high/low sebagai fondasi analisis pola. Detektor pola "
        "(double top, head & shoulders, dst.) belum diimplementasikan - "
        "menunggu spesifikasi aturan dari buku pemilik (lihat CLAUDE.md)."
    )

    with st.container(border=True):
        c = st.columns(4)
        with c[0]:
            period = st.selectbox("Periode data", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
        with c[1]:
            interval = st.selectbox("Interval", ["1d", "1wk"], index=0)
        with c[2]:
            distance = st.slider(
                "Jarak minimum antar swing (bar)",
                min_value=1,
                max_value=30,
                value=5,
                help="Diteruskan ke scipy.signal.find_peaks - kalibrasi sesuai contoh di buku.",
            )
        with c[3]:
            use_prominence = st.checkbox("Gunakan prominence")
            prominence = (
                st.number_input("Prominence", min_value=0.0, value=0.0)
                if use_prominence
                else None
            )

    if st.button("Ambil & tampilkan chart", type="primary", width="stretch"):
        prices_df = get_price_history(ticker, period=period, interval=interval)

        if prices_df.empty:
            st.error(f"Data harga untuk {ticker} tidak ditemukan.")
        else:
            highs_idx, lows_idx = get_extrema(
                prices_df["Close"], distance=distance, prominence=prominence
            )
            fig = plot_candlestick(
                prices_df, title=ticker, highs_idx=highs_idx, lows_idx=lows_idx
            )
            st.plotly_chart(fig, width="stretch")

            patterns_found = scan(prices_df["Close"], highs_idx, lows_idx)
            if patterns_found:
                for pattern in patterns_found:
                    st.write(pattern)
            else:
                st.info(
                    "Belum ada pola terdeteksi - registry detektor pola masih "
                    "kosong sampai aturan dari buku Edianto Ong ditambahkan."
                )

# --- Portofolio: alokasi & cash (CLAUDE.md bagian E) ---
with tab_portofolio:
    st.caption(
        "Bandingkan komposisi portofolio dengan aturan alokasi (5-6 saham beda "
        "sektor, bobot 25-30% per saham, cash cadangan 20%). Hasilnya daftar "
        "penyimpangan untuk ditinjau, bukan instruksi jual/beli."
    )

    c = st.columns([3, 1])
    with c[0]:
        positions_df = st.data_editor(
            pd.DataFrame(
                {"Ticker": ["BBCA.JK"], "Sektor": ["Bank"], "Nilai posisi (Rp)": [0.0]}
            ),
            num_rows="dynamic",
            width="stretch",
            key="portfolio_editor",
            column_config={
                "Nilai posisi (Rp)": st.column_config.NumberColumn(
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
                value=float(row["Nilai posisi (Rp)"] or 0),
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
        st.markdown("**Checklist kualitatif (dijawab manual)**")
        st.caption(
            "Kriteria ini sengaja tidak dinilai otomatis - tidak ada skor, tidak ada "
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
        st.markdown("**Status eksekusi & horizon**")
        st.caption(
            "Spesifikasi sumber memakai label BUY/HOLD/EXIT; di sini ditampilkan "
            "sebagai kondisi yang aktif/tidak, keputusannya tetap milikmu. "
            "Evaluasi ulang tiap rilis laporan keuangan kuartalan."
        )

        screening = st.session_state.get("screening")
        if not screening or screening["valuation"] is None:
            st.info(
                "Jalankan screening fundamental (dengan BVPS terisi) di tab Lapis 1 "
                "dulu - status ini memakai harga wajar & max buy dari sana."
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
