"""Fungsi chart plotly untuk visualisasi harga saham.

Sumbu x memakai POSISI BAR (0, 1, 2, ...), bukan tanggal, dan label tanggal
dipasang sebagai tick. Alasannya dua-duanya soal ketepatan gambar:

1. Semua garis analisis (trendline, channel line, level) dihitung per bar.
   Di sumbu tanggal, akhir pekan & libur bursa membuat jarak antar-bar tidak
   seragam sehingga garis lurus tampak patah.
2. Lebar candle dihitung Plotly dari jarak antar-nilai x. Kalau celah tanggal
   disembunyikan (rangebreaks), lebar candle tetap dihitung dari jarak
   aslinya - candle jadi melebar dan tidak lagi duduk tepat di posisinya.

Dengan sumbu posisi bar, candle, marker swing, garis, dan semua penanda
memakai koordinat yang sama persis, untuk interval apa pun (harian maupun
mingguan) dan untuk parameter apa pun.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from screener.levels import SUPPORT
from screener.trend import UP_TRENDLINE
from screener.volume import VOLUME_HIGH, VOLUME_LOW, volume_profiles

MAX_DATE_TICKS = 10
# Panel volume dibuat jauh lebih pendek daripada panel harga: perannya
# pendukung, bukan pusat perhatian (buku menaruhnya "di bagian bawah chart").
PRICE_PANEL_SHARE = 0.78
VOLUME_PANEL_SHARE = 0.22


def bar_positions(df: pd.DataFrame) -> np.ndarray:
    """Koordinat x untuk tiap bar: posisinya di deret, bukan tanggalnya."""
    return np.arange(len(df))


def _date_labels(df: pd.DataFrame) -> list[str]:
    return [d.strftime("%Y-%m-%d") for d in pd.DatetimeIndex(df.index)]


def _apply_date_axis(fig: go.Figure, df: pd.DataFrame) -> go.Figure:
    """Pasang label tanggal pada sumbu posisi bar, dijarangkan agar terbaca."""
    labels = _date_labels(df)
    if not labels:
        return fig
    step = max(1, len(labels) // MAX_DATE_TICKS)
    positions = list(range(0, len(labels), step))
    fig.update_xaxes(
        tickmode="array",
        tickvals=positions,
        ticktext=[labels[i] for i in positions],
    )
    return fig


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "",
    highs_idx: np.ndarray | None = None,
    lows_idx: np.ndarray | None = None,
    show_volume: bool = True,
) -> go.Figure:
    """Buat candlestick chart dari DataFrame OHLC (kolom Open/High/Low/Close).

    `highs_idx`/`lows_idx` opsional: indeks posisi (hasil `get_extrema`) untuk
    menandai swing high/low di atas chart, sebagai bantuan visual sebelum
    detektor pola (Lapis 2) tersedia.

    `show_volume` menambahkan panel volume bar di bawah panel harga (sesuai
    penempatan di buku), memakai sumbu x yang sama sehingga tiap batang
    volume sejajar dengan candle-nya. Panel volume dilewati bila kolom
    Volume tidak ada.
    """
    x = bar_positions(df)
    dates = _date_labels(df)
    with_volume = show_volume and "Volume" in df.columns

    if with_volume:
        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.03,
            row_heights=[PRICE_PANEL_SHARE, VOLUME_PANEL_SHARE],
        )
    else:
        fig = go.Figure()

    fig.add_trace(
        go.Candlestick(
            x=x,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            name="Harga",
            text=dates,  # tanggal tetap terbaca di hover meski sumbunya posisi
        )
    )
    if with_volume:
        add_volume_bars(fig, df)

    if highs_idx is not None and len(highs_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=highs_idx,
                y=df["High"].iloc[highs_idx],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=9),
                name="Swing high",
            )
        )

    if lows_idx is not None and len(lows_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=lows_idx,
                y=df["Low"].iloc[lows_idx],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=9),
                name="Swing low",
            )
        )

    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    if with_volume:
        fig.update_yaxes(title_text="Volume", row=2, col=1)
    _apply_date_axis(fig, df)
    return fig


def add_volume_bars(fig: go.Figure, df: pd.DataFrame, row: int = 2, col: int = 1) -> go.Figure:
    """Panel volume bar di bawah chart harga (aturan buku: volume dicatat
    sebagai batang di bagian bawah chart; batang tinggi = perdagangan ramai,
    batang pendek = minat kurang).

    Warna mengikuti arah candle-nya (hijau bila Close >= Open) supaya mudah
    dibaca bersama harga - itu konvensi umum, bukan aturan buku. Batang yang
    volumenya jauh di atas kebiasaan diberi warna pekat supaya "tinggi"
    terbaca sebagai ukuran, bukan kesan; ambangnya asumsi yang bisa diatur
    lewat `screener.volume`.
    """
    profiles = volume_profiles(df)
    labels = [p.label for p in profiles]
    ratios = [p.ratio for p in profiles]
    rising = df["Close"] >= df["Open"]

    colors, opacities = [], []
    for up, label in zip(rising, labels, strict=True):
        colors.append("green" if up else "red")
        opacities.append(
            1.0 if label == VOLUME_HIGH else (0.35 if label == VOLUME_LOW else 0.65)
        )

    fig.add_trace(
        go.Bar(
            x=bar_positions(df),
            y=df["Volume"],
            marker=dict(
                color=colors,
                opacity=opacities,
                line=dict(width=0),
            ),
            name="Volume",
            customdata=[
                [label, "-" if ratio is None else f"{ratio:.1f}x"]
                for label, ratio in zip(labels, ratios, strict=True)
            ],
            hovertemplate=(
                "volume %{y:,.0f}<br>%{customdata[0]} "
                "(%{customdata[1]} rata-rata)<extra></extra>"
            ),
        ),
        row=row,
        col=col,
    )
    return fig


def add_trendline(fig: go.Figure, df: pd.DataFrame, trendline) -> go.Figure:
    """Tambahkan up-/down-trendline (hasil `trend.build_trendline`) ke chart.

    Garis digambar dari titik acuan pertama sampai `end_index` (bar terkini),
    supaya terlihat sebagai level tempat tren diuji, bukan cuma penghubung
    dua titik. Nilai dihitung per bar, sejalan dengan model yang memakai
    indeks bar sebagai sumbu x; `_hide_nontrading_gaps` yang membuat jarak
    antar-bar di chart seragam sehingga garisnya tampak lurus.
    """
    bar_indices = np.arange(trendline.start_index, trendline.end_index + 1)
    color = "green" if trendline.kind == UP_TRENDLINE else "red"

    fig.add_trace(
        go.Scatter(
            x=bar_indices,
            y=[trendline.value_at(i) for i in bar_indices],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=trendline.kind,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[p.index for p in trendline.points],
            y=[p.price for p in trendline.points],
            mode="markers",
            marker=dict(symbol="circle-open", color=color, size=14, line=dict(width=2)),
            name=f"Titik {trendline.kind}",
        )
    )
    return fig


def add_break_markers(fig: go.Figure, df: pd.DataFrame, trendline, break_check) -> go.Figure:
    """Tandai whipsaw (silang) dan valid break (berlian) hasil
    `trend.check_trendline_break` pada chart.

    Marker diletakkan di harga yang menembus: Low untuk up-trendline, High
    untuk down-trendline; valid break di Close, karena Close-lah yang
    menentukan keabsahan penembusan menurut buku.
    """
    pierce_col = "Low" if trendline.kind == UP_TRENDLINE else "High"

    if break_check.whipsaw_indices:
        fig.add_trace(
            go.Scatter(
                x=break_check.whipsaw_indices,
                y=df[pierce_col].iloc[break_check.whipsaw_indices],
                mode="markers",
                marker=dict(symbol="x", color="orange", size=11),
                name="False break (intraday)",
            )
        )
    if break_check.gap_back_indices:
        fig.add_trace(
            go.Scatter(
                x=break_check.gap_back_indices,
                y=df["Close"].iloc[break_check.gap_back_indices],
                mode="markers",
                marker=dict(symbol="diamond-open", color="magenta", size=12),
                name="Break, gap kembali (2nd day gagal)",
            )
        )
    if break_check.valid_break_index is not None:
        i = break_check.valid_break_index
        fig.add_trace(
            go.Scatter(
                x=[i],
                y=[df["Close"].iloc[i]],
                mode="markers",
                marker=dict(symbol="diamond", color="magenta", size=14),
                name="Valid break (Close)",
            )
        )
    return fig


def add_levels(fig: go.Figure, df: pd.DataFrame, levels) -> go.Figure:
    """Gambar level support/resistance horizontal (hasil `levels.track_level`).

    Tiap level digambar dari bar terbentuknya sampai bar terkini, warnanya
    mengikuti PERAN SAAT INI (hijau support, merah resistance) karena buku
    menyatakan peran berbalik setelah ditembus. Valid break ditandai berlian,
    pullback yang bertahan (harga kembali menguji level yang sudah dilewati)
    ditandai lingkaran, keduanya di harga level.
    """
    for level in levels:
        color = "green" if level.role == SUPPORT else "red"
        x = [level.origin_index, level.end_index]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[level.price, level.price],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"{level.role.capitalize()} {level.price:g}",
                showlegend=False,
                hovertemplate=(
                    f"{level.role} {level.price:g}<br>awal: {level.initial_role}"
                    f"<br>usia: {level.age_bars} bar<extra></extra>"
                ),
            )
        )
        breaks = level.valid_breaks
        if breaks:
            fig.add_trace(
                go.Scatter(
                    x=[e.index for e in breaks],
                    y=[level.price] * len(breaks),
                    mode="markers",
                    marker=dict(symbol="diamond-open", color=color, size=10),
                    showlegend=False,
                    hovertemplate="valid break, peran berbalik<extra></extra>",
                )
            )
        pullbacks = level.pullbacks_held
        if pullbacks:
            fig.add_trace(
                go.Scatter(
                    x=[e.index for e in pullbacks],
                    y=[level.price] * len(pullbacks),
                    mode="markers",
                    marker=dict(symbol="circle", color=color, size=8),
                    showlegend=False,
                    hovertemplate=f"pullback: {level.role} diuji & bertahan<extra></extra>",
                )
            )
    return fig


def add_breakout_plan(fig: go.Figure, df: pd.DataFrame, signal, plan, position) -> go.Figure:
    """Tandai satu rencana breakout (hasil `breakout.*`): titik masuk (segitiga),
    garis cut-loss dari bar masuk sampai bar keluar / terkini, dan titik
    keluar (silang) bila rencana sudah menghasilkan keluar."""
    if signal.entry_index is None:
        return fig
    end = position.exit_index if position.exit_index is not None else len(df) - 1
    fig.add_trace(
        go.Scatter(
            x=[signal.entry_index],
            y=[signal.entry_price],
            mode="markers",
            marker=dict(symbol="triangle-up", color="cyan", size=13),
            name=f"Masuk 2nd day {signal.entry_price:g}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[signal.entry_index, end],
            y=[plan.cut_loss, plan.cut_loss],
            mode="lines",
            line=dict(color="cyan", width=1, dash="dashdot"),
            name=f"Cut-loss {plan.cut_loss:g}",
        )
    )
    if position.exit_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[position.exit_index],
                y=[position.exit_price],
                mode="markers",
                marker=dict(symbol="x", color="cyan", size=13),
                name=f"Keluar: {position.status}",
            )
        )
    return fig


def add_channel(fig: go.Figure, df: pd.DataFrame, channel) -> go.Figure:
    """Gambar channel line (proyeksi sejajar basic trendline) + titik acuannya.

    Basic trendline-nya sendiri sudah digambar `add_trendline`; di sini hanya
    sisi seberang koridor, dengan area di antara keduanya diberi arsiran tipis
    supaya "saluran"-nya terlihat.
    """
    bar_indices = np.arange(channel.basic.start_index, channel.basic.end_index + 1)
    x = bar_indices
    color = "green" if channel.is_uptrend else "red"

    fig.add_trace(
        go.Scatter(
            x=x,
            y=[channel.basic_value_at(i) for i in bar_indices],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[channel.channel_value_at(i) for i in bar_indices],
            mode="lines",
            line=dict(color=color, width=2, dash="dot"),
            fill="tonexty",
            fillcolor="rgba(128,128,128,0.12)",
            name="Channel line",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[channel.anchor.index],
            y=[channel.anchor.price],
            mode="markers",
            marker=dict(symbol="square-open", color=color, size=12, line=dict(width=2)),
            name="Titik channel line",
        )
    )
    return fig


def add_fan(fig: go.Figure, df: pd.DataFrame, fan) -> go.Figure:
    """Gambar tiga garis kipas (hasil `fan.detect_fan`) dari titik pangkal yang sama.

    Garis yang sudah tertembus digambar lebih tipis (perannya berbalik jadi
    penghalang), titik penembusan ditandai silang, dan penembusan garis
    ketiga - konfirmasi reversal - ditandai bintang.
    """
    color = "orange" if fan.direction == "bearish" else "blue"
    for line in fan.lines:
        end = line.break_index if line.is_broken else fan.lines[-1].anchor.index
        end = max(end, line.anchor.index)
        bar_indices = np.arange(line.origin.index, min(end + 1, len(df)))
        fig.add_trace(
            go.Scatter(
                x=bar_indices,
                y=[line.value_at(i) for i in bar_indices],
                mode="lines",
                line=dict(color=color, width=1 if line.is_broken else 2),
                opacity=0.6 if line.is_broken else 1.0,
                name=f"Fan line {line.order}"
                + (f" ({line.role_after_break})" if line.is_broken else ""),
            )
        )
        if line.is_broken:
            fig.add_trace(
                go.Scatter(
                    x=[line.break_index],
                    y=[df["Close"].iloc[line.break_index]],
                    mode="markers",
                    marker=dict(symbol="x-thin", color=color, size=10, line=dict(width=2)),
                    showlegend=False,
                    hovertemplate=f"garis {line.order} tertembus<extra></extra>",
                )
            )
        if line.retest_indices:
            fig.add_trace(
                go.Scatter(
                    x=line.retest_indices,
                    y=[line.value_at(i) for i in line.retest_indices],
                    mode="markers",
                    marker=dict(symbol="circle-open", color=color, size=7),
                    showlegend=False,
                    hovertemplate=(
                        f"uji ulang garis {line.order} sebagai "
                        f"{line.role_after_break}<extra></extra>"
                    ),
                )
            )

    fig.add_trace(
        go.Scatter(
            x=[fan.origin.index],
            y=[fan.origin.price],
            mode="markers",
            marker=dict(symbol="star", color=color, size=12),
            name="Pangkal kipas",
        )
    )
    if fan.confirmation_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[fan.confirmation_index],
                y=[df["Close"].iloc[fan.confirmation_index]],
                mode="markers",
                marker=dict(symbol="star-diamond", color=color, size=15),
                name=f"Reversal {fan.direction} terkonfirmasi",
            )
        )
    return fig


def add_gaps(fig: go.Figure, df: pd.DataFrame, gaps, row: int = 1, col: int = 1) -> go.Figure:
    """Arsir celah kosong (hasil `gaps.detect_gaps`) di panel harga.

    Yang digambar adalah zona yang BENAR-BENAR tersisa di chart setelah sesi
    pembukanya selesai (`Gap.visible_zone`) - sesuai buku, lompatan yang
    tertutup pergerakan sesi itu sendiri tidak meninggalkan gap. Zona
    dibentang dari bar pembuka sampai bar yang menutupnya (atau bar terakhir
    bila belum tertutup), supaya terlihat berapa lama celah itu bertahan.
    """
    for gap in gaps:
        zone = gap.visible_zone
        if zone is None:
            continue
        lower, upper = zone
        end = gap.filled_index if gap.filled_index is not None else len(df) - 1
        color = "rgba(0,200,120,0.18)" if gap.is_up else "rgba(230,70,70,0.18)"
        fig.add_shape(
            type="rect",
            x0=gap.index - 0.5, x1=max(end, gap.index) + 0.5,
            y0=lower, y1=upper,
            fillcolor=color, line=dict(width=0), layer="below",
            row=row, col=col,
        )
        # Titik tak terlihat untuk hover: shape tidak bisa menampilkan tooltip.
        fig.add_trace(
            go.Scatter(
                x=[gap.index],
                y=[(lower + upper) / 2],
                mode="markers",
                marker=dict(symbol="diamond-wide", size=9,
                            color="green" if gap.is_up else "red", opacity=0.8),
                name=f"{gap.kind} {gap.size_pct:.1f}%",
                showlegend=False,
                hovertemplate=(
                    f"{gap.kind} {gap.size_pct:.1f}%<br>"
                    f"zona {lower:,.0f}–{upper:,.0f}<br>"
                    f"{'sudah tertutup' if gap.is_filled else 'masih terbuka'}"
                    "<extra></extra>"
                ),
            ),
            row=row, col=col,
        )
    return fig
