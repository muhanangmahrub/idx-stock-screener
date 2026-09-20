"""Fungsi chart plotly untuk visualisasi harga saham."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go


def plot_candlestick(
    df: pd.DataFrame,
    title: str = "",
    highs_idx: np.ndarray | None = None,
    lows_idx: np.ndarray | None = None,
) -> go.Figure:
    """Buat candlestick chart dari DataFrame OHLC (kolom Open/High/Low/Close).

    `highs_idx`/`lows_idx` opsional: indeks posisi (hasil `get_extrema`) untuk
    menandai swing high/low di atas chart, sebagai bantuan visual sebelum
    detektor pola (Lapis 2) tersedia.
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df.index,
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
                name="Harga",
            )
        ]
    )

    if highs_idx is not None and len(highs_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=df.index[highs_idx],
                y=df["High"].iloc[highs_idx],
                mode="markers",
                marker=dict(symbol="triangle-down", color="red", size=9),
                name="Swing high",
            )
        )

    if lows_idx is not None and len(lows_idx) > 0:
        fig.add_trace(
            go.Scatter(
                x=df.index[lows_idx],
                y=df["Low"].iloc[lows_idx],
                mode="markers",
                marker=dict(symbol="triangle-up", color="green", size=9),
                name="Swing low",
            )
        )

    fig.update_layout(title=title, xaxis_rangeslider_visible=False)
    return fig


def add_trendline(fig: go.Figure, df: pd.DataFrame, trendline) -> go.Figure:
    """Tambahkan up-/down-trendline (hasil `trend.build_trendline`) ke chart.

    Garis digambar dari titik acuan pertama sampai `end_index` (bar terkini),
    supaya terlihat sebagai level tempat tren diuji, bukan cuma penghubung
    dua titik. Nilai dihitung per bar (bukan hanya dua ujung) karena sumbu x
    tanggal punya celah libur bursa - garis lurus di ruang indeks bar akan
    tampak patah kalau hanya dua ujungnya yang diplot.
    """
    bar_indices = np.arange(trendline.start_index, trendline.end_index + 1)
    color = "green" if trendline.kind == "up-trendline" else "red"

    fig.add_trace(
        go.Scatter(
            x=df.index[bar_indices],
            y=[trendline.value_at(i) for i in bar_indices],
            mode="lines",
            line=dict(color=color, width=2, dash="dash"),
            name=trendline.kind,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=df.index[[p.index for p in trendline.points]],
            y=[p.price for p in trendline.points],
            mode="markers",
            marker=dict(symbol="circle-open", color=color, size=14, line=dict(width=2)),
            name=f"titik {trendline.kind}",
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
    pierce_col = "Low" if trendline.kind == "up-trendline" else "High"

    if break_check.whipsaw_indices:
        fig.add_trace(
            go.Scatter(
                x=df.index[break_check.whipsaw_indices],
                y=df[pierce_col].iloc[break_check.whipsaw_indices],
                mode="markers",
                marker=dict(symbol="x", color="orange", size=11),
                name="false break (intraday)",
            )
        )
    if break_check.valid_break_index is not None:
        i = break_check.valid_break_index
        fig.add_trace(
            go.Scatter(
                x=[df.index[i]],
                y=[df["Close"].iloc[i]],
                mode="markers",
                marker=dict(symbol="diamond", color="magenta", size=14),
                name="valid break (Close)",
            )
        )
    return fig
