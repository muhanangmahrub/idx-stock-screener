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
    if break_check.gap_back_indices:
        fig.add_trace(
            go.Scatter(
                x=df.index[break_check.gap_back_indices],
                y=df["Close"].iloc[break_check.gap_back_indices],
                mode="markers",
                marker=dict(symbol="diamond-open", color="magenta", size=12),
                name="break, gap kembali (2nd day gagal)",
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


def add_levels(fig: go.Figure, df: pd.DataFrame, levels) -> go.Figure:
    """Gambar level support/resistance horizontal (hasil `levels.track_level`).

    Tiap level digambar dari bar terbentuknya sampai bar terkini, warnanya
    mengikuti PERAN SAAT INI (hijau support, merah resistance) karena buku
    menyatakan peran berbalik setelah ditembus. Valid break ditandai berlian,
    pullback yang bertahan (harga kembali menguji level yang sudah dilewati)
    ditandai lingkaran, keduanya di harga level.
    """
    for level in levels:
        color = "green" if level.role == "support" else "red"
        x = [df.index[level.origin_index], df.index[level.end_index]]
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[level.price, level.price],
                mode="lines",
                line=dict(color=color, width=1, dash="dot"),
                name=f"{level.role} {level.price:g}",
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
                    x=df.index[[e.index for e in breaks]],
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
                    x=df.index[[e.index for e in pullbacks]],
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
            x=[df.index[signal.entry_index]],
            y=[signal.entry_price],
            mode="markers",
            marker=dict(symbol="triangle-up", color="cyan", size=13),
            name=f"masuk 2nd day {signal.entry_price:g}",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df.index[signal.entry_index], df.index[end]],
            y=[plan.cut_loss, plan.cut_loss],
            mode="lines",
            line=dict(color="cyan", width=1, dash="dashdot"),
            name=f"cut-loss {plan.cut_loss:g}",
        )
    )
    if position.exit_index is not None:
        fig.add_trace(
            go.Scatter(
                x=[df.index[position.exit_index]],
                y=[position.exit_price],
                mode="markers",
                marker=dict(symbol="x", color="cyan", size=13),
                name=f"keluar: {position.status}",
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
    x = df.index[bar_indices]
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
            name="channel line",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[df.index[channel.anchor.index]],
            y=[channel.anchor.price],
            mode="markers",
            marker=dict(symbol="square-open", color=color, size=12, line=dict(width=2)),
            name="titik channel line",
        )
    )
    return fig
