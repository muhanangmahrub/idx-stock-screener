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
