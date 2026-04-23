"""
Multi-period KDJ resonance indicator.

Computes KDJ on four periods (13, 21, 34, 55) and highlights bars where all
four K/D pairs are below 20 (bottom resonance, DIBU) or above 80 (top
resonance, TOBU).

Port of the TDX formula:

    VAR3  := (CLOSE - LLV(LOW, 13)) / (HHV(HIGH, 13) - LLV(LOW, 13)) * 100
    K13 :  SMA(VAR3, 3, 1)
    D14 :  SMA(K13, 3, 1)
    ... (same for periods 21, 34, 55)
    K55 :  SMA(VAR55, 5, 1), COLORGREEN, LINETHICK1
    D55 :  SMA(K55, 5, 1),   COLORFF9933

    DIBU := K13<20 AND D14<20 AND ... AND K55<20 AND D55<20
    TOBU := K13>80 AND D14>80 AND ... AND K55>80 AND D55>80

Only K13/D14/K55/D55 are visible lines (matching the original `:` outputs);
K21/D21/K34/D34 are internal intermediates for resonance detection.
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import HHV, LLV, SMA
from app.services.tdx.indicators.base import (
    IndicatorBand,
    IndicatorHLine,
    IndicatorLine,
    IndicatorResult,
    series_to_json,
)


name = "multi_kdj"
label = "多周期 KDJ 共振"
pane = "sub"
min_bars = 55


def _kdj(close: pd.Series, high: pd.Series, low: pd.Series,
         period: int, sma_window: int) -> tuple[pd.Series, pd.Series]:
    rsv = (close - LLV(low, period)) / (HHV(high, period) - LLV(low, period)) * 100
    k = SMA(rsv, sma_window, 1)
    d = SMA(k, sma_window, 1)
    return k, d


def compute(df: pd.DataFrame) -> IndicatorResult:
    """
    df must contain columns: timestamp, open, high, low, close (ordered oldest->newest).
    """
    close, high, low = df["close"], df["high"], df["low"]

    k13, d14 = _kdj(close, high, low, 13, 3)
    k21, d21 = _kdj(close, high, low, 21, 3)
    k34, d34 = _kdj(close, high, low, 34, 3)
    k55, d55 = _kdj(close, high, low, 55, 5)

    dibu = (k13 < 20) & (d14 < 20) & (k21 < 20) & (d21 < 20) & \
           (k34 < 20) & (d34 < 20) & (k55 < 20) & (d55 < 20)
    tobu = (k13 > 80) & (d14 > 80) & (k21 > 80) & (d21 > 80) & \
           (k34 > 80) & (d34 > 80) & (k55 > 80) & (d55 > 80)

    ts_col = df["timestamp"].astype("int64")

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(0, 100),
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("K13", series_to_json(k13), "#FF6B6B", thickness=1),
            IndicatorLine("D14", series_to_json(d14), "#FFE66D", thickness=1),
            IndicatorLine("K55", series_to_json(k55), "#00FF00", thickness=1),
            IndicatorLine("D55", series_to_json(d55), "#FF9933", thickness=1),
        ],
        hlines=[
            IndicatorHLine("天线", 100, "#FFFFFF", dashed=True),
            IndicatorHLine("顶部", 80, "#55AA77"),
            IndicatorHLine("中轴线", 50, "#FFFFFF", dashed=True),
            IndicatorHLine("底部", 20, "#55AA77"),
            IndicatorHLine("底线", 0, "#FFFFFF", dashed=True),
        ],
        bands=[
            IndicatorBand(
                timestamps=ts_col[dibu.fillna(False)].tolist(),
                y1=60, y2=80,
                color="#FF0000", opacity=0.6,
            ),
            IndicatorBand(
                timestamps=ts_col[tobu.fillna(False)].tolist(),
                y1=20, y2=40,
                color="#00FF00", opacity=0.6,
            ),
        ],
    )
