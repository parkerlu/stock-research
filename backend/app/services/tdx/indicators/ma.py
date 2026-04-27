"""
均线 (Moving Average) — multi-period simple moving average overlaid on the
main candle pane. Standard A-share configuration: MA5, MA10, MA20, MA60.

TDX formula:
    MA5  : MA(CLOSE, 5)
    MA10 : MA(CLOSE, 10)
    MA20 : MA(CLOSE, 20)
    MA60 : MA(CLOSE, 60)
"""
from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import MA
from app.services.tdx.indicators.base import (
    IndicatorLine,
    IndicatorResult,
    series_to_json,
)


name = "ma"
label = "均线 (MA)"
pane = "main"
min_bars = 60


def compute(df: pd.DataFrame) -> IndicatorResult:
    close = df["close"].astype(float)
    ts_col = df["timestamp"].astype("int64")

    ma5 = MA(close, 5)
    ma10 = MA(close, 10)
    ma20 = MA(close, 20)
    ma60 = MA(close, 60)

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("MA5", series_to_json(ma5), "#FFFFFF", thickness=1),
            IndicatorLine("MA10", series_to_json(ma10), "#FFE66D", thickness=1),
            IndicatorLine("MA20", series_to_json(ma20), "#FF6B6B", thickness=1),
            IndicatorLine("MA60", series_to_json(ma60), "#00CFFF", thickness=1),
        ],
    )
