"""
动力线 (Dongli Xian) / 阶段指标 — "Stage Indicator" from 超级极品底 family.

Port of the TDX formula:

    VAR2     := LLV(LOW, 10)
    VAR33    := HHV(HIGH, 25)
    动力线    := EMA((CLOSE - VAR2) / (VAR33 - VAR2) * 4, 4)

    底部2    := 0.2           ; reference line
    关注     := 0.5
    阶段卖出  := 3.2
    BB       := 3.5
    清仓卖出  := 3.5

    阶段底部  := CROSS(动力线, 底部2)       ; 动力线 crosses above 0.2  → red stick + icon
    阶段关注  := CROSS(动力线, 关注)        ; 动力线 crosses above 0.5  → magenta stick
    清仓     := CROSS(清仓卖出, 动力线)     ; 动力线 crosses below 3.5 → blue stick
    短线卖出  := CROSS(阶段卖出, 动力线)     ; 动力线 crosses below 3.2 → green stick

    STICKLINE(阶段底部, 0.2,  49, 1, 0), COLORRED
    STICKLINE(阶段关注, 0.1,  20, 1, 0), COLORMAGENTA
    STICKLINE(清仓,    100,  70, 1, 0), COLORBLUE
    STICKLINE(短线卖出, 100,  90, 2, 0), COLORGREEN
    DRAWICON(阶段底部, 50, 11)

The Y-axis is 0..100. 动力线's natural range is 0..4 so the line sits near
the bottom; signal bands are drawn in the otherwise-empty upper half of
the pane, matching the original TDX layout.
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import CROSS, EMA, HHV, LLV
from app.services.tdx.indicators.base import (
    IndicatorBand,
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "dongli_xian"
label = "动力线（阶段指标）"
pane = "sub"
min_bars = 25  # HHV(HIGH, 25) is the longest window


def compute(df: pd.DataFrame) -> IndicatorResult:
    close, high, low = df["close"], df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")

    var2 = LLV(low, 10)
    var33 = HHV(high, 25)
    dongli = EMA((close - var2) / (var33 - var2) * 4, 4)

    # Signal crossovers
    stage_bottom = CROSS(dongli, pd.Series(0.2, index=dongli.index)).fillna(0)
    stage_watch = CROSS(dongli, pd.Series(0.5, index=dongli.index)).fillna(0)
    liquidate = CROSS(pd.Series(3.5, index=dongli.index), dongli).fillna(0)
    short_sell = CROSS(pd.Series(3.2, index=dongli.index), dongli).fillna(0)

    stage_bottom_ts = ts_col[stage_bottom.astype(bool)].tolist()
    stage_watch_ts = ts_col[stage_watch.astype(bool)].tolist()
    liquidate_ts = ts_col[liquidate.astype(bool)].tolist()
    short_sell_ts = ts_col[short_sell.astype(bool)].tolist()

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(0, 100),
        timestamps=ts_col.tolist(),
        lines=[
            # 动力线 (gray) — the main line; naturally in 0..4 range
            IndicatorLine("动力线", series_to_json(dongli), "#A8A8A8", thickness=1),
        ],
        hlines=[
            IndicatorHLine("底部2", 0.2, "#70DB93"),
            IndicatorHLine("关注", 0.5, "#FFFF00"),
            IndicatorHLine("阶段卖出", 3.2, "#C6C600"),
            IndicatorHLine("BB / 清仓卖出", 3.5, "#0088FF"),
        ],
        bands=[
            # 阶段底部 (buy signal) — red stick from 0.2 to 49
            IndicatorBand(
                timestamps=stage_bottom_ts,
                y1=0.2, y2=49,
                color="#FF0000", opacity=0.5,
            ),
            # 阶段关注 (watch signal) — magenta stick from 0.1 to 20
            IndicatorBand(
                timestamps=stage_watch_ts,
                y1=0.1, y2=20,
                color="#FF00FF", opacity=0.5,
            ),
            # 清仓 (sell all) — blue stick from 70 to 100
            IndicatorBand(
                timestamps=liquidate_ts,
                y1=70, y2=100,
                color="#0088FF", opacity=0.5,
            ),
            # 短线卖出 (short sell) — green stick from 90 to 100
            IndicatorBand(
                timestamps=short_sell_ts,
                y1=90, y2=100,
                color="#00FF00", opacity=0.6,
            ),
        ],
        markers=[
            # DRAWICON(阶段底部, 50, 11) — icon at y=50 when 阶段底部 fires
            IndicatorMarker(
                timestamp=ts, value=50,
                color="#FF0000", icon="triangle_up",
            )
            for ts in stage_bottom_ts
        ],
    )
