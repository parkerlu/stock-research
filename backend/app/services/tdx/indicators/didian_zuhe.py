"""
低点组合 (Didian Zuhe) — composite indicator merging 多周期KDJ共振 and 动力线阶段
into a single sub-pane view.

This is a 1:1 merge of the existing `multi_kdj` and `dongli_xian` indicators:
the calculations are identical; only the rendering is combined into one pane
with shared 0..100 y-axis. The hidden `A1` variable from the 信号灯 section
(9-day RSV, never displayed in the original TDX formula) is intentionally
omitted.

Y-axis layout (0..100):

    100 ── 天线        ┐
     90 ── 短线卖出 (green band)
     80 ── 顶部 / KDJ TOBU (green band 20..40 → mirror at top? see below)
     70 ── 清仓 (blue band 70..100)
     50 ── 中轴线
     40 ── TOBU upper
     20 ── TOBU lower / 底部
      0 ── 底线
           动力线 (0..4) sits near the bottom; 0.2/3.5 thresholds marked

Signal bands (rendered as per-bar rectangles):
  - DIBU       (multi-KDJ bottom resonance) : red,    y 60..80
  - TOBU       (multi-KDJ top resonance)    : green,  y 20..40
  - 阶段底部    (dongli crosses above 0.2)    : red,    y 0.2..49
  - 阶段关注    (dongli crosses above 0.5)    : magenta, y 0.1..20
  - 清仓       (dongli crosses below 3.5)    : blue,   y 70..100
  - 短线卖出    (dongli crosses below 3.2)    : green,  y 90..100
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import CROSS, EMA, HHV, LLV, SMA
from app.services.tdx.indicators.base import (
    IndicatorBand,
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "didian_zuhe"
label = "低点组合"
pane = "sub"
min_bars = 55  # KDJ 55-period is the longest window


def _kdj(close: pd.Series, high: pd.Series, low: pd.Series,
         period: int, sma_window: int) -> tuple[pd.Series, pd.Series]:
    rsv = (close - LLV(low, period)) / (HHV(high, period) - LLV(low, period)) * 100
    k = SMA(rsv, sma_window, 1)
    d = SMA(k, sma_window, 1)
    return k, d


def compute(df: pd.DataFrame) -> IndicatorResult:
    close, high, low = df["close"], df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")

    # --- Multi-period KDJ section ---
    k13, d14 = _kdj(close, high, low, 13, 3)
    k21, d21 = _kdj(close, high, low, 21, 3)
    k34, d34 = _kdj(close, high, low, 34, 3)
    k55, d55 = _kdj(close, high, low, 55, 5)

    dibu = (k13 < 20) & (d14 < 20) & (k21 < 20) & (d21 < 20) & \
           (k34 < 20) & (d34 < 20) & (k55 < 20) & (d55 < 20)
    tobu = (k13 > 80) & (d14 > 80) & (k21 > 80) & (d21 > 80) & \
           (k34 > 80) & (d34 > 80) & (k55 > 80) & (d55 > 80)

    # --- 动力线 / 阶段指标 section ---
    var2 = LLV(low, 10)
    var33 = HHV(high, 25)
    dongli = EMA((close - var2) / (var33 - var2) * 4, 4)

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
            IndicatorLine("K13", series_to_json(k13), "#FF6B6B", thickness=1),
            IndicatorLine("D14", series_to_json(d14), "#FFE66D", thickness=1),
            IndicatorLine("K55", series_to_json(k55), "#00FF00", thickness=1),
            IndicatorLine("D55", series_to_json(d55), "#FF9933", thickness=1),
            IndicatorLine("动力线", series_to_json(dongli), "#A8A8A8", thickness=1),
        ],
        hlines=[
            # KDJ reference lines (primary 0..100 scale)
            IndicatorHLine("天线", 100, "#FFFFFF", dashed=True),
            IndicatorHLine("顶部", 80, "#55AA77"),
            IndicatorHLine("中轴线", 50, "#FFFFFF", dashed=True),
            IndicatorHLine("底部", 20, "#55AA77"),
            IndicatorHLine("底线", 0, "#FFFFFF", dashed=True),
            # 动力线 key thresholds (0.5/3.2 omitted to reduce clutter)
            IndicatorHLine("动力线底", 0.2, "#70DB93"),
            IndicatorHLine("动力线顶", 3.5, "#0088FF"),
        ],
        bands=[
            # Multi-KDJ resonance bands
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
            # 动力线 阶段 bands
            IndicatorBand(
                timestamps=stage_bottom_ts,
                y1=0.2, y2=49,
                color="#FF0000", opacity=0.35,
            ),
            IndicatorBand(
                timestamps=stage_watch_ts,
                y1=0.1, y2=20,
                color="#FF00FF", opacity=0.35,
            ),
            IndicatorBand(
                timestamps=liquidate_ts,
                y1=70, y2=100,
                color="#0088FF", opacity=0.5,
            ),
            IndicatorBand(
                timestamps=short_sell_ts,
                y1=90, y2=100,
                color="#00FF00", opacity=0.6,
            ),
        ],
        markers=[
            IndicatorMarker(
                timestamp=ts, value=50,
                color="#FF0000", icon="triangle_up",
            )
            for ts in stage_bottom_ts
        ],
    )
