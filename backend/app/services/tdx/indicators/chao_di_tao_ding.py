"""
抄底逃顶 — Pine Momentum Line 移植 + 信号质量优化版.

基础公式同 dongli_xian:
    动力线 = EMA((CLOSE - LLV(LOW,10)) / (HHV(HIGH,25) - LLV(LOW,10)) × 4, 4)

水平参考线:
    0.2  深底
    0.5  关注线
    1.75 强弱分界
    3.2  阶段卖出
    3.5  清仓

相对原 dongli_xian 的改进 (买点质量):
    1. 🟢 深 V 反转 (大三角)：动量上穿 0.5 + 过去 20 bar 触及过 0.2 + 趋势 + 量能
       → 高准买点 (回测 dt-30 在 8 股池上 80% 胜率, +9.78%/笔)
    2. 🟣 W 双底 (三角)：第一深坑 → 反弹 ≥ 1.5 → 第二浅坑 → 反弹日确认
    3. 🟠 底背驰 (圆点)：价格新低 + 动量未新低 + 反弹日 + SMA20 上穿
       (实验性 — 大票回测仅 3 笔, 大池上可作为补充提示)

卖点 (沿用原 TDX 信号):
    🔵 阶段卖出: 动量下穿 3.2  (cyan 倒三角)
    🔴 清仓:    动量下穿 3.5  (red 倒三角)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.strategy_templates.diao_di import (
    compute_momentum_line,
    detect_bullish_divergence,
    detect_double_dip,
)
from app.services.tdx.functions import CROSS
from app.services.tdx.indicators.base import (
    IndicatorBand,
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "chao_di_tao_ding"
label = "抄底逃顶（深V+背驰+W底）"
pane = "sub"
min_bars = 60


def _shift1(a):
    return np.concatenate(([a[0]], a[:-1]))


def compute(df: pd.DataFrame) -> IndicatorResult:
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    open_ = df["open"].astype(float).values
    vol_col = "volume" if "volume" in df.columns else "vol"
    vol = df[vol_col].astype(float).values
    ts_col = df["timestamp"].astype("int64")

    momentum = compute_momentum_line(close, low, high)
    mom_series = pd.Series(momentum, index=df.index)

    # 双色动力线: 上行段橙 (#f36421), 下行段绿 — 沿用 Pine 原版配色
    prev_mom = _shift1(momentum)
    is_up = momentum > prev_mom
    line_up = np.where(is_up, momentum, np.nan)
    line_down = np.where(~is_up, momentum, np.nan)

    # ---------- 买点 ----------
    cross_attention = pd.Series(prev_mom <= 0.5, index=df.index) & (mom_series > 0.5)
    was_bottomed = (
        pd.Series((momentum <= 0.2).astype(float), index=df.index)
        .rolling(20, min_periods=1).max() > 0
    )
    sma60 = pd.Series(close, index=df.index).rolling(60, min_periods=10).mean()
    vma20 = pd.Series(vol, index=df.index).rolling(20, min_periods=5).mean()
    trend_ok = (close > (sma60 * 0.95).fillna(0).values)
    vol_ok = (vol > (vma20 * 1.05).fillna(0).values)
    deep_v_mask = (cross_attention & was_bottomed).values & trend_ok & vol_ok

    div_mask = detect_bullish_divergence(close, open_, high, low, vol, momentum)
    w_mask = detect_double_dip(close, open_, vol, momentum)

    # ---------- 卖点 ----------
    sell_stage = CROSS(pd.Series(3.2, index=df.index), mom_series).fillna(0).astype(bool).values
    sell_all = CROSS(pd.Series(3.5, index=df.index), mom_series).fillna(0).astype(bool).values

    # 标记的 y 值 = 当时的动量线值 (信号点准确位于线上)
    def _markers_at_line(mask, color, icon, y_offset=0.0):
        return [
            IndicatorMarker(
                timestamp=int(ts_col.iloc[i]),
                value=float(momentum[i]) + y_offset,
                color=color, icon=icon,
            )
            for i in np.where(mask)[0]
        ]

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=None,  # 自适应 — 动力线天然 0..4 范围
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("动力线↑", series_to_json(pd.Series(line_up)),
                          "#f36421", thickness=2),
            IndicatorLine("动力线↓", series_to_json(pd.Series(line_down)),
                          "#3CB371", thickness=2),
        ],
        hlines=[
            IndicatorHLine("深底 0.2", 0.2, "#70DB93"),
            IndicatorHLine("关注 0.5", 0.5, "#FFFF00"),
            IndicatorHLine("强弱 1.75", 1.75, "#70DB93", dashed=True),
            IndicatorHLine("阶段卖 3.2", 3.2, "#00FFFF"),
            IndicatorHLine("清仓 3.5", 3.5, "#0088FF"),
        ],
        bands=[],  # 不画大块色柱，避免遮挡
        markers=[
            *_markers_at_line(deep_v_mask, "#FF3030", "triangle_up", -0.15),
            *_markers_at_line(w_mask, "#9933FF", "triangle_up", -0.15),
            *_markers_at_line(div_mask, "#FF8800", "dot", -0.15),
            *_markers_at_line(sell_stage, "#00CC00", "triangle_down", 0.15),
            *_markers_at_line(sell_all, "#0088FF", "triangle_down", 0.15),
        ],
    )
