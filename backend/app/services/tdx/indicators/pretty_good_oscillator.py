"""
完美振荡器 (Pretty Good Oscillator, PGO) — Mark Johnson.

Pine 源码逐行移植 (f13end/tradingview-custom-indicators/oscillators/
pretty_good_oscillator.pine, MIT):

    pgo = (src - sma(src, length)) / atr(length)

在算什么
--------
收盘价偏离 N 日均线多少个 ATR。分子是"偏离多远"，分母是"这只票平时波动
多大" —— 相当于用波动率把偏离标准化，所以不同价位、不同波动的股票可以横向
比较（这点比 BIAS / 乖离率 强）。

    PGO < -2  : 偏离均线超过 2 个 ATR 的下方 → 超卖
    PGO > +3  : Johnson 原意的做多突破区

⚠️ ATR 用 Wilder 平滑 (RMA)，与 Pine 的 atr() 一致；不是简单均值。

为什么收录
----------
全 universe 4306 只、2015-2026 回测: 买入信号(PGO 上穿 -2) 净边际
+0.25%/笔 (t=4.8, OOS +0.34%)，在移植的 24 个 Pine 指标里排第 2
—— "净"指已扣除同频随机入场在同一出场规则下的基线 (约 +0.36%)。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.tdx.functions import ABS, MA, MAX, REF
from app.services.tdx.indicators.base import (
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "pretty_good_osc"
label = "完美振荡器 (PGO)"
pane = "sub"
min_bars = 30  # SMA(14) + ATR(14) 的 Wilder 平滑需要预热

LENGTH = 14
OVERSOLD = -2.0
BREAKOUT = 3.0


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int) -> pd.Series:
    """Wilder ATR — 与 Pine 的 atr() 一致 (RMA 而非 SMA)。"""
    pc = REF(close, 1)
    tr = MAX(MAX(high - low, ABS(high - pc)), ABS(low - pc))
    return tr.ewm(alpha=1.0 / n, adjust=False).mean()


def compute_pgo(
    high: pd.Series, low: pd.Series, close: pd.Series, n: int = LENGTH
) -> pd.Series:
    close = close.astype(float)
    atr = _atr(high.astype(float), low.astype(float), close, n)
    return (close - MA(close, n)) / atr.replace(0.0, np.nan)


def compute_edges(
    high: pd.Series, low: pd.Series, close: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """边沿信号: 买 = 上穿 -2 (超卖修复), 卖 = 下穿 +3 (突破段结束)。"""
    pgo = compute_pgo(high, low, close)
    buy = (pgo.shift(1) <= OVERSOLD) & (pgo > OVERSOLD)
    sell = (pgo.shift(1) >= BREAKOUT) & (pgo < BREAKOUT)
    return buy.fillna(False), sell.fillna(False)


def compute(df: pd.DataFrame) -> IndicatorResult:
    high, low, close = df["high"], df["low"], df["close"]
    ts_col = df["timestamp"].astype("int64")

    pgo = compute_pgo(high, low, close)
    buy, sell = compute_edges(high, low, close)

    markers = [
        *(IndicatorMarker(timestamp=ts, value=OVERSOLD, color="#FF3333",
                          icon="triangle_up")
          for ts in ts_col[buy].tolist()),
        *(IndicatorMarker(timestamp=ts, value=BREAKOUT, color="#00CC00",
                          icon="triangle_down")
          for ts in ts_col[sell].tolist()),
    ]

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        timestamps=ts_col.tolist(),
        lines=[IndicatorLine("PGO", series_to_json(pgo), "#3D85C6", thickness=2)],
        hlines=[
            IndicatorHLine("突破 +3", BREAKOUT, "#888888", dashed=True),
            IndicatorHLine("零轴", 0.0, "#989898", dashed=True),
            IndicatorHLine("超卖 -2", OVERSOLD, "#888888", dashed=True),
        ],
        markers=markers,
    )
