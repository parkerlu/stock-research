"""
钱德动量摆动指标 (Chande Momentum Oscillator, CMO) — Tushar Chande.

Pine 源码逐行移植 (f13end/tradingview-custom-indicators/oscillators/
chande_momentum_oscillator.pine, MIT):

    momm  = change(src)
    m1    = momm >= 0 ? momm : 0.0
    m2    = momm >= 0 ? 0.0  : -momm
    sm1   = sum(m1, length)
    sm2   = sum(m2, length)
    cmo   = 100 * (sm1 - sm2) / (sm1 + sm2)

在算什么
--------
和 RSI 同源但分子不同: RSI 是 涨幅/(涨幅+跌幅)，CMO 是
(涨幅−跌幅)/(涨幅+跌幅) —— 直接给出净动量，取值 -100..+100 且以 0 为中轴。
比 RSI 更敏感、不做平滑，所以拐点更早但也更毛躁。

    CMO < -50 : 单边下跌，跌幅几乎垄断了全部波动 → 超卖
    CMO > +50 : 单边上涨

为什么收录
----------
全 universe 4306 只、2015-2026 回测: 买入信号(CMO 上穿 -50) 净边际
+0.20%/笔 (t=5.0)，在移植的 24 个 Pine 指标里排第 3
—— "净"指已扣除同频随机入场在同一出场规则下的基线 (约 +0.32%)。
注意 OOS 净边际只有 +0.08%，样本外衰减明显，比 RVI/PGO 弱。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.tdx.functions import SUM
from app.services.tdx.indicators.base import (
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "chande_cmo"
label = "钱德动量 (CMO)"
pane = "sub"
min_bars = 15

LENGTH = 9
OVERSOLD = -50.0
OVERBOUGHT = 50.0


def compute_cmo(close: pd.Series, n: int = LENGTH) -> pd.Series:
    d = close.astype(float).diff()
    sm1 = SUM(d.clip(lower=0), n)
    sm2 = SUM((-d).clip(lower=0), n)
    total = (sm1 + sm2).replace(0.0, np.nan)
    return 100 * (sm1 - sm2) / total


def compute_edges(close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """边沿信号: 买 = 上穿 -50 (单边下跌结束), 卖 = 下穿 +50。"""
    cmo = compute_cmo(close)
    buy = (cmo.shift(1) <= OVERSOLD) & (cmo > OVERSOLD)
    sell = (cmo.shift(1) >= OVERBOUGHT) & (cmo < OVERBOUGHT)
    return buy.fillna(False), sell.fillna(False)


def compute(df: pd.DataFrame) -> IndicatorResult:
    close = df["close"]
    ts_col = df["timestamp"].astype("int64")

    cmo = compute_cmo(close)
    buy, sell = compute_edges(close)

    markers = [
        *(IndicatorMarker(timestamp=ts, value=OVERSOLD, color="#FF3333",
                          icon="triangle_up")
          for ts in ts_col[buy].tolist()),
        *(IndicatorMarker(timestamp=ts, value=OVERBOUGHT, color="#00CC00",
                          icon="triangle_down")
          for ts in ts_col[sell].tolist()),
    ]

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(-100.0, 100.0),
        timestamps=ts_col.tolist(),
        lines=[IndicatorLine("CMO", series_to_json(cmo), "#E8B54D", thickness=2)],
        hlines=[
            IndicatorHLine("超买 +50", OVERBOUGHT, "#E69138", dashed=True),
            IndicatorHLine("零轴", 0.0, "#989898", dashed=True),
            IndicatorHLine("超卖 -50", OVERSOLD, "#E69138", dashed=True),
        ],
        markers=markers,
    )
