"""
相对波动率指数 (Relative Volatility Index, RVI) — Donald Dorsey.

Pine 源码逐行移植 (f13end/tradingview-custom-indicators/oscillators/
relative_volatility_index.pine, MIT):

    sampleStdev(src, length) =>
        dev = src - sma(src, length)
        variance = sum(dev * dev, length) / (length - 1)
        sqrt(variance)
    up(src)   => change(src) > 0 ? selectedStdev : 0
    down(src) => change(src) > 0 ? 0 : selectedStdev
    rviOriginal(src) =>
        upSum   = ema(up(src), emaLength)
        downSum = ema(down(src), emaLength)
        100 * upSum / (upSum + downSum)
    rvi = (rviOriginal(high) + rviOriginal(low)) / 2      ; refined 版

在算什么
--------
和 RSI 的分子分母用"涨跌幅"不同，RVI 用的是**标准差** —— 上涨日的波动
占总波动的比例。所以它衡量的不是"跌了多少"，而是"下跌的动能有多强"。
RVI < 20 表示下跌波动已经衰竭，是超卖信号；> 80 表示上涨动能过热。

refined 版对 HIGH 和 LOW 各算一遍再取平均，比只用收盘价稳定。

为什么收录
----------
全 universe 4306 只、2015-2026、300 只抽样回测: 买入信号(RVI 上穿 20)
净边际 +0.55%/笔 (t=7.6, OOS +0.82%)，是移植的 24 个 Pine 指标里最高的
—— "净"指已扣除同频随机入场在同一出场规则下的基线 (约 +0.35%)。

⚠️ 但它在**组合层面是负贡献**: 单独跑 40 仓位只有年化 11.3%、回撤 46.8%,
加进 chan+kdj 组合会把年化从 21.8% 拖到 18.5%。原因是触发太频繁(全市场
73814 笔)且持仓偏长(28天), 会挤占更优信号的仓位。**只作看图参考, 不建议
直接照着做交易。**
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.tdx.functions import EMA
from app.services.tdx.indicators.base import (
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "rvi_dorsey"
label = "相对波动率指数 (RVI)"
pane = "sub"
# stdev(10) 之后再 EMA(14)，约 25 根才稳定
min_bars = 30

STDEV_LEN = 10
EMA_LEN = 14
OVERSOLD = 20.0
OVERBOUGHT = 80.0


def _rvi_one(src: pd.Series) -> pd.Series:
    """单条序列的 RVI。Pine 的 sampleStdev 是样本标准差 (ddof=1)。"""
    sd = src.rolling(STDEV_LEN, min_periods=STDEV_LEN).std(ddof=1)
    ch = src.diff()
    up = EMA(sd.where(ch > 0, 0.0), EMA_LEN)
    dn = EMA(sd.where(ch <= 0, 0.0), EMA_LEN)
    total = (up + dn).replace(0.0, np.nan)
    return 100 * up / total


def compute_rvi(high: pd.Series, low: pd.Series) -> pd.Series:
    """refined RVI = (RVI(HIGH) + RVI(LOW)) / 2。"""
    return (_rvi_one(high.astype(float)) + _rvi_one(low.astype(float))) / 2


def compute_edges(high: pd.Series, low: pd.Series) -> tuple[pd.Series, pd.Series]:
    """边沿信号: 买 = 上穿 20 (下跌波动衰竭), 卖 = 下穿 80。"""
    rvi = compute_rvi(high, low)
    buy = (rvi.shift(1) <= OVERSOLD) & (rvi > OVERSOLD)
    sell = (rvi.shift(1) >= OVERBOUGHT) & (rvi < OVERBOUGHT)
    return buy.fillna(False), sell.fillna(False)


def compute(df: pd.DataFrame) -> IndicatorResult:
    high, low = df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")

    rvi = compute_rvi(high, low)
    buy, sell = compute_edges(high, low)

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
        y_axis_range=(0.0, 100.0),
        timestamps=ts_col.tolist(),
        lines=[IndicatorLine("RVI", series_to_json(rvi), "#008000", thickness=2)],
        hlines=[
            IndicatorHLine("超买 80", OVERBOUGHT, "#E69138", dashed=True),
            IndicatorHLine("中轴 50", 50.0, "#989898", dashed=True),
            IndicatorHLine("超卖 20", OVERSOLD, "#E69138", dashed=True),
        ],
        markers=markers,
    )
