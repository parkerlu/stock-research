"""
缠论买点 (1类 / 2类) — 复用 services/chanlun.py 的引擎，与策略同源.

图上画什么
----------
    笔 (Stroke)   主图上的折线, 上涨笔红 / 下跌笔绿 —— 含包处理 + 分型识别后
                  连成的骨架, 缠论的所有判断都建立在它之上
    1类买点        红色上三角, 画在该 bar 的低点下方
                  = 最后一笔下跌创了新低, 但 MACD 负面积小于前一笔下跌 (底背驰)
    2类买点        黄色上三角
                  = 1类之后第一次反弹(顶分型)再回踩(底分型), 且新低不破 1类的低点

为什么单独收录 2 类
------------------
全 universe 4306 只 / 2015-2026 / 34830 个买点的体检:
  chan-2buy  超额 +0.59%/笔  t=7.8  OOS +0.96%  持仓 13d
在 47 个策略里排第 2, 且持仓最短(资金效率最高)。

组合层(20 仓位/复利/含冲击成本)单独跑更突出: 35.87x / 年化 36.0% / 回撤 13.9%,
是全部信号源里最好的一个。

出场参数的实测结论 (扫了 60+ 组合, 写在这里免得以后重复踩坑):
  · 止盈位越低越好 —— 次批 +8% 年化 47.4%, +25% 只有 28.9%, 不封顶更是掉到 9.4%。
    原因是仓位周转: 不封顶 11.6 年只成交 1175 笔, +8% 能成交 4463 笔。
    "让利润跑"在有仓位约束的组合里是灾难。
  · 止损 6% 是甜点, 3% 太紧 / 12% 太松, 两边都掉 2 个百分点。
  · 首批止盈后必须把止损上移到成本价 —— 开/关差 8 个百分点 (28.9% vs 20.6%)。
  · 最优: 止损6% + 首批 +4% 出一半(保本上移) + 次批 +8% 全出
    -> 年化 45.7% / 回撤 6.4% / 夏普 5.12 (原版策略自带出场是 34.6% / 10.1% / 2.64)
  ⚠️ 该参数在同一批数据上精调所得, OOS 只有 3.7 年; 且 +4% 就减仓意味着
     对手续费和滑点非常敏感。实盘前务必按自己的真实成本重估。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.chanlun import (
    find_class1_buys,
    find_class2_buys,
    find_fractals,
    find_strokes,
    merge_inclusion,
)
from app.services.tdx.indicators.base import (
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "chan_buy"
label = "缠论买点 (1类/2类)"
pane = "main"          # 笔要画在 K 线上
# 笔需要至少 5 根原始K, 背驰判断要 MACD(26) 预热, 再加几笔的历史
min_bars = 120


def compute_points(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> tuple[list, list]:
    """返回 (1类买点, 2类买点)。与 chanlun_strategy 完全同源。"""
    c = close.astype(float).values
    h = high.astype(float).values
    l = low.astype(float).values
    c1 = find_class1_buys(c, h, l)
    c2 = find_class2_buys(c, h, l, {b.bar_idx for b in c1})
    return c1, c2


def compute(df: pd.DataFrame) -> IndicatorResult:
    close, high, low = df["close"], df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")
    n = len(df)

    c = close.astype(float).values
    h = high.astype(float).values
    l = low.astype(float).values

    # ---- 笔: 把折线还原成与原始 K 线对齐的序列 ----
    # ⚠️ 只在转折点给值、中间留 None 是行不通的: klinecharts 的 line 不会跨
    # null 连线, 反而把 null 当 0, 结果是笔不显示且 Y 轴被拉到 0。
    # 必须把每一笔的两端之间线性插值填满。
    bars = merge_inclusion(h, l)
    strokes = find_strokes(find_fractals(bars), min_bars=5)
    up: list[float | None] = [None] * n     # 上涨笔
    dn: list[float | None] = [None] * n     # 下跌笔
    for s in strokes:
        i, j = s.start.bar_idx, s.end.bar_idx
        if not (0 <= i < n and 0 <= j < n) or j <= i:
            continue
        tgt = up if s.direction == "up" else dn
        p0, p1 = float(s.start.price), float(s.end.price)
        for k in range(i, j + 1):
            tgt[k] = p0 + (p1 - p0) * (k - i) / (j - i)

    c1, c2 = compute_points(close, high, low)

    # 三角画在 bar 低点下方一点, 免得压住 K 线
    span = float(np.nanmax(h) - np.nanmin(l)) if n else 0.0
    off = span * 0.012 if span > 0 else 0.0

    markers = [
        *(IndicatorMarker(timestamp=int(ts_col.iloc[b.bar_idx]),
                          value=float(l[b.bar_idx]) - off,
                          color="#FF3333", icon="triangle_up")
          for b in c1 if 0 <= b.bar_idx < n),
        *(IndicatorMarker(timestamp=int(ts_col.iloc[b.bar_idx]),
                          value=float(l[b.bar_idx]) - off,
                          color="#E8B54D", icon="triangle_up")
          for b in c2 if 0 <= b.bar_idx < n),
    ]

    warnings: list[str] = []
    if not strokes:
        warnings.append("笔识别为空 —— 该区间波动不足以形成有效分型")

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("上涨笔", series_to_json(pd.Series(up)), "#E94560", thickness=2),
            IndicatorLine("下跌笔", series_to_json(pd.Series(dn)), "#4CAF50", thickness=2),
        ],
        markers=markers,
        warnings=warnings,
    )
