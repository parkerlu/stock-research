"""买卖很准（合并版）— 纯指标策略，无 ML.

信号与副图指标 `tdx/indicators/maimai_henzhun` **同一实现**
(共用 compute_edges, 指标图上的三角 = 本策略的买卖点, 永远一致):

    买入 = 买线 从非0回落到0    { 连续超卖状态刚结束 → 反转确认 }
    卖出 = 卖线 从<100回到100   { 强势段被打破 → 止盈离场 }

信号发射方式: 每个边沿都发出 (不做持仓配对) ——
  · 回测引擎本身只认"空仓时的 buy / 持仓时的 sell", 多余信号自动忽略;
  · 选股扫描取最近一个 buy, 这样不会因"上一笔还没平仓"而漏掉新信号。

唯一的现实约束: 信号日收盘涨幅 > 9.9% (涨停) 时跳过买入 —— 涨停买不进,
与其他模板保持一致。
"""
from __future__ import annotations

import pandas as pd

from app.services.tdx.indicators.maimai_henzhun import compute_edges
from .base import StrategyTemplate


class MaimaiZhun(StrategyTemplate):
    template_id = "mmhz"

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return "买卖很准 (合并版边沿信号)"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 60:
            return []
        close = df["close"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        dates = df["trade_date"]

        buy_fire, sell_fire = compute_edges(close, high, low)

        # 涨停日买不进
        limit_up = close > close.shift(1) * 1.099
        buy_fire = buy_fire & ~limit_up.fillna(False)

        signals: list[dict] = []
        for i in range(len(df)):
            if buy_fire.iloc[i]:
                signals.append({"date": dates.iloc[i], "action": "buy"})
            elif sell_fire.iloc[i]:
                signals.append({"date": dates.iloc[i], "action": "sell"})
        return signals
