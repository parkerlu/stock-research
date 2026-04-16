"""RSI overbought/oversold strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class RSIOverboughtOversold(StrategyTemplate):
    template_id = "rsi"

    def __init__(self, period: int, overbought: int, oversold: int):
        self.period = period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return f"RSI_{self.period}_{self.overbought}_{self.oversold}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.period + 2:
            return []

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)

        avg_gain = gain.rolling(window=self.period, min_periods=self.period).mean()
        avg_loss = loss.rolling(window=self.period, min_periods=self.period).mean()

        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
                continue
            if rsi.iloc[i - 1] >= self.oversold and rsi.iloc[i] < self.oversold:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif rsi.iloc[i - 1] <= self.overbought and rsi.iloc[i] > self.overbought:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        periods = list(range(6, 29, 4))  # [6,10,14,18,22,26]
        overboughts = list(range(65, 86, 10))  # [65,75,85]
        oversolds = list(range(15, 36, 10))  # [15,25,35]
        return [
            {"period": period, "overbought": ob, "oversold": os_}
            for period, ob, os_ in product(periods, overboughts, oversolds)
            if os_ < ob - 15
        ]
