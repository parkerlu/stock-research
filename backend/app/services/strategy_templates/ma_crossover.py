"""MA dual crossover strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MACrossover(StrategyTemplate):
    template_id = "ma_crossover"

    def __init__(self, fast_period: int, slow_period: int):
        self.fast_period = fast_period
        self.slow_period = slow_period

    @property
    def name(self) -> str:
        return f"MA_{self.fast_period}_{self.slow_period}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.slow_period + 1:
            return []

        fast_ma = df["close"].rolling(window=self.fast_period).mean()
        slow_ma = df["close"].rolling(window=self.slow_period).mean()

        signals = []
        for i in range(1, len(df)):
            if pd.isna(fast_ma.iloc[i]) or pd.isna(slow_ma.iloc[i]):
                continue
            if pd.isna(fast_ma.iloc[i - 1]) or pd.isna(slow_ma.iloc[i - 1]):
                continue

            if fast_ma.iloc[i - 1] <= slow_ma.iloc[i - 1] and fast_ma.iloc[i] > slow_ma.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif fast_ma.iloc[i - 1] >= slow_ma.iloc[i - 1] and fast_ma.iloc[i] < slow_ma.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fast_values = list(range(3, 31, 3))
        slow_values = list(range(10, 121, 10))
        return [
            {"fast_period": fast, "slow_period": slow}
            for fast, slow in product(fast_values, slow_values)
            if fast < slow
        ]
