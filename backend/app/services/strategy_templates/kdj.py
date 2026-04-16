"""KDJ golden/dead cross strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class KDJCrossover(StrategyTemplate):
    template_id = "kdj"

    def __init__(self, k_period: int, d_period: int):
        self.k_period = k_period
        self.d_period = d_period

    @property
    def name(self) -> str:
        return f"KDJ_{self.k_period}_{self.d_period}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.k_period + self.d_period + 1:
            return []

        low_min = df["low"].rolling(window=self.k_period, min_periods=self.k_period).min()
        high_max = df["high"].rolling(window=self.k_period, min_periods=self.k_period).max()

        rsv = (df["close"] - low_min) / (high_max - low_min).replace(0, 1e-10) * 100

        k = rsv.ewm(com=self.d_period - 1, adjust=False).mean()
        d = k.ewm(com=self.d_period - 1, adjust=False).mean()

        signals = []
        for i in range(self.k_period + self.d_period, len(df)):
            if pd.isna(k.iloc[i]) or pd.isna(d.iloc[i]):
                continue
            if k.iloc[i - 1] <= d.iloc[i - 1] and k.iloc[i] > d.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif k.iloc[i - 1] >= d.iloc[i - 1] and k.iloc[i] < d.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        k_values = list(range(5, 22, 2))
        d_values = list(range(3, 10))
        return [
            {"k_period": k, "d_period": d}
            for k, d in product(k_values, d_values)
            if k > d
        ]
