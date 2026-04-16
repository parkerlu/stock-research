"""Bollinger Band breakout strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class BollingerBreakout(StrategyTemplate):
    template_id = "bollinger"

    def __init__(self, period: int, std_dev: float):
        self.period = period
        self.std_dev = std_dev

    @property
    def name(self) -> str:
        return f"BB_{self.period}_{self.std_dev}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.period + 1:
            return []

        ma = df["close"].rolling(window=self.period).mean()
        std = df["close"].rolling(window=self.period).std()
        upper = ma + self.std_dev * std
        lower = ma - self.std_dev * std

        signals = []
        for i in range(1, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(lower.iloc[i]):
                continue
            if df["close"].iloc[i - 1] >= lower.iloc[i - 1] and df["close"].iloc[i] < lower.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif df["close"].iloc[i - 1] <= upper.iloc[i - 1] and df["close"].iloc[i] > upper.iloc[i]:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        periods = list(range(10, 31, 2))
        std_devs = [1.5, 2.0, 2.5, 3.0]
        return [
            {"period": p, "std_dev": s}
            for p, s in product(periods, std_devs)
        ]
