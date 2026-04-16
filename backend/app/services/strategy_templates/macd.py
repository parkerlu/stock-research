"""MACD signal crossover strategy template."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MACDCrossover(StrategyTemplate):
    template_id = "macd"

    def __init__(self, fast: int, slow: int, signal: int):
        self.fast = fast
        self.slow = slow
        self.signal = signal

    @property
    def name(self) -> str:
        return f"MACD_{self.fast}_{self.slow}_{self.signal}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < self.slow + self.signal + 1:
            return []

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()

        signals = []
        for i in range(self.slow + self.signal, len(df)):
            prev_diff = macd_line.iloc[i - 1] - signal_line.iloc[i - 1]
            curr_diff = macd_line.iloc[i] - signal_line.iloc[i]

            if prev_diff <= 0 and curr_diff > 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif prev_diff >= 0 and curr_diff < 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fasts = list(range(8, 17, 4))  # [8,12,16]
        slows = list(range(20, 31, 5))  # [20,25,30]
        signal_vals = list(range(5, 13, 2))  # [5,7,9,11]
        return [
            {"fast": f, "slow": s, "signal": sig}
            for f, s, sig in product(fasts, slows, signal_vals)
            if f < s
        ]
