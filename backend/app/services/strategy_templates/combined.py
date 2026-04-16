"""Combined strategy templates: MA+RSI, MACD+Volume, Bollinger+RSI."""
from __future__ import annotations

from itertools import product

import pandas as pd

from .base import StrategyTemplate


class MARSICombined(StrategyTemplate):
    """Buy when price above MA AND RSI crosses below oversold. Sell when RSI crosses above overbought."""
    template_id = "ma_rsi"

    def __init__(self, ma_period: int, rsi_period: int, overbought: int, oversold: int):
        self.ma_period = ma_period
        self.rsi_period = rsi_period
        self.overbought = overbought
        self.oversold = oversold

    @property
    def name(self) -> str:
        return f"MA{self.ma_period}_RSI{self.rsi_period}_{self.overbought}_{self.oversold}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = max(self.ma_period, self.rsi_period) + 2
        if len(df) < n:
            return []

        ma = df["close"].rolling(window=self.ma_period).mean()

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(ma.iloc[i]) or pd.isna(rsi.iloc[i]) or pd.isna(rsi.iloc[i - 1]):
                continue
            if (rsi.iloc[i - 1] >= self.oversold and rsi.iloc[i] < self.oversold
                    and df["close"].iloc[i] > ma.iloc[i]):
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif rsi.iloc[i - 1] <= self.overbought and rsi.iloc[i] > self.overbought:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        ma_periods = [5, 10, 20, 60]
        rsi_periods = [6, 14, 20]
        obs = [70, 80]
        oss = [20, 30]
        return [
            {"ma_period": ma, "rsi_period": rsi, "overbought": ob, "oversold": os_}
            for ma, rsi, ob, os_ in product(ma_periods, rsi_periods, obs, oss)
            if os_ < ob - 20
        ]


class MACDVolumeCombined(StrategyTemplate):
    """MACD crossover confirmed by volume above its MA."""
    template_id = "macd_vol"

    def __init__(self, fast: int, slow: int, signal: int, vol_ma: int):
        self.fast = fast
        self.slow = slow
        self.signal = signal
        self.vol_ma = vol_ma

    @property
    def name(self) -> str:
        return f"MACD{self.fast}_{self.slow}_{self.signal}_V{self.vol_ma}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = self.slow + self.signal + 1
        if len(df) < n:
            return []

        ema_fast = df["close"].ewm(span=self.fast, adjust=False).mean()
        ema_slow = df["close"].ewm(span=self.slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=self.signal, adjust=False).mean()
        vol_ma = df["vol"].rolling(window=self.vol_ma).mean()

        signals = []
        for i in range(n, len(df)):
            if pd.isna(vol_ma.iloc[i]):
                continue
            prev_diff = macd_line.iloc[i - 1] - signal_line.iloc[i - 1]
            curr_diff = macd_line.iloc[i] - signal_line.iloc[i]
            vol_confirm = df["vol"].iloc[i] > vol_ma.iloc[i]

            if prev_diff <= 0 and curr_diff > 0 and vol_confirm:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif prev_diff >= 0 and curr_diff < 0:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        fasts = [8, 10, 12]
        slows = [20, 24, 26, 30]
        sigs = [7, 9, 11]
        vol_mas = [5, 10, 20]
        return [
            {"fast": f, "slow": s, "signal": sig, "vol_ma": vm}
            for f, s, sig, vm in product(fasts, slows, sigs, vol_mas)
            if f < s
        ]


class BollingerRSICombined(StrategyTemplate):
    """Bollinger Band breakout confirmed by RSI extremes."""
    template_id = "bb_rsi"

    def __init__(self, bb_period: int, bb_std: float, rsi_period: int, rsi_ob: int, rsi_os: int):
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_ob = rsi_ob
        self.rsi_os = rsi_os

    @property
    def name(self) -> str:
        return f"BB{self.bb_period}_{self.bb_std}_RSI{self.rsi_period}_{self.rsi_ob}_{self.rsi_os}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        n = max(self.bb_period, self.rsi_period) + 2
        if len(df) < n:
            return []

        ma = df["close"].rolling(window=self.bb_period).mean()
        std = df["close"].rolling(window=self.bb_period).std()
        upper = ma + self.bb_std * std
        lower = ma - self.bb_std * std

        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=self.rsi_period).mean()
        rs = avg_gain / avg_loss.replace(0, 1e-10)
        rsi = 100 - (100 / (1 + rs))

        signals = []
        for i in range(1, len(df)):
            if pd.isna(upper.iloc[i]) or pd.isna(rsi.iloc[i]):
                continue
            if df["close"].iloc[i] < lower.iloc[i] and rsi.iloc[i] < self.rsi_os:
                signals.append({"date": df["trade_date"].iloc[i], "action": "buy"})
            elif df["close"].iloc[i] > upper.iloc[i] and rsi.iloc[i] > self.rsi_ob:
                signals.append({"date": df["trade_date"].iloc[i], "action": "sell"})

        return signals

    @staticmethod
    def parameter_candidates() -> list[dict]:
        bb_periods = [10, 15, 20, 25]
        bb_stds = [1.5, 2.0, 2.5]
        rsi_periods = [6, 10, 14]
        obs = [70, 80]
        oss = [20, 30]
        return [
            {"bb_period": bp, "bb_std": bs, "rsi_period": rp, "rsi_ob": ob, "rsi_os": os_}
            for bp, bs, rp, ob, os_ in product(bb_periods, bb_stds, rsi_periods, obs, oss)
        ]
