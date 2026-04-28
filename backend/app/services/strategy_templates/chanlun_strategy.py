"""缠论 Chan Theory strategies — 1 类 + 2 类买点."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyTemplate
from app.services.chanlun import find_class1_buys, find_class2_buys, find_all_buys


def _walk_atr_greedy(i, close, high, low, p, n):
    entry = close[i]; hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.06)
    time_stop = p.get("time_stop", 90)
    peak = high[i]; activated = False
    alpha = 1.0 / 14.0
    atr = max(high[i] - low[i], 1e-9)
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        prev_c = close[k - 1] if k - 1 >= 0 else close[k]
        tr = max(high[k] - low[k], abs(high[k] - prev_c), abs(low[k] - prev_c))
        atr = (1 - alpha) * atr + alpha * tr
        peak = max(peak, high[k])
        if peak / entry - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return k, hard_stop
        if activated and atr > 0:
            stop = peak - atr_mult * atr
            if low[k] <= stop:
                return k, max(stop, hard_stop)
    return end, close[end]


class _ChanlunBase(StrategyTemplate):
    template_id = "chan-base"
    _classes_to_use: tuple[str, ...] = ("1", "2")
    _params: dict = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 90}
    _display = "Chan_Base"

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return self._display

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 130:
            return []
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        dates = df["trade_date"]

        if "1" in self._classes_to_use and "2" in self._classes_to_use:
            buys = find_all_buys(close, high, low)
        elif "1" in self._classes_to_use:
            buys = find_class1_buys(close, high, low)
        else:
            buys = find_class2_buys(
                close, high, low,
                {b.bar_idx for b in find_class1_buys(close, high, low)},
            )

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(close)
        buy_idx_set = {b.bar_idx for b in buys if b.type in self._classes_to_use}

        for i in range(20, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if i not in buy_idx_set:
                continue
            if i > 0 and close[i] > close[i - 1] * 1.099:
                continue
            exit_idx, _ = _walk_atr_greedy(i, close, high, low, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


class Chan1Buy(_ChanlunBase):
    """缠论 1 类买点 — 底背驰反转，每段下跌结束 + MACD 面积衰减."""
    template_id = "chan-1buy"
    _classes_to_use = ("1",)
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 75}
    _display = "缠论_一类买点"


class Chan2Buy(_ChanlunBase):
    """缠论 2 类买点 — 1 类后回踩不破底."""
    template_id = "chan-2buy"
    _classes_to_use = ("2",)
    _params = {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60}
    _display = "缠论_二类买点"


class Chan12Buy(_ChanlunBase):
    """缠论 1+2 类买点合并 — 抄底 + 确认双重信号."""
    template_id = "chan-12"
    _classes_to_use = ("1", "2")
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 75}
    _display = "缠论_一二类合并"


class Chan1BuyWide(_ChanlunBase):
    """1 类买点 + 宽 ATR(2.5)，吃大波段反弹."""
    template_id = "chan-1buy-wide"
    _classes_to_use = ("1",)
    _params = {"atr_mult": 2.5, "trail_activation": 0.10, "time_stop": 90}
    _display = "缠论_一类买点_宽幅"
