"""TDX 经典策略实现 — 通达信社区流传的高口碑选股公式。

收录：
  1. 三金叉共振 (Triple Gold Cross): MA(5/10) 金叉 + MACD 金叉 + KDJ 金叉，2 日内
  2. MACD 黄金坑: DIF & DEA 金叉 + DIF < 0 + 量能放大
  3. EXPMA 三步擒牛: EXPMA(12/50) 多头 + MACD 金叉 + BOLL 中轨上方
  4. 日周 KDJ 共振: 日 KDJ 金叉 + 周 KDJ 金叉

退出统一用贪婪 ATR + 硬止损 -10%。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyTemplate


def _shift1(arr):
    return np.concatenate(([arr[0]], arr[:-1]))


def _cross_above_var(s1, s2):
    s1 = np.asarray(s1, dtype=float); s2 = np.asarray(s2, dtype=float)
    p1, p2 = _shift1(s1), _shift1(s2)
    return (p1 <= p2) & (s1 > s2)


def _cross_above(s, level):
    s = np.asarray(s, dtype=float); p = _shift1(s)
    return (p <= level) & (s > level)


# =========================================================================
# Indicator pack
# =========================================================================

def _compute_pack(close, high, low, vol):
    n = len(close)
    sc, sh, sl, sv = pd.Series(close), pd.Series(high), pd.Series(low), pd.Series(vol)

    # SMAs
    sma5 = sc.rolling(5, min_periods=1).mean().values
    sma10 = sc.rolling(10, min_periods=1).mean().values
    sma20 = sc.rolling(20, min_periods=1).mean().values

    # EXPMA(12/50) — exponential moving averages
    expma12 = sc.ewm(span=12, adjust=False).mean().values
    expma50 = sc.ewm(span=50, adjust=False).mean().values

    # MACD
    ema12 = sc.ewm(span=12, adjust=False).mean().values
    ema26 = sc.ewm(span=26, adjust=False).mean().values
    diff = ema12 - ema26
    dea = pd.Series(diff).ewm(span=9, adjust=False).mean().values
    macd_hist = (diff - dea) * 2

    # KDJ (daily, n=9)
    h9 = sh.rolling(9, min_periods=1).max().values
    l9 = sl.rolling(9, min_periods=1).min().values
    rng9 = h9 - l9
    rsv = np.where(rng9 > 0, (close - l9) / np.where(rng9 > 0, rng9, 1) * 100, 50.0)
    k = np.zeros(n); d = np.zeros(n); k[0] = 50; d[0] = 50
    for i in range(1, n):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    j = 3 * k - 2 * d

    # ⚠️ 周线 KDJ 已整段移除 (2026-09)。
    # 原实现把整周聚合值广播回该周每一天, 周一就用到周五收盘 —— 偷看 4 天。
    # 修好之后仍决定弃用: 多周期这一类构造出错面太大, 而检验工具已两次给出
    # 假阴性(幸存者偏差、切点密度不足)。在验证能力有盲区时, 对整类高风险构造
    # 一刀切比逐个甄别更稳妥。详见 scripts/screening/README.md 的排除项。

    # Bollinger bands (20, 2)
    mid = sc.rolling(20, min_periods=1).mean().values
    std = sc.rolling(20, min_periods=1).std().fillna(0).values

    # ATR(14) Wilder
    prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values

    # Volume MA
    vma5 = sv.rolling(5, min_periods=1).mean().values

    return {
        "sma5": sma5, "sma10": sma10, "sma20": sma20,
        "expma12": expma12, "expma50": expma50,
        "diff": diff, "dea": dea, "macd_hist": macd_hist,
        "k": k, "d": d, "j": j,
        "boll_mid": mid,
        "atr14": atr14,
        "vma5": vma5,
    }


# =========================================================================
# Entry triggers
# =========================================================================

def _entry_triple_gold(close, ind):
    """Three-way gold cross within last 2 bars: SMA(5)/SMA(10), MACD DIF/DEA, KDJ K/D."""
    ma_cross = _cross_above_var(ind["sma5"], ind["sma10"])
    macd_cross = _cross_above_var(ind["diff"], ind["dea"])
    kdj_cross = _cross_above_var(ind["k"], ind["d"])
    # Within last 2 bars (any of t, t-1)
    ma_recent = pd.Series(ma_cross.astype(float)).rolling(2, min_periods=1).max().values > 0
    macd_recent = pd.Series(macd_cross.astype(float)).rolling(2, min_periods=1).max().values > 0
    kdj_recent = pd.Series(kdj_cross.astype(float)).rolling(2, min_periods=1).max().values > 0
    return ma_recent & macd_recent & kdj_recent


def _entry_macd_gold_pit(close, ind, vol):
    """MACD 黄金坑: DIF/DEA 金叉 + DIF < 0 + 量能 > 5日均量."""
    macd_cross = _cross_above_var(ind["diff"], ind["dea"])
    below_zero = ind["diff"] < 0
    vol_expand = vol > ind["vma5"] * 1.0
    return macd_cross & below_zero & vol_expand


def _entry_expma_trio(close, ind):
    """EXPMA 三步擒牛: EXPMA(12) > EXPMA(50) AND DIFF > DEA AND close > BOLL中轨.
    Uses 'fresh setup': condition true today, was false yesterday on at least one."""
    bullish = (ind["expma12"] > ind["expma50"]) & (ind["diff"] > ind["dea"]) & \
              (close > ind["boll_mid"])
    prev = _shift1(bullish.astype(float))
    return bullish & (prev == 0)   # newly entered the regime



# =========================================================================
# Exit walker (greedy ATR + hard -10%)
# =========================================================================

def _walk_atr_greedy(i, close, high, low, p, n):
    entry = close[i]
    hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.06)
    time_stop = p.get("time_stop", 60)
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


# =========================================================================
# Strategy classes
# =========================================================================

class _TDXClassicBase(StrategyTemplate):
    template_id = "tdx-base"
    _entry_name = ""
    _params: dict = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 60}

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return self._entry_name

    def _entry_mask(self, close, high, low, vol, ind) -> np.ndarray:
        raise NotImplementedError

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 60:
            return []
        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        vol = df["vol"].astype(float).values
        dates = df["trade_date"]
        ind = _compute_pack(close, high, low, vol)
        sig_mask = self._entry_mask(close, high, low, vol, ind)

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(close)
        for i in range(40, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if not sig_mask[i]:
                continue
            if i > 0 and close[i] > close[i - 1] * 1.099:
                continue
            exit_idx, _ = _walk_atr_greedy(i, close, high, low, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


class TDXTripleGold(_TDXClassicBase):
    template_id = "tdx-triple-gold"
    _entry_name = "三金叉共振 (MA+MACD+KDJ)"
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 75}

    def _entry_mask(self, close, high, low, vol, ind):
        return _entry_triple_gold(close, ind)


class TDXMACDGoldPit(_TDXClassicBase):
    template_id = "tdx-macd-pit"
    _entry_name = "MACD 黄金坑抄底"
    _params = {"atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 45}

    def _entry_mask(self, close, high, low, vol, ind):
        return _entry_macd_gold_pit(close, ind, vol)


class TDXExpmaTrio(_TDXClassicBase):
    template_id = "tdx-expma-trio"
    _entry_name = "EXPMA 三步擒牛"
    _params = {"atr_mult": 2.5, "trail_activation": 0.08, "time_stop": 90}

    def _entry_mask(self, close, high, low, vol, ind):
        return _entry_expma_trio(close, ind)


