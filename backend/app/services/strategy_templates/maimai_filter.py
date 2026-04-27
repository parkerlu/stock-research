"""Maimai-filter strategy: ML-filtered 买卖很准 'all-clear' transitions.

Entry: at bar i where (mm_jibuy + mm_duanbuy + mm_zhunbei) just transitioned
       from > 0 to == 0, AND a trained classifier scores the bar's features
       above `score_threshold`.

Exit: hard stop -10% / trail / time stop.

Model trained by scripts/train_maimai_filter.py over all 1925 active stocks.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import StrategyTemplate
from .ml_direct import (
    _FEATURE_NAMES, _compute_features, _csf_for, _candidate_paths,
)


_FILTER_CACHE: xgb.Booster | None = None


def _load_filter() -> xgb.Booster | None:
    global _FILTER_CACHE
    if _FILTER_CACHE is not None:
        return _FILTER_CACHE
    for p in _candidate_paths("maimai_filter_multi.json"):
        if os.path.exists(p):
            b = xgb.Booster()
            b.load_model(p)
            _FILTER_CACHE = b
            return b
    return None


def _shift1(arr):
    return np.concatenate(([arr[0]], arr[:-1]))


def _maimai_transition_mask(close, high, low) -> np.ndarray:
    """True at bar i where (jibuy + duanbuy + zhunbei) just hit 0 from > 0.

    Mirrors the TDX 买卖很准 indicator exactly:
      买卖 = LLV(MA((H+L+C)/3, 5), 10)        # buy threshold
      急买奇准 = LLV(close < 买卖, 5)            # close below threshold ALL 5 bars
      短买奇准 = LLV(close < 买卖, 10)           # ... ALL 10 bars
      准备现金 = (动向趋势线 > 88) AND (神偷线 < 5.8)
    """
    n = len(close)
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    maimai_thr = ban_s.rolling(10, min_periods=10).min().values
    below_buy = np.where(np.isnan(maimai_thr), 0.0,
                         (close < maimai_thr).astype(float))
    bb = pd.Series(below_buy)
    # LLV(bool, n) = 1 iff ALL n bars are true → use rolling min
    jibuy = (bb.rolling(5, min_periods=5).min() > 0).astype(float).values
    duanbuy = (bb.rolling(10, min_periods=10).min() > 0).astype(float).values
    # 准备 = DMI(5)-based  (mirror of precompute_maimai)
    prev_h = _shift1(high); prev_l = _shift1(low); prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    hd = high - prev_h; ld = prev_l - low
    pos_hd = np.where((hd > 0) & (hd > ld), hd, 0.0)
    pos_ld = np.where((ld > 0) & (ld > hd), ld, 0.0)
    td = pd.Series(tr).rolling(5, min_periods=5).sum().values
    dmp = pd.Series(pos_hd).rolling(5, min_periods=5).sum().values
    dmm = pd.Series(pos_ld).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td > 0, dmp * 100.0 / td, 0.0)
        fuzhu = np.where(td > 0, dmm * 100.0 / td, 0.0)
        denom = fuzhu + shentou
        dx = np.where(denom > 0, np.abs(fuzhu - shentou) / denom * 100.0, 0.0)
    dongxiang = pd.Series(dx).rolling(3, min_periods=1).mean().values
    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    sum_now = jibuy + duanbuy + zhunbei
    sum_prev = _shift1(sum_now)
    sum_prev[0] = 0.0
    return (sum_now == 0) & (sum_prev > 0)


def _walk_classic(i, close, high, low, p, n):
    entry = close[i]; hard_stop = entry * 0.90
    trail_pct = p.get("trail_pct", 0.04)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 30)
    peak = high[i]; activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        if peak / entry - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return k, hard_stop
        if activated:
            stop = peak * (1 - trail_pct)
            if low[k] <= stop:
                return k, max(stop, hard_stop)
    return end, close[end]


def _walk_atr_greedy(i, close, high, low, p, n):
    """Greedy ATR-based trailing stop. Lets winners run further than classic %.

    - hard stop -10% (always)
    - activate trailing only after gain >= activation
    - trailing stop = peak - atr_mult × ATR(14)
    - time stop falls back at end
    """
    entry = close[i]
    hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.08)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False

    # Wilder ATR(14) using close-aware true range, computed on the fly via
    # a leaky integrator α = 1/14.
    alpha = 1.0 / 14.0
    atr = max(high[i] - low[i], 1e-9)

    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        # Update ATR incrementally
        prev_close = close[k - 1] if k - 1 >= 0 else close[k]
        tr = max(
            high[k] - low[k],
            abs(high[k] - prev_close),
            abs(low[k] - prev_close),
        )
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


class MaimaiFilterStrategy(StrategyTemplate):
    """Filtered maimai strategy. Score threshold tunable per preset."""

    template_id = "mm-filter"
    _score_threshold = 0.50
    _exit_mode = "atr_greedy"   # "classic" or "atr_greedy"
    _params: dict = {"atr_mult": 2.0, "trail_activation": 0.08, "time_stop": 75}

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return f"MaimaiFilter_thr{int(self._score_threshold*100)}"

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 130:
            return []
        booster = _load_filter()
        if booster is None:
            return []

        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        vol = df["vol"].astype(float).values
        dates = df["trade_date"]

        sig_mask = _maimai_transition_mask(close, high, low)

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(df)

        # Compute features only for signal bars (saves time)
        # Score in a batch for performance.
        candidate_idx = [i for i in range(120, n - 1) if sig_mask[i]]
        if not candidate_idx:
            return []

        feat_rows = []
        valid_idx = []
        for i in candidate_idx:
            f = _compute_features(close, high, low, vol, i)
            if f is None:
                continue
            f.update(_csf_for(self.ts_code, dates.iloc[i]))
            feat_rows.append([f[k] for k in _FEATURE_NAMES])
            valid_idx.append(i)
        if not feat_rows:
            return []

        X = np.array(feat_rows, dtype=np.float32)
        dmat = xgb.DMatrix(X, feature_names=_FEATURE_NAMES)
        scores_at_signal = booster.predict(dmat)
        score_map = dict(zip(valid_idx, scores_at_signal))

        for i in range(120, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if not sig_mask[i]:
                continue
            sc = float(score_map.get(i, 0.0))
            if sc < self._score_threshold:
                continue
            if i > 0 and close[i] > close[i - 1] * 1.099:
                continue
            walker = _walk_atr_greedy if self._exit_mode == "atr_greedy" else _walk_classic
            exit_idx, _ = walker(i, close, high, low, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


# ---- Presets at different score thresholds ----

class MaimaiFilter40(MaimaiFilterStrategy):
    """thr 0.40 + 贪婪 ATR(1.8)：信号多，给波段空间。"""
    template_id = "mm-40"
    _score_threshold = 0.40
    _exit_mode = "atr_greedy"
    _params = {"atr_mult": 1.8, "trail_activation": 0.06, "time_stop": 60}

    @property
    def name(self) -> str:
        return "MaimaiFilter_40_Greedy"


class MaimaiFilter50(MaimaiFilterStrategy):
    """thr 0.50 + 贪婪 ATR(2.0)：sweet spot — 高胜率 + 吃波段."""
    template_id = "mm-50"
    _score_threshold = 0.50
    _exit_mode = "atr_greedy"
    _params = {"atr_mult": 2.0, "trail_activation": 0.08, "time_stop": 75}

    @property
    def name(self) -> str:
        return "MaimaiFilter_50_Greedy"


class MaimaiFilter55(MaimaiFilterStrategy):
    """thr 0.55 + 贪婪 ATR(2.5)：严格 + 最宽 trail，吃最大波段."""
    template_id = "mm-55"
    _score_threshold = 0.55
    _exit_mode = "atr_greedy"
    _params = {"atr_mult": 2.5, "trail_activation": 0.10, "time_stop": 90}

    @property
    def name(self) -> str:
        return "MaimaiFilter_55_GreedyWide"
