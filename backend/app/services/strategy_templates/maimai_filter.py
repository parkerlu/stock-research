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


def _compute_maimai_signals(close, high, low):
    """Compute all 5 TDX 买卖很准 lines + zhunbei. Returns dict of arrays."""
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    maimai_thr = ban_s.rolling(10, min_periods=10).min().values     # 买卖 (buy)
    hao_thr = ban_s.rolling(10, min_periods=10).max().values         # 好  (sell)

    # buy-side: close < maimai for ALL N bars
    bb = pd.Series(
        np.where(np.isnan(maimai_thr), 0.0, (close < maimai_thr).astype(float))
    )
    jibuy = (bb.rolling(5, min_periods=5).min() > 0).astype(float).values
    duanbuy = (bb.rolling(10, min_periods=10).min() > 0).astype(float).values

    # sell-side: close < hao for any of N (HHV per TDX)
    bs = pd.Series(
        np.where(np.isnan(hao_thr), 0.0, (close < hao_thr).astype(float))
    )
    jisell = (bs.rolling(5, min_periods=1).max() > 0).astype(float).values
    duansell = (bs.rolling(10, min_periods=1).max() > 0).astype(float).values

    # 准备现金 (DMI-5)
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

    return {
        "jibuy": jibuy, "duanbuy": duanbuy, "zhunbei": zhunbei,
        "jisell": jisell, "duansell": duansell,
    }


def _maimai_transition_mask(close, high, low) -> np.ndarray:
    """STRICT mode: all 3 buy-side signals at 0, ≥1 was active prev bar."""
    sigs = _compute_maimai_signals(close, high, low)
    sum_now = sigs["jibuy"] + sigs["duanbuy"] + sigs["zhunbei"]
    sum_prev = _shift1(sum_now)
    sum_prev[0] = 0.0
    return (sum_now == 0) & (sum_prev > 0)


def _maimai_broad_mask(close, high, low) -> np.ndarray:
    """BROAD mode: ANY of (急买/短买/准备/急卖/短卖) just turned off.

    Captures every "fear-or-buy-signal-just-cleared" moment — much more
    candidates per year. Quality varies more, so the ML filter does the heavy
    lifting.
    """
    sigs = _compute_maimai_signals(close, high, low)
    j_off = (_shift1(sigs["jibuy"]) > 0) & (sigs["jibuy"] == 0)
    d_off = (_shift1(sigs["duanbuy"]) > 0) & (sigs["duanbuy"] == 0)
    z_off = (_shift1(sigs["zhunbei"]) > 0) & (sigs["zhunbei"] == 0)
    js_off = (_shift1(sigs["jisell"]) > 0) & (sigs["jisell"] == 0)
    ds_off = (_shift1(sigs["duansell"]) > 0) & (sigs["duansell"] == 0)
    return j_off | d_off | z_off | js_off | ds_off


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


def _walk_with_mm_sell_exit(i, close, high, low, sigs, p, n):
    """Exit on EITHER:
       - 买卖很准 卖出信号 fires (急卖 50→100 or 短卖 50→100), OR
       - greedy ATR trailing stop, OR
       - hard stop -10%, OR
       - time_stop fallback.
    Whichever triggers first.
    """
    entry = close[i]
    hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.08)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False
    alpha = 1.0 / 14.0
    atr = max(high[i] - low[i], 1e-9)
    end = min(i + time_stop, n - 1)

    jisell = sigs["jisell"]
    duansell = sigs["duansell"]

    for k in range(i + 1, end + 1):
        # Update ATR(14)
        prev_c = close[k - 1] if k - 1 >= 0 else close[k]
        tr = max(high[k] - low[k], abs(high[k] - prev_c), abs(low[k] - prev_c))
        atr = (1 - alpha) * atr + alpha * tr
        peak = max(peak, high[k])
        if peak / entry - 1 >= activation:
            activated = True

        # Hard stop
        if low[k] <= hard_stop:
            return k, hard_stop

        # 买卖很准 sell signal fires (transitions to active state)
        # 急卖: 50→100 (entered overbought 5d)
        # 短卖: 50→100 (entered overbought 10d)
        if (jisell[k] > 0 and jisell[k - 1] == 0) or (
            duansell[k] > 0 and duansell[k - 1] == 0
        ):
            return k, close[k]

        # ATR trail
        if activated and atr > 0:
            stop = peak - atr_mult * atr
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
    # exit_mode: "atr_greedy" or "mm_sell" (uses 买卖很准 sell signals + ATR + hard stop)
    _exit_mode = "atr_greedy"
    _trigger_mode = "strict"       # "strict" (all 3 buy = 0) or "broad" (any signal off)
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

        if self._trigger_mode == "broad":
            sig_mask = _maimai_broad_mask(close, high, low)
        else:
            sig_mask = _maimai_transition_mask(close, high, low)
        # Pre-compute all signals once for the mm-sell exit walker
        all_sigs = _compute_maimai_signals(close, high, low) if self._exit_mode == "mm_sell" else None

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
            if self._exit_mode == "mm_sell":
                exit_idx, _ = _walk_with_mm_sell_exit(
                    i, close, high, low, all_sigs, self._params, n
                )
            elif self._exit_mode == "atr_greedy":
                exit_idx, _ = _walk_atr_greedy(i, close, high, low, self._params, n)
            else:
                exit_idx, _ = _walk_classic(i, close, high, low, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


# ---- Presets at different score thresholds ----

class MaimaiFilter30(MaimaiFilterStrategy):
    """thr 0.30 + 贪婪 ATR：高频版，最多信号。"""
    template_id = "mm-30"
    _score_threshold = 0.30
    _exit_mode = "atr_greedy"
    _trigger_mode = "strict"
    _params = {"atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 60}

    @property
    def name(self) -> str:
        return "MaimaiFilter_30_HighFreq"


class MaimaiFilter40(MaimaiFilterStrategy):
    """thr 0.40 + 贪婪 ATR(1.8)：信号多，给波段空间。"""
    template_id = "mm-40"
    _score_threshold = 0.40
    _exit_mode = "atr_greedy"
    _trigger_mode = "strict"
    _params = {"atr_mult": 1.8, "trail_activation": 0.06, "time_stop": 60}

    @property
    def name(self) -> str:
        return "MaimaiFilter_40_Greedy"


class MaimaiFilterBroad40(MaimaiFilterStrategy):
    """BROAD 触发 (任一信号关闭) + thr 0.40 + 贪婪：~25/年候选."""
    template_id = "mm-broad-40"
    _score_threshold = 0.40
    _exit_mode = "atr_greedy"
    _trigger_mode = "broad"
    _params = {"atr_mult": 1.8, "trail_activation": 0.06, "time_stop": 60}

    @property
    def name(self) -> str:
        return "MaimaiBroad_40"


class MaimaiPure40(MaimaiFilterStrategy):
    """纯 mm 买入 + 纯 mm 卖出（急卖/短卖触发） + ATR + 硬止损 -10%。"""
    template_id = "mm-pure-40"
    _score_threshold = 0.40
    _exit_mode = "mm_sell"
    _trigger_mode = "broad"
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 90}

    @property
    def name(self) -> str:
        return "MaimaiPure_40"


class MaimaiPure50(MaimaiFilterStrategy):
    """纯 mm 双向：buy on 买入信号关闭 + ML, exit on 卖出信号触发 OR ATR OR -10%."""
    template_id = "mm-pure-50"
    _score_threshold = 0.50
    _exit_mode = "mm_sell"
    _trigger_mode = "broad"
    _params = {"atr_mult": 2.5, "trail_activation": 0.08, "time_stop": 90}

    @property
    def name(self) -> str:
        return "MaimaiPure_50"


class MaimaiFilterBroad50(MaimaiFilterStrategy):
    """BROAD 触发 + thr 0.50 + 贪婪 ATR(2.0)：高频 sweet spot."""
    template_id = "mm-broad-50"
    _score_threshold = 0.50
    _exit_mode = "atr_greedy"
    _trigger_mode = "broad"
    _params = {"atr_mult": 2.0, "trail_activation": 0.08, "time_stop": 75}

    @property
    def name(self) -> str:
        return "MaimaiBroad_50"


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
