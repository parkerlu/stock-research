"""Multi-source reversal filter strategy.

Candidate triggers (UNION of):
  1. 买卖很准 broad: any of 急买/短买/准备/急卖/短卖 just turned off
  2. 动力线 cross above 0.2 (stage_bottom)
  3. KDJ J cross above 0
  4. RSI(14) cross above 30

Each candidate is scored by a 5-seed XGBoost ensemble (trained on 345K events
across 1926 stocks). Buy if score >= threshold.

Exit: greedy ATR(14)-based trailing with hard stop -10%.

Goal: 10+ candidates/year per stock so user has more action.
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
    for p in _candidate_paths("reversal_filter_multi.json"):
        if os.path.exists(p):
            b = xgb.Booster()
            b.load_model(p)
            _FILTER_CACHE = b
            return b
    return None


def _shift1(a):
    return np.concatenate(([a[0]], a[:-1]))


def _candidate_mask(close, high, low) -> np.ndarray:
    """Union of all 4 reversal-buy event types. Mirror of train_reversal_filter."""
    n = len(close)

    # ---- maimai (TDX-correct LLV) ----
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    maimai_thr = ban_s.rolling(10, min_periods=10).min().values
    hao_thr = ban_s.rolling(10, min_periods=10).max().values
    bb_buy = pd.Series(np.where(np.isnan(maimai_thr), 0.0,
                                 (close < maimai_thr).astype(float)))
    jibuy = (bb_buy.rolling(5, min_periods=5).min() > 0).astype(float).values
    duanbuy = (bb_buy.rolling(10, min_periods=10).min() > 0).astype(float).values
    bs_sell = pd.Series(np.where(np.isnan(hao_thr), 0.0,
                                  (close < hao_thr).astype(float)))
    jisell = (bs_sell.rolling(5, min_periods=1).max() > 0).astype(float).values
    duansell = (bs_sell.rolling(10, min_periods=1).max() > 0).astype(float).values

    # 准备 (DMI-5)
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

    # 动力线
    var2 = pd.Series(low).rolling(10, min_periods=1).min().values
    var33 = pd.Series(high).rolling(25, min_periods=1).max().values
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values
    pdl = _shift1(dongli)
    dl_cross = ((pdl <= 0.2) & (dongli > 0.2))

    # KDJ J
    h9 = pd.Series(high).rolling(9, min_periods=1).max().values
    l9 = pd.Series(low).rolling(9, min_periods=1).min().values
    rng9 = h9 - l9
    rsv = np.where(rng9 > 0, (close - l9) / np.where(rng9 > 0, rng9, 1) * 100, 50.0)
    k_ = np.zeros(n); d_ = np.zeros(n); k_[0] = 50; d_[0] = 50
    for i in range(1, n):
        k_[i] = (2 / 3) * k_[i - 1] + (1 / 3) * rsv[i]
        d_[i] = (2 / 3) * d_[i - 1] + (1 / 3) * k_[i]
    j_ = 3 * k_ - 2 * d_
    pj = _shift1(j_)
    kdj_cross = ((pj <= 0) & (j_ > 0))

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
    rs = np.where(al > 0, ag / np.where(al > 0, al, 1), 100.0)
    rsi = 100 - 100 / (1 + rs)
    prsi = _shift1(rsi)
    rsi_cross = ((prsi <= 30) & (rsi > 30))

    # Maimai turn-off events
    j_off = (_shift1(jibuy) > 0) & (jibuy == 0)
    d_off = (_shift1(duanbuy) > 0) & (duanbuy == 0)
    z_off = (_shift1(zhunbei) > 0) & (zhunbei == 0)
    js_off = (_shift1(jisell) > 0) & (jisell == 0)
    ds_off = (_shift1(duansell) > 0) & (duansell == 0)
    mm_any = j_off | d_off | z_off | js_off | ds_off

    return mm_any | dl_cross | kdj_cross | rsi_cross


def _walk_atr_greedy(i, close, high, low, p, n):
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


class ReversalFilterStrategy(StrategyTemplate):
    template_id = "rev-filter"
    _score_threshold = 0.50
    _params: dict = {"atr_mult": 2.0, "trail_activation": 0.08, "time_stop": 75}

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return f"Reversal_thr{int(self._score_threshold*100)}"

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

        sig_mask = _candidate_mask(close, high, low)

        n = len(df)
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
        scores = booster.predict(dmat)
        score_map = dict(zip(valid_idx, scores))

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
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
            exit_idx, _ = _walk_atr_greedy(i, close, high, low, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


# ---- Presets ----
class Reversal30(ReversalFilterStrategy):
    """thr 0.30: 高频版，~10/yr 候选."""
    template_id = "rev-30"
    _score_threshold = 0.30
    _params = {"atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 60}

    @property
    def name(self) -> str:
        return "Reversal_30_HighFreq"


class Reversal40(ReversalFilterStrategy):
    template_id = "rev-40"
    _score_threshold = 0.40
    _params = {"atr_mult": 1.8, "trail_activation": 0.06, "time_stop": 60}

    @property
    def name(self) -> str:
        return "Reversal_40"


class Reversal50(ReversalFilterStrategy):
    """Sweet spot: 多源候选 + thr 0.50 + 贪婪 ATR(2.0)."""
    template_id = "rev-50"
    _score_threshold = 0.50
    _params = {"atr_mult": 2.0, "trail_activation": 0.08, "time_stop": 75}

    @property
    def name(self) -> str:
        return "Reversal_50_Greedy"


class Reversal55(ReversalFilterStrategy):
    template_id = "rev-55"
    _score_threshold = 0.55
    _params = {"atr_mult": 2.5, "trail_activation": 0.10, "time_stop": 90}

    @property
    def name(self) -> str:
        return "Reversal_55_Wide"


class Reversal60(ReversalFilterStrategy):
    template_id = "rev-60"
    _score_threshold = 0.60
    _params = {"atr_mult": 2.5, "trail_activation": 0.10, "time_stop": 90}

    @property
    def name(self) -> str:
        return "Reversal_60_Strict"
