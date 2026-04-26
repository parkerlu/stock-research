"""Mined-426 V2 strategy presets.

5 conceptually orthogonal strategies output by the diverse-mining pipeline
(50 family × 20 LHS × 1004 stocks × 7 years, then Jaccard-diverse top-5).

Each preset uses:
  - A unique technical TRIGGER EVENT (cross/regime change) — not ML score
  - ML score as an optional QUALITY GATE (`ml_gate` param)
  - Hard stop at -10%, ATR or classic trail
  - Time stop 53-96 bars

Pairwise buy-event Jaccard ≤ 17.7% — these are 5 truly different signal sources.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb

from .base import StrategyTemplate
from .ml_direct import (
    _FEATURE_NAMES, _load_ensemble, _compute_features, _csf_for,
)


# =========================================================================
# Vectorized indicator computation (mirror of scripts/mine_426_step3)
# =========================================================================

def _shift1(arr):
    return np.concatenate(([arr[0]], arr[:-1]))


def _compute_indicators(close, high, low, vol) -> dict:
    """Vectorized indicator pack — all arrays length N."""
    n = len(close)
    sh = pd.Series(high)
    sl = pd.Series(low)
    sc = pd.Series(close)

    # 动力线
    var2 = sl.rolling(10, min_periods=1).min().values
    var33 = sh.rolling(25, min_periods=1).max().values
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values

    # KDJ (n=9)
    h9 = sh.rolling(9, min_periods=1).max().values
    l9 = sl.rolling(9, min_periods=1).min().values
    rng9 = h9 - l9
    rsv = np.where(rng9 > 0, (close - l9) / np.where(rng9 > 0, rng9, 1) * 100, 50.0)
    k = np.zeros(n); d = np.zeros(n); k[0] = 50.0; d[0] = 50.0
    for i in range(1, n):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]

    # ATR(14)
    prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values

    # RSI(14)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
    rs = np.where(al > 0, ag / np.where(al > 0, al, 1), 100.0)
    rsi = 100 - 100 / (1 + rs)

    # Bollinger (20, 2)
    mid = sc.rolling(20, min_periods=1).mean().values
    std = sc.rolling(20, min_periods=1).std().fillna(0).values
    bb_lower = mid - 2.0 * std

    # 买卖很准 准备
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    floor10 = ban_s.rolling(10, min_periods=10).min().values
    below_floor = np.where(np.isnan(floor10), 0.0, (close < floor10).astype(float))
    bf = pd.Series(below_floor)
    jibuy = (bf.rolling(5, min_periods=1).max() > 0).astype(float).values
    # DMI (5-bar) for 准备
    prev_h = _shift1(high); prev_l = _shift1(low)
    hd = high - prev_h; ld = prev_l - low
    pos_hd = np.where((hd > 0) & (hd > ld), hd, 0.0)
    pos_ld = np.where((ld > 0) & (ld > hd), ld, 0.0)
    td = pd.Series(tr).rolling(5, min_periods=5).sum().values
    dmp = pd.Series(pos_hd).rolling(5, min_periods=5).sum().values
    dmm = pd.Series(pos_ld).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td > 0, dmp * 100 / td, 0.0)
        fuzhu = np.where(td > 0, dmm * 100 / td, 0.0)
        denom = fuzhu + shentou
        dx_raw = np.where(denom > 0, np.abs(fuzhu - shentou) / denom * 100, 0.0)
    dongxiang = pd.Series(dx_raw).rolling(3, min_periods=1).mean().values
    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    return {
        "dl_value": dongli,
        "kdj_k": k, "kdj_d": d, "kdj_j": 3 * k - 2 * d,
        "atr14": atr14,
        "rsi14": rsi,
        "bb_lower": bb_lower,
        "mm_zhunbei_active": zhunbei,
        "mm_jibuy_active": jibuy,
    }


# =========================================================================
# Entry trigger functions (one per preset)
# =========================================================================

def _entry_kdj_kd_cross(close, ind, ml_score, ml_gate):
    k, d = ind["kdj_k"], ind["kdj_d"]
    pk, pd_ = _shift1(k), _shift1(d)
    cross = (pk <= pd_) & (k > d)
    if ml_gate is not None:
        cross = cross & (ml_score >= ml_gate)
    return cross


def _entry_bb_lower_bounce(close, ind, ml_score, ml_gate):
    lower = ind["bb_lower"]
    cross = (_shift1(close) < _shift1(lower)) & (close > lower)
    if ml_gate is not None:
        cross = cross & (ml_score >= ml_gate)
    return cross


def _entry_dl_cross_05(close, ind, ml_score, ml_gate):
    d = ind["dl_value"]
    cross = (_shift1(d) <= 0.5) & (d > 0.5)
    if ml_gate is not None:
        cross = cross & (ml_score >= ml_gate)
    return cross


def _entry_rsi_30(close, ind, ml_score, ml_gate):
    r = ind["rsi14"]
    cross = (_shift1(r) <= 30.0) & (r > 30.0)
    if ml_gate is not None:
        cross = cross & (ml_score >= ml_gate)
    return cross


def _entry_zhunbei_first(close, ind, ml_score, ml_gate):
    z = ind["mm_zhunbei_active"]
    first = (_shift1(z) <= 0) & (z > 0)
    if ml_gate is not None:
        first = first & (ml_score >= ml_gate)
    return first


# =========================================================================
# Walking exit logic (CLASSIC and ATR_TRAIL with hard stop -10%)
# =========================================================================

def _walk_classic(i, close, high, low, atr, p, n):
    entry = close[i]
    hard_stop = entry * 0.90
    trail_pct = p["trail_pct"]
    activation = p["trail_activation"]
    time_stop = p["time_stop"]
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


def _walk_atr(i, close, high, low, atr, p, n):
    entry = close[i]
    hard_stop = entry * 0.90
    atr_mult = p["atr_mult"]
    activation = p["trail_activation"]
    time_stop = p["time_stop"]
    peak = high[i]; activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        if peak / entry - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return k, hard_stop
        if activated and atr[k] > 0:
            stop = peak - atr_mult * atr[k]
            if low[k] <= stop:
                return k, max(stop, hard_stop)
    return end, close[end]


# =========================================================================
# Base class — runs the (entry, exit, params) pipeline + score caching
# =========================================================================

class _MinedV2Base(StrategyTemplate):
    """Base template that scores all bars, computes indicators, applies
    entry trigger, walks exits. Subclasses set:
      _entry_fn:   callable(close, ind, score, ml_gate) → bool array
      _exit_walker: 'classic' or 'atr'
      _params:    dict
    """

    _entry_fn = staticmethod(lambda *a: np.array([]))
    _exit_walker = "classic"
    _params: dict = {}

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 130:
            return []
        models = _load_ensemble()
        if not models:
            return []

        close = df["close"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        vol = df["vol"].astype(float).values
        dates = df["trade_date"]

        # Indicators (vectorized)
        ind = _compute_indicators(close, high, low, vol)

        # ML score per bar (need 120-bar warmup; 0 elsewhere)
        score = np.zeros(len(close))
        rows = []
        valid_idx = []
        for i in range(120, len(df)):
            f = _compute_features(close, high, low, vol, i)
            if f is None:
                continue
            f.update(_csf_for(self.ts_code, dates.iloc[i]))
            rows.append([f[k] for k in _FEATURE_NAMES])
            valid_idx.append(i)
        if rows:
            X = np.array(rows)
            dmat = xgb.DMatrix(X, feature_names=_FEATURE_NAMES)
            preds = np.array([m.predict(dmat) for m in models]).mean(axis=0)
            for i, p in zip(valid_idx, preds):
                score[i] = float(p)

        # Entry mask
        ml_gate = self._params.get("ml_gate")
        entry_mask = self._entry_fn(close, ind, score, ml_gate)

        # Walk forward
        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(close)
        atr14 = ind["atr14"]

        for i in range(120, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if not entry_mask[i]:
                continue
            if i > 0 and close[i] > close[i - 1] * 1.099:
                continue
            walker = _walk_atr if self._exit_walker == "atr" else _walk_classic
            exit_idx, _exit_price = walker(i, close, high, low, atr14, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx

        return signals


# =========================================================================
# 5 Diverse Top-5 presets
# =========================================================================

class Mined426V2_1(_MinedV2Base):
    """KDJ K/D 金叉 + ML quality gate.
    IS 75.9% win, +12.0% avg, OOS 83.1% win, +12.8% avg (highest avg).
    """
    template_id = "426-1"
    _entry_fn = staticmethod(_entry_kdj_kd_cross)
    _exit_walker = "atr"
    _params = {
        "ml_gate": 0.4612, "trail_activation": 0.1416,
        "atr_mult": 2.9141, "time_stop": 53,
    }

    @property
    def name(self) -> str:
        return "Mined426V2_1_KDJGoldenCross"


class Mined426V2_2(_MinedV2Base):
    """Bollinger 下轨反弹 + ML quality gate.
    IS 75.4% win, +6.4% avg, OOS 83.6% win.
    """
    template_id = "426-2"
    _entry_fn = staticmethod(_entry_bb_lower_bounce)
    _exit_walker = "classic"
    _params = {
        "ml_gate": 0.4971, "trail_activation": 0.1283,
        "trail_pct": 0.0416, "time_stop": 57,
    }

    @property
    def name(self) -> str:
        return "Mined426V2_2_BBLowerBounce"


class Mined426V2_3(_MinedV2Base):
    """动力线 上穿 0.5 (stage_watch) + ML quality gate.
    IS 83.4% win, +6.7% avg, OOS 86.3% win.
    """
    template_id = "426-3"
    _entry_fn = staticmethod(_entry_dl_cross_05)
    _exit_walker = "classic"
    _params = {
        "ml_gate": 0.4469, "trail_activation": 0.1112,
        "trail_pct": 0.0427, "time_stop": 96,
    }

    @property
    def name(self) -> str:
        return "Mined426V2_3_DongliWatch"


class Mined426V2_4(_MinedV2Base):
    """RSI 上穿 30 (oversold bounce) + ML quality gate (高门槛).
    IS 91.0% win (最高胜率), +6.0% avg, OOS 89.9% win.
    """
    template_id = "426-4"
    _entry_fn = staticmethod(_entry_rsi_30)
    _exit_walker = "classic"
    _params = {
        "ml_gate": 0.5448, "trail_activation": 0.0829,
        "trail_pct": 0.0581, "time_stop": 75,
    }

    @property
    def name(self) -> str:
        return "Mined426V2_4_RSIOversoldBounce"


class Mined426V2_5(_MinedV2Base):
    """买卖很准 准备首发 (DMI 极端横盘准备) + ML quality gate.
    IS 77.9% win, +6.6% avg, OOS 88.4% win.
    """
    template_id = "426-5"
    _entry_fn = staticmethod(_entry_zhunbei_first)
    _exit_walker = "atr"
    _params = {
        "ml_gate": 0.5774, "trail_activation": 0.1153,
        "atr_mult": 1.2598, "time_stop": 61,
    }

    @property
    def name(self) -> str:
        return "Mined426V2_5_ZhunbeiFirst"
