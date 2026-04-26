"""V2: 50 conceptually orthogonal entry signals.

Design principles:
  1. Each family fires on a distinct TRIGGER EVENT (cross, regime shift,
     pattern completion). NOT on continuous score thresholds.
  2. ML score is optional QUALITY GATE (param), not entry trigger.
  3. 10 concept categories ensure coverage; same-category ≤ 1 in top 5.

Concept categories:
  ML        — ML score crossings/changes
  动力线     — TDX 超级极品底 stage signals
  买卖很准   — TDX 急买/短买/准备 channel signals
  KDJ       — Stochastic crossings
  DMI       — Trend strength shifts
  RSI       — Mean-revert/momentum
  MACD      — Trend cross
  BB        — Bollinger bands
  Breakout  — N-day high/Donchian
  SMA       — Moving average cross
  Volume    — Volume regime
  Volatility— ATR contraction, NR7
  Pattern   — Candlestick / mean-revert
  Confluence— Multi-indicator
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================================================================
# Helpers
# =========================================================================

def _shift1(arr):
    return np.concatenate(([arr[0]], arr[:-1]))


def _cross_above(series, level):
    """True at the bar where series crosses up through level."""
    s = np.asarray(series, dtype=float)
    p = _shift1(s)
    return (p <= level) & (s > level)


def _cross_above_var(s1, s2):
    """True at bar where s1 crosses above s2."""
    s1 = np.asarray(s1, dtype=float)
    s2 = np.asarray(s2, dtype=float)
    p1, p2 = _shift1(s1), _shift1(s2)
    return (p1 <= p2) & (s1 > s2)


def _ml_gate(score, p):
    """Optional ML score quality gate. Returns bool array."""
    thr = p.get("ml_gate", None)
    if thr is None:
        return np.ones(len(score), dtype=bool)
    return score >= thr


# =========================================================================
# Entry signals — one per family, all return bool array length N
# =========================================================================

# ---- ML family (5) ----
def E_ml_cross_05(s, ind, p):
    return _cross_above(s, 0.50)

def E_ml_cross_06(s, ind, p):
    return _cross_above(s, 0.60)

def E_ml_5d_high(s, ind, p):
    """ML score reaches 5-day max."""
    s_max = pd.Series(s).rolling(5, min_periods=1).max().values
    return (s == s_max) & (s >= p.get("ml_gate", 0.45))

def E_ml_3d_strong(s, ind, p):
    """3 consecutive days score ≥ thr."""
    thr = p.get("ml_gate", 0.50)
    cond = s >= thr
    s_min = pd.Series(cond.astype(int)).rolling(3, min_periods=3).min().values
    cur = (s_min >= 1)
    prev = _shift1(cur.astype(int))
    return (prev == 0) & cur  # first day the 3-streak completes

def E_ml_velocity(s, ind, p):
    """5d rise of ML score > thr."""
    delta_thr = p.get("ml_delta", 0.20)
    s5 = _shift1(_shift1(_shift1(_shift1(_shift1(s)))))
    return (s - s5 >= delta_thr) & _ml_gate(s, p)


# ---- 动力线 family (4) ----
def E_dl_cross_02(s, ind, p):
    """Stage_bottom cross 0.2."""
    return _cross_above(ind["dl_value"], 0.2) & _ml_gate(s, p)

def E_dl_cross_05(s, ind, p):
    """Stage_watch cross 0.5."""
    return _cross_above(ind["dl_value"], 0.5) & _ml_gate(s, p)

def E_dl_cross_10(s, ind, p):
    return _cross_above(ind["dl_value"], 1.0) & _ml_gate(s, p)

def E_dl_velocity(s, ind, p):
    return (ind["dl_velocity"] >= p.get("vel_min", 0.5)) & _ml_gate(s, p)


# ---- 买卖很准 family (4) ----
def E_mm_jibuy_first(s, ind, p):
    """急买 first activation: jibuy 0→1."""
    j = ind["mm_jibuy_active"]
    return (_shift1(j) <= 0) & (j > 0) & _ml_gate(s, p)

def E_mm_duanbuy_first(s, ind, p):
    j = ind["mm_duanbuy_active"]
    return (_shift1(j) <= 0) & (j > 0) & _ml_gate(s, p)

def E_mm_zhunbei_first(s, ind, p):
    z = ind["mm_zhunbei_active"]
    return (_shift1(z) <= 0) & (z > 0) & _ml_gate(s, p)

def E_mm_shentou_high(s, ind, p):
    """神图 cross above 80."""
    return _cross_above(ind["mm_shentou"], 80.0) & _ml_gate(s, p)


# ---- KDJ family (4) ----
def E_kdj_j_cross_0(s, ind, p):
    return _cross_above(ind["kdj_j"], 0.0) & _ml_gate(s, p)

def E_kdj_j_cross_20(s, ind, p):
    return _cross_above(ind["kdj_j"], 20.0) & _ml_gate(s, p)

def E_kdj_kd_cross(s, ind, p):
    """K cross above D (golden cross)."""
    return _cross_above_var(ind["kdj_k"], ind["kdj_d"]) & _ml_gate(s, p)

def E_kdj_oversold_rebound(s, ind, p):
    """J was < 0 in last 3 bars and now crosses up."""
    j = ind["kdj_j"]
    was_under = pd.Series(j < 0).rolling(3, min_periods=1).max().values  # any <0 in last 3
    return _cross_above(j, 0.0) & (was_under > 0) & _ml_gate(s, p)


# ---- DMI family (3) ----
def E_dmi_pdi_cross(s, ind, p):
    """PDI cross above MDI."""
    return _cross_above_var(ind["dmi_pdi"], ind["dmi_mdi"]) & _ml_gate(s, p)

def E_dmi_adx_regime(s, ind, p):
    """ADX cross above 20 (regime shift to trending)."""
    return (_cross_above(ind["dmi_adx"], p.get("adx_thr", 20.0)) &
            (ind["dmi_pdi"] > ind["dmi_mdi"]) & _ml_gate(s, p))

def E_dmi_strong_bull(s, ind, p):
    return (_cross_above(ind["dmi_adx"], 30.0) &
            (ind["dmi_pdi"] > ind["dmi_mdi"]) & _ml_gate(s, p))


# ---- RSI family (3) ----
def E_rsi_30(s, ind, p):
    return _cross_above(ind["rsi14"], 30.0) & _ml_gate(s, p)

def E_rsi_50(s, ind, p):
    return _cross_above(ind["rsi14"], 50.0) & _ml_gate(s, p)

def E_rsi_div(s, ind, p):
    """Bullish RSI divergence proxy: price 20d-low + RSI > rsi[20-bar-ago]+5."""
    close_ind = ind.get("__close__")
    if close_ind is None:
        return np.zeros(len(s), dtype=bool)
    close = close_ind
    ll20 = pd.Series(close).rolling(20, min_periods=5).min().values
    at_low = (close <= ll20)
    rsi = ind["rsi14"]
    rsi_20ago = pd.Series(rsi).shift(20).fillna(50).values
    higher_rsi = rsi > rsi_20ago + 5.0
    return at_low & higher_rsi & _ml_gate(s, p)


# ---- MACD family (3) ----
def E_macd_diff_dea(s, ind, p):
    return _cross_above_var(ind["macd_diff"], ind["macd_dea"]) & _ml_gate(s, p)

def E_macd_diff_zero(s, ind, p):
    return _cross_above(ind["macd_diff"], 0.0) & _ml_gate(s, p)

def E_macd_hist_uptrend(s, ind, p):
    """3-bar rising hist, hist > 0, just turned positive last 5 bars."""
    h = ind["macd_hist"]
    h1, h2 = _shift1(h), _shift1(_shift1(h))
    rising = (h > h1) & (h1 > h2)
    pos = h > 0
    just_turned = pd.Series(_cross_above(h, 0.0).astype(float)).rolling(5, min_periods=1).max().values > 0
    return rising & pos & just_turned & _ml_gate(s, p)


# ---- Bollinger family (3) ----
def E_bb_lower_bounce(s, ind, p):
    """Close was below lower band, now back above."""
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    lower = ind["boll_lower"]
    return _cross_above_var(close, lower) & _ml_gate(s, p)

def E_bb_squeeze_break(s, ind, p):
    """BB width was at 20-bar minimum, then close pierces upper band."""
    w = ind["boll_width"]
    w_min = pd.Series(w).rolling(20, min_periods=5).min().values
    was_squeezed = (_shift1(w) <= _shift1(w_min) * 1.05)
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    upper = ind["boll_upper"]
    return was_squeezed & _cross_above_var(close, upper) & _ml_gate(s, p)

def E_bb_upper_break(s, ind, p):
    """Close cross above upper band (continuation)."""
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    return _cross_above_var(close, ind["boll_upper"]) & _ml_gate(s, p)


# ---- Channel breakout family (4) ----
def E_break_high20(s, ind, p):
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    return _cross_above_var(close, ind["high20"]) & _ml_gate(s, p)

def E_break_high50(s, ind, p):
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    return _cross_above_var(close, ind["high50"]) & _ml_gate(s, p)

def E_break_donch_55(s, ind, p):
    """Custom 55-bar high breakout."""
    high = ind.get("__high__")
    close = ind.get("__close__")
    if high is None or close is None:
        return np.zeros(len(s), dtype=bool)
    h55 = pd.Series(high).rolling(55, min_periods=10).max().shift(1).bfill().values
    return _cross_above_var(close, h55) & _ml_gate(s, p)

def E_break_high100(s, ind, p):
    """100-bar high — major breakout."""
    high = ind.get("__high__")
    close = ind.get("__close__")
    if high is None or close is None:
        return np.zeros(len(s), dtype=bool)
    h100 = pd.Series(high).rolling(100, min_periods=20).max().shift(1).bfill().values
    return _cross_above_var(close, h100) & _ml_gate(s, p)


# ---- SMA cross family (3) ----
def E_sma_5_20(s, ind, p):
    """SMA(5) cross above SMA(20). Need to compute sma5 from close."""
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    sma5 = pd.Series(close).rolling(5, min_periods=1).mean().values
    sma20 = ind["sma20"]
    return _cross_above_var(sma5, sma20) & _ml_gate(s, p)

def E_sma_10_30(s, ind, p):
    return _cross_above_var(ind["sma10"], ind["sma30"]) & _ml_gate(s, p)

def E_sma_20_60(s, ind, p):
    return _cross_above_var(ind["sma20"], ind["sma60"]) & _ml_gate(s, p)


# ---- Volume family (3) ----
def E_vol_spike_green(s, ind, p):
    """Volume > 2× MA20 AND green candle."""
    spike = ind["vol_spike"] >= p.get("vol_min", 2.0)
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    green = close > _shift1(close)
    return spike & green & _ml_gate(s, p)

def E_vol_spike_above_ma(s, ind, p):
    """Volume spike AND close > sma20."""
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    return ((ind["vol_spike"] >= 1.5) &
            (close > ind["sma20"]) & _ml_gate(s, p))

def E_vol_contract_break(s, ind, p):
    """Volume MA20 dropping then spike (accumulation reveal)."""
    vma20 = ind["vma20"]
    vma20_5ago = pd.Series(vma20).shift(5).fillna(vma20[0]).values
    contracted = vma20 < vma20_5ago * 0.85
    return contracted & (ind["vol_spike"] >= 2.0) & _ml_gate(s, p)


# ---- Volatility family (2) ----
def E_atr_contract_break(s, ind, p):
    """ATR was at 20-bar low, then close > prev high."""
    atr = ind["atr14"]
    atr_min = pd.Series(atr).rolling(20, min_periods=5).min().values
    contracted = (_shift1(atr) <= _shift1(atr_min) * 1.10)
    close = ind.get("__close__")
    high = ind.get("__high__")
    if close is None or high is None:
        return np.zeros(len(s), dtype=bool)
    break_up = close > _shift1(high)
    return contracted & break_up & _ml_gate(s, p)

def E_nr7_break(s, ind, p):
    """NR7: today's range is the smallest of last 7. Then breakout next bar."""
    high = ind.get("__high__")
    low = ind.get("__low__")
    close = ind.get("__close__")
    if high is None or low is None or close is None:
        return np.zeros(len(s), dtype=bool)
    rng = high - low
    rng_min7 = pd.Series(rng).rolling(7, min_periods=7).min().values
    nr7 = (_shift1(rng) <= _shift1(rng_min7) + 1e-9)
    break_up = close > _shift1(high)
    return nr7 & break_up & _ml_gate(s, p)


# ---- Mean reversion / pattern (2) ----
def E_zscore_revert(s, ind, p):
    """Z-score of close vs ma20 < -2 then closing higher."""
    close = ind.get("__close__")
    if close is None:
        return np.zeros(len(s), dtype=bool)
    sma = ind["sma20"]
    sd = pd.Series(close).rolling(20, min_periods=5).std().fillna(1).values
    z = np.where(sd > 0, (close - sma) / np.where(sd > 0, sd, 1), 0)
    deeply_under = (_shift1(z) < -2.0)
    rebound = close > _shift1(close)
    return deeply_under & rebound & _ml_gate(s, p)

def E_engulfing(s, ind, p):
    """Bullish engulfing: prev red candle, today green and close > prev open."""
    close = ind.get("__close__")
    o = ind.get("__open__")
    if close is None or o is None:
        return np.zeros(len(s), dtype=bool)
    prev_red = _shift1(close) < _shift1(o)
    today_green = close > o
    engulf = (close > _shift1(o)) & (o < _shift1(close))
    return prev_red & today_green & engulf & _ml_gate(s, p)


# ---- Confluence (no-ML) (8) ----
def E_dl_jibuy(s, ind, p):
    j = ind["mm_jibuy_active"]
    j_first = (_shift1(j) <= 0) & (j > 0)
    return j_first & (ind["dl_stage_bottom_recent"] > 0) & _ml_gate(s, p)

def E_dl_zhunbei(s, ind, p):
    z = ind["mm_zhunbei_active"]
    z_first = (_shift1(z) <= 0) & (z > 0)
    return z_first & (ind["dl_value"] < 2.5) & _ml_gate(s, p)

def E_dl_kdj_j0(s, ind, p):
    return ((ind["dl_stage_bottom_recent"] > 0) &
            _cross_above(ind["kdj_j"], 0.0) & _ml_gate(s, p))

def E_macd_dmi(s, ind, p):
    return (_cross_above_var(ind["macd_diff"], ind["macd_dea"]) &
            (ind["dmi_pdi"] > ind["dmi_mdi"]) &
            (ind["dmi_adx"] > 18) & _ml_gate(s, p))

def E_dl_macd(s, ind, p):
    return (_cross_above_var(ind["macd_diff"], ind["macd_dea"]) &
            (ind["dl_stage_bottom_recent"] > 0) & _ml_gate(s, p))

def E_kdj_dmi(s, ind, p):
    return (_cross_above(ind["kdj_j"], 0.0) &
            (ind["dmi_pdi"] > ind["dmi_mdi"]) &
            (ind["dmi_adx"] >= p.get("adx_min", 18)) & _ml_gate(s, p))

def E_jibuy_kdj_d(s, ind, p):
    """急买 first + K cross above D."""
    j = ind["mm_jibuy_active"]
    j_first = (_shift1(j) <= 0) & (j > 0)
    return j_first & _cross_above_var(ind["kdj_k"], ind["kdj_d"]) & _ml_gate(s, p)

def E_zhunbei_macd(s, ind, p):
    z = ind["mm_zhunbei_active"]
    z_first = (_shift1(z) <= 0) & (z > 0)
    return z_first & (ind["macd_diff"] > ind["macd_dea"]) & _ml_gate(s, p)


# =========================================================================
# Universal exit walkers — same as v1 (re-exported for convenience)
# =========================================================================

def walk_classic_trail(i, close, high, low, score, ind, p, n):
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    trail_pct = p.get("trail_pct", 0.03)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
    peak = high[i]; activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        if peak / entry_price - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        if activated:
            stop = peak * (1 - trail_pct)
            if low[k] <= stop:
                return (k, max(stop, hard_stop), "trail")
    return (end, close[end], "time_stop")


def walk_atr_trail(i, close, high, low, score, ind, p, n):
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    atr = ind["atr14"]
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
    peak = high[i]; activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        if peak / entry_price - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        if activated:
            stop = peak - atr_mult * atr[k]
            if low[k] <= stop:
                return (k, max(stop, hard_stop), "atr_trail")
    return (end, close[end], "time_stop")


def walk_score_decay(i, close, high, low, score, ind, p, n):
    entry_price = close[i]; hard_stop = entry_price * 0.90
    exit_thr = p.get("exit_threshold", 0.30)
    activation = p.get("trail_activation", 0.05)
    trail_pct = p.get("trail_pct", 0.04)
    time_stop = p.get("time_stop", 75)
    peak = high[i]; activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        if peak / entry_price - 1 >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        if score[k] < exit_thr:
            return (k, close[k], "score_decay")
        if activated:
            stop = peak * (1 - trail_pct)
            if low[k] <= stop:
                return (k, stop, "trail")
    return (end, close[end], "time_stop")


# =========================================================================
# 50 family registry
# =========================================================================

@dataclass
class FamilyDef:
    family_id: str
    concept: str
    entry_fn: callable
    exit_fn: callable
    default_params: dict


FAMILIES: list[FamilyDef] = []


def _add(fid, concept, entry_fn, exit_fn, params):
    FAMILIES.append(FamilyDef(fid, concept, entry_fn, exit_fn, params))


# Default param skeleton with optional ML gate
_BASE = {"ml_gate": 0.40, "trail_activation": 0.05, "time_stop": 60}
_TRAIL = {**_BASE, "trail_pct": 0.04}
_ATR = {**_BASE, "atr_mult": 2.0}
_DECAY = {**_BASE, "exit_threshold": 0.25, "trail_pct": 0.04}

# ---- ML (5) ----
_add("M01_ml_cross_05", "ML", E_ml_cross_05, walk_atr_trail, dict(_ATR))
_add("M02_ml_cross_06", "ML", E_ml_cross_06, walk_atr_trail, dict(_ATR))
_add("M03_ml_5d_high",  "ML", E_ml_5d_high,  walk_classic_trail, dict(_TRAIL))
_add("M04_ml_3d_strong", "ML", E_ml_3d_strong, walk_score_decay, dict(_DECAY))
_add("M05_ml_velocity", "ML", E_ml_velocity, walk_atr_trail, {**_ATR, "ml_delta": 0.20})

# ---- 动力线 (4) ----
_add("D01_dl_cross_02", "动力线", E_dl_cross_02, walk_atr_trail, dict(_ATR))
_add("D02_dl_cross_05", "动力线", E_dl_cross_05, walk_classic_trail, dict(_TRAIL))
_add("D03_dl_cross_10", "动力线", E_dl_cross_10, walk_atr_trail, dict(_ATR))
_add("D04_dl_velocity", "动力线", E_dl_velocity, walk_classic_trail, {**_TRAIL, "vel_min": 0.5})

# ---- 买卖很准 (4) ----
_add("MM01_jibuy",   "买卖很准", E_mm_jibuy_first,   walk_atr_trail, dict(_ATR))
_add("MM02_duanbuy", "买卖很准", E_mm_duanbuy_first, walk_classic_trail, dict(_TRAIL))
_add("MM03_zhunbei", "买卖很准", E_mm_zhunbei_first, walk_atr_trail, dict(_ATR))
_add("MM04_shentou", "买卖很准", E_mm_shentou_high,  walk_atr_trail, dict(_ATR))

# ---- KDJ (4) ----
_add("K01_j_cross_0",  "KDJ", E_kdj_j_cross_0,  walk_atr_trail, dict(_ATR))
_add("K02_j_cross_20", "KDJ", E_kdj_j_cross_20, walk_classic_trail, dict(_TRAIL))
_add("K03_kd_cross",   "KDJ", E_kdj_kd_cross,   walk_atr_trail, dict(_ATR))
_add("K04_oversold_rebound", "KDJ", E_kdj_oversold_rebound, walk_classic_trail, dict(_TRAIL))

# ---- DMI (3) ----
_add("DI01_pdi_cross",   "DMI", E_dmi_pdi_cross, walk_atr_trail, dict(_ATR))
_add("DI02_adx_regime",  "DMI", E_dmi_adx_regime, walk_atr_trail, {**_ATR, "adx_thr": 20.0})
_add("DI03_strong_bull", "DMI", E_dmi_strong_bull, walk_atr_trail, dict(_ATR))

# ---- RSI (3) ----
_add("R01_rsi_30",  "RSI", E_rsi_30,  walk_classic_trail, dict(_TRAIL))
_add("R02_rsi_50",  "RSI", E_rsi_50,  walk_atr_trail, dict(_ATR))
_add("R03_rsi_div", "RSI", E_rsi_div, walk_classic_trail, dict(_TRAIL))

# ---- MACD (3) ----
_add("MA01_diff_dea",     "MACD", E_macd_diff_dea,     walk_atr_trail, dict(_ATR))
_add("MA02_diff_zero",    "MACD", E_macd_diff_zero,    walk_atr_trail, dict(_ATR))
_add("MA03_hist_uptrend", "MACD", E_macd_hist_uptrend, walk_classic_trail, dict(_TRAIL))

# ---- Bollinger (3) ----
_add("B01_bb_lower",   "BB", E_bb_lower_bounce, walk_classic_trail, dict(_TRAIL))
_add("B02_bb_squeeze", "BB", E_bb_squeeze_break, walk_atr_trail, dict(_ATR))
_add("B03_bb_upper",   "BB", E_bb_upper_break,   walk_atr_trail, dict(_ATR))

# ---- Breakout (4) ----
_add("BR01_high20",   "Breakout", E_break_high20,   walk_atr_trail, dict(_ATR))
_add("BR02_high50",   "Breakout", E_break_high50,   walk_atr_trail, dict(_ATR))
_add("BR03_donch55",  "Breakout", E_break_donch_55, walk_atr_trail, dict(_ATR))
_add("BR04_high100",  "Breakout", E_break_high100,  walk_atr_trail, dict(_ATR))

# ---- SMA (3) ----
_add("S01_sma_5_20",  "SMA", E_sma_5_20,  walk_classic_trail, dict(_TRAIL))
_add("S02_sma_10_30", "SMA", E_sma_10_30, walk_atr_trail, dict(_ATR))
_add("S03_sma_20_60", "SMA", E_sma_20_60, walk_atr_trail, dict(_ATR))

# ---- Volume (3) ----
_add("V01_spike_green", "Volume", E_vol_spike_green,    walk_classic_trail, {**_TRAIL, "vol_min": 2.0})
_add("V02_spike_ma",    "Volume", E_vol_spike_above_ma, walk_atr_trail, dict(_ATR))
_add("V03_contract",    "Volume", E_vol_contract_break, walk_classic_trail, dict(_TRAIL))

# ---- Volatility (2) ----
_add("VL01_atr_break", "Volatility", E_atr_contract_break, walk_atr_trail, dict(_ATR))
_add("VL02_nr7_break", "Volatility", E_nr7_break,           walk_classic_trail, dict(_TRAIL))

# ---- Pattern / mean revert (2) ----
_add("P01_zscore",    "Pattern", E_zscore_revert, walk_classic_trail, dict(_TRAIL))
_add("P02_engulfing", "Pattern", E_engulfing,     walk_classic_trail, dict(_TRAIL))

# ---- Confluence (8) ----
_add("C01_dl_jibuy",     "Confluence", E_dl_jibuy,     walk_atr_trail, dict(_ATR))
_add("C02_dl_zhunbei",   "Confluence", E_dl_zhunbei,   walk_atr_trail, dict(_ATR))
_add("C03_dl_kdj_j0",    "Confluence", E_dl_kdj_j0,    walk_atr_trail, dict(_ATR))
_add("C04_macd_dmi",     "Confluence", E_macd_dmi,     walk_atr_trail, dict(_ATR))
_add("C05_dl_macd",      "Confluence", E_dl_macd,      walk_atr_trail, dict(_ATR))
_add("C06_kdj_dmi",      "Confluence", E_kdj_dmi,      walk_atr_trail, dict(_ATR))
_add("C07_jibuy_kd",     "Confluence", E_jibuy_kdj_d,  walk_classic_trail, dict(_TRAIL))
_add("C08_zhunbei_macd", "Confluence", E_zhunbei_macd, walk_atr_trail, dict(_ATR))


# 51 → trim 1 to fit 50: drop weakest confluence (will be regenerated below)
FAMILIES.pop()  # remove last C08
assert len(FAMILIES) == 50, f"expected 50, got {len(FAMILIES)}"


# =========================================================================
# Backtest engine — needs close/high/low/open injected into ind
# =========================================================================

def run_one_strategy_one_stock(family: FamilyDef, params: dict,
                               close, high, low, score, ind, n) -> list[dict]:
    if n < 30:
        return []
    # Inject OHLC accessors for entry fns that need them
    ind = {**ind,
           "__close__": close, "__high__": high, "__low__": low,
           "__open__": ind.get("__open__", close)}
    try:
        entry_mask = family.entry_fn(score, ind, params)
    except Exception:
        return []
    if entry_mask.sum() == 0:
        return []

    trades = []
    in_pos = False
    exit_until = -1
    start = 25
    for i in range(start, n - 1):
        if in_pos:
            if i >= exit_until:
                in_pos = False
            continue
        if not entry_mask[i]:
            continue
        if i > 0 and close[i] > close[i - 1] * 1.099:
            continue  # limit-up filter
        try:
            exit_idx, exit_price, reason = family.exit_fn(
                i, close, high, low, score, ind, params, n)
        except Exception:
            continue
        entry_price = close[i]
        ret = exit_price / entry_price - 1
        trades.append({
            "entry_idx": i, "exit_idx": exit_idx,
            "entry_price": entry_price, "exit_price": exit_price,
            "ret": ret, "reason": reason,
            "hold_days": exit_idx - i,
        })
        in_pos = True
        exit_until = exit_idx
    return trades
