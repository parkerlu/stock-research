"""Mined-426 V2 — generic executor + top-10 presets.

This file is self-contained: it inlines the entry-signal functions and
exit walkers from scripts/mine_426_v2_strategies.py so the backend container
can run them without needing the scripts/ folder mounted.

Top 10 winners are exposed as 426-1 through 426-10. Adding more is a one-line
entry in TOP_PRESETS.
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
# Indicator pack — vectorized, mirror of mine_426_step3
# =========================================================================

def _shift1(arr):
    return np.concatenate(([arr[0]], arr[:-1]))


def _compute_indicators(close, high, low, vol) -> dict:
    n = len(close)
    sh, sl, sc, sv = pd.Series(high), pd.Series(low), pd.Series(close), pd.Series(vol)

    # 动力线
    var2 = sl.rolling(10, min_periods=1).min().values
    var33 = sh.rolling(25, min_periods=1).max().values
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values
    prev_d = _shift1(dongli)
    dl_stage_bottom_recent = pd.Series(((prev_d <= 0.2) & (dongli > 0.2)).astype(float)).rolling(5, min_periods=1).max().values
    dl_velocity = np.zeros(n)
    if n >= 5:
        dl_velocity[5:] = dongli[5:] - dongli[:-5]

    # KDJ
    h9 = sh.rolling(9, min_periods=1).max().values
    l9 = sl.rolling(9, min_periods=1).min().values
    rng9 = h9 - l9
    rsv = np.where(rng9 > 0, (close - l9) / np.where(rng9 > 0, rng9, 1) * 100, 50.0)
    k = np.zeros(n); d = np.zeros(n); k[0] = 50.0; d[0] = 50.0
    for i in range(1, n):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    j = 3 * k - 2 * d

    # ATR (Wilder)
    prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values

    # DMI
    prev_h = _shift1(high); prev_l = _shift1(low)
    hd = high - prev_h; ld = prev_l - low
    pos_dm = np.where((hd > 0) & (hd > ld), hd, 0.0)
    neg_dm = np.where((ld > 0) & (ld > hd), ld, 0.0)
    tr14 = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    pdm14 = pd.Series(pos_dm).ewm(alpha=1 / 14, adjust=False).mean().values
    ndm14 = pd.Series(neg_dm).ewm(alpha=1 / 14, adjust=False).mean().values
    pdi = np.where(tr14 > 0, 100 * pdm14 / np.where(tr14 > 0, tr14, 1), 0)
    mdi = np.where(tr14 > 0, 100 * ndm14 / np.where(tr14 > 0, tr14, 1), 0)
    denom = pdi + mdi
    dx = np.where(denom > 0, 100 * np.abs(pdi - mdi) / np.where(denom > 0, denom, 1), 0)
    adx = pd.Series(dx).ewm(alpha=1 / 14, adjust=False).mean().values

    # RSI
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
    rs = np.where(al > 0, ag / np.where(al > 0, al, 1), 100.0)
    rsi = 100 - 100 / (1 + rs)

    # MACD
    ema12 = sc.ewm(span=12, adjust=False).mean().values
    ema26 = sc.ewm(span=26, adjust=False).mean().values
    macd_diff = ema12 - ema26
    macd_dea = pd.Series(macd_diff).ewm(span=9, adjust=False).mean().values
    macd_hist = macd_diff - macd_dea

    # Bollinger (20, 2)
    mid = sc.rolling(20, min_periods=1).mean().values
    std = sc.rolling(20, min_periods=1).std().fillna(0).values
    bb_upper = mid + 2.0 * std
    bb_lower = mid - 2.0 * std
    bb_width = np.where(mid > 0, (bb_upper - bb_lower) / np.where(mid > 0, mid, 1), 0.0)

    # Channel (Donchian)
    high20 = sh.rolling(20, min_periods=1).max().shift(1).bfill().values
    high50 = sh.rolling(50, min_periods=1).max().shift(1).bfill().values

    # SMAs
    sma5 = sc.rolling(5, min_periods=1).mean().values
    sma10 = sc.rolling(10, min_periods=1).mean().values
    sma20 = sc.rolling(20, min_periods=1).mean().values
    sma30 = sc.rolling(30, min_periods=1).mean().values
    sma60 = sc.rolling(60, min_periods=1).mean().values

    # Volume
    vma20 = sv.rolling(20, min_periods=1).mean().values
    vol_spike = np.where(vma20 > 0, vol / np.where(vma20 > 0, vma20, 1), 1.0)

    # 买卖很准 (准备 / 急买)
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    floor10 = ban_s.rolling(10, min_periods=10).min().values
    below_floor = np.where(np.isnan(floor10), 0.0, (close < floor10).astype(float))
    bf = pd.Series(below_floor)
    jibuy = (bf.rolling(5, min_periods=1).max() > 0).astype(float).values
    duanbuy = (bf.rolling(10, min_periods=1).max() > 0).astype(float).values
    # DMI(5) for 准备
    td5 = pd.Series(tr).rolling(5, min_periods=5).sum().values
    pdm5 = pd.Series(pos_dm).rolling(5, min_periods=5).sum().values
    ndm5 = pd.Series(neg_dm).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td5 > 0, pdm5 * 100 / td5, 0.0)
        fuzhu = np.where(td5 > 0, ndm5 * 100 / td5, 0.0)
        dx5_denom = fuzhu + shentou
        dx5 = np.where(dx5_denom > 0, np.abs(fuzhu - shentou) / dx5_denom * 100, 0.0)
    dongxiang = pd.Series(dx5).rolling(3, min_periods=1).mean().values
    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    return {
        "dl_value": dongli,
        "dl_stage_bottom_recent": dl_stage_bottom_recent,
        "dl_velocity": dl_velocity,
        "kdj_k": k, "kdj_d": d, "kdj_j": j,
        "atr14": atr14,
        "dmi_pdi": pdi, "dmi_mdi": mdi, "dmi_adx": adx,
        "rsi14": rsi,
        "macd_diff": macd_diff, "macd_dea": macd_dea, "macd_hist": macd_hist,
        "boll_upper": bb_upper, "boll_lower": bb_lower, "boll_width": bb_width,
        "high20": high20, "high50": high50,
        "sma5": sma5, "sma10": sma10, "sma20": sma20, "sma30": sma30, "sma60": sma60,
        "vma20": vma20, "vol_spike": vol_spike,
        "mm_jibuy_active": jibuy, "mm_duanbuy_active": duanbuy,
        "mm_zhunbei_active": zhunbei, "mm_shentou": shentou,
    }


# =========================================================================
# Entry signal functions — mirror of mine_426_v2_strategies.py
# Each returns bool array length N. score & ind are positional, params dict.
# =========================================================================

def _cross_above(s, level):
    s = np.asarray(s, dtype=float); p = _shift1(s)
    return (p <= level) & (s > level)


def _cross_above_var(s1, s2):
    s1 = np.asarray(s1, dtype=float); s2 = np.asarray(s2, dtype=float)
    p1, p2 = _shift1(s1), _shift1(s2)
    return (p1 <= p2) & (s1 > s2)


def _ml_gate(score, p):
    thr = p.get("ml_gate", None)
    if thr is None:
        return np.ones(len(score), dtype=bool)
    return score >= thr


def E_dl_cross_05(s, ind, p):
    return _cross_above(ind["dl_value"], 0.5) & _ml_gate(s, p)


def E_dl_cross_10(s, ind, p):
    return _cross_above(ind["dl_value"], 1.0) & _ml_gate(s, p)


def E_kdj_kd_cross(s, ind, p):
    return _cross_above_var(ind["kdj_k"], ind["kdj_d"]) & _ml_gate(s, p)


def E_kdj_j_cross_20(s, ind, p):
    return _cross_above(ind["kdj_j"], 20.0) & _ml_gate(s, p)


def E_rsi_30(s, ind, p):
    return _cross_above(ind["rsi14"], 30.0) & _ml_gate(s, p)


def E_sma_5_20(s, ind, p):
    return _cross_above_var(ind["sma5"], ind["sma20"]) & _ml_gate(s, p)


def E_break_high20(s, ind, p):
    close = ind["__close__"]
    return _cross_above_var(close, ind["high20"]) & _ml_gate(s, p)


def E_break_high50(s, ind, p):
    close = ind["__close__"]
    return _cross_above_var(close, ind["high50"]) & _ml_gate(s, p)


def E_bb_upper_break(s, ind, p):
    close = ind["__close__"]
    return _cross_above_var(close, ind["boll_upper"]) & _ml_gate(s, p)


# Family registry — only contains entries used by current presets.
# More can be added freely; mining results elsewhere reference these IDs.
FAMILY_ENTRY_FNS = {
    "D02_dl_cross_05": E_dl_cross_05,
    "D03_dl_cross_10": E_dl_cross_10,
    "K02_j_cross_20": E_kdj_j_cross_20,
    "K03_kd_cross": E_kdj_kd_cross,
    "R01_rsi_30": E_rsi_30,
    "S01_sma_5_20": E_sma_5_20,
    "BR01_high20": E_break_high20,
    "BR02_high50": E_break_high50,
    "B03_bb_upper": E_bb_upper_break,
}


# =========================================================================
# Exit walkers (CLASSIC + ATR; both apply hard stop -10%)
# =========================================================================

def _walk_classic(i, close, high, low, atr, p, n):
    entry = close[i]; hard_stop = entry * 0.90
    trail_pct = p.get("trail_pct", 0.04)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
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
    entry = close[i]; hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
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


def _pick_walker(family_id: str):
    """Each family had a default exit type during mining. Defaults preserved."""
    classic_families = {"D02_dl_cross_05", "K02_j_cross_20", "R01_rsi_30",
                        "S01_sma_5_20", "B01_bb_lower"}
    return _walk_classic if family_id in classic_families else _walk_atr


# =========================================================================
# Generic strategy class
# =========================================================================

class _MinedV2Generic(StrategyTemplate):
    """Generic V2 strategy: parameterized by family_id + params + display."""

    template_id = "mined_v2_generic"
    _family_id = ""
    _params: dict = {}
    _display_name = ""

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return self._display_name or self.template_id

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

        ind = _compute_indicators(close, high, low, vol)
        ind["__close__"] = close
        ind["__high__"] = high
        ind["__low__"] = low

        # ML score per bar (120-bar warmup)
        score = np.zeros(len(close))
        rows = []; valid_idx = []
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

        entry_fn = FAMILY_ENTRY_FNS.get(self._family_id)
        if entry_fn is None:
            return []
        entry_mask = entry_fn(score, ind, self._params)

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(close)
        atr14 = ind["atr14"]
        walker = _pick_walker(self._family_id)

        for i in range(120, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if not entry_mask[i]:
                continue
            if i > 0 and close[i] > close[i - 1] * 1.099:
                continue  # limit-up filter
            exit_idx, _exit_price = walker(i, close, high, low, atr14, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx

        return signals


# =========================================================================
# Top-10 presets — output of mining round 1 (2031 stocks, 2026-04-27)
# =========================================================================

TOP_PRESETS: list[dict] = [
    {"id": "426-1", "family": "D02_dl_cross_05", "concept": "动力线",
     "name": "Mined426V2_1_DongliCross05",
     "metrics": {"is_win": 84.6, "is_avg": 6.7, "oos_win": 87.8, "oos_avg": 7.3},
     "params": {"ml_gate": 0.5345, "trail_activation": 0.1053, "time_stop": 85, "trail_pct": 0.0619}},
    {"id": "426-2", "family": "K03_kd_cross", "concept": "KDJ",
     "name": "Mined426V2_2_KDJGoldenCross",
     "metrics": {"is_win": 76.9, "is_avg": 7.6, "oos_win": 80.7, "oos_avg": 7.3},
     "params": {"ml_gate": 0.5993, "trail_activation": 0.1012, "time_stop": 72, "atr_mult": 2.0010}},
    {"id": "426-3", "family": "R01_rsi_30", "concept": "RSI",
     "name": "Mined426V2_3_RSI30Bounce",
     "metrics": {"is_win": 75.1, "is_avg": 7.3, "oos_win": 79.3, "oos_avg": 8.1},
     "params": {"ml_gate": 0.5873, "trail_activation": 0.1411, "time_stop": 91, "trail_pct": 0.0397}},
    {"id": "426-4", "family": "K02_j_cross_20", "concept": "KDJ",
     "name": "Mined426V2_4_KDJJ20",
     "metrics": {"is_win": 80.8, "is_avg": 9.6, "oos_win": 85.7, "oos_avg": 10.7},
     "params": {"ml_gate": 0.5989, "trail_activation": 0.1445, "time_stop": 76, "trail_pct": 0.0313}},
    {"id": "426-5", "family": "D03_dl_cross_10", "concept": "动力线",
     "name": "Mined426V2_5_DongliCross10",
     "metrics": {"is_win": 78.6, "is_avg": 9.0, "oos_win": 83.2, "oos_avg": 10.0},
     "params": {"ml_gate": 0.5088, "trail_activation": 0.1204, "time_stop": 88, "atr_mult": 2.2225}},
    {"id": "426-6", "family": "S01_sma_5_20", "concept": "SMA",
     "name": "Mined426V2_6_SMA5_20Cross",
     "metrics": {"is_win": 94.9, "is_avg": 7.3, "oos_win": 91.7, "oos_avg": 16.5},
     "params": {"ml_gate": 0.5181, "trail_activation": 0.0408, "time_stop": 46, "trail_pct": 0.0323}},
    {"id": "426-7", "family": "BR01_high20", "concept": "Breakout",
     "name": "Mined426V2_7_High20Break",
     "metrics": {"is_win": 77.1, "is_avg": 7.2, "oos_win": 77.3, "oos_avg": 11.7},
     "params": {"ml_gate": 0.5109, "trail_activation": 0.0955, "time_stop": 86, "atr_mult": 1.8498}},
    {"id": "426-8", "family": "S01_sma_5_20", "concept": "SMA",
     "name": "Mined426V2_8_SMA5_20Slow",
     "metrics": {"is_win": 90.6, "is_avg": 8.7, "oos_win": 93.8, "oos_avg": 23.6},
     "params": {"ml_gate": 0.5548, "trail_activation": 0.0508, "time_stop": 61, "trail_pct": 0.0328}},
    {"id": "426-9", "family": "BR02_high50", "concept": "Breakout",
     "name": "Mined426V2_9_High50Break",
     "metrics": {"is_win": 80.4, "is_avg": 7.9, "oos_win": 77.8, "oos_avg": 11.6},
     "params": {"ml_gate": 0.5142, "trail_activation": 0.0864, "time_stop": 45, "atr_mult": 1.8630}},
    {"id": "426-10", "family": "B03_bb_upper", "concept": "BB",
     "name": "Mined426V2_10_BBUpperBreak",
     "metrics": {"is_win": 77.6, "is_avg": 6.8, "oos_win": 82.1, "oos_avg": 13.3},
     "params": {"ml_gate": 0.5738, "trail_activation": 0.0513, "time_stop": 51, "atr_mult": 1.3242}},
]


def _make_class(spec: dict) -> type:
    cls = type(
        f"MinedV2_{spec['id'].replace('-', '_')}",
        (_MinedV2Generic,),
        {
            "template_id": spec["id"],
            "_family_id": spec["family"],
            "_params": spec["params"],
            "_display_name": spec["name"],
        },
    )
    return cls


# Build the 10 classes dynamically and expose them by name
_classes = {spec["id"]: _make_class(spec) for spec in TOP_PRESETS}
Mined426V2_1 = _classes["426-1"]
Mined426V2_2 = _classes["426-2"]
Mined426V2_3 = _classes["426-3"]
Mined426V2_4 = _classes["426-4"]
Mined426V2_5 = _classes["426-5"]
Mined426V2_6 = _classes["426-6"]
Mined426V2_7 = _classes["426-7"]
Mined426V2_8 = _classes["426-8"]
Mined426V2_9 = _classes["426-9"]
Mined426V2_10 = _classes["426-10"]
