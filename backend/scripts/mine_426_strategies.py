"""50 strategy families for the 426 mining.

Each family is identified by (entry_id, exit_id, default_params). The mining
script runs each family across 1004 stocks with 20 parameter variants → 1000
candidates total.

Entry signals (15) produce a length-N bool array of buy events.
Exit rules (5) take an entry index + per-stock arrays and return the exit index
+ exit price + reason. A hard stop of -10% always applies on top.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# =========================================================================
# Entry signal generators (return bool array length N)
# =========================================================================

def entry_ml_high(score, ind, p):
    return score >= p["buy_threshold"]


def entry_ml_dongli(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["dl_stage_bottom_recent"] > 0)


def entry_ml_jibuy(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["mm_jibuy_active"] > 0)


def entry_ml_kdj(score, ind, p):
    j = ind["kdj_j"]
    j_prev = np.concatenate(([j[0]], j[:-1]))
    return (score >= p["buy_threshold"]) & ((j_prev <= 0) & (j > 0))


def entry_dongli_bottom(score, ind, p):
    d = ind["dl_value"]
    d_prev = np.concatenate(([d[0]], d[:-1]))
    return (d_prev <= 0.2) & (d > 0.2)


def entry_dongli_watch(score, ind, p):
    d = ind["dl_value"]
    d_prev = np.concatenate(([d[0]], d[:-1]))
    return (d_prev <= 0.5) & (d > 0.5)


def entry_jibuy(score, ind, p):
    return ind["mm_jibuy_active"] > 0


def entry_duanbuy(score, ind, p):
    return ind["mm_duanbuy_active"] > 0


def entry_zhunbei(score, ind, p):
    return ind["mm_zhunbei_active"] > 0


def entry_kdj_j_cross(score, ind, p):
    j = ind["kdj_j"]
    j_prev = np.concatenate(([j[0]], j[:-1]))
    return (j_prev <= 0) & (j > 0)


def entry_dmi_bull(score, ind, p):
    pdi = ind["dmi_pdi"]
    mdi = ind["dmi_mdi"]
    adx = ind["dmi_adx"]
    pdi_prev = np.concatenate(([pdi[0]], pdi[:-1]))
    mdi_prev = np.concatenate(([mdi[0]], mdi[:-1]))
    cross = (pdi_prev <= mdi_prev) & (pdi > mdi)
    return cross & (adx >= p.get("adx_min", 20))


def entry_rsi_oversold(score, ind, p):
    rsi = ind["rsi14"]
    rsi_prev = np.concatenate(([rsi[0]], rsi[:-1]))
    return (rsi_prev <= p.get("rsi_low", 30)) & (rsi > p.get("rsi_low", 30))


def entry_macd_cross(score, ind, p):
    diff = ind["macd_diff"]
    dea = ind["macd_dea"]
    diff_prev = np.concatenate(([diff[0]], diff[:-1]))
    dea_prev = np.concatenate(([dea[0]], dea[:-1]))
    return (diff_prev <= dea_prev) & (diff > dea) & (diff > 0)


def entry_boll_bounce(close, ind, p):
    lower = ind["boll_lower"]
    return (np.concatenate(([close[0]], close[:-1])) < lower) & (close > lower * 0.998)


def entry_donch20(close, high, ind, p):
    h20 = ind["high20"]
    return close > h20


# =========================================================================
# Exit rule walkers
# =========================================================================
# Each takes entry index i, returns (exit_idx, exit_price, exit_reason).
# Universal: hard stop_loss=-10% from entry close.
# All exits use next-bar OPEN to trade.

def walk_classic_trail(i, close, high, low, score, ind, p, n):
    """Trailing stop based on running peak * (1 - trail_pct)."""
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    trail_pct = p.get("trail_pct", 0.03)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        gain = peak / entry_price - 1
        if gain >= activation:
            activated = True
        # Hard stop
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        if activated:
            stop = peak * (1 - trail_pct)
            if low[k] <= stop:
                return (k, stop, "trail")
    return (end, close[end], "time_stop")


def walk_atr_trail(i, close, high, low, score, ind, p, n):
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    atr = ind["atr14"]
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.05)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        gain = peak / entry_price - 1
        if gain >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        if activated:
            stop = peak - atr_mult * atr[k]
            if low[k] <= stop:
                return (k, max(stop, hard_stop), "atr_trail")
    return (end, close[end], "time_stop")


def walk_chandelier(i, close, high, low, score, ind, p, n):
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    atr = ind["atr14"]
    atr_mult = p.get("atr_mult", 2.5)
    time_stop = p.get("time_stop", 75)
    end = min(i + time_stop, n - 1)
    look = p.get("chandelier_look", 20)
    for k in range(i + 1, end + 1):
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        # HHV(high, look) since entry
        lo_idx = max(i, k - look + 1)
        hhv = high[lo_idx:k + 1].max()
        stop = hhv - atr_mult * atr[k]
        if low[k] <= stop:
            return (k, max(stop, hard_stop), "chandelier")
    return (end, close[end], "time_stop")


def walk_score_decay(i, close, high, low, score, ind, p, n):
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    exit_threshold = p.get("exit_threshold", 0.30)
    activation = p.get("trail_activation", 0.05)
    trail_pct = p.get("trail_pct", 0.04)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        gain = peak / entry_price - 1
        if gain >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        # Score-based exit
        if score[k] < exit_threshold:
            return (k, close[k], "score_decay")
        if activated:
            stop = peak * (1 - trail_pct)
            if low[k] <= stop:
                return (k, stop, "trail")
    return (end, close[end], "time_stop")


def walk_indicator_sell(i, close, high, low, score, ind, p, n):
    """Exit when 动力线 crosses below 3.5 OR 急卖 signal."""
    entry_price = close[i]
    hard_stop = entry_price * 0.90
    activation = p.get("trail_activation", 0.05)
    trail_pct = p.get("trail_pct", 0.04)
    time_stop = p.get("time_stop", 75)
    peak = high[i]
    activated = False
    end = min(i + time_stop, n - 1)
    dl = ind["dl_value"]
    for k in range(i + 1, end + 1):
        peak = max(peak, high[k])
        gain = peak / entry_price - 1
        if gain >= activation:
            activated = True
        if low[k] <= hard_stop:
            return (k, hard_stop, "hard_stop")
        # Indicator sell: 动力线 cross down 3.5
        if dl[k - 1] >= 3.5 and dl[k] < 3.5:
            return (k, close[k], "dl_sell")
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
    entry_fn: callable
    exit_fn: callable
    default_params: dict
    entry_input: tuple = ("score", "ind")  # what entry_fn needs
    description: str = ""


def _entry_with_close(fn, close, score, ind, p):
    """Wrapper for entries that need close (e.g., boll_bounce, donch)."""
    return fn(close, ind, p)


# Build 50 families: 15 entries × variants of exits = 50 deliberate combinations
FAMILIES: list[FamilyDef] = []


def _add(fid, entry_fn, exit_fn, params, entry_input=("score", "ind"), desc=""):
    FAMILIES.append(FamilyDef(fid, entry_fn, exit_fn, params, entry_input, desc))


# ---- A: ML score + various exits (10) ----
_add("A01_ml_classic",  entry_ml_high, walk_classic_trail,
     {"buy_threshold": 0.50, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 75},
     desc="ML high score + classic trail")
_add("A02_ml_atr",      entry_ml_high, walk_atr_trail,
     {"buy_threshold": 0.50, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 75})
_add("A03_ml_chand",    entry_ml_high, walk_chandelier,
     {"buy_threshold": 0.50, "atr_mult": 2.5, "chandelier_look": 20, "time_stop": 75})
_add("A04_ml_decay",    entry_ml_high, walk_score_decay,
     {"buy_threshold": 0.55, "exit_threshold": 0.30, "trail_pct": 0.04, "time_stop": 75})
_add("A05_ml_dlexit",   entry_ml_high, walk_indicator_sell,
     {"buy_threshold": 0.50, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 75})
_add("A06_mldl_classic", entry_ml_dongli, walk_classic_trail,
     {"buy_threshold": 0.45, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 75})
_add("A07_mldl_atr",    entry_ml_dongli, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 75})
_add("A08_mljibuy_classic", entry_ml_jibuy, walk_classic_trail,
     {"buy_threshold": 0.45, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("A09_mljibuy_atr", entry_ml_jibuy, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("A10_mlkdj",       entry_ml_kdj, walk_atr_trail,
     {"buy_threshold": 0.50, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})

# ---- C: TDX indicator entries (10) ----
_add("C01_dl_bottom_classic", entry_dongli_bottom, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("C02_dl_bottom_atr", entry_dongli_bottom, walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("C03_dl_bottom_dlexit", entry_dongli_bottom, walk_indicator_sell,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("C04_dl_watch_classic", entry_dongli_watch, walk_classic_trail,
     {"trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 45})
_add("C05_dl_watch_atr", entry_dongli_watch, walk_atr_trail,
     {"atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 45})
_add("C06_jibuy_classic", entry_jibuy, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("C07_duanbuy_classic", entry_duanbuy, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("C08_zhunbei_atr", entry_zhunbei, walk_atr_trail,
     {"atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 60})
_add("C09_kdjj_classic", entry_kdj_j_cross, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 45})
_add("C10_dmi_bull", entry_dmi_bull, walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60, "adx_min": 20})

# ---- D: Classic momentum (10) ----
_add("D01_rsi_classic",  entry_rsi_oversold, walk_classic_trail,
     {"rsi_low": 30, "trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 45})
_add("D02_rsi_atr",      entry_rsi_oversold, walk_atr_trail,
     {"rsi_low": 30, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 45})
_add("D03_macd_classic", entry_macd_cross, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("D04_macd_atr",     entry_macd_cross, walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("D05_boll_bounce",  lambda s, i, p: entry_boll_bounce(i["__close__"], i, p),
     walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 45})
_add("D06_donch20",      lambda s, i, p: entry_donch20(i["__close__"], i["__high__"], i, p),
     walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})
_add("D07_donch20_atr",  lambda s, i, p: entry_donch20(i["__close__"], i["__high__"], i, p),
     walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("D08_rsi_chand",    entry_rsi_oversold, walk_chandelier,
     {"rsi_low": 30, "atr_mult": 2.5, "time_stop": 60})
_add("D09_macd_chand",   entry_macd_cross, walk_chandelier,
     {"atr_mult": 2.5, "time_stop": 60})
_add("D10_dmi_classic",  entry_dmi_bull, walk_classic_trail,
     {"adx_min": 20, "trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 60})

# ---- E: ML score + indicator confluence (10) — different combinations ----
def _entry_ml_dmi(score, ind, p):
    pdi = ind["dmi_pdi"]
    mdi = ind["dmi_mdi"]
    return (score >= p["buy_threshold"]) & (pdi > mdi) & (ind["dmi_adx"] >= p.get("adx_min", 18))

def _entry_ml_macd(score, ind, p):
    diff = ind["macd_diff"]
    dea = ind["macd_dea"]
    diff_prev = np.concatenate(([diff[0]], diff[:-1]))
    dea_prev = np.concatenate(([dea[0]], dea[:-1]))
    return (score >= p["buy_threshold"]) & (diff_prev <= dea_prev) & (diff > dea)

def _entry_ml_macd_pos(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["macd_diff"] > 0)

def _entry_ml_rsi(score, ind, p):
    rsi = ind["rsi14"]
    return (score >= p["buy_threshold"]) & (rsi > p.get("rsi_min", 50)) & (rsi < p.get("rsi_max", 70))

def _entry_ml_above_ma60(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["__close__"] > ind["sma60"])

def _entry_ml_dl_watch(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["dl_stage_watch_recent"] > 0)

def _entry_ml_zhunbei(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["mm_zhunbei_active"] > 0)

def _entry_ml_volspike(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["vol_spike"] >= p.get("vol_min", 1.5))

def _entry_ml_donch(score, ind, p):
    return (score >= p["buy_threshold"]) & (ind["__close__"] > ind["high20"])

def _entry_dl_kdj(score, ind, p):
    j = ind["kdj_j"]
    j_prev = np.concatenate(([j[0]], j[:-1]))
    return (ind["dl_stage_bottom_recent"] > 0) & ((j_prev <= 0) & (j > 0))


_add("E01_ml_dmi", _entry_ml_dmi, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60, "adx_min": 18})
_add("E02_ml_macd", _entry_ml_macd, walk_classic_trail,
     {"buy_threshold": 0.45, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("E03_ml_macd_pos", _entry_ml_macd_pos, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("E04_ml_rsi", _entry_ml_rsi, walk_classic_trail,
     {"buy_threshold": 0.45, "rsi_min": 50, "rsi_max": 70,
      "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("E05_ml_ma60", _entry_ml_above_ma60, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("E06_ml_dlwatch", _entry_ml_dl_watch, walk_classic_trail,
     {"buy_threshold": 0.45, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("E07_ml_zhunbei", _entry_ml_zhunbei, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 60})
_add("E08_ml_volspike", _entry_ml_volspike, walk_classic_trail,
     {"buy_threshold": 0.45, "vol_min": 1.5, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("E09_ml_donch20", _entry_ml_donch, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("E10_dl_kdj", _entry_dl_kdj, walk_classic_trail,
     {"trail_pct": 0.04, "trail_activation": 0.05, "time_stop": 45})


# ---- F: Score + Indicator confluence (TDX × ML hybrid) (10) ----
def _entry_ml_dlbottom_macd(score, ind, p):
    diff = ind["macd_diff"]
    return ((score >= p["buy_threshold"]) &
            (ind["dl_stage_bottom_recent"] > 0) & (diff > 0))

def _entry_ml_dlwatch_dmi(score, ind, p):
    return ((score >= p["buy_threshold"]) &
            (ind["dl_stage_watch_recent"] > 0) &
            (ind["dmi_pdi"] > ind["dmi_mdi"]))

def _entry_ml_jibuy_kdj(score, ind, p):
    j = ind["kdj_j"]
    j_prev = np.concatenate(([j[0]], j[:-1]))
    return ((score >= p["buy_threshold"]) &
            (ind["mm_jibuy_active"] > 0) & ((j_prev <= 20) & (j > 20)))

def _entry_dlbottom_jibuy(score, ind, p):
    return (ind["dl_stage_bottom_recent"] > 0) & (ind["mm_jibuy_active"] > 0)

def _entry_dlwatch_macd(score, ind, p):
    diff = ind["macd_diff"]
    dea = ind["macd_dea"]
    return (ind["dl_stage_watch_recent"] > 0) & (diff > dea)

def _entry_ml_rsi_donch(score, ind, p):
    rsi = ind["rsi14"]
    return ((score >= p["buy_threshold"]) &
            (rsi > 40) & (rsi < 70) &
            (ind["__close__"] > ind["high20"] * 0.995))

def _entry_ml_volspike_dl(score, ind, p):
    return ((score >= p["buy_threshold"]) &
            (ind["vol_spike"] >= p.get("vol_min", 1.5)) &
            (ind["dl_value"] < 2.5))

def _entry_score_top_pct(score, ind, p):
    return score >= p["buy_threshold"]  # uses high threshold ≥0.55

def _entry_dl_velocity(score, ind, p):
    return ind["dl_velocity"] >= p.get("vel_min", 0.5)

def _entry_dmi_macd(score, ind, p):
    diff = ind["macd_diff"]
    dea = ind["macd_dea"]
    return ((ind["dmi_pdi"] > ind["dmi_mdi"]) & (ind["dmi_adx"] > 18) &
            (diff > dea) & (diff > 0))


_add("F01_ml_dl_macd", _entry_ml_dlbottom_macd, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("F02_ml_dlw_dmi", _entry_ml_dlwatch_dmi, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("F03_ml_ji_kdj", _entry_ml_jibuy_kdj, walk_classic_trail,
     {"buy_threshold": 0.40, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("F04_dl_ji",     _entry_dlbottom_jibuy, walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("F05_dlw_macd",  _entry_dlwatch_macd, walk_classic_trail,
     {"trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 45})
_add("F06_ml_rsi_donch", _entry_ml_rsi_donch, walk_atr_trail,
     {"buy_threshold": 0.45, "atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})
_add("F07_ml_vol_dl", _entry_ml_volspike_dl, walk_classic_trail,
     {"buy_threshold": 0.45, "vol_min": 1.5, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 60})
_add("F08_score_top_atr", _entry_score_top_pct, walk_atr_trail,
     {"buy_threshold": 0.60, "atr_mult": 1.8, "trail_activation": 0.05, "time_stop": 75})
_add("F09_dl_velocity", _entry_dl_velocity, walk_classic_trail,
     {"vel_min": 0.5, "trail_pct": 0.03, "trail_activation": 0.05, "time_stop": 45})
_add("F10_dmi_macd",  _entry_dmi_macd, walk_atr_trail,
     {"atr_mult": 2.0, "trail_activation": 0.05, "time_stop": 60})


# Sanity: ensure exactly 50 families
assert len(FAMILIES) == 50, f"expected 50 families, got {len(FAMILIES)}"


# =========================================================================
# Backtest engine
# =========================================================================

def run_one_strategy_one_stock(family: FamilyDef, params: dict,
                                close, high, low, score, ind, n) -> list[dict]:
    """Run a single (family, params, stock) — return list of trades."""
    # Build entry mask
    if "__close__" in str(family.entry_fn) or family.entry_fn.__name__.startswith("_entry"):
        # E-family: needs ind with __close__ injected
        ind = {**ind, "__close__": close, "__high__": high}
    try:
        entry_mask = family.entry_fn(score, ind, params)
    except Exception:
        return []

    if entry_mask.sum() == 0:
        return []

    trades = []
    in_pos = False
    exit_until = -1
    # Walk bars; need at least 25 history for indicators
    start = max(25, 1)
    for i in range(start, n - 1):
        if in_pos:
            if i >= exit_until:
                in_pos = False
            continue
        if not entry_mask[i]:
            continue
        # Buy at next-bar open
        if i + 1 >= n:
            continue
        # Limit-up filter: skip if today's close > yesterday close * 1.099
        if i > 0 and close[i] > close[i - 1] * 1.099:
            continue
        # Use close[i] as proxy for next-day open entry (consistency with rest of repo)
        exit_idx, exit_price, reason = family.exit_fn(
            i, close, high, low, score, ind, params, n)
        entry_price = close[i]
        ret = exit_price / entry_price - 1
        trades.append({
            "entry_idx": i,
            "exit_idx": exit_idx,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "ret": ret,
            "reason": reason,
            "hold_days": exit_idx - i,
        })
        in_pos = True
        exit_until = exit_idx

    return trades
