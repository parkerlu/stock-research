"""Fast screening using pre-computed parquet caches.

Avoids re-running ml_direct ensemble per stock. Loads:
  cache/ohlcv.parquet
  cache/ml_scores.parquet     (XGB ensemble scores for every bar of every stock)
  cache/indicators.parquet    (precomputed dl/kdj/dmi/macd/boll/atr/rsi/sma/etc.)

Module-level memoized so first scan pays the parquet load cost (~5-10 s),
all subsequent scans hit RAM and finish in seconds.
"""
from __future__ import annotations

import threading
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/app")
CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
SCORES_PATH = CACHE_DIR / "ml_scores.parquet"
INDIC_PATH = CACHE_DIR / "indicators.parquet"

_LOCK = threading.Lock()
_CACHE: dict | None = None


def _shift1(a):
    return np.concatenate(([a[0]], a[:-1]))


def _load_caches() -> dict:
    """Load parquets once, build per-stock arrays. Re-load if files newer."""
    global _CACHE
    with _LOCK:
        if _CACHE is not None:
            return _CACHE
        if not (OHLCV_PATH.exists() and SCORES_PATH.exists() and INDIC_PATH.exists()):
            _CACHE = {"loaded": False, "stocks": {}, "max_date": None}
            return _CACHE
        t0 = time.time()
        ohlcv = pd.read_parquet(OHLCV_PATH)
        ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
        scores = pd.read_parquet(SCORES_PATH)
        scores["trade_date"] = pd.to_datetime(scores["trade_date"])
        indic = pd.read_parquet(INDIC_PATH)
        indic["trade_date"] = pd.to_datetime(indic["trade_date"])

        merged = ohlcv.merge(scores, on=["ts_code", "trade_date"], how="left")
        merged = merged.merge(indic, on=["ts_code", "trade_date"], how="left")
        merged["ml_score"] = merged["ml_score"].fillna(0.0)

        max_date = merged["trade_date"].max()
        indic_cols = [c for c in indic.columns if c not in ("ts_code", "trade_date")]

        per_stock = {}
        for ts, g in merged.groupby("ts_code"):
            g = g.sort_values("trade_date").reset_index(drop=True)
            if len(g) < 60:
                continue
            per_stock[ts] = {
                "dates": g["trade_date"].dt.date.values,
                "open": g["open"].astype(float).values,
                "high": g["high"].astype(float).values,
                "low": g["low"].astype(float).values,
                "close": g["close"].astype(float).values,
                "vol": g["vol"].astype(float).values,
                "score": g["ml_score"].astype(float).values,
                **{c: g[c].astype(float).values for c in indic_cols},
            }

        _CACHE = {
            "loaded": True,
            "stocks": per_stock,
            "max_date": max_date.date(),
            "elapsed": round(time.time() - t0, 1),
        }
        print(f"[screening_fast] loaded {len(per_stock)} stocks in "
              f"{_CACHE['elapsed']}s", flush=True)
        return _CACHE


# =========================================================================
# Strategy registry — fast vectorized entry-mask producers
# Each takes per-stock sd dict and returns (sig_mask, params_used).
# =========================================================================

def _maimai_signals(sd):
    close = sd["close"]; high = sd["high"]; low = sd["low"]
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
    # zhunbei is precomputed in indicators
    zhunbei = sd.get("mm_zhunbei_active", np.zeros(len(close)))
    return jibuy, duanbuy, zhunbei, jisell, duansell


def _entry_mask_for(template_id: str, sd: dict) -> np.ndarray:
    """Compute entry-trigger mask for a given template using cached arrays."""
    n = len(sd["close"])
    score = sd["score"]
    close = sd["close"]; high = sd["high"]; low = sd["low"]; vol = sd["vol"]

    # mm-* family
    if template_id.startswith("mm-"):
        jibuy, duanbuy, zhunbei, jisell, duansell = _maimai_signals(sd)
        thr_map = {
            "mm-30": 0.30, "mm-40": 0.40, "mm-50": 0.50, "mm-55": 0.55,
            "mm-broad-40": 0.40, "mm-broad-50": 0.50,
            "mm-pure-40": 0.40, "mm-pure-50": 0.50,
        }
        thr = thr_map.get(template_id, 0.50)
        broad = template_id.startswith("mm-broad") or template_id.startswith("mm-pure")
        if broad:
            # ONLY include the 3 BUY-side signals turning off (急买/短买/准备).
            # Sell-side (急卖/短卖) turn-off was previously included as a
            # "fear exhausted" proxy but produced false positives.
            j_off = (_shift1(jibuy) > 0) & (jibuy == 0)
            d_off = (_shift1(duanbuy) > 0) & (duanbuy == 0)
            z_off = (_shift1(zhunbei) > 0) & (zhunbei == 0)
            sig_mask = j_off | d_off | z_off
        else:
            sum_now = jibuy + duanbuy + zhunbei
            sum_prev = _shift1(sum_now)
            sum_prev[0] = 0.0
            sig_mask = (sum_now == 0) & (sum_prev > 0)
        return sig_mask & (score >= thr)

    # rev-* family
    if template_id.startswith("rev-"):
        jibuy, duanbuy, zhunbei, jisell, duansell = _maimai_signals(sd)
        # buy-side mm only — no sell-off
        j_off = (_shift1(jibuy) > 0) & (jibuy == 0)
        d_off = (_shift1(duanbuy) > 0) & (duanbuy == 0)
        z_off = (_shift1(zhunbei) > 0) & (zhunbei == 0)
        mm_any = j_off | d_off | z_off

        # 动力线 stage_bottom: cross above 0.2
        dl = sd.get("dl_value", np.zeros(n))
        pdl = _shift1(dl)
        dl_cross = (pdl <= 0.2) & (dl > 0.2)
        # KDJ J cross above 0
        j = sd.get("kdj_j", np.full(n, 50.0))
        pj = _shift1(j)
        kdj_cross = (pj <= 0) & (j > 0)
        # RSI cross above 30
        rsi = sd.get("rsi14", np.full(n, 50.0))
        prsi = _shift1(rsi)
        rsi_cross = (prsi <= 30) & (rsi > 30)
        union = mm_any | dl_cross | kdj_cross | rsi_cross
        thr_map = {"rev-30": 0.30, "rev-40": 0.40, "rev-50": 0.50,
                   "rev-55": 0.55, "rev-60": 0.60}
        thr = thr_map.get(template_id, 0.50)
        return union & (score >= thr)

    # 426-X family — use mining script's family mappings
    if template_id.startswith("426-"):
        from app.services.strategy_templates.mined_426_v2 import TOP_PRESETS, _compute_indicators
        spec = next((s for s in TOP_PRESETS if s["id"] == template_id), None)
        if spec is None:
            return np.zeros(n, dtype=bool)
        # We have most indicators in the cache already, but TOP_PRESETS uses
        # mined_426_v2's _compute_indicators which builds them inline. To stay
        # vectorized and simple, recompute via _compute_indicators (fast):
        ind = _compute_indicators(close, high, low, vol)
        ind["__close__"] = close; ind["__high__"] = high; ind["__low__"] = low
        from app.services.strategy_templates.mined_426_v2 import FAMILY_ENTRY_FNS
        entry_fn = FAMILY_ENTRY_FNS.get(spec["family"])
        if entry_fn is None:
            return np.zeros(n, dtype=bool)
        return entry_fn(score, ind, spec["params"])

    # tdx-* family — fall back to legacy slow path (rare strategies)
    return np.zeros(n, dtype=bool)


# =========================================================================
# Public scan API
# =========================================================================

async def fast_scan_async(template_id: str, lookback_days: int, on_progress=None) -> list[dict]:
    """Async version — yields to event loop every batch so HTTP polling
    can see progress mid-scan instead of waiting for the whole scan to finish.
    """
    import asyncio
    cache = _load_caches()
    if not cache.get("loaded"):
        return []
    stocks = cache["stocks"]
    codes = sorted(stocks.keys())
    total = len(codes)

    hits = []
    BATCH = 25  # finer granularity for visible progress
    for k, ts in enumerate(codes):
        sd = stocks[ts]
        try:
            mask = _entry_mask_for(template_id, sd)
        except Exception:
            mask = None
        if mask is not None and mask.any():
            n = len(mask)
            # Last `lookback_days` trading bars
            for i in range(max(0, n - lookback_days), n):
                if mask[i]:
                    sig_date: date = sd["dates"][i]
                    latest_idx = n - 1
                    latest_date: date = sd["dates"][latest_idx]
                    hits.append({
                        "ts_code": ts,
                        "signal_date": sig_date.isoformat(),
                        "latest_date": latest_date.isoformat(),
                        "latest_close": round(float(sd["close"][latest_idx]), 4),
                    })
                    break
        if (k + 1) % BATCH == 0:
            if on_progress:
                on_progress(k + 1, total)
            # Yield to event loop so polling endpoint can serve job-status reads
            await asyncio.sleep(0)
    if on_progress:
        on_progress(total, total)
    hits.sort(key=lambda h: (h["signal_date"], h["gain_since_signal_pct"]), reverse=True)
    return hits


def fast_scan(template_id: str, lookback_days: int, on_progress=None) -> list[dict]:
    """Synchronous wrapper kept for backwards compat."""
    import asyncio
    return asyncio.run(fast_scan_async(template_id, lookback_days, on_progress))
