"""Step 4: Generate 1000 candidates × backtest × filter × output Top 5.

Outputs:
  docs/strategy_mine/all_passing_strategies.json   — all candidates passing the gate
  docs/strategy_mine/top_5.json                    — top 5 with composite score
  docs/strategy_mine/REPORT.md                     — summary
"""
from __future__ import annotations

import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mine_426_strategies import FAMILIES, run_one_strategy_one_stock  # noqa: E402

CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
SCORES_PATH = CACHE_DIR / "ml_scores.parquet"
INDIC_PATH = CACHE_DIR / "indicators.parquet"
OUT_DIR = ROOT.parent / "docs" / "strategy_mine"

# ---- gating thresholds (user spec) ----
WIN_RATE_MIN = 0.75
AVG_PROFIT_MIN = 0.06
MAX_LOSS_LIMIT = -0.1005  # floating-point tolerance (hard stop is -10%)
TOTAL_RETURN_MIN = 0.10
MIN_TRADES = 20
MIN_STOCKS = 5

# ---- OOS split ----
OOS_DATE = pd.Timestamp("2024-09-01")


# ===================================================================
# LHS sampling per family
# ===================================================================
def lhs_param_grid(family, n_variants: int = 20, rng_seed: int = 42) -> list[dict]:
    """Generate n_variants LHS parameter sets around the family default."""
    import hashlib
    h = int(hashlib.md5(family.family_id.encode()).hexdigest(), 16) % 1000
    rng = np.random.RandomState(rng_seed + h)
    base = family.default_params
    variants = []
    for k in range(n_variants):
        p = dict(base)
        # Perturb continuous params
        if "buy_threshold" in p:
            p["buy_threshold"] = float(np.clip(
                base["buy_threshold"] + rng.uniform(-0.10, 0.15), 0.30, 0.70))
        if "trail_pct" in p:
            p["trail_pct"] = float(np.clip(
                base["trail_pct"] + rng.uniform(-0.02, 0.03), 0.015, 0.07))
        if "atr_mult" in p:
            p["atr_mult"] = float(np.clip(
                base["atr_mult"] + rng.uniform(-0.8, 1.5), 1.0, 4.0))
        if "trail_activation" in p:
            p["trail_activation"] = float(np.clip(
                base["trail_activation"] + rng.uniform(-0.03, 0.10), 0.03, 0.20))
        if "time_stop" in p:
            p["time_stop"] = int(np.clip(
                base["time_stop"] + rng.randint(-30, 40), 20, 120))
        if "exit_threshold" in p:
            p["exit_threshold"] = float(np.clip(
                base["exit_threshold"] + rng.uniform(-0.10, 0.10), 0.10, 0.45))
        if "rsi_low" in p:
            p["rsi_low"] = int(np.clip(base["rsi_low"] + rng.randint(-10, 10), 20, 40))
        if "rsi_min" in p:
            p["rsi_min"] = int(np.clip(base["rsi_min"] + rng.randint(-10, 10), 35, 60))
        if "rsi_max" in p:
            p["rsi_max"] = int(np.clip(base["rsi_max"] + rng.randint(-5, 10), 60, 85))
        if "adx_min" in p:
            p["adx_min"] = int(np.clip(base["adx_min"] + rng.randint(-8, 8), 12, 35))
        if "vol_min" in p:
            p["vol_min"] = float(np.clip(base["vol_min"] + rng.uniform(-0.5, 1.0), 1.0, 3.0))
        if "vel_min" in p:
            p["vel_min"] = float(np.clip(base["vel_min"] + rng.uniform(-0.3, 0.4), 0.2, 1.5))
        if "chandelier_look" in p:
            p["chandelier_look"] = int(np.clip(base["chandelier_look"] + rng.randint(-8, 12), 8, 40))
        variants.append(p)
    return variants


# ===================================================================
# Per-stock backtest (apply one strategy across all stocks)
# ===================================================================
def evaluate_candidate(family, params, per_stock_data: dict, dates: list) -> dict:
    """Run one (family, params) across all stocks, aggregate metrics."""
    all_trades = []
    stocks_traded = set()

    for ts, sd in per_stock_data.items():
        n = len(sd["close"])
        if n < 60:
            continue
        try:
            trades = run_one_strategy_one_stock(
                family, params,
                sd["close"], sd["high"], sd["low"],
                sd["score"], sd["ind"], n,
            )
        except Exception:
            continue
        for t in trades:
            t["ts_code"] = ts
            t["entry_date"] = dates[ts][t["entry_idx"]]
            t["exit_date"] = dates[ts][t["exit_idx"]]
            all_trades.append(t)
        if trades:
            stocks_traded.add(ts)

    if not all_trades:
        return {"n_trades": 0, "passed": False}

    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    # Compute metrics — full and OOS
    def _metrics(sub):
        if sub.empty:
            return None
        rets = sub["ret"].values
        wins = (rets > 0).sum()
        n_trades = len(rets)
        win_rate = wins / n_trades
        avg = rets.mean()
        max_loss = rets.min()
        # Cumulative return: equal-weight per trade, simple sum (not compounded — protects against leverage illusion)
        total_ret = rets.sum()
        return {
            "n_trades": int(n_trades),
            "win_rate": float(win_rate),
            "avg_ret": float(avg),
            "max_loss": float(max_loss),
            "total_ret": float(total_ret),
        }

    full = _metrics(df)
    oos = _metrics(df[df["entry_date"] >= OOS_DATE])
    if oos is None:
        oos = {"n_trades": 0, "win_rate": 0, "avg_ret": 0, "max_loss": 0, "total_ret": 0}

    passed = (
        full["win_rate"] >= WIN_RATE_MIN
        and full["avg_ret"] >= AVG_PROFIT_MIN
        and full["max_loss"] >= MAX_LOSS_LIMIT
        and full["total_ret"] >= TOTAL_RETURN_MIN
        and full["n_trades"] >= MIN_TRADES
        and len(stocks_traded) >= MIN_STOCKS
        and oos["n_trades"] >= 5
        and oos["win_rate"] >= WIN_RATE_MIN
        and oos["avg_ret"] >= AVG_PROFIT_MIN
        and oos["total_ret"] >= 0.0
    )

    composite = (full["win_rate"] *
                 math.log(1 + max(full["total_ret"], 0.001)) *
                 math.sqrt(full["n_trades"]) /
                 max(abs(full["max_loss"]) + 0.01, 0.05))

    return {
        "passed": passed,
        "stocks_traded": int(len(stocks_traded)),
        "full": full,
        "oos": oos,
        "composite": float(composite),
    }


# ===================================================================
# Main
# ===================================================================
def load_per_stock_data():
    print("Loading caches...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    scores = pd.read_parquet(SCORES_PATH)
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    indic = pd.read_parquet(INDIC_PATH)
    indic["trade_date"] = pd.to_datetime(indic["trade_date"])

    # Merge: ohlcv ⟕ scores ⟕ indic (left)
    print(f"  OHLCV: {len(ohlcv):,}, scores: {len(scores):,}, indic: {len(indic):,}", flush=True)
    merged = ohlcv.merge(scores, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(indic, on=["ts_code", "trade_date"], how="left")
    merged["ml_score"] = merged["ml_score"].fillna(0.0)
    print(f"  merged: {len(merged):,} rows", flush=True)

    # Group by stock
    indic_cols = [c for c in indic.columns if c not in ("ts_code", "trade_date")]

    per_stock = {}
    dates_per_stock = {}
    for ts, g in merged.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if len(g) < 60:
            continue
        ind = {c: g[c].values.astype(float) for c in indic_cols}
        per_stock[ts] = {
            "close": g["close"].values.astype(float),
            "high": g["high"].values.astype(float),
            "low": g["low"].values.astype(float),
            "score": g["ml_score"].values.astype(float),
            "ind": ind,
        }
        dates_per_stock[ts] = [d.strftime("%Y-%m-%d") for d in g["trade_date"]]

    print(f"  per-stock data ready for {len(per_stock)} stocks", flush=True)
    return per_stock, dates_per_stock


def main():
    t0 = time.time()
    per_stock, dates = load_per_stock_data()

    # Generate 1000 candidates: 50 families × 20 variants each
    print(f"\nGenerating 1000 candidates ({len(FAMILIES)} families × 20 variants)...", flush=True)
    candidates = []
    for fam in FAMILIES:
        for k, p in enumerate(lhs_param_grid(fam, 20)):
            candidates.append({
                "id": f"{fam.family_id}_v{k:02d}",
                "family": fam.family_id,
                "family_obj": fam,
                "params": p,
            })
    print(f"  {len(candidates)} candidates ready", flush=True)

    # Evaluate each
    print(f"\nMining 1000 candidates × {len(per_stock)} stocks...", flush=True)
    t_mine = time.time()
    results = []
    for ci, c in enumerate(candidates):
        r = evaluate_candidate(c["family_obj"], c["params"], per_stock, dates)
        c.pop("family_obj")
        results.append({**c, **r})
        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_mine
            eta = elapsed / (ci + 1) * (len(candidates) - ci - 1)
            n_pass = sum(1 for r in results if r.get("passed"))
            print(f"  {ci+1}/{len(candidates)} done | passed: {n_pass} | elapsed: {elapsed:.0f}s | eta: {eta:.0f}s", flush=True)

    # Filter & rank
    passing = [r for r in results if r.get("passed")]
    passing.sort(key=lambda r: r["composite"], reverse=True)
    print(f"\nDone in {time.time()-t_mine:.0f}s", flush=True)
    print(f"  Total: {len(results)}, Passing all gates: {len(passing)}", flush=True)

    # Top 5 with same-family cap
    top5 = []
    family_count: dict[str, int] = {}
    for r in passing:
        fam = r["family"]
        if family_count.get(fam, 0) >= 2:
            continue
        top5.append(r)
        family_count[fam] = family_count.get(fam, 0) + 1
        if len(top5) >= 5:
            break

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with (OUT_DIR / "all_passing_strategies.json").open("w") as f:
        json.dump(passing, f, indent=2, default=str)
    with (OUT_DIR / "top_5.json").open("w") as f:
        json.dump(top5, f, indent=2, default=str)
    with (OUT_DIR / "all_results.json").open("w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print(f"\n{'='*80}", flush=True)
    print(f"TOP 5 (same-family ≤ 2)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'rank':<5}{'id':<28}{'win%':>7}{'avgRet%':>9}{'totRet%':>9}"
          f"{'maxLoss%':>10}{'trades':>8}{'stocks':>8}{'composite':>11}", flush=True)
    for i, r in enumerate(top5, 1):
        f = r["full"]
        print(f"{i:<5}{r['id']:<28}{f['win_rate']*100:>6.1f}%"
              f"{f['avg_ret']*100:>+8.2f}%{f['total_ret']*100:>+8.2f}%"
              f"{f['max_loss']*100:>+9.2f}%{f['n_trades']:>8d}"
              f"{r['stocks_traded']:>8d}{r['composite']:>11.3f}", flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s", flush=True)
    print(f"Wrote: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
