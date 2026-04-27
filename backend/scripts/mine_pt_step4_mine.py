"""V2 mining driver, but score is loaded from cache/pt_scores.parquet
(PatchTST output) instead of cache/ml_scores.parquet (XGBoost ensemble).

Same 50 families × 20 LHS = 1000 candidates, same gates, same Jaccard top-5.
Output: docs/strategy_mine/pt/

Run after mine_pt_step2_cache_scores.py has produced cache/pt_scores.parquet.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.mine_426_v2_strategies import FAMILIES, run_one_strategy_one_stock  # noqa
from scripts.mine_426_v2_mine import (  # noqa: reuse helpers
    WIN_RATE_MIN, AVG_PROFIT_MIN, MAX_LOSS_LIMIT, TOTAL_RETURN_MIN,
    MIN_TRADES, MIN_STOCKS, JACCARD_MAX, TOP_K, OOS_DATE,
    lhs_param_grid, evaluate_candidate, jaccard,
)


CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
SCORES_PATH = CACHE_DIR / "pt_scores.parquet"   # ← only difference vs v2_mine
INDIC_PATH = CACHE_DIR / "indicators.parquet"
OUT_DIR = ROOT.parent / "docs" / "strategy_mine" / "pt"


def load_per_stock_data():
    print("Loading caches...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    scores = pd.read_parquet(SCORES_PATH)
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    indic = pd.read_parquet(INDIC_PATH)
    indic["trade_date"] = pd.to_datetime(indic["trade_date"])
    print(f"  OHLCV: {len(ohlcv):,}, scores: {len(scores):,}, indic: {len(indic):,}", flush=True)
    merged = ohlcv.merge(scores, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(indic, on=["ts_code", "trade_date"], how="left")
    merged["ml_score"] = merged["ml_score"].fillna(0.0)

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
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = []
    for fam in FAMILIES:
        for k, p in enumerate(lhs_param_grid(fam, 20)):
            candidates.append({
                "id": f"{fam.family_id}_v{k:02d}",
                "family": fam.family_id,
                "concept": fam.concept,
                "_fam_obj": fam,
                "params": p,
            })
    print(f"Total candidates: {len(candidates)}", flush=True)

    print(f"\nMining (PatchTST scores)... ", flush=True)
    t_mine = time.time()
    results = []
    for ci, c in enumerate(candidates):
        r = evaluate_candidate(c["_fam_obj"], c["params"], per_stock, dates)
        c.pop("_fam_obj")
        results.append({**c, **r})
        if (ci + 1) % 50 == 0:
            elapsed = time.time() - t_mine
            n_pass = sum(1 for x in results if x.get("passed"))
            print(f"  {ci+1}/{len(candidates)} | passed: {n_pass} | {elapsed:.0f}s", flush=True)

    passing = [r for r in results if r.get("passed")]
    passing.sort(key=lambda r: r["composite"], reverse=True)
    print(f"\nPassing: {len(passing)}/{len(results)}", flush=True)

    from collections import Counter
    print("Concept distribution of passing:")
    for k, v in Counter(r["concept"] for r in passing).most_common():
        print(f"  {k:<14} {v}", flush=True)

    chosen = []
    chosen_sets = []
    chosen_concepts = set()
    for r in passing:
        bs = r.get("_buy_set", set())
        if r["concept"] in chosen_concepts:
            continue
        if any(jaccard(bs, cs) >= JACCARD_MAX for cs in chosen_sets):
            continue
        chosen.append(r)
        chosen_sets.append(bs)
        chosen_concepts.add(r["concept"])
        if len(chosen) >= TOP_K:
            break

    print(f"\n{'='*100}")
    print(f"PT-MINING DIVERSE TOP {TOP_K} (Jaccard ≤ {JACCARD_MAX}, max 1 per concept):")
    print(f"{'='*100}")
    print(f"{'rank':<5}{'id':<30}{'concept':<14}{'win%':>7}{'avgR%':>8}{'totR%':>10}"
          f"{'maxL%':>8}{'trd':>5}{'stk':>5}{'comp':>9}", flush=True)
    for i, r in enumerate(chosen, 1):
        f = r["full"]
        print(f"{i:<5}{r['id']:<30}{r['concept']:<14}"
              f"{f['win_rate']*100:>6.1f}%{f['avg_ret']*100:>+7.2f}%"
              f"{f['total_ret']*100:>+9.0f}%{f['max_loss']*100:>+7.2f}%"
              f"{f['n_trades']:>5}{r['stocks_traded']:>5}{r['composite']:>9.2f}", flush=True)

    def _strip(rs):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]

    with (OUT_DIR / "all_results.json").open("w") as f:
        json.dump(_strip(results), f, indent=2, default=str)
    with (OUT_DIR / "passing.json").open("w") as f:
        json.dump(_strip(passing), f, indent=2, default=str)
    with (OUT_DIR / "top_5.json").open("w") as f:
        json.dump(_strip(chosen), f, indent=2, default=str)

    print(f"\nWrote: {OUT_DIR}/")
    print(f"Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
