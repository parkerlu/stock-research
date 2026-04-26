"""V2 mining: 50 conceptually orthogonal families × 20 LHS variants = 1000 candidates.

Output: docs/strategy_mine/v2/
  all_results.json — every candidate's metrics
  passing.json     — strategies passing all hard gates
  top_5.json       — Jaccard-diverse top 5 (max 1 per concept)
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
from scripts.mine_426_step4_mine import load_per_stock_data  # noqa

OUT_DIR = ROOT.parent / "docs" / "strategy_mine" / "v2"

WIN_RATE_MIN = 0.75
AVG_PROFIT_MIN = 0.06
MAX_LOSS_LIMIT = -0.1005
TOTAL_RETURN_MIN = 0.10
MIN_TRADES = 20
MIN_STOCKS = 5
JACCARD_MAX = 0.30
TOP_K = 5

OOS_DATE = pd.Timestamp("2024-09-01")


def lhs_param_grid(family, n_variants: int = 20) -> list[dict]:
    h = int(hashlib.md5(family.family_id.encode()).hexdigest(), 16) % 1000
    rng = np.random.RandomState(42 + h)
    base = family.default_params
    out = []
    for k in range(n_variants):
        p = dict(base)
        # ml_gate: vary in {None, 0.30, 0.40, 0.50, 0.55, 0.60} so some variants drop ML entirely
        if rng.rand() < 0.20:
            p["ml_gate"] = None  # 20% chance of NO ML gate
        else:
            p["ml_gate"] = float(np.clip(
                base.get("ml_gate", 0.40) + rng.uniform(-0.10, 0.20), 0.30, 0.65))
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
        if "vel_min" in p:
            p["vel_min"] = float(np.clip(p["vel_min"] + rng.uniform(-0.3, 0.4), 0.2, 1.5))
        if "vol_min" in p:
            p["vol_min"] = float(np.clip(p["vol_min"] + rng.uniform(-0.5, 1.0), 1.0, 3.5))
        if "adx_thr" in p:
            p["adx_thr"] = float(np.clip(p["adx_thr"] + rng.uniform(-5, 8), 12, 35))
        if "adx_min" in p:
            p["adx_min"] = float(np.clip(p["adx_min"] + rng.uniform(-5, 8), 12, 35))
        if "ml_delta" in p:
            p["ml_delta"] = float(np.clip(p["ml_delta"] + rng.uniform(-0.10, 0.10), 0.05, 0.40))
        out.append(p)
    return out


def evaluate_candidate(family, params, per_stock, dates):
    all_trades = []
    stocks_traded = set()
    for ts, sd in per_stock.items():
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
            all_trades.append(t)
        if trades:
            stocks_traded.add(ts)

    if not all_trades:
        return {"passed": False, "n_trades": 0, "stocks_traded": 0,
                "full": None, "oos": None, "composite": 0.0}

    df = pd.DataFrame(all_trades)
    df["entry_date"] = pd.to_datetime(df["entry_date"])

    def _m(sub):
        if sub.empty:
            return None
        r = sub["ret"].values
        return {
            "n_trades": int(len(r)),
            "win_rate": float((r > 0).mean()),
            "avg_ret": float(r.mean()),
            "max_loss": float(r.min()),
            "total_ret": float(r.sum()),
        }

    full = _m(df); oos = _m(df[df["entry_date"] >= OOS_DATE])
    if oos is None:
        oos = {"n_trades": 0, "win_rate": 0, "avg_ret": 0, "max_loss": 0, "total_ret": 0}
    if full is None:
        return {"passed": False, "n_trades": 0, "stocks_traded": 0,
                "full": None, "oos": None, "composite": 0.0}

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
    # Capture buy events for Jaccard later
    buy_set = {(t["ts_code"], int(t["entry_idx"])) for t in all_trades}
    return {
        "passed": passed,
        "stocks_traded": int(len(stocks_traded)),
        "full": full, "oos": oos,
        "composite": float(composite),
        "_buy_set": buy_set,  # internal, not serialized
    }


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


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

    print(f"\nMining... ", flush=True)
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

    # Filter & rank
    passing = [r for r in results if r.get("passed")]
    passing.sort(key=lambda r: r["composite"], reverse=True)
    print(f"\nPassing: {len(passing)}/{len(results)}", flush=True)

    # Concept distribution of passing
    from collections import Counter
    print("Concept distribution of passing:")
    for k, v in Counter(r["concept"] for r in passing).most_common():
        print(f"  {k:<14} {v}", flush=True)

    # Diverse top 5: greedy with Jaccard < 0.30 AND max 1 per concept
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

    # Print
    print(f"\n{'='*100}", flush=True)
    print(f"DIVERSE TOP {TOP_K} (Jaccard ≤ {JACCARD_MAX}, max 1 per concept):", flush=True)
    print(f"{'='*100}", flush=True)
    print(f"{'rank':<5}{'id':<30}{'concept':<14}{'win%':>7}{'avgR%':>8}{'totR%':>10}"
          f"{'maxL%':>8}{'trd':>5}{'stk':>5}{'comp':>9}", flush=True)
    for i, r in enumerate(chosen, 1):
        f = r["full"]
        print(f"{i:<5}{r['id']:<30}{r['concept']:<14}"
              f"{f['win_rate']*100:>6.1f}%{f['avg_ret']*100:>+7.2f}%"
              f"{f['total_ret']*100:>+9.0f}%{f['max_loss']*100:>+7.2f}%"
              f"{f['n_trades']:>5}{r['stocks_traded']:>5}{r['composite']:>9.2f}", flush=True)

    # Strip _buy_set before save (not JSON serializable as int keys ok, just remove it)
    def _strip(rs):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rs]

    with (OUT_DIR / "all_results.json").open("w") as f:
        json.dump(_strip(results), f, indent=2, default=str)
    with (OUT_DIR / "passing.json").open("w") as f:
        json.dump(_strip(passing), f, indent=2, default=str)
    with (OUT_DIR / "top_5.json").open("w") as f:
        json.dump(_strip(chosen), f, indent=2, default=str)

    # Pairwise Jaccard for chosen
    if len(chosen) > 1:
        print(f"\nPairwise Jaccard of chosen top {len(chosen)}:")
        ids = [r["id"][:18] for r in chosen]
        print(" "*8, "  ".join(f"{x:>10}" for x in ids), flush=True)
        for i, idi in enumerate(chosen):
            row = [f"{ids[i]:<8}"]
            for j, jdj in enumerate(chosen):
                row.append(f"{jaccard(chosen_sets[i], chosen_sets[j])*100:>9.1f}%")
            print("  ".join(row), flush=True)

    print(f"\nTotal time: {time.time()-t0:.0f}s\nWrote: {OUT_DIR}/", flush=True)


if __name__ == "__main__":
    main()
