"""Step 5: Re-pick Top 5 from passing strategies with Jaccard diversity constraint.

Greedy selection: highest composite first, then iteratively add the next-highest
that overlaps < JACCARD_MAX with all previously picked.

Output: docs/strategy_mine/top_5_diverse.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.mine_426_step4_mine import load_per_stock_data
from scripts.mine_426_strategies import FAMILIES, run_one_strategy_one_stock

JACCARD_MAX = 0.30
TOP_K = 5


def buy_set(fam, params, per_stock) -> set:
    """Return set of (ts_code, entry_idx) buy events."""
    s = set()
    for ts, sd in per_stock.items():
        try:
            trades = run_one_strategy_one_stock(
                fam, params,
                sd["close"], sd["high"], sd["low"],
                sd["score"], sd["ind"], len(sd["close"]),
            )
            for t in trades:
                s.add((ts, t["entry_idx"]))
        except Exception:
            pass
    return s


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def main():
    out_dir = Path(__file__).parent.parent.parent / "docs" / "strategy_mine"
    with (out_dir / "all_passing_strategies.json").open() as f:
        passing = json.load(f)
    passing.sort(key=lambda r: r["composite"], reverse=True)
    print(f"Loaded {len(passing)} passing strategies", flush=True)

    print("Loading per-stock data + computing buy sets...", flush=True)
    per_stock, _ = load_per_stock_data()
    fam_map = {f.family_id: f for f in FAMILIES}

    print(f"Computing buy events for all {len(passing)} passing...", flush=True)
    buy_sets = []
    for r in passing:
        fam = fam_map[r["family"]]
        bs = buy_set(fam, r["params"], per_stock)
        buy_sets.append(bs)

    # Greedy diversity-aware top-K
    chosen_idx = []
    chosen_sets: list[set] = []
    for i, (r, bs) in enumerate(zip(passing, buy_sets)):
        max_j = max((jaccard(bs, cs) for cs in chosen_sets), default=0.0)
        if max_j < JACCARD_MAX:
            chosen_idx.append(i)
            chosen_sets.append(bs)
            if len(chosen_idx) >= TOP_K:
                break

    if len(chosen_idx) < TOP_K:
        print(f"WARNING: only {len(chosen_idx)} strategies satisfy Jaccard < {JACCARD_MAX}.", flush=True)

    top = [passing[i] for i in chosen_idx]

    print()
    print(f"Diverse Top-{TOP_K} (Jaccard ≤ {JACCARD_MAX*100:.0f}%):")
    print(f"{'rank':<5}{'id':<28}{'family':<28}{'win%':>7}{'avgR%':>7}{'totR%':>10}{'trd':>6}{'stk':>5}{'comp':>9}", flush=True)
    for r_idx, idx in enumerate(chosen_idx, 1):
        r = passing[idx]
        f = r["full"]
        print(f"{r_idx:<5}{r['id']:<28}{r['family']:<28}"
              f"{f['win_rate']*100:>6.1f}%{f['avg_ret']*100:>+6.2f}%"
              f"{f['total_ret']*100:>+9.0f}%{f['n_trades']:>6}"
              f"{r['stocks_traded']:>5}{r['composite']:>9.2f}",
              flush=True)

    # Pairwise Jaccard
    print()
    print(f"Pairwise Jaccard (chosen):")
    header = "         " + "  ".join(f"{passing[i]['id'][:10]:>10}" for i in chosen_idx)
    print(header, flush=True)
    for i, idx_i in enumerate(chosen_idx):
        row = [f"{passing[idx_i]['id'][:8]:<8}"]
        for j, idx_j in enumerate(chosen_idx):
            jv = jaccard(buy_sets[idx_i], buy_sets[idx_j]) * 100
            row.append(f"{jv:>9.1f}%")
        print("  ".join(row), flush=True)

    # Save
    with (out_dir / "top_5_diverse.json").open("w") as f:
        json.dump(top, f, indent=2, default=str)
    print(f"\nSaved {len(top)} → {out_dir}/top_5_diverse.json", flush=True)


if __name__ == "__main__":
    main()
