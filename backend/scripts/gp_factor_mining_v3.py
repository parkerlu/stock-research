"""
GP V3 — diversity-aware deep mining for distinct factor families.

Improvements over V2:
  1. Stronger niching: tournament rejects parents with similar formula prefixes
  2. Hall-of-fame archive: tracks the best UNIQUE families discovered
  3. Periodic injection: every 15 generations, inject 30% fresh random trees
     to escape local optima
  4. Larger population (350) and generations (60)
  5. Final selection picks 1 representative per "family" (signature-based)
"""
from __future__ import annotations

import asyncio
import hashlib
import random
import time

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from scripts.gp_factor_mining_v2 import (
    Node, INPUTS, BINARY_OPS, UNARY_OPS, TS_OPS, TS_BINARY, WINDOWS,
    random_tree, clone, crossover, mutate, evaluate_array,
    cross_sectional_ic, fitness as base_fitness,
)


MAX_DEPTH = 5
POP_SIZE = 350
N_GENERATIONS = 60
TOURNAMENT_SIZE = 7
ELITISM = 8
MUTATION_RATE = 0.35
CROSSOVER_RATE = 0.55
INJECTION_RATE = 0.30        # share of pop replaced every INJECTION_INTERVAL
INJECTION_INTERVAL = 15
LABEL_HORIZON = 5
random.seed(2024)
np.random.seed(2024)


def signature(node: Node) -> str:
    """A coarse signature: the multiset of (op, depth) pairs.
    Two trees with same signature are considered the same 'family'."""
    parts = []
    def walk(n, d):
        parts.append(f"{n.op}@{d}")
        if n.children:
            for c in n.children:
                walk(c, d + 1)
    walk(node, 0)
    parts.sort()
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def select_with_niching(pop_with_fit: list, exclude_sigs: set) -> Node:
    sample = random.sample(pop_with_fit, min(TOURNAMENT_SIZE, len(pop_with_fit)))
    sample.sort(key=lambda x: -x[1])
    for cand, _ in sample:
        if signature(cand) not in exclude_sigs:
            return clone(cand)
    return clone(sample[0][0])


async def get_codes() -> list[str]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == 3)
        )).scalars().all()
    await eng.dispose()
    return list(rows)


async def load_panel(codes: list[str]) -> tuple[dict, np.ndarray, list[str]]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    per_stock = {}
    for code in codes:
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == code)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) < 250:
            continue
        latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
        df = pd.DataFrame([{
            "trade_date": r.trade_date,
            "open": float(r.open) * (float(r.adj_factor or 1.0) / latest_adj),
            "high": float(r.high) * (float(r.adj_factor or 1.0) / latest_adj),
            "low":  float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj),
            "close":float(r.close)* (float(r.adj_factor or 1.0) / latest_adj),
            "vol": float(r.vol or 0),
            "amount": float(r.amount or 0),
        } for r in rows]).set_index("trade_date")
        per_stock[code] = df
    await eng.dispose()
    if not per_stock:
        return {}, np.array([]), []
    common = sorted(set.intersection(*[set(df.index) for df in per_stock.values()]))
    codes_used = list(per_stock.keys())
    T, N = len(common), len(codes_used)
    panel = {}
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        arr = np.zeros((T, N))
        for j, c in enumerate(codes_used):
            arr[:, j] = per_stock[c].reindex(common)[col].values
        panel[col] = arr
    panel["vwap"] = panel["amount"] / np.where(panel["vol"] > 0, panel["vol"], 1)
    panel["ret"] = np.zeros((T, N))
    panel["ret"][1:] = panel["close"][1:] / panel["close"][:-1] - 1
    future = np.zeros((T, N))
    future[:-LABEL_HORIZON] = (
        panel["close"][LABEL_HORIZON:] / panel["close"][:-LABEL_HORIZON] - 1
    )
    return panel, future, codes_used


async def main() -> None:
    codes = sorted(await get_codes())[:60]
    print(f"Building panel for {len(codes)} stocks...", flush=True)
    panel, future_ret, _ = await load_panel(codes)
    if not panel:
        print("no panel"); return
    print(f"  panel: T={panel['close'].shape[0]} N={panel['close'].shape[1]}\n", flush=True)

    # Hall of fame: keep best UNIQUE family found across all generations
    hof: dict[str, tuple[Node, float, float, float]] = {}  # sig → (tree, score, ic, ir)

    def update_hof(tree: Node, score: float):
        sig = signature(tree)
        try:
            fac = evaluate_array(tree, panel)
            ic, ir, n_days = cross_sectional_ic(fac, future_ret)
            if n_days < 100 or abs(ic) < 0.01:
                return
        except Exception:
            return
        existing = hof.get(sig)
        if existing is None or score > existing[1]:
            hof[sig] = (clone(tree), score, ic, ir)

    print(f"Initializing population of {POP_SIZE}...", flush=True)
    pop = []
    for k in range(POP_SIZE):
        t = random_tree()
        attempts = 0
        while t.depth() < 3 and attempts < 6:
            t = random_tree(force_complex=True)
            attempts += 1
        pop.append(t)
    fits = [base_fitness(t, panel, future_ret) for t in pop]
    pop_with_fit = list(zip(pop, fits))
    pop_with_fit.sort(key=lambda x: -x[1])
    for t, f in pop_with_fit:
        if f > 0:
            update_hof(t, f)
    t0 = time.time()
    best = pop_with_fit[0]
    print(f"  Gen  0: best={best[1]:+.3f}  hof={len(hof)}  | {best[0].to_str()[:80]}",
          flush=True)

    for gen in range(1, N_GENERATIONS + 1):
        # Periodic injection of fresh random trees
        keep_n = int(POP_SIZE * (1 - INJECTION_RATE)) if gen % INJECTION_INTERVAL == 0 else POP_SIZE
        new_pop = [clone(p) for p, _ in pop_with_fit[:ELITISM]]
        # Build niching exclusion based on current top
        top_sigs = set(signature(p) for p, _ in pop_with_fit[:5])

        attempts = 0
        while len(new_pop) < keep_n and attempts < POP_SIZE * 4:
            attempts += 1
            if random.random() < CROSSOVER_RATE:
                p1 = select_with_niching(pop_with_fit, top_sigs if random.random() < 0.4 else set())
                p2 = select_with_niching(pop_with_fit, set())
                child = crossover(p1, p2)
            else:
                child = clone(select_with_niching(pop_with_fit, set()))
            if random.random() < MUTATION_RATE:
                child = mutate(child)
            if child.depth() > MAX_DEPTH + 1:
                continue
            new_pop.append(child)
        # Inject fresh random trees
        while len(new_pop) < POP_SIZE:
            new_pop.append(random_tree(force_complex=True))

        fits = [base_fitness(t, panel, future_ret) for t in new_pop]
        pop_with_fit = list(zip(new_pop, fits))
        pop_with_fit.sort(key=lambda x: -x[1])

        # Update hall of fame with everything from this gen
        for t, f in pop_with_fit[:50]:
            if f > 0:
                update_hof(t, f)

        if gen % 5 == 0 or gen == 1:
            best = pop_with_fit[0]
            elapsed = time.time() - t0
            inj = "INJ" if gen % INJECTION_INTERVAL == 0 else "  -"
            print(f"  Gen {gen:>2} {inj}: best={best[1]:+.3f}  hof={len(hof)}  ({elapsed:.0f}s)  | "
                  f"{best[0].to_str()[:80]}", flush=True)

    # Output: top hof entries, deduplicated by signature
    print(f"\n{'='*78}", flush=True)
    print(f"  HOF: {len(hof)} unique families discovered. Top 25:", flush=True)
    print(f"{'='*78}", flush=True)
    sorted_hof = sorted(hof.values(), key=lambda x: -x[1])[:25]
    print(f"{'#':>2}  {'IC':>7}  {'IR':>6}  {'depth':>5}  formula", flush=True)
    print("-" * 78, flush=True)
    for i, (tree, score, ic, ir) in enumerate(sorted_hof, 1):
        formula = tree.to_str()
        if len(formula) > 95:
            formula = formula[:92] + "..."
        print(f"{i:>2}  {ic:+.4f}  {ir:+.3f}  {tree.depth():>5}  {formula}",
              flush=True)

    # Save top 10 unique formulas to a file for later use
    import json
    out = []
    used_sigs = set()
    for tree, score, ic, ir in sorted_hof:
        sig_short = tree.to_str()[:50]
        if sig_short in used_sigs:
            continue
        used_sigs.add(sig_short)
        out.append({"formula": tree.to_str(), "ic": ic, "ir": ir,
                    "depth": tree.depth(), "score": score})
        if len(out) >= 12:
            break
    import os
    os.makedirs("models", exist_ok=True)
    with open("models/gp_v3_top_factors.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved top {len(out)} unique factors to models/gp_v3_top_factors.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
