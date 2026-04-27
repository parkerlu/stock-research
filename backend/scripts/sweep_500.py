"""
500-variant sweep on 100-stock pool, post-2024 OOS.
Sampled from a 6×5×2×5×5×4 grid (6000 combos) to give 500 diverse configs.
Picks top 3 for production presets.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from scripts.sweep_v8_to_v20 import (
    precompute_scores, evaluate_one, get_session_factory, HISTORY_START,
)


def gen_grid_sample(n: int = 500, seed: int = 42) -> list[tuple[str, dict]]:
    grid = list(itertools.product(
        [0.35, 0.40, 0.45, 0.50, 0.55, 0.60],   # buy_threshold
        [0.20, 0.25, 0.30, 0.35],                # exit_threshold
        [-0.07, -0.10],                          # stop_loss
        [0.05, 0.07, 0.09, 0.11, 0.13],          # trail_activation
        [0.03, 0.04, 0.05, 0.06, 0.07],          # trail_pct
        [45, 60, 75, 90],                        # time_stop
    ))
    random.seed(seed)
    sample = random.sample(grid, min(n, len(grid)))
    out = []
    for i, (buy, ex, sl, ta, tp, ts) in enumerate(sample, 71):
        out.append((f"V{i}",
            {"buy_threshold": buy, "exit_threshold": ex,
             "stop_loss": sl,
             "trail_activation": ta, "trail_pct": tp,
             "time_stop": ts, "use_adaptive": True}))
    return out


async def get_pool_codes(pool_id: int = 3) -> list[str]:
    Session = get_session_factory()
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == pool_id)
        )).scalars().all()
    return list(rows)


async def load_candles(symbol: str) -> pd.DataFrame:
    Session = get_session_factory()
    async with Session() as db:
        rows = (await db.execute(
            select(DailyCandle).where(DailyCandle.ts_code == symbol)
            .order_by(DailyCandle.trade_date)
        )).scalars().all()
    if not rows:
        return pd.DataFrame()
    latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
    return pd.DataFrame([{
        "trade_date": r.trade_date,
        "open": round(float(r.open) * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "high": round(float(r.high) * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "low":  round(float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "close":round(float(r.close)* (float(r.adj_factor or 1.0) / latest_adj), 4),
        "vol": float(r.vol or 0), "amount": float(r.amount) if r.amount else 0,
    } for r in rows])


async def main() -> None:
    variants = gen_grid_sample(500)
    print(f"Generated {len(variants)} variants V71-V570", flush=True)

    codes = await get_pool_codes(3)
    print(f"Pool: {len(codes)} stocks", flush=True)

    print("Loading candles...", flush=True)
    candles_by_sym = {}
    for i, c in enumerate(codes):
        df = await load_candles(c)
        if len(df) >= 130:
            candles_by_sym[c] = df
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(codes)}", flush=True)
    print(f"Loaded {len(candles_by_sym)}", flush=True)

    print("Pre-computing scores (~5 min)...", flush=True)
    score_cache = precompute_scores(candles_by_sym)
    print(f"Cached {len(score_cache)} stocks\n", flush=True)

    results = []
    for k, (name, params) in enumerate(variants, 1):
        r = evaluate_one(params, score_cache)
        r["name"] = name
        r["params"] = params
        results.append(r)
        if k % 25 == 0 or k == len(variants):
            print(f"  evaluated {k}/{len(variants)}", flush=True)

    ranked = sorted(results, key=lambda x: -x["composite"])
    print(f"\n{'='*78}\n  TOP 20 (out of {len(results)})\n{'='*78}", flush=True)
    for i, r in enumerate(ranked[:20], 1):
        p = r["params"]
        print(f"{i:>2}. {r['name']:<6}  composite={r['composite']:>6.2f}  "
              f"win={r['win_rate']:>5.1f}%  ≥5%={r['acc5_pct']:>5.1f}%  "
              f"≥10%={r['acc10_pct']:>5.1f}%  avg={r['avg_per_stock_ret']:+6.2f}%  "
              f"trades={r['total_trades']}  | "
              f"buy={p['buy_threshold']} ex={p['exit_threshold']} "
              f"sl={p['stop_loss']} act={p['trail_activation']} "
              f"trail={p['trail_pct']} ts={p['time_stop']}", flush=True)

    os.makedirs("models", exist_ok=True)
    with open("models/sweep_500_results.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)
    print(f"\nSaved to models/sweep_500_results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
