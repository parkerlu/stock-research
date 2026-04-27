"""
Sweep 50 strategy variants V21-V70. Target zone:
  - More signals (lower buy threshold 0.40-0.55)
  - Greedier exits (wider trail, lower exit threshold)
  - Wider stop loss (-10% allowed)
  - Longer time stops (45-90 bars)

Reuses score-cache approach from sweep_v8_to_v20.
Picks top 3 (or 5) as new production presets.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from app.services.backtest_engine import run_backtest
from scripts.sweep_v8_to_v20 import (
    precompute_scores, evaluate_one, get_session_factory,
    HISTORY_START, CUTOFF,
)


def gen_variants() -> list[tuple[str, dict]]:
    """50 variants in the 'more signals + greedy + 10% dd ok' region."""
    vs = []
    n = 21
    # Group A: lower buy threshold (more signals), default exits (12)
    for buy in [0.40, 0.45, 0.50, 0.55]:
        for trail_act, trail_pct in [(0.05, 0.04), (0.08, 0.05), (0.10, 0.06)]:
            vs.append((f"V{n}",
                {"buy_threshold": buy, "exit_threshold": 0.30,
                 "stop_loss": -0.10,
                 "trail_activation": trail_act, "trail_pct": trail_pct,
                 "time_stop": 60, "use_adaptive": True}))
            n += 1

    # Group B: aggressive trail (wider), -10% stop, long time stop (10)
    for trail_act in [0.04, 0.06, 0.08, 0.10, 0.12]:
        for trail_pct in [0.05, 0.07]:
            vs.append((f"V{n}",
                {"buy_threshold": 0.50, "exit_threshold": 0.30,
                 "stop_loss": -0.10,
                 "trail_activation": trail_act, "trail_pct": trail_pct,
                 "time_stop": 75, "use_adaptive": True}))
            n += 1

    # Group C: lowest buy thr 0.40, varied exits, never sell on low score (8)
    for exit_thr in [0.15, 0.20]:
        for ts in [45, 60, 75, 90]:
            vs.append((f"V{n}",
                {"buy_threshold": 0.40, "exit_threshold": exit_thr,
                 "stop_loss": -0.10,
                 "trail_activation": 0.08, "trail_pct": 0.05,
                 "time_stop": ts, "use_adaptive": True}))
            n += 1

    # Group D: very wide trail, no early exit (8)
    for trail_act in [0.10, 0.12, 0.15]:
        for trail_pct in [0.06, 0.08]:
            vs.append((f"V{n}",
                {"buy_threshold": 0.50, "exit_threshold": 0.20,
                 "stop_loss": -0.10,
                 "trail_activation": trail_act, "trail_pct": trail_pct,
                 "time_stop": 90, "use_adaptive": True}))
            n += 1
        if n > 70: break

    # Group E: catch many signals + classic exits (~12 so we hit 50)
    for buy in [0.35, 0.40, 0.45]:
        for sl in [-0.08, -0.10]:
            for trail_act in [0.06, 0.10]:
                if n > 70: break
                vs.append((f"V{n}",
                    {"buy_threshold": buy, "exit_threshold": 0.25,
                     "stop_loss": sl,
                     "trail_activation": trail_act, "trail_pct": 0.05,
                     "time_stop": 60, "use_adaptive": True}))
                n += 1

    return vs[:50]


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
    variants = gen_variants()
    print(f"Generated {len(variants)} variants V21-V{20 + len(variants)}", flush=True)

    codes = await get_pool_codes(3)
    print(f"Pool: {len(codes)} stocks", flush=True)

    print("Loading candles (adj-prices)...", flush=True)
    candles_by_sym = {}
    for i, c in enumerate(codes):
        df = await load_candles(c)
        if len(df) >= 130:
            candles_by_sym[c] = df
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/{len(codes)}", flush=True)
    print(f"Loaded {len(candles_by_sym)}", flush=True)

    print("Pre-computing scores (one-time)...", flush=True)
    score_cache = precompute_scores(candles_by_sym)
    print(f"Cached scores for {len(score_cache)} stocks\n", flush=True)

    results = []
    for name, params in variants:
        r = evaluate_one(params, score_cache)
        r["name"] = name
        r["params"] = params
        results.append(r)
        print(f"{name:<6}  trades={r['total_trades']:>4}  "
              f"win%={r['win_rate']:>5.1f}  ≥5%={r['acc5_pct']:>5.1f}  "
              f"≥10%={r['acc10_pct']:>5.1f}  avg/stock={r['avg_per_stock_ret']:+6.2f}%  "
              f"profit_stocks={r['stocks_profit_rate']:>5.1f}%  "
              f"composite={r['composite']:>6.2f}", flush=True)

    ranked = sorted(results, key=lambda x: -x["composite"])
    print(f"\n{'='*78}\n  TOP 10\n{'='*78}", flush=True)
    for i, r in enumerate(ranked[:10], 1):
        p = r["params"]
        print(f"{i:>2}. {r['name']:<6}  composite={r['composite']:>6.2f}  "
              f"win={r['win_rate']:>5.1f}%  ≥5%={r['acc5_pct']:>5.1f}%  "
              f"avg={r['avg_per_stock_ret']:+6.2f}%  trades={r['total_trades']}  | "
              f"buy={p['buy_threshold']} ex={p['exit_threshold']} "
              f"sl={p['stop_loss']} act={p['trail_activation']} "
              f"trail={p['trail_pct']} ts={p['time_stop']}", flush=True)

    os.makedirs("models", exist_ok=True)
    with open("models/sweep_v21_v70_results.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)


if __name__ == "__main__":
    asyncio.run(main())
