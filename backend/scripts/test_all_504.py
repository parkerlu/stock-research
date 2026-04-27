"""Run Top1 strategy across ALL 504 stocks in DB, post-2024 OOS."""
from __future__ import annotations

import asyncio
import pandas as pd
from datetime import date
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker

from app.config import settings
from app.models.schema import DailyCandle
from scripts.sweep_v8_to_v20 import (
    precompute_scores, evaluate_one, get_session_factory,
)


async def main() -> None:
    Session = get_session_factory()
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    print(f"Loading all {len(codes)} stocks...", flush=True)

    candles = {}
    for k, c in enumerate(codes):
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == c)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) >= 130:
            la = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
            df = pd.DataFrame([{
                "trade_date": r.trade_date,
                "open": round(float(r.open) * (float(r.adj_factor or 1.0) / la), 4),
                "high": round(float(r.high) * (float(r.adj_factor or 1.0) / la), 4),
                "low":  round(float(r.low)  * (float(r.adj_factor or 1.0) / la), 4),
                "close":round(float(r.close)* (float(r.adj_factor or 1.0) / la), 4),
                "vol": float(r.vol or 0),
                "amount": float(r.amount) if r.amount else 0,
            } for r in rows])
            candles[c] = df
        if (k + 1) % 100 == 0:
            print(f"  loaded {k + 1}/{len(codes)}", flush=True)
    print(f"Loaded {len(candles)} stocks\nPre-computing scores...", flush=True)
    cache = precompute_scores(candles)
    print(f"Cached {len(cache)}\n", flush=True)

    configs = [
        ("Top1 (adaptive)", {"buy_threshold": 0.50, "exit_threshold": 0.30,
                             "stop_loss": -0.10, "trail_activation": 0.11,
                             "trail_pct": 0.03, "time_stop": 75, "use_adaptive": True}),
        ("Tight (buy=0.55)", {"buy_threshold": 0.55, "exit_threshold": 0.30,
                              "stop_loss": -0.10, "trail_activation": 0.11,
                              "trail_pct": 0.03, "time_stop": 75, "use_adaptive": False}),
        ("HighFreq (buy=0.40)", {"buy_threshold": 0.40, "exit_threshold": 0.25,
                                  "stop_loss": -0.10, "trail_activation": 0.11,
                                  "trail_pct": 0.03, "time_stop": 75, "use_adaptive": False}),
        ("VeryHigh (buy=0.30)", {"buy_threshold": 0.30, "exit_threshold": 0.20,
                                  "stop_loss": -0.10, "trail_activation": 0.11,
                                  "trail_pct": 0.03, "time_stop": 75, "use_adaptive": False}),
    ]

    print(f"{'config':<22}  {'trades':>6}  {'win%':>6}  {'≥5%':>6}  "
          f"{'≥10%':>6}  {'avg/stk':>8}  {'profit_stks':>11}", flush=True)
    print("-" * 80, flush=True)
    for name, p in configs:
        r = evaluate_one(p, cache)
        print(f"{name:<22}  {r['total_trades']:>6}  {r['win_rate']:>5.1f}%  "
              f"{r['acc5_pct']:>5.1f}%  {r['acc10_pct']:>5.1f}%  "
              f"{r['avg_per_stock_ret']:>+7.2f}%  "
              f"{r['stocks_profit_rate']:>10.1f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
