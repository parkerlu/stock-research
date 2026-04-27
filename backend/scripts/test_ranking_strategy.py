"""
D — Run RankingStrategy (top-decile daily rebalance) on the 100-stock pool.

Goal: high-frequency signal generation (every day) + accuracy from
ranking selection.
"""
from __future__ import annotations

import asyncio
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from app.services.strategy_templates.ranking_strategy import (
    composite_score, panel_backtest,
)


CUTOFF = date(2024, 1, 1)


async def get_pool_codes(pool_id: int = 3) -> list[str]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == pool_id)
        )).scalars().all()
    await eng.dispose()
    return list(rows)


async def load_panel(codes: list[str]):
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    per_stock = {}
    for code in codes:
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == code)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) < 130:
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
        return None, None, None
    common = sorted(set.intersection(*[set(df.index) for df in per_stock.values()]))
    codes_used = list(per_stock.keys())
    T, N = len(common), len(codes_used)
    panel = {}
    for col in ["open", "high", "low", "close", "vol", "amount"]:
        arr = np.full((T, N), np.nan)
        for j, c in enumerate(codes_used):
            arr[:, j] = per_stock[c].reindex(common)[col].values
        panel[col] = arr
    return panel, common, codes_used


async def main() -> None:
    print("Loading 100-stock panel...", flush=True)
    codes = await get_pool_codes(3)
    panel, dates, codes_used = await load_panel(codes)
    if panel is None:
        print("no panel"); return
    T, N = panel["close"].shape
    print(f"  panel: {T} dates × {N} stocks", flush=True)

    print("Computing composite score...", flush=True)
    composite = composite_score(panel)
    print(f"  composite shape: {composite.shape}", flush=True)

    # Sweep through different top-pct + drop-pct + time-stop configs
    configs = [
        ("top10_drop30_ts60", dict(top_pct=0.10, drop_pct=0.30, time_stop=60)),
        ("top10_drop20_ts60", dict(top_pct=0.10, drop_pct=0.20, time_stop=60)),
        ("top15_drop30_ts60", dict(top_pct=0.15, drop_pct=0.30, time_stop=60)),
        ("top20_drop30_ts45", dict(top_pct=0.20, drop_pct=0.30, time_stop=45)),
        ("top10_drop40_ts90", dict(top_pct=0.10, drop_pct=0.40, time_stop=90)),
    ]

    print(f"\n{'='*78}\n  Ranking strategy results (post-2024 OOS)\n{'='*78}", flush=True)
    print(f"{'config':<22} {'trades':>7} {'win%':>6} {'≥5%':>6} {'≥10%':>6} "
          f"{'avg_pnl':>8} {'median':>7}", flush=True)
    print("-" * 78, flush=True)
    for name, kwargs in configs:
        result = panel_backtest(panel, dates, codes_used, CUTOFF, composite,
                                stop_loss=-0.07, take_profit_trail=0.05,
                                trail_act=0.05, **kwargs)
        print(f"{name:<22} {result['n_trades']:>7} "
              f"{result['win_rate']:>5.1f}% {result['acc5']:>5.1f}% "
              f"{result['acc10']:>5.1f}% {result['avg_pnl']:>+7.2f}% "
              f"{result['median_pnl']:>+6.2f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
