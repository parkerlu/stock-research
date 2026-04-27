"""Expand candle coverage to 2000 stocks across SH+SZ.

Strategy: pick from 主板 + 创业板 listed before 2020-01-01, ordered by
list_date asc (oldest first = richest history). Skip stocks already synced
with ≥ 500 bars.

Usage:  python -m scripts.backfill_to_2000 --target 2000
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.models.schema import DailyCandle, StockBasic


async def pick_universe(target: int, list_before: date) -> list[str]:
    """Pick `target` stocks: main board + ChiNext, listed before list_before,
    ordered by list_date asc."""
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        stmt = (select(StockBasic.ts_code)
                .where(StockBasic.market.in_(["主板", "创业板"]),
                       StockBasic.list_date.is_not(None),
                       StockBasic.list_date < list_before)
                .order_by(StockBasic.list_date.asc(), StockBasic.ts_code)
                .limit(target))
        codes = (await db.execute(stmt)).scalars().all()
    await engine.dispose()
    return list(codes)


async def already_synced(codes: list[str], min_bars: int) -> set[str]:
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        stmt = (select(DailyCandle.ts_code, func.count())
                .where(DailyCandle.ts_code.in_(codes))
                .group_by(DailyCandle.ts_code))
        rows = (await db.execute(stmt)).all()
    await engine.dispose()
    return {ts_code for ts_code, n in rows if n >= min_bars}


async def fetch_and_persist(provider: TuShareProvider, ts_code: str,
                            start: date, end: date) -> int:
    df = await provider.fetch_daily(ts_code, start, end)
    if df.empty:
        return 0
    df = df.copy()
    df["source"] = "tushare"
    rows = df.to_dict("records")

    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        stmt = pg_insert(DailyCandle).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
        await db.execute(stmt)
        await db.commit()
    await engine.dispose()
    return len(rows)


async def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--target", type=int, default=2000)
    p.add_argument("--years", type=int, default=7)
    p.add_argument("--rate-sec", type=float, default=0.6)
    p.add_argument("--list-before", type=str, default="2020-01-01")
    p.add_argument("--min-bars", type=int, default=500)
    args = p.parse_args()

    list_before = date.fromisoformat(args.list_before)
    end = date.today()
    start = date(end.year - args.years, 1, 1)

    print(f"Picking up to {args.target} stocks listed before {list_before}...")
    codes = await pick_universe(args.target, list_before)
    print(f"Picked {len(codes)} (主板 + 创业板)")

    print(f"Checking which are already synced (≥{args.min_bars} bars)...")
    synced = await already_synced(codes, args.min_bars)
    todo = [c for c in codes if c not in synced]
    print(f"  Already synced: {len(synced)}")
    print(f"  To fetch:       {len(todo)}\n")
    if not todo:
        print("Nothing to do."); return

    provider = TuShareProvider(token=settings.tushare_token)
    t_start = time.time()
    failed: list[tuple[str, str]] = []
    total_rows = 0

    for i, code in enumerate(todo, 1):
        t0 = time.time()
        try:
            n = await fetch_and_persist(provider, code, start, end)
            total_rows += n
            elapsed = time.time() - t_start
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"[{i:>4}/{len(todo)}] {code}  rows={n:>5}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  total={total_rows}",
                  flush=True)
        except Exception as e:
            failed.append((code, str(e)[:80]))
            print(f"[{i:>4}/{len(todo)}] {code}  FAILED: {str(e)[:80]}", flush=True)
        dt = time.time() - t0
        if dt < args.rate_sec:
            await asyncio.sleep(args.rate_sec - dt)

    print(f"\n{'='*60}")
    print(f"Done. {len(todo) - len(failed)}/{len(todo)} succeeded.")
    print(f"Total rows inserted: {total_rows}")
    print(f"Elapsed: {time.time() - t_start:.0f}s")
    if failed:
        print(f"\nFirst 10 failures ({len(failed)} total):")
        for code, msg in failed[:10]:
            print(f"  {code}: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
