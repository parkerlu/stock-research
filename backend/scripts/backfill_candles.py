"""
Backfill daily_candle for ~200 mainstream stocks (main-board, listed pre-2015).

Picks stocks deterministically from stock_basic, fetches 7 years of daily data
via TuShare, and bulk-inserts into daily_candle. Rate-limited to ~1 stock/sec
to stay under TuShare's 200 calls/min limit.

Run with:  python -m scripts.backfill_candles --limit 200
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date, datetime

from sqlalchemy import select, distinct, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.models.schema import DailyCandle, StockBasic


async def pick_universe(limit: int) -> list[str]:
    """Pick `limit` stocks: main board, listed before 2015 (=> >7y history)."""
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        # Main board only (market='主板'), older listings have richer history
        stmt = (select(StockBasic.ts_code)
                .where(StockBasic.market == "主板",
                       StockBasic.list_date.is_not(None),
                       StockBasic.list_date < date(2015, 1, 1))
                .order_by(StockBasic.ts_code)
                .limit(limit))
        codes = (await db.execute(stmt)).scalars().all()
    await engine.dispose()
    return list(codes)


async def already_synced(codes: list[str], min_bars: int = 100) -> set[str]:
    """Return codes that already have at least min_bars of candles."""
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
    """Fetch daily candles for one stock and upsert. Returns rows inserted."""
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--years", type=int, default=7)
    parser.add_argument("--rate-sec", type=float, default=0.6,
                        help="Min seconds between TuShare calls")
    args = parser.parse_args()

    end = date.today()
    start = date(end.year - args.years, 1, 1)

    print(f"Picking up to {args.limit} stocks listed before 2015...")
    codes = await pick_universe(args.limit)
    print(f"Picked {len(codes)} stocks")

    print("Checking which are already synced...")
    synced = await already_synced(codes)
    todo = [c for c in codes if c not in synced]
    print(f"Already synced: {len(synced)}.  To fetch: {len(todo)}.\n")

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
            print(f"[{i:>3}/{len(todo)}] {code}  rows={n:>5}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  total_rows={total_rows}")
        except Exception as e:
            failed.append((code, str(e)[:80]))
            print(f"[{i:>3}/{len(todo)}] {code}  FAILED: {str(e)[:80]}")
        # Rate limit
        dt = time.time() - t0
        if dt < args.rate_sec:
            await asyncio.sleep(args.rate_sec - dt)

    print(f"\n{'='*60}")
    print(f"Done. {len(todo) - len(failed)}/{len(todo)} succeeded.")
    print(f"Total rows inserted: {total_rows}")
    print(f"Elapsed: {time.time() - t_start:.0f}s")
    if failed:
        print(f"\nFailed ({len(failed)}):")
        for code, msg in failed[:10]:
            print(f"  {code}: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
