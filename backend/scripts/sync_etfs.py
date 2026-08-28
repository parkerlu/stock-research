"""Pull ETFs (market='E') from TuShare and backfill daily candles.

ETFs share the daily_candle table with stocks. We mark them in stock_basic
with market='ETF' so screening / mining can include or exclude them by
filter.

Usage: python -m scripts.sync_etfs --min-history-years 1
"""
from __future__ import annotations

import argparse
import asyncio
import time
from datetime import date

import pandas as pd
import tushare as ts
from sqlalchemy import select, func
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockBasic


async def upsert_etfs(df: pd.DataFrame):
    eng = create_async_engine(settings.database_url, echo=False)
    S = async_sessionmaker(eng, expire_on_commit=False)
    rows = []
    for _, r in df.iterrows():
        rows.append({
            "ts_code": r["ts_code"],
            "symbol": r["ts_code"].split(".")[0],
            "name": r.get("name") or "",
            "area": None,
            "industry": (r.get("invest_type") or r.get("type") or "ETF")[:20],
            "market": "ETF",
            "list_date": pd.to_datetime(r["list_date"]).date()
                          if pd.notna(r.get("list_date")) else None,
            "is_active": True,
        })
    async with S() as db:
        stmt = pg_insert(StockBasic).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ts_code"],
            set_={
                "name": stmt.excluded.name,
                "industry": stmt.excluded.industry,
                "market": stmt.excluded.market,
                "list_date": stmt.excluded.list_date,
                "is_active": stmt.excluded.is_active,
            },
        )
        await db.execute(stmt)
        await db.commit()
    await eng.dispose()


async def synced_codes(codes: list[str]) -> dict[str, int]:
    eng = create_async_engine(settings.database_url, echo=False)
    S = async_sessionmaker(eng, expire_on_commit=False)
    async with S() as db:
        rows = (await db.execute(
            select(DailyCandle.ts_code, func.count())
            .where(DailyCandle.ts_code.in_(codes))
            .group_by(DailyCandle.ts_code)
        )).all()
    await eng.dispose()
    return {ts: int(n) for ts, n in rows}


async def fetch_one(pro, ts_code: str, start: date, end: date) -> int:
    """Fetch ETF daily candles via TuShare pro.fund_daily."""
    df = pro.fund_daily(
        ts_code=ts_code,
        start_date=start.strftime("%Y%m%d"),
        end_date=end.strftime("%Y%m%d"),
    )
    if df is None or df.empty:
        return 0
    df = df.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.rename(columns={
        "vol": "vol",     # already vol (in lots? — TuShare returns 手, fine)
        "amount": "amount",
    })
    df["adj_factor"] = 1.0
    df["source"] = "tushare"
    keep = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount", "adj_factor", "source"]
    rows = df[keep].to_dict("records")

    eng = create_async_engine(settings.database_url, echo=False)
    S = async_sessionmaker(eng, expire_on_commit=False)
    async with S() as db:
        stmt = pg_insert(DailyCandle).values(rows)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["ts_code", "trade_date"]
        )
        await db.execute(stmt)
        await db.commit()
    await eng.dispose()
    return len(rows)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=int, default=7)
    p.add_argument("--rate-sec", type=float, default=0.4)
    p.add_argument("--top-n", type=int, default=50,
                   help="Pick top-N ETFs by recent trading amount")
    args = p.parse_args()

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()

    print("Fetching ETF list + recent volume...", flush=True)
    basic = pro.fund_basic(market="E")
    print(f"  {len(basic)} ETFs total", flush=True)

    # Most recent trading-day amounts → rank popularity
    today = date.today()
    end_str = today.strftime("%Y%m%d")
    daily = pro.fund_daily(trade_date=end_str)
    if daily is None or daily.empty:
        # Try yesterday if today is non-trading
        from datetime import timedelta
        for back in range(1, 5):
            d = (today - timedelta(days=back)).strftime("%Y%m%d")
            daily = pro.fund_daily(trade_date=d)
            if daily is not None and not daily.empty:
                print(f"  using volume from {d}", flush=True)
                break

    # Filter by ≥1 year history
    basic["list_date_dt"] = pd.to_datetime(basic["list_date"], errors="coerce")
    cutoff = pd.Timestamp(date(today.year - 1, today.month, today.day))
    long_history = basic[basic["list_date_dt"] <= cutoff][["ts_code", "name", "list_date"]]

    # Rank and take top-N
    daily_sorted = (daily[daily["ts_code"].isin(long_history["ts_code"])]
                    .sort_values("amount", ascending=False)
                    .head(args.top_n))
    eligible = daily_sorted.merge(long_history, on="ts_code")

    # Force-include 588060 if user mentioned it and it isn't already there
    must_have = ["588060.SH"]
    for code in must_have:
        if code not in set(eligible["ts_code"]):
            extra = long_history[long_history["ts_code"] == code]
            if not extra.empty:
                eligible = pd.concat(
                    [eligible, extra.assign(amount=0)], ignore_index=True
                )

    print(f"\nSelected top {len(eligible)} ETFs (by amount, ≥1y history):", flush=True)
    for _, r in eligible.iterrows():
        amt = r.get("amount", 0) or 0
        print(f"  {r['ts_code']}  {r['name']:<30}  amount={amt/1e8:.1f}亿", flush=True)

    print("Upserting ETF metadata into stock_basic...", flush=True)
    await upsert_etfs(eligible)

    codes = eligible["ts_code"].tolist()
    sync_map = await synced_codes(codes)
    todo = [c for c in codes if sync_map.get(c, 0) < 100]
    print(f"To fetch: {len(todo)} (of {len(codes)} eligible)", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    end = date.today()
    start = date(end.year - args.years, 1, 1)
    t0 = time.time()
    failed: list[tuple[str, str]] = []
    total_rows = 0
    for i, code in enumerate(todo, 1):
        t_st = time.time()
        try:
            n = await fetch_one(pro, code, start, end)
            total_rows += n
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(todo) - i) / rate if rate > 0 else 0
            print(f"[{i:>4}/{len(todo)}] {code}  rows={n:>5}  "
                  f"rate={rate:.1f}/s  eta={eta:.0f}s  total={total_rows}",
                  flush=True)
        except Exception as e:
            failed.append((code, str(e)[:80]))
            print(f"[{i:>4}/{len(todo)}] {code}  FAILED: {str(e)[:80]}", flush=True)
        dt = time.time() - t_st
        if dt < args.rate_sec:
            await asyncio.sleep(args.rate_sec - dt)

    print(f"\nDone. {len(todo) - len(failed)}/{len(todo)} succeeded.")
    print(f"Total rows: {total_rows}")
    print(f"Elapsed: {time.time() - t0:.0f}s")
    if failed:
        print(f"\nFirst 10 failures ({len(failed)} total):")
        for c, msg in failed[:10]:
            print(f"  {c}: {msg}")


if __name__ == "__main__":
    asyncio.run(main())
