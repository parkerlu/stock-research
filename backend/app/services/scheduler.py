"""Daily 15:30 scheduler — backfill all stock + ETF candles after market close."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta

import pandas as pd
import tushare as ts
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.db import async_session
from app.models.schema import DailyCandle, StockBasic

log = logging.getLogger("scheduler")

_scheduler: AsyncIOScheduler | None = None


async def _sync_stocks():
    """Backfill stale stock candles (same logic as /api/system/backfill/start)."""
    log.info("=== scheduled stock sync started ===")
    today = date.today()

    async with async_session() as db:
        rows = (await db.execute(
            select(StockBasic.ts_code, StockBasic.is_active)
        )).all()
        latest_rows = (await db.execute(
            select(DailyCandle.ts_code, func.max(DailyCandle.trade_date).label("latest"))
            .group_by(DailyCandle.ts_code)
        )).all()

    latest_map = {ts_code: d for ts_code, d in latest_rows}
    targets = []
    for ts_code, active in rows:
        if active is False:
            continue
        latest = latest_map.get(ts_code)
        if latest is None or latest < today:
            targets.append((ts_code, latest))

    log.info("stocks to update: %d", len(targets))
    if not targets:
        return

    provider = TuShareProvider(token=settings.tushare_token)
    inserted = 0

    for k, (ts_code, latest) in enumerate(targets):
        t_st = time.time()
        try:
            start = (latest + timedelta(days=1)) if latest else date(today.year - 7, 1, 1)
            if start <= today:
                df = await provider.fetch_daily(ts_code, start, today)
                if not df.empty:
                    df = df.copy()
                    df["source"] = "tushare"
                    records = df.to_dict("records")
                    async with async_session() as db:
                        stmt = pg_insert(DailyCandle).values(records)
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["ts_code", "trade_date"]
                        )
                        await db.execute(stmt)
                        await db.commit()
                    inserted += len(records)
        except Exception as e:
            log.warning("stock %s failed: %s", ts_code, str(e)[:100])

        elapsed = time.time() - t_st
        if elapsed < 0.4:
            await asyncio.sleep(0.4 - elapsed)

        if (k + 1) % 100 == 0:
            log.info("  progress: %d/%d  rows=%d", k + 1, len(targets), inserted)

    log.info("=== stock sync done: %d rows inserted ===", inserted)


async def _sync_etfs():
    """Backfill ETF candles (top-50 by volume)."""
    log.info("=== scheduled ETF sync started ===")
    today = date.today()

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()

    try:
        basic = pro.fund_basic(market="E")
    except Exception as e:
        log.error("ETF basic fetch failed: %s", e)
        return

    basic["list_date_dt"] = pd.to_datetime(basic["list_date"], errors="coerce")
    cutoff = pd.Timestamp(date(today.year - 1, today.month, today.day))
    long_history = basic[basic["list_date_dt"] <= cutoff][["ts_code", "name", "list_date"]]

    daily = None
    for back in range(0, 5):
        d = (today - timedelta(days=back)).strftime("%Y%m%d")
        daily = pro.fund_daily(trade_date=d)
        if daily is not None and not daily.empty:
            break

    if daily is None or daily.empty:
        log.warning("no recent ETF daily data found")
        return

    eligible = (daily[daily["ts_code"].isin(long_history["ts_code"])]
                .sort_values("amount", ascending=False)
                .head(50)
                .merge(long_history, on="ts_code"))

    must_have = ["588060.SH"]
    for code in must_have:
        if code not in set(eligible["ts_code"]):
            extra = long_history[long_history["ts_code"] == code]
            if not extra.empty:
                eligible = pd.concat(
                    [eligible, extra.assign(amount=0)], ignore_index=True
                )

    # Upsert ETF metadata
    meta_rows = []
    for _, r in eligible.iterrows():
        meta_rows.append({
            "ts_code": r["ts_code"],
            "symbol": r["ts_code"].split(".")[0],
            "name": r.get("name") or "",
            "area": None,
            "industry": "ETF",
            "market": "ETF",
            "list_date": pd.to_datetime(r["list_date"]).date()
                          if pd.notna(r.get("list_date")) else None,
            "is_active": True,
        })
    async with async_session() as db:
        stmt = pg_insert(StockBasic).values(meta_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ts_code"],
            set_={
                "name": stmt.excluded.name,
                "industry": stmt.excluded.industry,
                "market": stmt.excluded.market,
                "is_active": stmt.excluded.is_active,
            },
        )
        await db.execute(stmt)
        await db.commit()

    # Fetch candles for ETFs with <100 bars
    codes = eligible["ts_code"].tolist()
    async with async_session() as db:
        bar_rows = (await db.execute(
            select(DailyCandle.ts_code, func.count())
            .where(DailyCandle.ts_code.in_(codes))
            .group_by(DailyCandle.ts_code)
        )).all()
    sync_map = {ts_code: int(n) for ts_code, n in bar_rows}
    todo = [c for c in codes if sync_map.get(c, 0) < 100]

    if not todo:
        log.info("all ETFs up to date")
        return

    start = date(today.year - 7, 1, 1)
    inserted = 0
    for i, code in enumerate(todo, 1):
        t_st = time.time()
        try:
            df = pro.fund_daily(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=today.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.copy()
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                df["adj_factor"] = 1.0
                df["source"] = "tushare"
                keep = ["ts_code", "trade_date", "open", "high", "low", "close",
                        "vol", "amount", "adj_factor", "source"]
                records = df[keep].to_dict("records")
                async with async_session() as db:
                    stmt = pg_insert(DailyCandle).values(records)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["ts_code", "trade_date"]
                    )
                    await db.execute(stmt)
                    await db.commit()
                inserted += len(records)
        except Exception as e:
            log.warning("ETF %s failed: %s", code, str(e)[:100])
        dt = time.time() - t_st
        if dt < 0.4:
            await asyncio.sleep(0.4 - dt)

    log.info("=== ETF sync done: %d rows inserted ===", inserted)


async def daily_sync_job():
    """Main scheduled job: sync stocks then ETFs."""
    log.info("====== daily sync triggered at 15:30 ======")
    await _sync_stocks()
    await _sync_etfs()
    log.info("====== daily sync complete ======")


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
    _scheduler.add_job(
        daily_sync_job,
        CronTrigger(hour=15, minute=30, timezone="Asia/Shanghai"),
        id="daily_sync",
        replace_existing=True,
    )
    _scheduler.start()
    log.info("scheduler started — daily sync at 15:30 CST")


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
