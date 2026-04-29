"""System admin: DB status + on-demand candle backfill."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import date, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlalchemy import distinct, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.db import async_session
from app.models.schema import DailyCandle, StockBasic

router = APIRouter(prefix="/api/system", tags=["system"])


@router.get("/data-status")
async def data_status():
    """Per-stock summary: latest local date, bars count, days stale."""
    async with async_session() as db:
        max_row = await db.execute(
            select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
        )
        market_latest = max_row.scalar_one_or_none()
        if not market_latest:
            return {"market_latest": None, "totals": {}, "stale": []}

        # Per-stock latest + count
        sub = (
            select(
                DailyCandle.ts_code,
                func.max(DailyCandle.trade_date).label("latest"),
                func.count().label("bars"),
            )
            .group_by(DailyCandle.ts_code)
            .subquery()
        )
        rows = (await db.execute(
            select(sub.c.ts_code, sub.c.latest, sub.c.bars,
                   StockBasic.name, StockBasic.is_active)
            .outerjoin(StockBasic, StockBasic.ts_code == sub.c.ts_code)
        )).all()

    n = len(rows)
    up_to_date = sum(1 for r in rows if r[1] == market_latest)
    stale_7d = sum(1 for r in rows if r[1] and (market_latest - r[1]).days >= 7)
    stale_30d = sum(1 for r in rows if r[1] and (market_latest - r[1]).days >= 30)
    inactive = sum(1 for r in rows if r[4] is False)

    # List stocks needing backfill (active + stale OR < market latest)
    stale = []
    for ts_code, latest, bars, name, active in rows:
        if not latest:
            continue
        gap = (market_latest - latest).days
        if gap > 0:
            stale.append({
                "ts_code": ts_code,
                "name": name,
                "latest_date": latest.isoformat(),
                "bars": int(bars),
                "days_stale": gap,
                "active": bool(active) if active is not None else True,
            })
    stale.sort(key=lambda r: (-r["days_stale"], r["ts_code"]))

    return {
        "market_latest": market_latest.isoformat(),
        "totals": {
            "total_stocks": n,
            "up_to_date": up_to_date,
            "stale_7d": stale_7d,
            "stale_30d": stale_30d,
            "inactive": inactive,
        },
        "stale": stale[:200],   # cap to avoid huge payloads
    }


# =========================================================================
# Backfill job (async, with progress + cancel)
# =========================================================================

class BackfillRequest(BaseModel):
    include_inactive: bool = False
    only_stale: bool = True   # if True, only fill stocks whose latest < market latest


_JOBS: dict[str, dict] = {}
_CANCEL: dict[str, bool] = {}


async def _do_backfill(job_id: str, include_inactive: bool, only_stale: bool):
    job = _JOBS[job_id]
    job["status"] = "running"
    t0 = time.time()
    try:
        async with async_session() as db:
            market_latest_row = await db.execute(
                select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
            )
            market_latest: date | None = market_latest_row.scalar_one_or_none()
            if not market_latest:
                job["status"] = "completed"
                return

            # Pick the universe
            stmt = select(StockBasic.ts_code, StockBasic.is_active)
            rows = (await db.execute(stmt)).all()
            codes_all = [(c, a) for c, a in rows]

            # Per-stock latest
            sub = (
                select(DailyCandle.ts_code, func.max(DailyCandle.trade_date).label("latest"))
                .group_by(DailyCandle.ts_code).subquery()
            )
            latest_rows = (await db.execute(
                select(sub.c.ts_code, sub.c.latest)
            )).all()
            latest_map = {ts: d for ts, d in latest_rows}

        # `end` should be TODAY, not max-DB-date — otherwise we never advance
        # past the last successful backfill. TuShare returns up-to-the-most-
        # recent trading day automatically (skips weekends/holidays).
        from datetime import date as _date
        today = _date.today()
        end = today

        # Filter: a stock is "stale" if its latest is < end (today) — no longer
        # gated on market_latest (which is a chicken-and-egg metric).
        targets: list[tuple[str, date | None]] = []
        for ts_code, active in codes_all:
            if active is False and not include_inactive:
                continue
            latest = latest_map.get(ts_code)
            if only_stale:
                if latest is None or latest >= end:
                    continue
            targets.append((ts_code, latest))

        job["total"] = len(targets)
        if not targets:
            job["status"] = "completed"
            job["elapsed_sec"] = time.time() - t0
            return

        provider = TuShareProvider(token=settings.tushare_token)
        rows_inserted = 0

        for k, (ts_code, latest) in enumerate(targets):
            if _CANCEL.get(job_id):
                job["status"] = "cancelled"
                job["progress"] = k
                job["elapsed_sec"] = time.time() - t0
                job["rows_inserted"] = rows_inserted
                return

            t_st = time.time()
            try:
                # Start day-after-latest, or 7 years back if no data
                start = (latest + timedelta(days=1)) if latest else date(end.year - 7, 1, 1)
                if start <= end:
                    df = await provider.fetch_daily(ts_code, start, end)
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
                        rows_inserted += len(records)
            except Exception as e:
                job.setdefault("failures", []).append({
                    "ts_code": ts_code, "error": str(e)[:120],
                })

            job["progress"] = k + 1
            job["rows_inserted"] = rows_inserted
            job["elapsed_sec"] = time.time() - t0

            # Rate limit: ≥0.4 s between calls
            elapsed = time.time() - t_st
            if elapsed < 0.4:
                await asyncio.sleep(0.4 - elapsed)

            if (k + 1) % 25 == 0:
                await asyncio.sleep(0)

        job["status"] = "completed"
        job["elapsed_sec"] = time.time() - t0
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["elapsed_sec"] = time.time() - t0


@router.post("/backfill/start")
async def backfill_start(req: BackfillRequest):
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id, "status": "pending",
        "progress": 0, "total": 0,
        "rows_inserted": 0, "elapsed_sec": 0.0,
        "error": None, "failures": [],
    }
    _CANCEL[job_id] = False
    asyncio.create_task(_do_backfill(job_id, req.include_inactive, req.only_stale))
    return {"job_id": job_id}


@router.get("/backfill/{job_id}")
async def backfill_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return {
        "job_id": job_id,
        "status": job["status"],
        "progress": job["progress"],
        "total": job["total"],
        "rows_inserted": job.get("rows_inserted", 0),
        "elapsed_sec": round(job.get("elapsed_sec", 0.0), 1),
        "error": job.get("error"),
        "failures": job.get("failures", [])[-10:],   # tail
    }


@router.post("/backfill/{job_id}/cancel")
async def backfill_cancel(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(404, "Unknown job")
    _CANCEL[job_id] = True
    return {"job_id": job_id, "cancelling": True}
