"""Screening API — async scan job for recent strategy signals."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import distinct, select

from app.db import async_session, get_db
from app.models.schema import DailyCandle, StockBasic
from app.services.factory_service import get_candle_dicts
from app.services.strategy_templates import TEMPLATE_REGISTRY

router = APIRouter(prefix="/api/screening", tags=["screening"])


class ScanRequest(BaseModel):
    template_id: str
    lookback_days: int = 3


class StockHit(BaseModel):
    ts_code: str
    name: str | None = None
    signal_date: str
    latest_date: str
    latest_close: float
    gain_since_signal_pct: float
    max_drawdown_pct: float       # peak-to-trough since signal, on this stock alone


class JobState(BaseModel):
    job_id: str
    status: str            # "pending" | "running" | "completed" | "cancelled" | "error"
    progress: int = 0      # stocks scanned
    total: int = 0
    hits: list[StockHit] = []
    error: str | None = None
    elapsed_sec: float = 0.0


# In-memory job registry. Survives only while the process lives.
_JOBS: dict[str, dict] = {}
_CANCEL: dict[str, bool] = {}


async def _run_scan(job_id: str, template_id: str, lookback_days: int) -> None:
    job = _JOBS[job_id]
    job["status"] = "running"
    t0 = time.time()

    # Fast path: cached screening for known template families
    fast_supported = (
        template_id.startswith("mm-") or
        template_id.startswith("rev-") or
        template_id.startswith("426-")
    )
    if fast_supported:
        try:
            from app.services.screening_fast import fast_scan, _load_caches
            from sqlalchemy import select
            async with async_session() as db:
                names_rows = (await db.execute(
                    select(StockBasic.ts_code, StockBasic.name)
                )).all()
                active_rows = (await db.execute(
                    select(StockBasic.ts_code).where(StockBasic.is_active.is_(True))
                )).scalars().all()
            names = {ts: nm for ts, nm in names_rows}
            active_set = set(active_rows)

            cache = _load_caches()
            stocks = cache.get("stocks", {})
            # Filter to active only
            active_codes = [c for c in stocks.keys() if c in active_set]
            job["total"] = len(active_codes)

            def on_progress(scanned, _total):
                if _CANCEL.get(job_id):
                    raise InterruptedError()
                # Map raw progress to active subset
                job["progress"] = min(scanned, job["total"])
                job["elapsed_sec"] = time.time() - t0

            try:
                # Pre-filter stocks dict to active before scanning
                from app.services import screening_fast
                full_stocks = screening_fast._CACHE["stocks"] if screening_fast._CACHE else {}
                active_only = {k: v for k, v in full_stocks.items() if k in active_set}
                # Temporarily swap to active subset
                screening_fast._CACHE["stocks"] = active_only
                try:
                    hits_raw = fast_scan(template_id, lookback_days, on_progress)
                finally:
                    screening_fast._CACHE["stocks"] = full_stocks
            except InterruptedError:
                job["status"] = "cancelled"
                job["progress"] = job.get("progress", 0)
                job["elapsed_sec"] = time.time() - t0
                return

            hits = [
                StockHit(
                    ts_code=h["ts_code"],
                    name=names.get(h["ts_code"]),
                    signal_date=h["signal_date"],
                    latest_date=h["latest_date"],
                    latest_close=h["latest_close"],
                    gain_since_signal_pct=h["gain_since_signal_pct"],
                    max_drawdown_pct=h["max_drawdown_pct"],
                )
                for h in hits_raw
            ]
            job["hits"] = hits
            job["status"] = "completed"
            job["elapsed_sec"] = time.time() - t0
            return
        except Exception as e:
            job["error"] = f"fast scan failed: {e}; falling back to slow path"
            # fall through to slow path

    try:
        async with async_session() as db:
            # Active stocks only (excludes ST/*ST/suspended)
            codes = (await db.execute(
                select(distinct(DailyCandle.ts_code))
                .join(StockBasic, StockBasic.ts_code == DailyCandle.ts_code)
                .where(StockBasic.is_active.is_(True))
            )).scalars().all()
            names_rows = (await db.execute(
                select(StockBasic.ts_code, StockBasic.name)
            )).all()
            names = {ts: nm for ts, nm in names_rows}
            max_date = (await db.execute(
                select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
            )).scalar_one_or_none()

        if not max_date:
            job["status"] = "completed"
            return

        end = max_date
        start = end - timedelta(days=400)
        cls = TEMPLATE_REGISTRY[template_id]

        job["total"] = len(codes)
        hits: list[StockHit] = []

        for k, code in enumerate(codes):
            if _CANCEL.get(job_id):
                job["status"] = "cancelled"
                job["progress"] = k
                job["elapsed_sec"] = time.time() - t0
                job["hits"] = hits
                return

            try:
                async with async_session() as db:
                    candles = await get_candle_dicts(db, code, start, end)
                if len(candles) < 130:
                    continue
                df = pd.DataFrame(candles)
                template = cls()
                if hasattr(template, "ts_code"):
                    template.ts_code = code
                signals = template.generate_signals(df)
                buys = [s for s in signals if s.get("action") == "buy"]
                if not buys:
                    continue
                last_buy = buys[-1]
                d = last_buy["date"]
                d_str = d.isoformat() if hasattr(d, "isoformat") else str(d)
                buy_date = pd.to_datetime(d_str).date()
                df_dates = pd.to_datetime(df["trade_date"]).dt.date.tolist()
                try:
                    buy_idx = df_dates.index(buy_date)
                except ValueError:
                    continue
                recent_n = len(df_dates) - buy_idx
                if recent_n > lookback_days + 1:
                    continue
                latest_close = float(df.iloc[-1]["close"])
                buy_close = float(df.iloc[buy_idx]["close"])
                gain = (latest_close / buy_close - 1) * 100 if buy_close > 0 else 0.0
                # Per-stock max drawdown since the buy signal:
                # running peak of close, worst (peak − low)/peak observed.
                window = df.iloc[buy_idx:].reset_index(drop=True)
                closes = window["close"].astype(float).values
                lows = window["low"].astype(float).values
                running_peak = closes[0]
                worst_dd = 0.0
                for i in range(1, len(window)):
                    running_peak = max(running_peak, closes[i])
                    if running_peak > 0:
                        dd = (running_peak - lows[i]) / running_peak
                        if dd > worst_dd:
                            worst_dd = dd
                hits.append(StockHit(
                    ts_code=code,
                    name=names.get(code),
                    signal_date=buy_date.isoformat(),
                    latest_date=df_dates[-1].isoformat(),
                    latest_close=latest_close,
                    gain_since_signal_pct=round(gain, 2),
                    max_drawdown_pct=round(worst_dd * 100, 2),
                ))
            except Exception:
                pass

            job["progress"] = k + 1
            job["elapsed_sec"] = time.time() - t0
            # Update visible hits incrementally (sorted)
            hits.sort(key=lambda h: (h.signal_date, h.gain_since_signal_pct), reverse=True)
            job["hits"] = hits

            if (k + 1) % 25 == 0:
                await asyncio.sleep(0)  # yield to event loop

        job["status"] = "completed"
        job["elapsed_sec"] = time.time() - t0
    except Exception as e:
        job["status"] = "error"
        job["error"] = str(e)
        job["elapsed_sec"] = time.time() - t0


@router.post("/scan/start")
async def start_scan(req: ScanRequest):
    if req.template_id not in TEMPLATE_REGISTRY:
        raise HTTPException(400, f"Unknown template: {req.template_id}")
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {
        "job_id": job_id, "status": "pending",
        "progress": 0, "total": 0, "hits": [],
        "elapsed_sec": 0.0, "error": None,
        "template_id": req.template_id,
    }
    _CANCEL[job_id] = False
    asyncio.create_task(_run_scan(job_id, req.template_id, req.lookback_days))
    return {"job_id": job_id}


@router.get("/scan/{job_id}", response_model=JobState)
async def scan_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return JobState(
        job_id=job_id,
        status=job["status"],
        progress=job["progress"],
        total=job["total"],
        hits=job["hits"],
        error=job.get("error"),
        elapsed_sec=round(job.get("elapsed_sec", 0.0), 1),
    )


@router.post("/scan/{job_id}/cancel")
async def cancel_scan(job_id: str):
    if job_id not in _JOBS:
        raise HTTPException(404, "Unknown job")
    _CANCEL[job_id] = True
    return {"job_id": job_id, "cancelling": True}
