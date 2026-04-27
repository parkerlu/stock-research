"""Strategy factory service: generate candidates, backtest, filter, persist."""
from __future__ import annotations

import uuid
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime

import pandas as pd
from sqlalchemy import delete, select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import DailyCandle, FactoryJob, Strategy
from app.services.backtest_engine import run_backtest
from app.services.strategy_templates import TEMPLATE_REGISTRY, generate_all_candidates


def evaluate_single_candidate(
    candles: list[dict],
    candidate: dict,
) -> dict:
    """Evaluate a single strategy candidate (runs in worker process)."""
    template_cls = TEMPLATE_REGISTRY[candidate["template_id"]]
    template = template_cls(**candidate["params"])

    df = pd.DataFrame(candles)
    signals = template.generate_signals(df)
    metrics = run_backtest(candles, signals, initial_capital=10000)

    return {
        "template_id": candidate["template_id"],
        "name": candidate["name"],
        "params": candidate["params"],
        "metrics": metrics,
    }


def filter_and_rank(results: list[dict], top_n: int = 50) -> list[dict]:
    """Apply hard filters and return top N by annualized return.

    Filters: net_profit > 0 AND max_drawdown <= 0.35
    """
    passed = [
        r for r in results
        if r["metrics"]["net_profit"] > 0 and r["metrics"]["max_drawdown"] <= 0.35
    ]
    passed.sort(key=lambda r: r["metrics"]["annualized_return"], reverse=True)
    return passed[:top_n]


async def get_candle_dicts(db: AsyncSession, ts_code: str, start: date, end: date) -> list[dict]:
    """Fetch candle data as list of dicts for backtest.

    Applies BACKWARD adjustment (latest price preserved) to match exactly
    what the K-line chart shows. Without this, dividends/splits cause the
    backtest to compute returns from raw prices that don't match the chart.
    """
    stmt = (
        select(DailyCandle)
        .where(and_(
            DailyCandle.ts_code == ts_code,
            DailyCandle.trade_date >= start,
            DailyCandle.trade_date <= end,
        ))
        .order_by(DailyCandle.trade_date)
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    if not rows:
        return []
    # Use GLOBAL latest adj_factor (not within range) so prices match the chart
    latest_row = (await db.execute(
        select(DailyCandle.adj_factor)
        .where(DailyCandle.ts_code == ts_code)
        .order_by(DailyCandle.trade_date.desc())
        .limit(1)
    )).scalar_one_or_none()
    latest_adj = float(latest_row) if latest_row else 1.0
    return [
        {
            "trade_date": r.trade_date,
            # Apply factor = adj / latest_adj — same formula as quote_service
            "open": round(float(r.open) * (float(r.adj_factor or 1.0) / latest_adj), 4),
            "high": round(float(r.high) * (float(r.adj_factor or 1.0) / latest_adj), 4),
            "low":  round(float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj), 4),
            "close":round(float(r.close)* (float(r.adj_factor or 1.0) / latest_adj), 4),
            "vol": r.vol,
            "amount": float(r.amount),
        }
        for r in rows
    ]


async def run_factory(
    db: AsyncSession,
    ts_code: str,
    cutoff_date: date,
    job_id: str | None = None,
) -> str:
    """Run the strategy factory: generate candidates, backtest, filter, persist top 50."""
    candidates = generate_all_candidates()

    # Use existing job or create new one
    if job_id:
        job = await db.get(FactoryJob, job_id)
        if not job:
            return job_id
        job.status = "running"
        job.total_candidates = len(candidates)
        await db.commit()
    else:
        job_id = str(uuid.uuid4())
        job = FactoryJob(
            id=job_id,
            ts_code=ts_code,
            status="running",
            total_candidates=len(candidates),
            config={"cutoff_date": cutoff_date.isoformat()},
        )
        db.add(job)
        await db.commit()

    # Fetch candle data: cutoff_date minus 10 years
    start_date = date(cutoff_date.year - 10, cutoff_date.month, cutoff_date.day)
    candles = await get_candle_dicts(db, ts_code, start_date, cutoff_date)

    if not candles:
        job.status = "failed"
        job.error = "No candle data available"
        job.completed_at = datetime.now()
        await db.commit()
        return job_id

    # Run backtests in parallel
    results = []
    try:
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(evaluate_single_candidate, candles, c)
                for c in candidates
            ]
            for i, future in enumerate(futures):
                try:
                    result = future.result(timeout=30)
                    results.append(result)
                except Exception:
                    pass

                if (i + 1) % 50 == 0 or i == len(futures) - 1:
                    job.evaluated = i + 1
                    await db.commit()

        top_results = filter_and_rank(results)
        job.evaluated = len(candidates)
        job.passed = len(top_results)

        # Delete old non-pinned strategies for this stock
        await db.execute(
            delete(Strategy).where(
                and_(Strategy.ts_code == ts_code, Strategy.is_pinned == False)  # noqa: E712
            )
        )

        # Insert new top strategies
        for r in top_results:
            m = r["metrics"]
            strategy = Strategy(
                ts_code=ts_code,
                name=r["name"],
                template=r["template_id"],
                parameters=r["params"],
                annualized_return=m["annualized_return"],
                net_profit=m["net_profit"],
                max_drawdown=m["max_drawdown"],
                win_rate=m["win_rate"],
                total_trades=m["total_trades"],
                profit_factor=m["profit_factor"],
                final_capital=m["final_capital"],
                metrics=m,
                job_id=job_id,
            )
            db.add(strategy)

        job.status = "completed"
        job.completed_at = datetime.now()
        await db.commit()

    except Exception as e:
        job.status = "failed"
        job.error = str(e)
        job.completed_at = datetime.now()
        await db.commit()

    return job_id
