"""Backtest API: run backtests, retrieve reports."""
from __future__ import annotations

import uuid
from datetime import date, datetime

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import BacktestRun, Strategy
from app.services.backtest_engine import run_backtest
from app.services.factory_service import get_candle_dicts
from app.services.strategy_templates import TEMPLATE_REGISTRY

router = APIRouter(prefix="/api/backtests", tags=["backtests"])


class RunBacktestRequest(BaseModel):
    ts_code: str
    strategy_id: int | None = None
    start_date: date
    end_date: date
    initial_capital: float = 10000.0
    position_ratios: list[int] = [40, 30, 30]


async def _execute_backtest(run_id: str, request: RunBacktestRequest):
    """Background task to execute a backtest."""
    from app.db import async_session

    async with async_session() as db:
        bt = await db.get(BacktestRun, run_id)
        if not bt:
            return

        try:
            candles = await get_candle_dicts(db, request.ts_code, request.start_date, request.end_date)
            if not candles:
                bt.status = "failed"
                bt.completed_at = datetime.now()
                await db.commit()
                return

            signals = []
            if request.strategy_id:
                strategy = await db.get(Strategy, request.strategy_id)
                if strategy and strategy.template in TEMPLATE_REGISTRY:
                    import pandas as pd
                    template_cls = TEMPLATE_REGISTRY[strategy.template]
                    template = template_cls(**strategy.parameters)
                    df = pd.DataFrame(candles)
                    signals = template.generate_signals(df)

            result = run_backtest(
                candles, signals,
                initial_capital=request.initial_capital,
                position_ratios=request.position_ratios,
            )

            bt.status = "completed"
            bt.metrics = {
                k: v for k, v in result.items()
                if k not in ("trades", "equity_curve")
            }
            bt.trades = result["trades"]
            bt.equity_curve = result["equity_curve"]
            bt.completed_at = datetime.now()
            await db.commit()

        except Exception:
            bt.status = "failed"
            bt.completed_at = datetime.now()
            await db.commit()


@router.post("/run")
async def run(
    body: RunBacktestRequest,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
):
    run_id = str(uuid.uuid4())
    bt = BacktestRun(
        id=run_id,
        ts_code=body.ts_code,
        strategy_id=body.strategy_id,
        start_date=body.start_date,
        end_date=body.end_date,
        initial_capital=body.initial_capital,
        position_ratios=body.position_ratios,
        status="running",
    )
    db.add(bt)
    await db.commit()

    background_tasks.add_task(_execute_backtest, run_id, body)
    return {"id": run_id, "status": "running"}


@router.get("/{run_id}/report")
async def report(run_id: str, db: AsyncSession = Depends(get_db)):
    bt = await db.get(BacktestRun, run_id)
    if not bt:
        raise HTTPException(status_code=404, detail="Backtest run not found")
    return {
        "id": bt.id,
        "status": bt.status,
        "ts_code": bt.ts_code,
        "strategy_id": bt.strategy_id,
        "metrics": bt.metrics,
        "trades": bt.trades,
        "equity_curve": bt.equity_curve,
        "created_at": bt.created_at.isoformat() if bt.created_at else None,
        "completed_at": bt.completed_at.isoformat() if bt.completed_at else None,
    }
