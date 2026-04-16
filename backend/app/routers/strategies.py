"""Strategy API: list strategies, pin/unpin."""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.models.schema import Strategy

router = APIRouter(prefix="/api/strategies", tags=["strategies"])


@router.get("")
async def list_strategies(
    ts_code: str = Query(...),
    db: AsyncSession = Depends(get_db),
):
    stmt = (
        select(Strategy)
        .where(Strategy.ts_code == ts_code)
        .order_by(Strategy.annualized_return.desc())
    )
    result = await db.execute(stmt)
    rows = result.scalars().all()
    return [
        {
            "id": s.id,
            "name": s.name,
            "template": s.template,
            "parameters": s.parameters,
            "is_pinned": s.is_pinned,
            "annualized_return": float(s.annualized_return) if s.annualized_return is not None else None,
            "net_profit": float(s.net_profit) if s.net_profit is not None else None,
            "max_drawdown": float(s.max_drawdown) if s.max_drawdown is not None else None,
            "win_rate": float(s.win_rate) if s.win_rate is not None else None,
            "total_trades": s.total_trades,
            "profit_factor": float(s.profit_factor) if s.profit_factor is not None else None,
            "final_capital": float(s.final_capital) if s.final_capital is not None else None,
            "job_id": s.job_id,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in rows
    ]


@router.post("/{strategy_id}/pin")
async def pin(strategy_id: int, db: AsyncSession = Depends(get_db)):
    strategy = await db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.is_pinned = True
    await db.commit()
    return {"ok": True}


@router.delete("/{strategy_id}/pin")
async def unpin(strategy_id: int, db: AsyncSession = Depends(get_db)):
    strategy = await db.get(Strategy, strategy_id)
    if not strategy:
        raise HTTPException(status_code=404, detail="Strategy not found")
    strategy.is_pinned = False
    await db.commit()
    return {"ok": True}
