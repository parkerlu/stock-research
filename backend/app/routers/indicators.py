from dataclasses import asdict
from datetime import date

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.datasources.manager import DataSourceManager
from app.db import get_db
from app.routers.quotes import get_manager
from app.services.quote_service import get_candles
from app.services.tdx import registry
from app.services.tdx.indicators.base import IndicatorResult

router = APIRouter(prefix="/api/indicators")


@router.get("")
def list_indicators():
    """List all registered TDX indicators."""
    return [
        {"name": m.name, "label": m.label, "pane": m.pane, "min_bars": m.min_bars}
        for m in registry.all_indicators()
    ]


@router.get("/{name}")
async def get_indicator(
    name: str,
    ts_code: str = Query(...),
    tf: str = Query("1d", pattern="^(1d|1w|1m)$"),
    start: date = Query(alias="from", default=None),
    end: date = Query(alias="to", default=None),
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    impl = registry.get(name)
    if impl is None:
        raise HTTPException(status_code=404, detail=f"Unknown indicator: {name}")

    if start is None:
        start = date(date.today().year - 2, date.today().month, date.today().day)
    if end is None:
        end = date.today()

    try:
        candles = await get_candles(db, manager, ts_code, tf, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    if not candles:
        return asdict(IndicatorResult(
            name=impl.name, label=impl.label, pane=impl.pane,
            warnings=["该股票在所选区间无 K 线数据"],
        ))

    df = pd.DataFrame(candles)

    if len(df) < impl.min_bars:
        return asdict(IndicatorResult(
            name=impl.name, label=impl.label, pane=impl.pane,
            warnings=[
                f"数据不足，{impl.label} 需要至少 {impl.min_bars} 个 bar，当前仅 {len(df)} 个"
            ],
        ))

    result = impl.compute(df)
    return asdict(result)
