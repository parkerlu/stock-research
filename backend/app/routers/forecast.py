"""LSTM forecast endpoint."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.db import get_db
from app.services.factory_service import get_candle_dicts
try:
    from app.services.lstm_forecaster import forecast
    _LSTM_OK = True
except Exception as _e:
    _LSTM_OK = False
    _import_err = str(_e)
    def forecast(df):  # noqa
        return None

router = APIRouter(prefix="/api/forecast", tags=["forecast"])


@router.get("/{ts_code}")
async def forecast_stock(
    ts_code: str,
    db: AsyncSession = Depends(get_db),
):
    end = date.today()
    start = end - timedelta(days=200)
    candles = await get_candle_dicts(db, ts_code, start, end)
    if not candles or len(candles) < 60:
        raise HTTPException(404, "Not enough history")
    df = pd.DataFrame(candles)
    result = forecast(df)
    if result is None:
        raise HTTPException(503, "Forecast model not available")
    result["ts_code"] = ts_code
    return result
