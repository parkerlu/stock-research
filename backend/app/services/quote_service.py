from datetime import date, timedelta
from itertools import groupby

from sqlalchemy import select, and_
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.datasources.manager import DataSourceManager
from app.models.schema import DailyCandle


def _iso_week_key(d: date) -> tuple[int, int]:
    """Return (iso_year, iso_week) for grouping."""
    cal = d.isocalendar()
    return (cal[0], cal[1])


def _month_key(d: date) -> tuple[int, int]:
    return (d.year, d.month)


def _aggregate(rows: list[dict], key_fn) -> list[dict]:
    """Generic aggregation: group rows by key_fn, produce OHLCV bars."""
    sorted_rows = sorted(rows, key=lambda r: r["trade_date"])
    result = []
    for _, group in groupby(sorted_rows, key=lambda r: key_fn(r["trade_date"])):
        candles = list(group)
        bar = {
            "timestamp": int(candles[0]["trade_date"].strftime("%s")) * 1000,
            "open": candles[0]["open"],
            "close": candles[-1]["close"],
            "high": max(c["high"] for c in candles),
            "low": min(c["low"] for c in candles),
            "volume": sum(c["vol"] for c in candles),
            "amount": sum(c["amount"] for c in candles),
        }
        result.append(bar)
    return result


def aggregate_weekly(rows: list[dict]) -> list[dict]:
    return _aggregate(rows, _iso_week_key)


def aggregate_monthly(rows: list[dict]) -> list[dict]:
    return _aggregate(rows, _month_key)


async def get_candles(
    db: AsyncSession,
    manager: DataSourceManager,
    ts_code: str,
    tf: str,
    start: date,
    end: date,
) -> list[dict]:
    """Fetch candles with cache-through. Returns list of dicts for JSON response."""
    # 1. Check what's cached
    stmt = (
        select(DailyCandle)
        .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
        .order_by(DailyCandle.trade_date)
    )
    result = await db.execute(stmt)
    cached = result.scalars().all()
    cached_dates = {c.trade_date for c in cached}

    # 2. If no cached data, fetch full range from source
    if not cached_dates:
        df = await manager.fetch_daily(ts_code, start, end)
        if not df.empty:
            rows = df.to_dict("records")
            stmt_upsert = pg_insert(DailyCandle).values(rows)
            stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
            await db.execute(stmt_upsert)
            await db.commit()
            result = await db.execute(
                select(DailyCandle)
                .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                .order_by(DailyCandle.trade_date)
            )
            cached = result.scalars().all()
    else:
        # 3. Check if we need incremental update
        last_cached = max(cached_dates)
        if last_cached < end:
            fetch_start = last_cached + timedelta(days=1)
            df = await manager.fetch_daily(ts_code, fetch_start, end)
            if not df.empty:
                rows = df.to_dict("records")
                stmt_upsert = pg_insert(DailyCandle).values(rows)
                stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
                await db.execute(stmt_upsert)
                await db.commit()
                result = await db.execute(
                    select(DailyCandle)
                    .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                    .order_by(DailyCandle.trade_date)
                )
                cached = result.scalars().all()

    # 4. Apply adj_factor for forward-adjusted prices
    daily_rows = []
    for c in cached:
        adj = float(c.adj_factor) if c.adj_factor else 1.0
        latest_adj = float(cached[-1].adj_factor) if cached[-1].adj_factor else 1.0
        factor = adj / latest_adj if latest_adj != 0 else 1.0
        daily_rows.append({
            "trade_date": c.trade_date,
            "open": round(float(c.open) * factor, 4),
            "high": round(float(c.high) * factor, 4),
            "low": round(float(c.low) * factor, 4),
            "close": round(float(c.close) * factor, 4),
            "vol": c.vol,
            "amount": float(c.amount),
        })

    # 5. Aggregate if needed
    if tf == "1w":
        return aggregate_weekly(daily_rows)
    elif tf == "1m":
        return aggregate_monthly(daily_rows)

    return [
        {
            "timestamp": int(r["trade_date"].strftime("%s")) * 1000,
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["vol"],
            "amount": r["amount"],
        }
        for r in daily_rows
    ]
