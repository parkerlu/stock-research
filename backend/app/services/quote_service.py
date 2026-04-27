import calendar
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
            "timestamp": calendar.timegm(candles[0]["trade_date"].timetuple()) * 1000,
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
        # 3. Check if we need incremental update (forward and backward)
        first_cached = min(cached_dates)
        last_cached = max(cached_dates)
        need_reload = False

        # Backward fill: fetch earlier history if requested range starts before cache
        # Skip if gap is small (<=5 days covers weekends/holidays)
        if first_cached > start and (first_cached - start).days > 5:
            try:
                fetch_end = first_cached - timedelta(days=1)
                df = await manager.fetch_daily(ts_code, start, fetch_end)
                if not df.empty:
                    rows = df.to_dict("records")
                    stmt_upsert = pg_insert(DailyCandle).values(rows)
                    stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
                    await db.execute(stmt_upsert)
                    need_reload = True
            except Exception:
                pass  # Use existing cache if data source fails

        # Forward fill: fetch newer data
        # Skip if gap is small (<=5 days covers weekends/holidays)
        if last_cached < end and (end - last_cached).days > 5:
            try:
                fetch_start = last_cached + timedelta(days=1)
                df = await manager.fetch_daily(ts_code, fetch_start, end)
                if not df.empty:
                    rows = df.to_dict("records")
                    stmt_upsert = pg_insert(DailyCandle).values(rows)
                    stmt_upsert = stmt_upsert.on_conflict_do_nothing(index_elements=["ts_code", "trade_date"])
                    await db.execute(stmt_upsert)
                    need_reload = True
            except Exception:
                pass  # Use existing cache if data source fails

        if need_reload:
            await db.commit()
            result = await db.execute(
                select(DailyCandle)
                .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                .order_by(DailyCandle.trade_date)
            )
            cached = result.scalars().all()

    # 4. Apply adj_factor for backward-adjusted prices.
    # IMPORTANT: latest_adj must be the GLOBAL latest for this symbol, not the
    # latest within the requested date range — otherwise charts show different
    # adjusted prices for the same bar depending on what window is queried.
    latest_row = (await db.execute(
        select(DailyCandle.adj_factor)
        .where(DailyCandle.ts_code == ts_code)
        .order_by(DailyCandle.trade_date.desc())
        .limit(1)
    )).scalar_one_or_none()
    latest_adj = float(latest_row) if latest_row else 1.0

    daily_rows = []
    for c in cached:
        adj = float(c.adj_factor) if c.adj_factor else 1.0
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
            "timestamp": calendar.timegm(r["trade_date"].timetuple()) * 1000,
            "open": r["open"],
            "high": r["high"],
            "low": r["low"],
            "close": r["close"],
            "volume": r["vol"],
            "amount": r["amount"],
        }
        for r in daily_rows
    ]
