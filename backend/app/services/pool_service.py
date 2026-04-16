from datetime import date, datetime, timedelta

from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.datasources.manager import DataSourceManager
from app.models.schema import DailyCandle, StockBasic, StockPool, StockPoolItem


async def create_pool(db: AsyncSession, name: str, description: str | None = None) -> dict:
    pool = StockPool(name=name, description=description)
    db.add(pool)
    await db.commit()
    await db.refresh(pool)
    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "created_at": pool.created_at.isoformat(),
    }


async def list_pools(db: AsyncSession) -> list[dict]:
    stmt = (
        select(
            StockPool,
            func.count(StockPoolItem.id).label("stock_count"),
        )
        .outerjoin(StockPoolItem, StockPool.id == StockPoolItem.pool_id)
        .group_by(StockPool.id)
        .order_by(StockPool.created_at.desc())
    )
    result = await db.execute(stmt)
    return [
        {
            "id": row.StockPool.id,
            "name": row.StockPool.name,
            "description": row.StockPool.description,
            "stock_count": row.stock_count,
            "created_at": row.StockPool.created_at.isoformat(),
        }
        for row in result.all()
    ]


async def get_pool_detail(db: AsyncSession, pool_id: int) -> dict | None:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return None

    stmt = (
        select(StockPoolItem, StockBasic.name, StockBasic.symbol)
        .join(StockBasic, StockPoolItem.ts_code == StockBasic.ts_code, isouter=True)
        .where(StockPoolItem.pool_id == pool_id)
        .order_by(StockPoolItem.added_at.desc())
    )
    result = await db.execute(stmt)
    items = result.all()

    stocks = []
    for row in items:
        item = row.StockPoolItem
        candle_stmt = (
            select(DailyCandle)
            .where(DailyCandle.ts_code == item.ts_code)
            .order_by(DailyCandle.trade_date.desc())
            .limit(2)
        )
        candle_result = await db.execute(candle_stmt)
        candles = candle_result.scalars().all()

        if candles:
            latest = candles[0]
            prev_close = float(candles[1].close) if len(candles) > 1 else None
            change_pct = (
                round((float(latest.close) - prev_close) / prev_close * 100, 2)
                if prev_close
                else None
            )
            stocks.append({
                "ts_code": item.ts_code,
                "name": row.name or "",
                "symbol": row.symbol or "",
                "close": round(float(latest.close), 2),
                "change_pct": change_pct,
                "volume": latest.vol,
                "turnover_rate": None,
                "trade_date": latest.trade_date.isoformat(),
            })
        else:
            stocks.append({
                "ts_code": item.ts_code,
                "name": row.name or "",
                "symbol": row.symbol or "",
                "close": None,
                "change_pct": None,
                "volume": None,
                "turnover_rate": None,
                "trade_date": None,
            })

    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "stocks": stocks,
    }


async def update_pool(
    db: AsyncSession, pool_id: int, name: str | None = None, description: str | None = None
) -> dict | None:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return None
    if name is not None:
        pool.name = name
    if description is not None:
        pool.description = description
    pool.updated_at = datetime.now()
    await db.commit()
    await db.refresh(pool)
    return {
        "id": pool.id,
        "name": pool.name,
        "description": pool.description,
        "updated_at": pool.updated_at.isoformat(),
    }


async def delete_pool(db: AsyncSession, pool_id: int) -> bool:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return False
    await db.delete(pool)
    await db.commit()
    return True


async def add_stock_to_pool(
    db: AsyncSession, manager: DataSourceManager, pool_id: int, ts_code: str
) -> bool:
    pool = await db.get(StockPool, pool_id)
    if not pool:
        return False

    stmt = select(StockPoolItem).where(
        StockPoolItem.pool_id == pool_id, StockPoolItem.ts_code == ts_code
    )
    existing = await db.execute(stmt)
    if existing.scalar_one_or_none():
        return True

    db.add(StockPoolItem(pool_id=pool_id, ts_code=ts_code))
    await db.commit()

    candle_check = await db.execute(
        select(DailyCandle.ts_code)
        .where(DailyCandle.ts_code == ts_code)
        .limit(1)
    )
    if not candle_check.scalar_one_or_none():
        try:
            end = date.today()
            start = end - timedelta(days=365)
            df = await manager.fetch_daily(ts_code, start, end)
            if not df.empty:
                rows = df.to_dict("records")
                stmt_upsert = pg_insert(DailyCandle).values(rows)
                stmt_upsert = stmt_upsert.on_conflict_do_nothing(
                    index_elements=["ts_code", "trade_date"]
                )
                await db.execute(stmt_upsert)
                await db.commit()
        except Exception:
            pass

    return True


async def remove_stock_from_pool(db: AsyncSession, pool_id: int, ts_code: str) -> bool:
    result = await db.execute(
        delete(StockPoolItem).where(
            StockPoolItem.pool_id == pool_id, StockPoolItem.ts_code == ts_code
        )
    )
    await db.commit()
    return result.rowcount > 0
