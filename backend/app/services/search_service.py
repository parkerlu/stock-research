from datetime import datetime

from sqlalchemy import delete, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import Favorite, SearchHistory, StockBasic


async def search_stocks(db: AsyncSession, query: str, limit: int = 10) -> list[dict]:
    pattern = f"%{query}%"
    stmt = (
        select(StockBasic)
        .where(
            StockBasic.is_active == True,  # noqa: E712
            or_(
                StockBasic.ts_code.like(pattern),
                StockBasic.symbol.like(pattern),
                StockBasic.name.like(pattern),
            ),
        )
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [
        {
            "ts_code": s.ts_code,
            "symbol": s.symbol,
            "name": s.name,
            "industry": s.industry,
        }
        for s in result.scalars().all()
    ]


async def add_search_history(db: AsyncSession, ts_code: str) -> None:
    db.add(SearchHistory(ts_code=ts_code, searched_at=datetime.now()))
    await db.commit()


async def get_search_history(db: AsyncSession, limit: int = 20) -> list[dict]:
    stmt = (
        select(SearchHistory, StockBasic.name)
        .join(StockBasic, SearchHistory.ts_code == StockBasic.ts_code, isouter=True)
        .order_by(SearchHistory.searched_at.desc())
        .limit(limit)
    )
    result = await db.execute(stmt)
    return [
        {
            "ts_code": row.SearchHistory.ts_code,
            "name": row.name or "",
            "searched_at": row.SearchHistory.searched_at.isoformat(),
        }
        for row in result.all()
    ]


async def add_favorite(db: AsyncSession, ts_code: str) -> None:
    existing = await db.execute(select(Favorite).where(Favorite.ts_code == ts_code))
    if existing.scalar_one_or_none():
        return
    db.add(Favorite(ts_code=ts_code, created_at=datetime.now()))
    await db.commit()


async def remove_favorite(db: AsyncSession, ts_code: str) -> None:
    await db.execute(delete(Favorite).where(Favorite.ts_code == ts_code))
    await db.commit()


async def get_favorites(db: AsyncSession) -> list[dict]:
    stmt = (
        select(Favorite, StockBasic.name)
        .join(StockBasic, Favorite.ts_code == StockBasic.ts_code, isouter=True)
        .order_by(Favorite.created_at.desc())
    )
    result = await db.execute(stmt)
    return [
        {
            "ts_code": row.Favorite.ts_code,
            "name": row.name or "",
        }
        for row in result.all()
    ]
