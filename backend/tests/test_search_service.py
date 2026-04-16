from datetime import datetime

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.schema import Base, Favorite, SearchHistory, StockBasic
from app.services.search_service import (
    search_stocks,
    add_search_history,
    get_search_history,
    add_favorite,
    remove_favorite,
    get_favorites,
)


@pytest.fixture
async def mem_db():
    """In-memory SQLite for fast unit tests."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    async with session_factory() as session:
        # Seed test stocks
        session.add_all([
            StockBasic(ts_code="600519.SH", symbol="600519", name="贵州茅台", industry="白酒", is_active=True),
            StockBasic(ts_code="000858.SZ", symbol="000858", name="五粮液", industry="白酒", is_active=True),
            StockBasic(ts_code="600500.SH", symbol="600500", name="中化国际", industry="化工", is_active=True),
        ])
        await session.commit()
        yield session
    await engine.dispose()


class TestSearchStocks:
    @pytest.mark.asyncio
    async def test_search_by_code(self, mem_db):
        results = await search_stocks(mem_db, "600519", limit=10)
        assert len(results) == 1
        assert results[0]["ts_code"] == "600519.SH"

    @pytest.mark.asyncio
    async def test_search_by_name(self, mem_db):
        results = await search_stocks(mem_db, "茅台", limit=10)
        assert len(results) == 1
        assert results[0]["name"] == "贵州茅台"

    @pytest.mark.asyncio
    async def test_search_partial_code(self, mem_db):
        results = await search_stocks(mem_db, "6005", limit=10)
        assert len(results) == 2  # 600519 and 600500


class TestSearchHistory:
    @pytest.mark.asyncio
    async def test_add_and_get_history(self, mem_db):
        await add_search_history(mem_db, "600519.SH")
        await add_search_history(mem_db, "000858.SZ")
        history = await get_search_history(mem_db, limit=20)

        assert len(history) == 2
        assert history[0]["ts_code"] == "000858.SZ"  # most recent first


class TestFavorites:
    @pytest.mark.asyncio
    async def test_add_and_get_favorite(self, mem_db):
        await add_favorite(mem_db, "600519.SH")
        favorites = await get_favorites(mem_db)

        assert len(favorites) == 1
        assert favorites[0]["ts_code"] == "600519.SH"

    @pytest.mark.asyncio
    async def test_remove_favorite(self, mem_db):
        await add_favorite(mem_db, "600519.SH")
        await remove_favorite(mem_db, "600519.SH")
        favorites = await get_favorites(mem_db)

        assert len(favorites) == 0

    @pytest.mark.asyncio
    async def test_add_duplicate_favorite_is_idempotent(self, mem_db):
        await add_favorite(mem_db, "600519.SH")
        await add_favorite(mem_db, "600519.SH")  # should not raise
        favorites = await get_favorites(mem_db)

        assert len(favorites) == 1
