from datetime import date

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db import get_db
from app.main import create_app
from app.models.schema import Base, StockBasic


@pytest.fixture
async def app_client():
    """App with in-memory SQLite for route testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    # Seed data
    async with session_factory() as session:
        session.add_all([
            StockBasic(ts_code="600519.SH", symbol="600519", name="贵州茅台", industry="白酒", is_active=True),
            StockBasic(ts_code="000858.SZ", symbol="000858", name="五粮液", industry="白酒", is_active=True),
        ])
        await session.commit()

    app = create_app()

    async def override_get_db():
        async with session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = override_get_db

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        yield client

    await engine.dispose()


class TestSearchRoute:
    @pytest.mark.asyncio
    async def test_search_returns_results(self, app_client):
        resp = await app_client.get("/api/quotes/search", params={"q": "茅台"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "贵州茅台"

    @pytest.mark.asyncio
    async def test_search_empty_query_returns_400(self, app_client):
        resp = await app_client.get("/api/quotes/search", params={"q": ""})
        assert resp.status_code in (400, 422)


class TestFavoritesRoute:
    @pytest.mark.asyncio
    async def test_add_and_list_favorites(self, app_client):
        resp = await app_client.post("/api/favorites/600519.SH")
        assert resp.status_code == 200

        resp = await app_client.get("/api/favorites")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 1
        assert data[0]["ts_code"] == "600519.SH"

    @pytest.mark.asyncio
    async def test_remove_favorite(self, app_client):
        await app_client.post("/api/favorites/600519.SH")
        resp = await app_client.delete("/api/favorites/600519.SH")
        assert resp.status_code == 200

        resp = await app_client.get("/api/favorites")
        assert len(resp.json()) == 0


class TestSearchHistoryRoute:
    @pytest.mark.asyncio
    async def test_get_history(self, app_client):
        resp = await app_client.get("/api/search-history")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
