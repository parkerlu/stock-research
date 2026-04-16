# Quote Module Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the market quote module — stock search, K-line charts with technical indicators, drawing tools, and multi-timeframe linked view.

**Architecture:** React frontend with KLineChart for charting, Zustand for state. FastAPI backend serving OHLCV data from PostgreSQL cache, sourced from TuShare Pro (primary) with AKShare fallback. Weekly/monthly candles aggregated from daily data server-side.

**Tech Stack:** React 18 + TypeScript + Vite + KLineChart + Zustand | FastAPI + SQLAlchemy 2.0 + Alembic + PostgreSQL | TuShare Pro + AKShare | Docker Compose

**Spec:** `docs/superpowers/specs/2026-04-16-quote-module-design.md`

---

## File Structure

```
stock/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py                     # FastAPI app factory, CORS, router mounting
│   │   ├── config.py                   # Settings via pydantic-settings (DB URL, TuShare token)
│   │   ├── db.py                       # async engine, sessionmaker, get_db dependency
│   │   ├── routers/
│   │   │   ├── __init__.py
│   │   │   └── quotes.py              # All /api/quotes/*, /api/favorites, /api/search-history routes
│   │   ├── services/
│   │   │   ├── __init__.py
│   │   │   ├── quote_service.py        # candle fetch + cache + weekly/monthly aggregation
│   │   │   └── search_service.py       # search, history, favorites CRUD
│   │   ├── datasources/
│   │   │   ├── __init__.py
│   │   │   ├── base.py                 # DataProvider abstract base
│   │   │   ├── tushare_provider.py     # TuShare daily + adj_factor + stock_basic
│   │   │   ├── akshare_provider.py     # AKShare fallback
│   │   │   └── manager.py             # DataSourceManager: try primary, fallback secondary
│   │   └── models/
│   │       ├── __init__.py
│   │       └── schema.py              # SQLAlchemy ORM: StockBasic, DailyCandle, SearchHistory, Favorite
│   ├── alembic/
│   │   ├── env.py
│   │   └── versions/                   # migration files
│   ├── alembic.ini
│   ├── tests/
│   │   ├── conftest.py                 # fixtures: test DB, async client, mock providers
│   │   ├── test_datasources.py         # DataSourceManager fallback logic
│   │   ├── test_quote_service.py       # aggregation, caching
│   │   ├── test_search_service.py      # search, history, favorites
│   │   └── test_routes.py             # API endpoint integration tests
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── main.tsx                    # React entry
│   │   ├── App.tsx                     # Router shell
│   │   ├── pages/
│   │   │   └── QuotePage.tsx           # Main page: SearchPanel + ChartArea
│   │   ├── components/
│   │   │   ├── SearchPanel/
│   │   │   │   ├── SearchBar.tsx       # Debounced autocomplete input
│   │   │   │   ├── RecentList.tsx      # Recent search history
│   │   │   │   ├── FavoriteList.tsx    # Starred favorites with price
│   │   │   │   ├── SnapshotCard.tsx    # OHLCV snapshot card
│   │   │   │   └── index.tsx           # Panel container with tab switching
│   │   │   └── ChartArea/
│   │   │       ├── MainChart.tsx       # Single KLineChart instance
│   │   │       ├── Toolbar.tsx         # Timeframe, indicators, drawing tools
│   │   │       ├── LinkedView.tsx      # Three linked KLineChart instances
│   │   │       └── index.tsx           # Container switching single/linked mode
│   │   ├── stores/
│   │   │   └── quoteStore.ts           # Zustand: current symbol, timeframe, favorites, etc.
│   │   ├── api/
│   │   │   └── quotes.ts              # fetch wrappers for all backend endpoints
│   │   └── types/
│   │       └── quote.ts               # Candle, Snapshot, StockInfo, etc.
│   ├── index.html
│   ├── package.json
│   ├── tsconfig.json
│   ├── tsconfig.app.json
│   └── vite.config.ts
├── docker-compose.yml                  # PostgreSQL 15
├── .env.example                        # TUSHARE_TOKEN, DATABASE_URL
└── .gitignore
```

---

### Task 1: Backend Project Scaffolding

**Files:**
- Create: `backend/pyproject.toml`
- Create: `backend/requirements.txt`
- Create: `backend/app/__init__.py`
- Create: `backend/app/main.py`
- Create: `backend/app/config.py`
- Create: `docker-compose.yml`
- Create: `.env.example`

- [ ] **Step 1: Create docker-compose.yml**

```yaml
# docker-compose.yml
services:
  db:
    image: postgres:15
    environment:
      POSTGRES_USER: stock
      POSTGRES_PASSWORD: __REDACTED_DB_PASSWORD__
      POSTGRES_DB: stock_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

- [ ] **Step 2: Create .env.example**

```bash
# .env.example
TUSHARE_TOKEN=your_tushare_token_here
DATABASE_URL=postgresql+asyncpg://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db
DATABASE_URL_SYNC=postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db
```

- [ ] **Step 3: Create backend/pyproject.toml**

```toml
[project]
name = "stock-backend"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 4: Create backend/requirements.txt**

```
fastapi==0.115.*
uvicorn[standard]==0.34.*
sqlalchemy[asyncio]==2.0.*
asyncpg==0.30.*
psycopg2-binary==2.9.*
alembic==1.15.*
pydantic-settings==2.8.*
tushare==1.4.*
akshare>=1.16
httpx==0.28.*
pytest==8.3.*
pytest-asyncio==0.25.*
python-dotenv==1.1.*
```

- [ ] **Step 5: Create backend/app/config.py**

```python
# backend/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    database_url: str = "postgresql+asyncpg://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db"
    database_url_sync: str = "postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db"
    tushare_token: str = ""

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
```

- [ ] **Step 6: Create backend/app/main.py**

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Quote API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
```

- [ ] **Step 7: Create backend/app/__init__.py**

Empty file.

- [ ] **Step 8: Start PostgreSQL and verify backend runs**

```bash
cd /Users/chunyuanlu/WebApp/stock
docker compose up -d
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
# Visit http://localhost:8000/api/health → {"status":"ok"}
```

- [ ] **Step 9: Commit**

```bash
git add docker-compose.yml .env.example backend/pyproject.toml backend/requirements.txt backend/app/__init__.py backend/app/config.py backend/app/main.py
git commit -m "feat: backend project scaffolding with FastAPI and PostgreSQL"
```

---

### Task 2: Database Models and Migration

**Files:**
- Create: `backend/app/db.py`
- Create: `backend/app/models/__init__.py`
- Create: `backend/app/models/schema.py`
- Create: `backend/alembic.ini`
- Create: `backend/alembic/env.py`

- [ ] **Step 1: Create backend/app/db.py**

```python
# backend/app/db.py
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

engine = create_async_engine(settings.database_url, echo=False)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session
```

- [ ] **Step 2: Create backend/app/models/schema.py**

```python
# backend/app/models/schema.py
from datetime import date, datetime

from sqlalchemy import Boolean, BigInteger, Date, DateTime, Index, Numeric, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class StockBasic(Base):
    __tablename__ = "stock_basic"

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    symbol: Mapped[str] = mapped_column(String(6), index=True)
    name: Mapped[str] = mapped_column(String(20), index=True)
    area: Mapped[str | None] = mapped_column(String(10))
    industry: Mapped[str | None] = mapped_column(String(20))
    market: Mapped[str | None] = mapped_column(String(10))
    list_date: Mapped[date | None] = mapped_column(Date)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class DailyCandle(Base):
    __tablename__ = "daily_candle"
    __table_args__ = (
        Index("ix_daily_candle_lookup", "ts_code", "trade_date"),
    )

    ts_code: Mapped[str] = mapped_column(String(12), primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    open: Mapped[float] = mapped_column(Numeric(12, 4))
    high: Mapped[float] = mapped_column(Numeric(12, 4))
    low: Mapped[float] = mapped_column(Numeric(12, 4))
    close: Mapped[float] = mapped_column(Numeric(12, 4))
    vol: Mapped[int] = mapped_column(BigInteger)
    amount: Mapped[float] = mapped_column(Numeric(18, 4))
    adj_factor: Mapped[float] = mapped_column(Numeric(12, 6), default=1.0)
    source: Mapped[str] = mapped_column(String(10), default="tushare")


class SearchHistory(Base):
    __tablename__ = "search_history"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(12), index=True)
    searched_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)


class Favorite(Base):
    __tablename__ = "favorite"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    ts_code: Mapped[str] = mapped_column(String(12), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
```

- [ ] **Step 3: Create backend/app/models/__init__.py**

```python
# backend/app/models/__init__.py
from app.models.schema import Base, DailyCandle, Favorite, SearchHistory, StockBasic

__all__ = ["Base", "DailyCandle", "Favorite", "SearchHistory", "StockBasic"]
```

- [ ] **Step 4: Initialize Alembic**

```bash
cd backend
alembic init alembic
```

- [ ] **Step 5: Edit backend/alembic.ini — set sqlalchemy.url**

Change the `sqlalchemy.url` line to:

```ini
sqlalchemy.url = postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db
```

- [ ] **Step 6: Edit backend/alembic/env.py — import models**

Replace the `target_metadata = None` line with:

```python
from app.models.schema import Base
target_metadata = Base.metadata
```

- [ ] **Step 7: Generate and run migration**

```bash
cd backend
alembic revision --autogenerate -m "create quote tables"
alembic upgrade head
```

- [ ] **Step 8: Verify tables exist**

```bash
docker exec -it stock-db-1 psql -U stock -d stock_db -c "\dt"
# Should list: stock_basic, daily_candle, search_history, favorite, alembic_version
```

- [ ] **Step 9: Commit**

```bash
git add backend/app/db.py backend/app/models/ backend/alembic.ini backend/alembic/
git commit -m "feat: database models and initial migration for quote module"
```

---

### Task 3: Data Provider Base and TuShare Provider

**Files:**
- Create: `backend/app/datasources/__init__.py`
- Create: `backend/app/datasources/base.py`
- Create: `backend/app/datasources/tushare_provider.py`
- Create: `backend/tests/__init__.py`
- Create: `backend/tests/conftest.py`
- Create: `backend/tests/test_datasources.py`

- [ ] **Step 1: Create backend/app/datasources/base.py**

```python
# backend/app/datasources/base.py
from abc import ABC, abstractmethod
from datetime import date

import pandas as pd


class DataProvider(ABC):
    """Abstract base for market data providers."""

    @abstractmethod
    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        """Fetch daily OHLCV data. Returns DataFrame with columns:
        ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
        """
        ...

    @abstractmethod
    async def fetch_stock_basic(self) -> pd.DataFrame:
        """Fetch all A-share stock basic info. Returns DataFrame with columns:
        ts_code, symbol, name, area, industry, market, list_date
        """
        ...

    @abstractmethod
    async def fetch_snapshot(self, ts_code: str) -> dict | None:
        """Fetch realtime snapshot for a single stock. Returns dict with keys:
        symbol, name, price, change, change_pct, open, high, low, vol, amount, turnover
        or None if unavailable.
        """
        ...
```

- [ ] **Step 2: Write failing test for TuShareProvider.fetch_daily**

```python
# backend/tests/conftest.py
import pytest


@pytest.fixture
def tushare_token():
    """Read token from .env or skip if not available."""
    import os
    from dotenv import load_dotenv
    load_dotenv()
    token = os.getenv("TUSHARE_TOKEN", "")
    if not token:
        pytest.skip("TUSHARE_TOKEN not set")
    return token
```

```python
# backend/tests/test_datasources.py
from datetime import date

import pytest

from app.datasources.tushare_provider import TuShareProvider


class TestTuShareProvider:
    @pytest.mark.asyncio
    async def test_fetch_daily_returns_dataframe(self, tushare_token):
        provider = TuShareProvider(token=tushare_token)
        df = await provider.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

        assert not df.empty
        assert set(df.columns) >= {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"}
        assert (df["ts_code"] == "000001.SZ").all()

    @pytest.mark.asyncio
    async def test_fetch_stock_basic_returns_dataframe(self, tushare_token):
        provider = TuShareProvider(token=tushare_token)
        df = await provider.fetch_stock_basic()

        assert not df.empty
        assert "ts_code" in df.columns
        assert "name" in df.columns
        assert len(df) > 3000  # A-share has 5000+ stocks
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_datasources.py::TestTuShareProvider -v
# Expected: FAIL — ImportError (tushare_provider.py doesn't exist)
```

- [ ] **Step 4: Implement TuShareProvider**

```python
# backend/app/datasources/tushare_provider.py
import asyncio
from datetime import date
from functools import partial

import pandas as pd
import tushare as ts

from app.datasources.base import DataProvider


class TuShareProvider(DataProvider):
    def __init__(self, token: str):
        self._api = ts.pro_api(token)

    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        # Fetch daily bars
        daily = await loop.run_in_executor(
            None,
            partial(
                self._api.daily,
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            ),
        )
        # Fetch adj factors for the same range
        adj = await loop.run_in_executor(
            None,
            partial(
                self._api.adj_factor,
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            ),
        )
        if daily.empty:
            return pd.DataFrame()

        daily["trade_date"] = pd.to_datetime(daily["trade_date"]).dt.date
        if not adj.empty:
            adj["trade_date"] = pd.to_datetime(adj["trade_date"]).dt.date
            daily = daily.merge(adj[["trade_date", "adj_factor"]], on="trade_date", how="left")
        if "adj_factor" not in daily.columns:
            daily["adj_factor"] = 1.0
        daily["adj_factor"] = daily["adj_factor"].fillna(1.0)

        return daily[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]]

    async def fetch_stock_basic(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            partial(
                self._api.stock_basic,
                exchange="",
                list_status="L",
                fields="ts_code,symbol,name,area,industry,market,list_date",
            ),
        )
        if not df.empty:
            df["list_date"] = pd.to_datetime(df["list_date"]).dt.date
        return df

    async def fetch_snapshot(self, ts_code: str) -> dict | None:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            partial(
                self._api.daily,
                ts_code=ts_code,
                limit=1,
            ),
        )
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "symbol": ts_code,
            "name": "",  # filled by caller from stock_basic
            "price": float(row["close"]),
            "change": float(row["change"]),
            "change_pct": float(row["pct_chg"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "vol": int(row["vol"]),
            "amount": float(row["amount"]),
            "turnover": 0.0,  # not available in daily endpoint
        }
```

- [ ] **Step 5: Create __init__.py**

```python
# backend/app/datasources/__init__.py
```

```python
# backend/tests/__init__.py
```

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_datasources.py::TestTuShareProvider -v
# Expected: PASS (requires valid TUSHARE_TOKEN in .env)
```

- [ ] **Step 7: Commit**

```bash
git add backend/app/datasources/ backend/tests/
git commit -m "feat: TuShare data provider with daily, stock_basic, snapshot"
```

---

### Task 4: AKShare Provider

**Files:**
- Create: `backend/app/datasources/akshare_provider.py`
- Modify: `backend/tests/test_datasources.py`

- [ ] **Step 1: Write failing test for AKShareProvider**

Append to `backend/tests/test_datasources.py`:

```python
from app.datasources.akshare_provider import AKShareProvider


class TestAKShareProvider:
    @pytest.mark.asyncio
    async def test_fetch_daily_returns_dataframe(self):
        provider = AKShareProvider()
        df = await provider.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

        assert not df.empty
        assert set(df.columns) >= {"ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"}

    @pytest.mark.asyncio
    async def test_fetch_stock_basic_returns_dataframe(self):
        provider = AKShareProvider()
        df = await provider.fetch_stock_basic()

        assert not df.empty
        assert "ts_code" in df.columns
        assert "name" in df.columns
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_datasources.py::TestAKShareProvider -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement AKShareProvider**

```python
# backend/app/datasources/akshare_provider.py
import asyncio
from datetime import date
from functools import partial

import akshare as ak
import pandas as pd

from app.datasources.base import DataProvider


def _ts_code_to_ak_symbol(ts_code: str) -> str:
    """Convert '000001.SZ' to '000001'."""
    return ts_code.split(".")[0]


def _ak_symbol_to_ts_code(symbol: str, market: str) -> str:
    """Convert symbol + market hint to ts_code."""
    if market in ("深A", "深B", "中小板", "创业板"):
        return f"{symbol}.SZ"
    return f"{symbol}.SH"


class AKShareProvider(DataProvider):
    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        symbol = _ts_code_to_ak_symbol(ts_code)
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(
            None,
            partial(
                ak.stock_zh_a_hist,
                symbol=symbol,
                period="daily",
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
                adjust="qfq",
            ),
        )
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={
            "日期": "trade_date",
            "开盘": "open",
            "最高": "high",
            "最低": "low",
            "收盘": "close",
            "成交量": "vol",
            "成交额": "amount",
        })
        df["ts_code"] = ts_code
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        df["adj_factor"] = 1.0  # AKShare qfq data has adjustment baked in
        df["amount"] = df["amount"] / 1000  # convert to 千元 to match TuShare

        return df[["ts_code", "trade_date", "open", "high", "low", "close", "vol", "amount", "adj_factor"]]

    async def fetch_stock_basic(self) -> pd.DataFrame:
        loop = asyncio.get_event_loop()
        df = await loop.run_in_executor(None, ak.stock_info_a_code_name)
        if df.empty:
            return pd.DataFrame()

        df = df.rename(columns={"code": "symbol", "name": "name"})
        df["ts_code"] = df["symbol"].apply(
            lambda s: f"{s}.SZ" if s.startswith(("0", "3")) else f"{s}.SH"
        )
        df["area"] = None
        df["industry"] = None
        df["market"] = None
        df["list_date"] = None

        return df[["ts_code", "symbol", "name", "area", "industry", "market", "list_date"]]

    async def fetch_snapshot(self, ts_code: str) -> dict | None:
        symbol = _ts_code_to_ak_symbol(ts_code)
        loop = asyncio.get_event_loop()
        try:
            df = await loop.run_in_executor(
                None,
                partial(ak.stock_zh_a_spot_em),
            )
        except Exception:
            return None

        row = df[df["代码"] == symbol]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "symbol": ts_code,
            "name": str(r.get("名称", "")),
            "price": float(r.get("最新价", 0)),
            "change": float(r.get("涨跌额", 0)),
            "change_pct": float(r.get("涨跌幅", 0)),
            "open": float(r.get("今开", 0)),
            "high": float(r.get("最高", 0)),
            "low": float(r.get("最低", 0)),
            "vol": int(r.get("成交量", 0)),
            "amount": float(r.get("成交额", 0)),
            "turnover": float(r.get("换手率", 0)),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_datasources.py::TestAKShareProvider -v
# Expected: PASS
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/datasources/akshare_provider.py backend/tests/test_datasources.py
git commit -m "feat: AKShare fallback data provider"
```

---

### Task 5: DataSourceManager (Primary/Fallback Switching)

**Files:**
- Create: `backend/app/datasources/manager.py`
- Modify: `backend/tests/test_datasources.py`

- [ ] **Step 1: Write failing test for DataSourceManager fallback logic**

Append to `backend/tests/test_datasources.py`:

```python
from unittest.mock import AsyncMock

from app.datasources.base import DataProvider
from app.datasources.manager import DataSourceManager


class TestDataSourceManager:
    @pytest.mark.asyncio
    async def test_uses_primary_when_available(self):
        primary = AsyncMock(spec=DataProvider)
        fallback = AsyncMock(spec=DataProvider)
        expected = pd.DataFrame({"ts_code": ["000001.SZ"]})
        primary.fetch_daily.return_value = expected

        manager = DataSourceManager(primary=primary, fallback=fallback)
        result = await manager.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

        assert result.equals(expected)
        primary.fetch_daily.assert_called_once()
        fallback.fetch_daily.assert_not_called()

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_fails(self):
        primary = AsyncMock(spec=DataProvider)
        fallback = AsyncMock(spec=DataProvider)
        primary.fetch_daily.side_effect = Exception("TuShare down")
        expected = pd.DataFrame({"ts_code": ["000001.SZ"]})
        fallback.fetch_daily.return_value = expected

        manager = DataSourceManager(primary=primary, fallback=fallback)
        result = await manager.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

        assert result.equals(expected)
        fallback.fetch_daily.assert_called_once()

    @pytest.mark.asyncio
    async def test_raises_when_both_fail(self):
        primary = AsyncMock(spec=DataProvider)
        fallback = AsyncMock(spec=DataProvider)
        primary.fetch_daily.side_effect = Exception("TuShare down")
        fallback.fetch_daily.side_effect = Exception("AKShare down")

        manager = DataSourceManager(primary=primary, fallback=fallback)
        with pytest.raises(Exception, match="All data sources failed"):
            await manager.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

    @pytest.mark.asyncio
    async def test_falls_back_when_primary_returns_empty(self):
        primary = AsyncMock(spec=DataProvider)
        fallback = AsyncMock(spec=DataProvider)
        primary.fetch_daily.return_value = pd.DataFrame()
        expected = pd.DataFrame({"ts_code": ["000001.SZ"]})
        fallback.fetch_daily.return_value = expected

        manager = DataSourceManager(primary=primary, fallback=fallback)
        result = await manager.fetch_daily("000001.SZ", date(2025, 1, 2), date(2025, 1, 10))

        assert result.equals(expected)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_datasources.py::TestDataSourceManager -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement DataSourceManager**

```python
# backend/app/datasources/manager.py
import logging
from datetime import date

import pandas as pd

from app.datasources.base import DataProvider

logger = logging.getLogger(__name__)


class DataSourceManager:
    def __init__(self, primary: DataProvider, fallback: DataProvider):
        self._primary = primary
        self._fallback = fallback

    async def _try_with_fallback(self, method_name: str, *args, **kwargs):
        """Try primary, fall back to secondary. Raises if both fail."""
        for label, provider in [("primary", self._primary), ("fallback", self._fallback)]:
            try:
                result = await getattr(provider, method_name)(*args, **kwargs)
                if isinstance(result, pd.DataFrame) and result.empty:
                    logger.warning(f"{label} returned empty for {method_name}")
                    continue
                if result is None:
                    logger.warning(f"{label} returned None for {method_name}")
                    continue
                return result
            except Exception:
                logger.warning(f"{label} failed for {method_name}", exc_info=True)
                continue
        raise Exception(f"All data sources failed for {method_name}")

    async def fetch_daily(self, ts_code: str, start: date, end: date) -> pd.DataFrame:
        return await self._try_with_fallback("fetch_daily", ts_code, start, end)

    async def fetch_stock_basic(self) -> pd.DataFrame:
        return await self._try_with_fallback("fetch_stock_basic")

    async def fetch_snapshot(self, ts_code: str) -> dict | None:
        return await self._try_with_fallback("fetch_snapshot", ts_code)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_datasources.py::TestDataSourceManager -v
# Expected: 4 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/datasources/manager.py backend/tests/test_datasources.py
git commit -m "feat: DataSourceManager with primary/fallback switching"
```

---

### Task 6: Quote Service (Cache + Aggregation)

**Files:**
- Create: `backend/app/services/__init__.py`
- Create: `backend/app/services/quote_service.py`
- Create: `backend/tests/test_quote_service.py`

- [ ] **Step 1: Write failing test for weekly aggregation**

```python
# backend/tests/test_quote_service.py
from datetime import date

import pytest

from app.services.quote_service import aggregate_weekly, aggregate_monthly


class TestAggregation:
    def test_aggregate_weekly(self):
        """Mon 2025-01-06 to Fri 2025-01-10 is one week."""
        rows = [
            {"trade_date": date(2025, 1, 6), "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 7), "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0, "vol": 150, "amount": 1500.0},
            {"trade_date": date(2025, 1, 8), "open": 12.0, "high": 14.0, "low": 11.0, "close": 13.0, "vol": 200, "amount": 2000.0},
            {"trade_date": date(2025, 1, 9), "open": 13.0, "high": 15.0, "low": 10.5, "close": 14.0, "vol": 120, "amount": 1200.0},
            {"trade_date": date(2025, 1, 10), "open": 14.0, "high": 16.0, "low": 12.0, "close": 15.0, "vol": 180, "amount": 1800.0},
        ]
        result = aggregate_weekly(rows)

        assert len(result) == 1
        week = result[0]
        assert week["open"] == 10.0     # first day's open
        assert week["close"] == 15.0    # last day's close
        assert week["high"] == 16.0     # max high
        assert week["low"] == 9.0       # min low
        assert week["volume"] == 750    # sum vol
        assert week["amount"] == 7500.0 # sum amount

    def test_aggregate_weekly_partial_week(self):
        """A week with only 3 trading days (e.g. holiday week)."""
        rows = [
            {"trade_date": date(2025, 1, 27), "open": 10.0, "high": 12.0, "low": 9.5, "close": 11.5, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 28), "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "vol": 110, "amount": 1100.0},
            {"trade_date": date(2025, 1, 29), "open": 12.5, "high": 14.0, "low": 12.0, "close": 13.5, "vol": 120, "amount": 1200.0},
        ]
        result = aggregate_weekly(rows)

        assert len(result) == 1
        assert result[0]["open"] == 10.0
        assert result[0]["close"] == 13.5

    def test_aggregate_monthly(self):
        """Two months of data should produce two bars."""
        rows = [
            {"trade_date": date(2025, 1, 2), "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 31), "open": 11.0, "high": 15.0, "low": 10.0, "close": 14.0, "vol": 200, "amount": 2000.0},
            {"trade_date": date(2025, 2, 3), "open": 14.0, "high": 16.0, "low": 13.0, "close": 15.0, "vol": 150, "amount": 1500.0},
            {"trade_date": date(2025, 2, 28), "open": 15.0, "high": 18.0, "low": 14.0, "close": 17.0, "vol": 180, "amount": 1800.0},
        ]
        result = aggregate_monthly(rows)

        assert len(result) == 2
        jan = result[0]
        assert jan["open"] == 10.0
        assert jan["close"] == 14.0
        assert jan["high"] == 15.0
        assert jan["low"] == 9.0
        assert jan["volume"] == 300
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_quote_service.py -v
# Expected: FAIL — ImportError
```

- [ ] **Step 3: Implement quote_service.py**

```python
# backend/app/services/__init__.py
```

```python
# backend/app/services/quote_service.py
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
            # Re-query to get ORM objects
            result = await db.execute(
                select(DailyCandle)
                .where(and_(DailyCandle.ts_code == ts_code, DailyCandle.trade_date >= start, DailyCandle.trade_date <= end))
                .order_by(DailyCandle.trade_date)
            )
            cached = result.scalars().all()
    else:
        # 3. Check if we need incremental update (missing recent days)
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

    # 4. Convert to dicts and apply adj_factor for forward-adjusted prices
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

    # Daily: convert to output format
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_quote_service.py -v
# Expected: 3 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/ backend/tests/test_quote_service.py
git commit -m "feat: quote service with cache-through and weekly/monthly aggregation"
```

---

### Task 7: Search Service

**Files:**
- Create: `backend/app/services/search_service.py`
- Create: `backend/tests/test_search_service.py`

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_search_service.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pip install aiosqlite  # needed for SQLite async tests
pytest tests/test_search_service.py -v
# Expected: FAIL — ImportError
```

Add `aiosqlite==0.21.*` to `requirements.txt` as a test dependency.

- [ ] **Step 3: Implement search_service.py**

```python
# backend/app/services/search_service.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_search_service.py -v
# Expected: 6 PASSED
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/services/search_service.py backend/tests/test_search_service.py backend/requirements.txt
git commit -m "feat: search service with stock search, history, and favorites"
```

---

### Task 8: API Routes

**Files:**
- Create: `backend/app/routers/__init__.py`
- Create: `backend/app/routers/quotes.py`
- Modify: `backend/app/main.py` (mount router)
- Create: `backend/tests/test_routes.py`

- [ ] **Step 1: Write failing test for API routes**

```python
# backend/tests/test_routes.py
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
        assert resp.status_code == 400


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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd backend
pytest tests/test_routes.py -v
# Expected: FAIL — routes don't exist yet
```

- [ ] **Step 3: Create router**

```python
# backend/app/routers/__init__.py
```

```python
# backend/app/routers/quotes.py
from datetime import date

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.datasources.akshare_provider import AKShareProvider
from app.datasources.manager import DataSourceManager
from app.datasources.tushare_provider import TuShareProvider
from app.db import get_db
from app.services.quote_service import get_candles
from app.services.search_service import (
    add_favorite,
    add_search_history,
    get_favorites,
    get_search_history,
    remove_favorite,
    search_stocks,
)

router = APIRouter(prefix="/api")

_manager: DataSourceManager | None = None


def get_manager() -> DataSourceManager:
    global _manager
    if _manager is None:
        primary = TuShareProvider(token=settings.tushare_token)
        fallback = AKShareProvider()
        _manager = DataSourceManager(primary=primary, fallback=fallback)
    return _manager


@router.get("/quotes/search")
async def search(
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=50),
    db: AsyncSession = Depends(get_db),
):
    if not q.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty")
    return await search_stocks(db, q.strip(), limit)


@router.get("/quotes/{symbol}/candles")
async def candles(
    symbol: str,
    tf: str = Query("1d", pattern="^(1d|1w|1m)$"),
    start: date = Query(alias="from", default=None),
    end: date = Query(alias="to", default=None),
    db: AsyncSession = Depends(get_db),
    manager: DataSourceManager = Depends(get_manager),
):
    if start is None:
        start = date(date.today().year - 1, date.today().month, date.today().day)
    if end is None:
        end = date.today()

    # Record search history
    await add_search_history(db, symbol)

    try:
        data = await get_candles(db, manager, symbol, tf, start, end)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    return {"symbol": symbol, "tf": tf, "candles": data}


@router.get("/quotes/{symbol}/snapshot")
async def snapshot(
    symbol: str,
    manager: DataSourceManager = Depends(get_manager),
):
    try:
        data = await manager.fetch_snapshot(symbol)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    if data is None:
        raise HTTPException(status_code=404, detail="Snapshot not available")
    return data


@router.get("/search-history")
async def history(
    limit: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    return await get_search_history(db, limit)


@router.post("/favorites/{ts_code}")
async def add_fav(ts_code: str, db: AsyncSession = Depends(get_db)):
    await add_favorite(db, ts_code)
    return {"ok": True}


@router.delete("/favorites/{ts_code}")
async def remove_fav(ts_code: str, db: AsyncSession = Depends(get_db)):
    await remove_favorite(db, ts_code)
    return {"ok": True}


@router.get("/favorites")
async def list_favs(db: AsyncSession = Depends(get_db)):
    return await get_favorites(db)
```

- [ ] **Step 4: Mount router in main.py**

Replace `backend/app/main.py`:

```python
# backend/app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.quotes import router as quotes_router


def create_app() -> FastAPI:
    app = FastAPI(title="Stock Quote API", version="0.1.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173"],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(quotes_router)

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    return app


app = create_app()
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd backend
pytest tests/test_routes.py -v
# Expected: 5 PASSED
```

- [ ] **Step 6: Commit**

```bash
git add backend/app/routers/ backend/app/main.py backend/tests/test_routes.py
git commit -m "feat: API routes for quotes, search, favorites, and history"
```

---

### Task 9: Stock Basic Data Sync Command

**Files:**
- Create: `backend/app/commands/sync_stocks.py`

This one-off command pre-loads `stock_basic` from TuShare so search works immediately.

- [ ] **Step 1: Create the sync script**

```python
# backend/app/commands/sync_stocks.py
"""One-off script to sync stock_basic from data source into PostgreSQL."""
import asyncio

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings
from app.datasources.akshare_provider import AKShareProvider
from app.datasources.manager import DataSourceManager
from app.datasources.tushare_provider import TuShareProvider
from app.models.schema import StockBasic


async def sync():
    engine = create_async_engine(settings.database_url)
    session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    primary = TuShareProvider(token=settings.tushare_token)
    fallback = AKShareProvider()
    manager = DataSourceManager(primary=primary, fallback=fallback)

    print("Fetching stock_basic from data source...")
    df = await manager.fetch_stock_basic()
    print(f"Got {len(df)} stocks. Upserting into database...")

    async with session_factory() as session:
        for _, row in df.iterrows():
            stock = StockBasic(
                ts_code=row["ts_code"],
                symbol=row["symbol"],
                name=row["name"],
                area=row.get("area"),
                industry=row.get("industry"),
                market=row.get("market"),
                list_date=row.get("list_date"),
                is_active=True,
            )
            await session.merge(stock)
        await session.commit()

    print("Done.")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(sync())
```

- [ ] **Step 2: Create `__init__.py`**

```python
# backend/app/commands/__init__.py
```

- [ ] **Step 3: Test the sync command**

```bash
cd backend
python -m app.commands.sync_stocks
# Expected: "Fetching stock_basic... Got 5000+ stocks. Upserting... Done."
```

- [ ] **Step 4: Commit**

```bash
git add backend/app/commands/
git commit -m "feat: stock_basic sync command for pre-loading search data"
```

---

### Task 10: Frontend Scaffolding

**Files:**
- Create: `frontend/` (via Vite)
- Create: `frontend/src/types/quote.ts`
- Create: `frontend/src/api/quotes.ts`

- [ ] **Step 1: Create Vite React TypeScript project**

```bash
cd /Users/chunyuanlu/WebApp/stock
npm create vite@latest frontend -- --template react-ts
cd frontend
npm install
```

- [ ] **Step 2: Install dependencies**

```bash
cd frontend
npm install klinecharts zustand
npm install -D @types/node
```

- [ ] **Step 3: Configure Vite proxy**

Replace `frontend/vite.config.ts`:

```typescript
// frontend/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
```

- [ ] **Step 4: Create TypeScript types**

```typescript
// frontend/src/types/quote.ts
export interface StockInfo {
  ts_code: string;
  symbol: string;
  name: string;
  industry: string | null;
}

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  amount: number;
}

export interface CandleResponse {
  symbol: string;
  tf: string;
  candles: Candle[];
}

export interface Snapshot {
  symbol: string;
  name: string;
  price: number;
  change: number;
  change_pct: number;
  open: number;
  high: number;
  low: number;
  vol: number;
  amount: number;
  turnover: number;
}

export interface SearchHistoryItem {
  ts_code: string;
  name: string;
  searched_at: string;
}

export interface FavoriteItem {
  ts_code: string;
  name: string;
}

export type Timeframe = "1d" | "1w" | "1m";
```

- [ ] **Step 5: Create API client**

```typescript
// frontend/src/api/quotes.ts
import type {
  CandleResponse,
  FavoriteItem,
  SearchHistoryItem,
  Snapshot,
  StockInfo,
  Timeframe,
} from "../types/quote";

const BASE = "/api";

async function json<T>(url: string, init?: RequestInit): Promise<T> {
  const res = await fetch(url, init);
  if (!res.ok) {
    throw new Error(`API error: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export async function searchStocks(
  q: string,
  limit = 10
): Promise<StockInfo[]> {
  return json(`${BASE}/quotes/search?q=${encodeURIComponent(q)}&limit=${limit}`);
}

export async function getCandles(
  symbol: string,
  tf: Timeframe = "1d",
  from?: string,
  to?: string
): Promise<CandleResponse> {
  const params = new URLSearchParams({ tf });
  if (from) params.set("from", from);
  if (to) params.set("to", to);
  return json(`${BASE}/quotes/${symbol}/candles?${params}`);
}

export async function getSnapshot(symbol: string): Promise<Snapshot> {
  return json(`${BASE}/quotes/${symbol}/snapshot`);
}

export async function getSearchHistory(
  limit = 20
): Promise<SearchHistoryItem[]> {
  return json(`${BASE}/search-history?limit=${limit}`);
}

export async function addFavorite(tsCode: string): Promise<void> {
  await json(`${BASE}/favorites/${tsCode}`, { method: "POST" });
}

export async function removeFavorite(tsCode: string): Promise<void> {
  await json(`${BASE}/favorites/${tsCode}`, { method: "DELETE" });
}

export async function getFavorites(): Promise<FavoriteItem[]> {
  return json(`${BASE}/favorites`);
}
```

- [ ] **Step 6: Verify frontend starts**

```bash
cd frontend
npm run dev
# Visit http://localhost:5173 — should see Vite + React default page
```

- [ ] **Step 7: Commit**

```bash
git add frontend/
git commit -m "feat: frontend scaffolding with Vite, React, KLineChart, types, and API client"
```

---

### Task 11: Zustand Store

**Files:**
- Create: `frontend/src/stores/quoteStore.ts`

- [ ] **Step 1: Create the store**

```typescript
// frontend/src/stores/quoteStore.ts
import { create } from "zustand";
import type {
  FavoriteItem,
  SearchHistoryItem,
  Snapshot,
  Timeframe,
} from "../types/quote";
import {
  addFavorite as apiFav,
  getFavorites,
  getSearchHistory,
  getSnapshot,
  removeFavorite as apiUnfav,
} from "../api/quotes";

interface QuoteState {
  // Current stock
  currentSymbol: string;
  currentName: string;
  setCurrentStock: (symbol: string, name: string) => void;

  // Timeframe
  timeframe: Timeframe;
  setTimeframe: (tf: Timeframe) => void;

  // Linked view mode
  linkedMode: boolean;
  toggleLinkedMode: () => void;

  // Snapshot
  snapshot: Snapshot | null;
  fetchSnapshot: () => Promise<void>;

  // Search history
  history: SearchHistoryItem[];
  fetchHistory: () => Promise<void>;

  // Favorites
  favorites: FavoriteItem[];
  fetchFavorites: () => Promise<void>;
  addFavorite: (tsCode: string) => Promise<void>;
  removeFavorite: (tsCode: string) => Promise<void>;
}

export const useQuoteStore = create<QuoteState>((set, get) => ({
  currentSymbol: "",
  currentName: "",
  setCurrentStock: (symbol, name) => {
    set({ currentSymbol: symbol, currentName: name });
    get().fetchSnapshot();
    get().fetchHistory();
  },

  timeframe: "1d",
  setTimeframe: (tf) => set({ timeframe: tf }),

  linkedMode: false,
  toggleLinkedMode: () => set((s) => ({ linkedMode: !s.linkedMode })),

  snapshot: null,
  fetchSnapshot: async () => {
    const { currentSymbol } = get();
    if (!currentSymbol) return;
    try {
      const data = await getSnapshot(currentSymbol);
      set({ snapshot: data });
    } catch {
      set({ snapshot: null });
    }
  },

  history: [],
  fetchHistory: async () => {
    try {
      const data = await getSearchHistory();
      set({ history: data });
    } catch {
      /* ignore */
    }
  },

  favorites: [],
  fetchFavorites: async () => {
    try {
      const data = await getFavorites();
      set({ favorites: data });
    } catch {
      /* ignore */
    }
  },
  addFavorite: async (tsCode) => {
    await apiFav(tsCode);
    await get().fetchFavorites();
  },
  removeFavorite: async (tsCode) => {
    await apiUnfav(tsCode);
    await get().fetchFavorites();
  },
}));
```

- [ ] **Step 2: Verify build**

```bash
cd frontend
npx tsc --noEmit
# Expected: no errors
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/stores/quoteStore.ts
git commit -m "feat: Zustand store for quote module state management"
```

---

### Task 12: SearchPanel Components

**Files:**
- Create: `frontend/src/components/SearchPanel/SearchBar.tsx`
- Create: `frontend/src/components/SearchPanel/RecentList.tsx`
- Create: `frontend/src/components/SearchPanel/FavoriteList.tsx`
- Create: `frontend/src/components/SearchPanel/SnapshotCard.tsx`
- Create: `frontend/src/components/SearchPanel/index.tsx`

- [ ] **Step 1: Create SearchBar**

```tsx
// frontend/src/components/SearchPanel/SearchBar.tsx
import { useCallback, useEffect, useRef, useState } from "react";
import { searchStocks } from "../../api/quotes";
import type { StockInfo } from "../../types/quote";
import { useQuoteStore } from "../../stores/quoteStore";

export function SearchBar() {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<StockInfo[]>([]);
  const [open, setOpen] = useState(false);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 1) {
      setResults([]);
      return;
    }
    try {
      const data = await searchStocks(q);
      setResults(data);
      setOpen(true);
    } catch {
      setResults([]);
    }
  }, []);

  useEffect(() => {
    clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => doSearch(query), 300);
    return () => clearTimeout(timerRef.current);
  }, [query, doSearch]);

  const handleSelect = (stock: StockInfo) => {
    setCurrentStock(stock.ts_code, stock.name);
    setQuery("");
    setOpen(false);
  };

  return (
    <div className="search-bar">
      <input
        type="text"
        placeholder="输入代码或名称..."
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        onFocus={() => results.length > 0 && setOpen(true)}
        onBlur={() => setTimeout(() => setOpen(false), 200)}
      />
      {open && results.length > 0 && (
        <ul className="search-dropdown">
          {results.map((s) => (
            <li key={s.ts_code} onMouseDown={() => handleSelect(s)}>
              <span className="code">{s.symbol}</span>
              <span className="name">{s.name}</span>
              {s.industry && <span className="industry">{s.industry}</span>}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create RecentList**

```tsx
// frontend/src/components/SearchPanel/RecentList.tsx
import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";

export function RecentList() {
  const history = useQuoteStore((s) => s.history);
  const fetchHistory = useQuoteStore((s) => s.fetchHistory);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  return (
    <ul className="stock-list">
      {history.map((item, i) => (
        <li
          key={`${item.ts_code}-${i}`}
          onClick={() => setCurrentStock(item.ts_code, item.name)}
        >
          <span className="name">{item.name}</span>
          <span className="code">{item.ts_code}</span>
        </li>
      ))}
      {history.length === 0 && (
        <li className="empty">暂无搜索记录</li>
      )}
    </ul>
  );
}
```

- [ ] **Step 3: Create FavoriteList**

```tsx
// frontend/src/components/SearchPanel/FavoriteList.tsx
import { useEffect } from "react";
import { useQuoteStore } from "../../stores/quoteStore";

export function FavoriteList() {
  const favorites = useQuoteStore((s) => s.favorites);
  const fetchFavorites = useQuoteStore((s) => s.fetchFavorites);
  const setCurrentStock = useQuoteStore((s) => s.setCurrentStock);
  const removeFavorite = useQuoteStore((s) => s.removeFavorite);

  useEffect(() => {
    fetchFavorites();
  }, [fetchFavorites]);

  return (
    <ul className="stock-list">
      {favorites.map((item) => (
        <li key={item.ts_code}>
          <span
            className="name clickable"
            onClick={() => setCurrentStock(item.ts_code, item.name)}
          >
            {item.name}
          </span>
          <button
            className="remove-btn"
            onClick={() => removeFavorite(item.ts_code)}
            title="取消收藏"
          >
            ✕
          </button>
        </li>
      ))}
      {favorites.length === 0 && (
        <li className="empty">暂无收藏</li>
      )}
    </ul>
  );
}
```

- [ ] **Step 4: Create SnapshotCard**

```tsx
// frontend/src/components/SearchPanel/SnapshotCard.tsx
import { useQuoteStore } from "../../stores/quoteStore";

export function SnapshotCard() {
  const snapshot = useQuoteStore((s) => s.snapshot);
  const currentName = useQuoteStore((s) => s.currentName);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const addFavorite = useQuoteStore((s) => s.addFavorite);

  if (!snapshot || !currentSymbol) {
    return <div className="snapshot-card empty">选择一只股票查看行情</div>;
  }

  const isUp = snapshot.change >= 0;
  const colorClass = isUp ? "up" : "down";

  return (
    <div className="snapshot-card">
      <div className="snapshot-header">
        <div>
          <div className="stock-name">{currentName}</div>
          <div className="stock-code">{currentSymbol}</div>
        </div>
        <button
          className="fav-btn"
          onClick={() => addFavorite(currentSymbol)}
          title="收藏"
        >
          ⭐
        </button>
      </div>
      <div className={`snapshot-price ${colorClass}`}>
        {snapshot.price.toFixed(2)}
      </div>
      <div className={`snapshot-change ${colorClass}`}>
        {isUp ? "+" : ""}
        {snapshot.change.toFixed(2)} ({isUp ? "+" : ""}
        {snapshot.change_pct.toFixed(2)}%)
      </div>
      <div className="snapshot-grid">
        <div><span className="label">开</span><span>{snapshot.open.toFixed(2)}</span></div>
        <div><span className="label">高</span><span>{snapshot.high.toFixed(2)}</span></div>
        <div><span className="label">低</span><span>{snapshot.low.toFixed(2)}</span></div>
        <div><span className="label">量</span><span>{(snapshot.vol / 10000).toFixed(1)}万</span></div>
        <div><span className="label">额</span><span>{(snapshot.amount / 100000000).toFixed(1)}亿</span></div>
        <div><span className="label">换</span><span>{snapshot.turnover.toFixed(2)}%</span></div>
      </div>
    </div>
  );
}
```

- [ ] **Step 5: Create SearchPanel container**

```tsx
// frontend/src/components/SearchPanel/index.tsx
import { useState } from "react";
import { SearchBar } from "./SearchBar";
import { RecentList } from "./RecentList";
import { FavoriteList } from "./FavoriteList";
import { SnapshotCard } from "./SnapshotCard";

type Tab = "recent" | "favorites";

export function SearchPanel() {
  const [tab, setTab] = useState<Tab>("recent");

  return (
    <div className="search-panel">
      <SearchBar />
      <div className="panel-tabs">
        <button
          className={tab === "recent" ? "active" : ""}
          onClick={() => setTab("recent")}
        >
          最近查看
        </button>
        <button
          className={tab === "favorites" ? "active" : ""}
          onClick={() => setTab("favorites")}
        >
          我的收藏 ⭐
        </button>
      </div>
      <div className="panel-list">
        {tab === "recent" ? <RecentList /> : <FavoriteList />}
      </div>
      <div className="panel-divider" />
      <SnapshotCard />
    </div>
  );
}
```

- [ ] **Step 6: Verify build**

```bash
cd frontend
npx tsc --noEmit
# Expected: no errors
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/components/SearchPanel/
git commit -m "feat: SearchPanel components — search bar, recent, favorites, snapshot"
```

---

### Task 13: ChartArea — MainChart

**Files:**
- Create: `frontend/src/components/ChartArea/MainChart.tsx`

- [ ] **Step 1: Create MainChart with KLineChart**

```tsx
// frontend/src/components/ChartArea/MainChart.tsx
import { useEffect, useRef } from "react";
import { init, dispose, type Chart } from "klinecharts";
import { getCandles } from "../../api/quotes";
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";

interface Props {
  timeframe?: Timeframe;
  className?: string;
}

export function MainChart({ timeframe: tfOverride, className }: Props) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<Chart | null>(null);
  const currentSymbol = useQuoteStore((s) => s.currentSymbol);
  const storeTimeframe = useQuoteStore((s) => s.timeframe);
  const tf = tfOverride ?? storeTimeframe;

  useEffect(() => {
    if (!containerRef.current) return;
    const chart = init(containerRef.current, {
      styles: {
        grid: {
          horizontal: { color: "#1e1e30" },
          vertical: { color: "#1e1e30" },
        },
        candle: {
          priceMark: { last: { show: true } },
        },
      },
    });
    chartRef.current = chart ?? null;

    // Default indicators
    chart?.createIndicator("VOL", false, { id: "volume_pane" });

    return () => {
      if (containerRef.current) {
        dispose(containerRef.current);
      }
      chartRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!currentSymbol || !chartRef.current) return;
    let cancelled = false;

    (async () => {
      try {
        const resp = await getCandles(currentSymbol, tf);
        if (cancelled || !chartRef.current) return;
        chartRef.current.applyNewData(resp.candles);
      } catch (err) {
        console.error("Failed to load candles:", err);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [currentSymbol, tf]);

  return (
    <div
      ref={containerRef}
      className={`main-chart ${className ?? ""}`}
      style={{ width: "100%", height: "100%" }}
    />
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChartArea/MainChart.tsx
git commit -m "feat: MainChart component with KLineChart integration"
```

---

### Task 14: ChartArea — Toolbar

**Files:**
- Create: `frontend/src/components/ChartArea/Toolbar.tsx`

- [ ] **Step 1: Create Toolbar**

```tsx
// frontend/src/components/ChartArea/Toolbar.tsx
import { useQuoteStore } from "../../stores/quoteStore";
import type { Timeframe } from "../../types/quote";

const TIMEFRAMES: { label: string; value: Timeframe }[] = [
  { label: "日", value: "1d" },
  { label: "周", value: "1w" },
  { label: "月", value: "1m" },
];

const INDICATORS = [
  { group: "均线", items: ["MA", "EMA", "BOLL"] },
  { group: "趋势", items: ["MACD", "DMI", "SAR"] },
  { group: "摆动", items: ["KDJ", "RSI", "WR"] },
  { group: "量能", items: ["OBV"] },
];

const OVERLAYS = [
  { label: "趋势线", type: "segment" },
  { label: "水平线", type: "horizontalStraightLine" },
  { label: "垂直线", type: "verticalStraightLine" },
  { label: "平行通道", type: "parallelStraightLine" },
  { label: "斐波那契", type: "fibonacciLine" },
  { label: "矩形", type: "rect" },
  { label: "文字", type: "simpleAnnotation" },
  { label: "箭头", type: "arrow" },
];

// These are the indicators that overlay on the main candle pane
const MAIN_PANE_INDICATORS = new Set(["MA", "EMA", "BOLL", "SAR"]);

interface Props {
  activeIndicators: string[];
  onToggleIndicator: (name: string, isMainPane: boolean) => void;
  onSelectOverlay: (type: string) => void;
}

export function Toolbar({
  activeIndicators,
  onToggleIndicator,
  onSelectOverlay,
}: Props) {
  const timeframe = useQuoteStore((s) => s.timeframe);
  const setTimeframe = useQuoteStore((s) => s.setTimeframe);
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const toggleLinkedMode = useQuoteStore((s) => s.toggleLinkedMode);

  return (
    <div className="chart-toolbar">
      {/* Timeframe buttons */}
      <div className="toolbar-group">
        {TIMEFRAMES.map((tf) => (
          <button
            key={tf.value}
            className={timeframe === tf.value ? "active" : ""}
            onClick={() => setTimeframe(tf.value)}
          >
            {tf.label}
          </button>
        ))}
      </div>

      <div className="toolbar-divider" />

      {/* Indicator dropdown */}
      <div className="toolbar-group dropdown-container">
        <button className="dropdown-trigger">📊 指标 ▾</button>
        <div className="dropdown-menu indicator-menu">
          {INDICATORS.map((group) => (
            <div key={group.group} className="indicator-group">
              <div className="group-label">{group.group}</div>
              {group.items.map((name) => (
                <label key={name} className="indicator-item">
                  <input
                    type="checkbox"
                    checked={activeIndicators.includes(name)}
                    onChange={() =>
                      onToggleIndicator(name, MAIN_PANE_INDICATORS.has(name))
                    }
                  />
                  {name}
                </label>
              ))}
            </div>
          ))}
        </div>
      </div>

      {/* Active indicator tags */}
      <div className="toolbar-group indicator-tags">
        {activeIndicators.map((name) => (
          <span key={name} className="indicator-tag">
            {name}
          </span>
        ))}
      </div>

      <div className="toolbar-divider" />

      {/* Drawing tools */}
      <div className="toolbar-group dropdown-container">
        <button className="dropdown-trigger">📐 画线 ▾</button>
        <div className="dropdown-menu">
          {OVERLAYS.map((o) => (
            <button
              key={o.type}
              className="overlay-item"
              onClick={() => onSelectOverlay(o.type)}
            >
              {o.label}
            </button>
          ))}
        </div>
      </div>

      {/* Linked mode toggle */}
      <div className="toolbar-group" style={{ marginLeft: "auto" }}>
        <button
          className={linkedMode ? "active" : ""}
          onClick={toggleLinkedMode}
        >
          ⊞ 多周期联动
        </button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChartArea/Toolbar.tsx
git commit -m "feat: Toolbar with timeframe, indicators, drawing tools, linked mode"
```

---

### Task 15: ChartArea — LinkedView

**Files:**
- Create: `frontend/src/components/ChartArea/LinkedView.tsx`

- [ ] **Step 1: Create LinkedView**

```tsx
// frontend/src/components/ChartArea/LinkedView.tsx
import { MainChart } from "./MainChart";
import type { Timeframe } from "../../types/quote";

const LINKED_TFS: { label: string; tf: Timeframe }[] = [
  { label: "日线", tf: "1d" },
  { label: "周线", tf: "1w" },
  { label: "月线", tf: "1m" },
];

export function LinkedView() {
  return (
    <div className="linked-view">
      {LINKED_TFS.map(({ label, tf }) => (
        <div key={tf} className="linked-pane">
          <div className="linked-label">{label}</div>
          <MainChart timeframe={tf} className="linked-chart" />
        </div>
      ))}
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChartArea/LinkedView.tsx
git commit -m "feat: LinkedView for multi-timeframe side-by-side display"
```

---

### Task 16: ChartArea Container

**Files:**
- Create: `frontend/src/components/ChartArea/index.tsx`

- [ ] **Step 1: Create ChartArea container**

```tsx
// frontend/src/components/ChartArea/index.tsx
import { useCallback, useRef, useState } from "react";
import { type Chart } from "klinecharts";
import { useQuoteStore } from "../../stores/quoteStore";
import { MainChart } from "./MainChart";
import { Toolbar } from "./Toolbar";
import { LinkedView } from "./LinkedView";

// Indicators that go on the main candle pane (overlays)
const MAIN_PANE_INDICATORS = new Set(["MA", "EMA", "BOLL", "SAR"]);

export function ChartArea() {
  const linkedMode = useQuoteStore((s) => s.linkedMode);
  const [activeIndicators, setActiveIndicators] = useState<string[]>(["MA"]);
  const chartRef = useRef<Chart | null>(null);

  // Track created sub-pane indicator IDs so we can remove them
  const indicatorPaneIds = useRef<Record<string, string>>({});

  const handleToggleIndicator = useCallback(
    (name: string, isMainPane: boolean) => {
      setActiveIndicators((prev) => {
        const chart = chartRef.current;
        if (prev.includes(name)) {
          // Remove indicator
          if (chart) {
            if (isMainPane) {
              chart.removeIndicator("candle_pane", name);
            } else if (indicatorPaneIds.current[name]) {
              chart.removeIndicator(indicatorPaneIds.current[name], name);
              delete indicatorPaneIds.current[name];
            }
          }
          return prev.filter((n) => n !== name);
        } else {
          // Add indicator
          if (chart) {
            if (isMainPane) {
              chart.createIndicator(name, false, { id: "candle_pane" });
            } else {
              const paneId = chart.createIndicator(name, true);
              if (paneId) {
                indicatorPaneIds.current[name] = paneId;
              }
            }
          }
          return [...prev, name];
        }
      });
    },
    []
  );

  const handleSelectOverlay = useCallback((type: string) => {
    chartRef.current?.createOverlay(type);
  }, []);

  return (
    <div className="chart-area">
      <Toolbar
        activeIndicators={activeIndicators}
        onToggleIndicator={handleToggleIndicator}
        onSelectOverlay={handleSelectOverlay}
      />
      <div className="chart-body">
        {linkedMode ? (
          <LinkedView />
        ) : (
          <MainChart />
        )}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd frontend
npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
git add frontend/src/components/ChartArea/index.tsx
git commit -m "feat: ChartArea container with indicator/overlay management"
```

---

### Task 17: QuotePage and App Assembly

**Files:**
- Create: `frontend/src/pages/QuotePage.tsx`
- Modify: `frontend/src/App.tsx`
- Create: `frontend/src/App.css` (styles)

- [ ] **Step 1: Create QuotePage**

```tsx
// frontend/src/pages/QuotePage.tsx
import { SearchPanel } from "../components/SearchPanel";
import { ChartArea } from "../components/ChartArea";

export function QuotePage() {
  return (
    <div className="quote-page">
      <SearchPanel />
      <ChartArea />
    </div>
  );
}
```

- [ ] **Step 2: Replace App.tsx**

```tsx
// frontend/src/App.tsx
import { QuotePage } from "./pages/QuotePage";
import "./App.css";

function App() {
  return <QuotePage />;
}

export default App;
```

- [ ] **Step 3: Create App.css with dark theme**

```css
/* frontend/src/App.css */
:root {
  --bg-primary: #141420;
  --bg-secondary: #1a1a2e;
  --bg-tertiary: #252540;
  --border: #2a2a40;
  --text-primary: #e0e0e0;
  --text-secondary: #888;
  --text-muted: #555;
  --accent: #4cc9f0;
  --up: #4caf50;
  --down: #e94560;
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
  font-family: -apple-system, "PingFang SC", "Microsoft YaHei", sans-serif;
  background: var(--bg-primary);
  color: var(--text-primary);
  font-size: 13px;
}

/* Layout */
.quote-page {
  display: flex;
  height: 100vh;
  overflow: hidden;
}

/* SearchPanel */
.search-panel {
  width: 240px;
  background: var(--bg-secondary);
  border-right: 1px solid var(--border);
  display: flex;
  flex-direction: column;
  flex-shrink: 0;
  overflow-y: auto;
}

.search-bar { padding: 12px; position: relative; }
.search-bar input {
  width: 100%;
  background: var(--bg-tertiary);
  border: none;
  border-radius: 6px;
  padding: 8px 10px;
  color: var(--text-primary);
  font-size: 13px;
  outline: none;
}
.search-bar input::placeholder { color: var(--text-secondary); }

.search-dropdown {
  position: absolute;
  top: 100%;
  left: 12px;
  right: 12px;
  background: var(--bg-tertiary);
  border-radius: 6px;
  list-style: none;
  z-index: 10;
  max-height: 200px;
  overflow-y: auto;
}
.search-dropdown li {
  padding: 6px 10px;
  cursor: pointer;
  display: flex;
  gap: 8px;
  align-items: center;
}
.search-dropdown li:hover { background: var(--bg-secondary); }
.search-dropdown .code { color: var(--accent); }
.search-dropdown .name { color: var(--text-primary); }
.search-dropdown .industry { color: var(--text-muted); font-size: 11px; }

.panel-tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  padding: 0 12px;
}
.panel-tabs button {
  background: none;
  border: none;
  color: var(--text-muted);
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
  border-bottom: 2px solid transparent;
}
.panel-tabs button.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
}

.panel-list { flex: 0 0 auto; }

.stock-list {
  list-style: none;
  padding: 4px 12px;
}
.stock-list li {
  padding: 6px 8px;
  border-radius: 4px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.stock-list li:hover { background: var(--bg-tertiary); }
.stock-list li.empty { color: var(--text-muted); cursor: default; text-align: center; }
.stock-list .remove-btn {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 12px;
}

.panel-divider {
  border-top: 1px solid var(--border);
  margin: 8px 12px;
}

/* SnapshotCard */
.snapshot-card { padding: 0 12px 12px; }
.snapshot-card.empty { color: var(--text-muted); text-align: center; padding: 20px 12px; }

.snapshot-header {
  display: flex;
  justify-content: space-between;
  align-items: start;
  margin-bottom: 8px;
}
.stock-name { font-size: 14px; font-weight: 600; }
.stock-code { color: var(--text-muted); font-size: 11px; }
.fav-btn { background: none; border: none; cursor: pointer; font-size: 16px; }

.snapshot-price { font-size: 22px; font-weight: bold; text-align: center; margin: 4px 0; }
.snapshot-change { text-align: center; font-size: 12px; margin-bottom: 8px; }
.up { color: var(--up); }
.down { color: var(--down); }

.snapshot-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 4px;
}
.snapshot-grid > div {
  background: var(--bg-tertiary);
  padding: 4px 6px;
  border-radius: 3px;
  display: flex;
  justify-content: space-between;
  font-size: 11px;
}
.snapshot-grid .label { color: var(--text-muted); }

/* ChartArea */
.chart-area {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}

.chart-toolbar {
  display: flex;
  align-items: center;
  padding: 6px 16px;
  background: var(--bg-secondary);
  border-bottom: 1px solid var(--border);
  gap: 12px;
  flex-shrink: 0;
}

.toolbar-group { display: flex; align-items: center; gap: 4px; }
.toolbar-divider { width: 1px; height: 16px; background: var(--border); }

.toolbar-group button {
  background: none;
  border: none;
  color: var(--text-secondary);
  padding: 4px 10px;
  border-radius: 4px;
  cursor: pointer;
  font-size: 12px;
}
.toolbar-group button.active {
  background: var(--accent);
  color: #000;
  font-weight: 600;
}
.toolbar-group button:hover:not(.active) { background: var(--bg-tertiary); }

/* Dropdown */
.dropdown-container { position: relative; }
.dropdown-menu {
  display: none;
  position: absolute;
  top: 100%;
  left: 0;
  background: var(--bg-secondary);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 8px;
  z-index: 20;
  min-width: 160px;
}
.dropdown-container:hover .dropdown-menu { display: block; }

.indicator-group { margin-bottom: 6px; }
.group-label { color: var(--text-muted); font-size: 10px; text-transform: uppercase; margin-bottom: 2px; }
.indicator-item {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 3px 0;
  cursor: pointer;
  font-size: 12px;
}
.indicator-tag {
  background: var(--bg-tertiary);
  padding: 2px 8px;
  border-radius: 3px;
  font-size: 10px;
  color: var(--accent);
}

.overlay-item {
  display: block;
  width: 100%;
  text-align: left;
  background: none;
  border: none;
  color: var(--text-primary);
  padding: 4px 8px;
  border-radius: 3px;
  cursor: pointer;
  font-size: 12px;
}
.overlay-item:hover { background: var(--bg-tertiary); }

/* Chart body */
.chart-body { flex: 1; padding: 8px; min-height: 0; }
.main-chart { border-radius: 6px; overflow: hidden; }

/* Linked view */
.linked-view {
  display: flex;
  gap: 8px;
  height: 100%;
}
.linked-pane {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
}
.linked-label {
  color: var(--accent);
  font-size: 11px;
  padding: 4px 8px;
  border-bottom: 1px solid var(--border);
  background: var(--bg-secondary);
  border-radius: 6px 6px 0 0;
}
.linked-chart { flex: 1; border-radius: 0 0 6px 6px; }
```

- [ ] **Step 4: Clean up default Vite files**

Delete `frontend/src/index.css` if it exists. Update `frontend/src/main.tsx` to remove the default CSS import:

```tsx
// frontend/src/main.tsx
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App.tsx";

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <App />
  </StrictMode>
);
```

- [ ] **Step 5: Verify build and dev server**

```bash
cd frontend
npx tsc --noEmit
npm run dev
# Open http://localhost:5173 — should see dark-themed page with left panel + empty chart area
```

- [ ] **Step 6: Commit**

```bash
git add frontend/src/pages/ frontend/src/App.tsx frontend/src/App.css frontend/src/main.tsx
git commit -m "feat: QuotePage assembly with dark theme styling"
```

---

### Task 18: End-to-End Smoke Test

**Files:** none (manual verification)

- [ ] **Step 1: Start all services**

```bash
# Terminal 1: PostgreSQL
docker compose up -d

# Terminal 2: Backend
cd backend && source .venv/bin/activate
# Copy .env.example to .env and fill in TUSHARE_TOKEN
uvicorn app.main:app --reload --port 8000

# Terminal 3: Sync stock_basic
cd backend && source .venv/bin/activate
python -m app.commands.sync_stocks

# Terminal 4: Frontend
cd frontend && npm run dev
```

- [ ] **Step 2: Verify search**

Open http://localhost:5173. Type "茅台" in the search bar. Should see "贵州茅台 600519" in the dropdown. Click it.

- [ ] **Step 3: Verify K-line chart**

After selecting a stock, the right side should display a K-line chart with daily candles and volume. Try switching to 周 and 月 timeframes.

- [ ] **Step 4: Verify snapshot**

Left panel bottom should show the stock's latest price, change, open/high/low/vol.

- [ ] **Step 5: Verify favorites**

Click the ⭐ button on the snapshot card. Switch to "我的收藏" tab. Stock should appear. Click ✕ to remove.

- [ ] **Step 6: Verify indicators**

Hover over "📊 指标", enable MACD, BOLL. BOLL should overlay on the main chart, MACD should appear as a sub-pane.

- [ ] **Step 7: Verify drawing tools**

Hover over "📐 画线", select "趋势线". Click and drag on the chart to draw.

- [ ] **Step 8: Verify linked view**

Click "⊞ 多周期联动". Three charts (日/周/月) should appear side by side.

- [ ] **Step 9: Run all backend tests**

```bash
cd backend
pytest tests/ -v
# Expected: all tests PASS
```

- [ ] **Step 10: Commit any final fixes**

```bash
git add -A
git commit -m "chore: final adjustments from smoke testing"
```
