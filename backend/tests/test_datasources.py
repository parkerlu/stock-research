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


import pandas as pd
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
