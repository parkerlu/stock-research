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
