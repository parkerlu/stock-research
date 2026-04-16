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
