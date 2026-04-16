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
