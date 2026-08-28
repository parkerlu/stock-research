import logging
from datetime import date

import pandas as pd

from app.datasources.base import DataProvider

logger = logging.getLogger(__name__)


class DataSourceManager:
    """按顺序尝试多个数据源, 前一个失败/返回空就换下一个。

    典型链路: Tencent(实时) → TuShare(历史) → AKShare(兜底)。
    Tencent 不支持 fetch_daily/fetch_stock_basic, 会立刻返回空 DataFrame,
    因此历史类调用会无网络开销地穿透到 TuShare。
    """

    def __init__(self, *providers: DataProvider,
                 primary: DataProvider | None = None,
                 fallback: DataProvider | None = None):
        chain = list(providers)
        if primary is not None:
            chain.append(primary)
        if fallback is not None:
            chain.append(fallback)
        if not chain:
            raise ValueError("DataSourceManager 至少需要一个数据源")
        self._providers = chain

    def provider_of(self, cls: type) -> DataProvider | None:
        """取链路里指定类型的数据源 (如需直接用腾讯的分时/批量接口)。"""
        for p in self._providers:
            if isinstance(p, cls):
                return p
        return None

    async def _try_with_fallback(self, method_name: str, *args, **kwargs):
        """依次尝试各数据源。全部失败才抛异常。"""
        for i, provider in enumerate(self._providers):
            label = f"{type(provider).__name__}[{i}]"
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
