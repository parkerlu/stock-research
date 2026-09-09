import logging
from datetime import date

import pandas as pd

from app.datasources.base import DataProvider

logger = logging.getLogger(__name__)


class DataSourceManager:
    """按顺序尝试多个数据源, 前一个失败/返回空就换下一个。

    典型链路: Tencent(实时) → TuShare(历史) → AKShare(兜底)。
    Tencent 不支持 fetch_stock_basic, 会立刻返回空 DataFrame 并穿透到 TuShare。

    ⚠️ fetch_daily 曾经也是空实现, 8c344fd 起腾讯【会返回数据】了, 于是历史
       日线的第一顺位变成了腾讯。它给前复权价 + adj_factor=1.0, 与库里
       "原始价 + TuShare 绝对 factor" 不是一个口径, 直接入库会毁掉整段历史的
       复权(2026-09-09 上汽集团)。落库路径统一走 quote_service._rows_for_db。
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
