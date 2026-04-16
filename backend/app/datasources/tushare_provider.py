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
        daily = await loop.run_in_executor(
            None,
            partial(
                self._api.daily,
                ts_code=ts_code,
                start_date=start.strftime("%Y%m%d"),
                end_date=end.strftime("%Y%m%d"),
            ),
        )
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
            "name": "",
            "price": float(row["close"]),
            "change": float(row["change"]),
            "change_pct": float(row["pct_chg"]),
            "open": float(row["open"]),
            "high": float(row["high"]),
            "low": float(row["low"]),
            "vol": int(row["vol"]),
            "amount": float(row["amount"]),
            "turnover": 0.0,
        }
