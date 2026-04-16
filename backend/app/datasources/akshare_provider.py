import asyncio
from datetime import date
from functools import partial

import akshare as ak
import pandas as pd

from app.datasources.base import DataProvider


def _ts_code_to_ak_symbol(ts_code: str) -> str:
    """Convert '000001.SZ' to '000001'."""
    return ts_code.split(".")[0]


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
        df["adj_factor"] = 1.0
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
