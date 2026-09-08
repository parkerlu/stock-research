"""按【交易日】整批补日线 —— tushare 一次调用拿全市场一天。

⚠️ 为什么要有这个: 原来的 scheduler._sync_stocks 是【逐只】拉的, 4400 只挨个
   调 API。两个后果, 2026-09-08 都撞上了:
     1. 慢 —— 实测跑 400 只用了 3 分钟, 全量要半小时
     2. 脆 —— 任何一次限流或网络抖动就留下缺口, 而且【静默】:
        界面上看不出来, 只是选股结果变少。09-07 就这样只入库 3279/5549 只,
        导致所有训练指标当天出不了完整信号。

   pro.daily(trade_date=) 一次拿一整天, 3 秒补完 5549 只。
   补历史缺口一律用这个, 别用逐只。

⚠️ 复权因子要一起拉 —— 只有价没有 adj_factor 的话, 除权日会算出假暴跌。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging

import pandas as pd
import tushare as ts
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import DailyCandle

log = logging.getLogger("fill_daily")
CH = 2000          # PostgreSQL 绑定参数上限 32767


async def fill(days: list[str]) -> int:
    """days: ['20260907', ...]。返回新增行数。"""
    pro = ts.pro_api(settings.tushare_token)
    eng = create_async_engine(settings.database_url)
    total = 0
    try:
        for d in days:
            try:
                df = pro.daily(trade_date=d)
                adj = pro.adj_factor(trade_date=d)
            except Exception as exc:  # noqa: BLE001
                log.warning("%s 拉取失败: %s", d, exc)
                continue
            if df is None or df.empty:
                log.info("%s: 源无数据(非交易日?)", d)
                continue
            if adj is not None and not adj.empty:
                df = df.merge(adj[["ts_code", "adj_factor"]], on="ts_code", how="left")
            if "adj_factor" not in df.columns:
                df["adj_factor"] = 1.0
            df["adj_factor"] = df["adj_factor"].fillna(1.0)
            rows = [{"ts_code": r.ts_code,
                     "trade_date": pd.to_datetime(r.trade_date).date(),
                     "open": float(r.open), "high": float(r.high),
                     "low": float(r.low), "close": float(r.close),
                     "vol": int(r.vol), "amount": float(r.amount),
                     "adj_factor": float(r.adj_factor), "source": "tushare"}
                    for r in df.itertuples() if pd.notna(r.close)]
            async with eng.begin() as c:
                for i in range(0, len(rows), CH):
                    st = pg_insert(DailyCandle).values(rows[i:i + CH])
                    await c.execute(st.on_conflict_do_nothing(
                        index_elements=["ts_code", "trade_date"]))
                n = (await c.execute(text(
                    "select count(*) from daily_candle where trade_date = :d"),
                    {"d": pd.to_datetime(d).date()})).scalar()
            total += len(rows)
            log.info("%s: 源给 %d 行, 库内 %d 只", d, len(rows), n)
    finally:
        await eng.dispose()
    return total


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5, help="回补最近几个交易日")
    ap.add_argument("--date", default=None, help="只补某一天, 如 20260907")
    a = ap.parse_args()

    if a.date:
        await fill([a.date])
        return
    pro = ts.pro_api(settings.tushare_token)
    end = dt.date.today()
    cal = pro.trade_cal(exchange="SSE",
                        start_date=(end - dt.timedelta(days=a.days * 3)).strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"), is_open="1")
    days = sorted(cal.cal_date.tolist())[-a.days:]
    log.info("回补 %s", days)
    await fill(days)


if __name__ == "__main__":
    asyncio.run(main())
