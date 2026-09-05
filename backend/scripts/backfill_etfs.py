"""一次性回补全市场 ETF 日线.

原来 _sync_etfs 只取成交额前 50, 库里只有 72 只 —— 做 ETF 策略时候选池
太小(6 个仓位就占 8%), 挑最好的几乎必然是运气。这里按门槛取全部。

门槛(宽松, 宁可多拉): 上市满 1 年、近期日均成交额 >= 500 万。
"""
from __future__ import annotations
import asyncio, sys, time
from datetime import date, timedelta

import pandas as pd
import tushare as ts
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

sys.path.insert(0, "/app")
from app.config import settings
from app.db import async_session
from app.models.schema import DailyCandle, StockBasic

MIN_AMOUNT_K = 500 * 10      # 500 万元 = 5000 千元
START = "20150101"


async def main():
    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()
    today = date.today()

    basic = pro.fund_basic(market="E")
    basic["ld"] = pd.to_datetime(basic["list_date"], errors="coerce")
    cutoff = pd.Timestamp(today - timedelta(days=365))
    old = basic[basic["ld"] <= cutoff][["ts_code", "name", "list_date"]]
    print(f"上市满 1 年的 ETF: {len(old)}")

    # 用最近一个有数据的交易日筛流动性
    daily = None
    for back in range(0, 8):
        d = (today - timedelta(days=back)).strftime("%Y%m%d")
        daily = pro.fund_daily(trade_date=d)
        if daily is not None and not daily.empty:
            print(f"流动性基准日 {d}, {len(daily)} 条")
            break
    if daily is None or daily.empty:
        print("拿不到近期数据"); return

    elig = (daily[daily.ts_code.isin(old.ts_code)]
            .query("amount >= @MIN_AMOUNT_K")
            .merge(old, on="ts_code"))
    print(f"日均额达标: {len(elig)} 只\n")

    async with async_session() as db:
        have = set((await db.execute(select(StockBasic.ts_code))).scalars().all())

    inserted_meta = 0
    async with async_session() as db:
        rows = []
        for _, r in elig.iterrows():
            if r.ts_code in have: continue
            rows.append({"ts_code": r.ts_code, "symbol": r.ts_code.split(".")[0],
                         "name": (r.get("name") or "")[:60], "industry": "ETF",
                         "market": "ETF", "is_active": True,
                         "list_date": pd.to_datetime(r.list_date).date()})
        if rows:
            await db.execute(pg_insert(StockBasic).values(rows)
                             .on_conflict_do_nothing(index_elements=["ts_code"]))
            await db.commit(); inserted_meta = len(rows)
    print(f"新增 ETF 元数据 {inserted_meta} 条\n")

    t0 = time.time(); total = 0
    for i, code in enumerate(elig.ts_code.tolist(), 1):
        try:
            df = pro.fund_daily(ts_code=code, start_date=START,
                                end_date=today.strftime("%Y%m%d"))
        except Exception as e:
            print(f"  {code} 拉取失败: {e}"); continue
        if df is None or df.empty: continue
        df = df.rename(columns={"vol": "vol", "amount": "amount"})
        payload = [{
            "ts_code": code,
            "trade_date": pd.to_datetime(r.trade_date).date(),
            "open": float(r.open), "high": float(r.high),
            "low": float(r.low), "close": float(r.close),
            "vol": float(r.vol or 0), "amount": float(r.amount or 0),
            "adj_factor": 1.0,          # ETF 不除权, 前复权因子恒为 1
            "source": "tushare",
        } for r in df.itertuples() if pd.notna(r.close)]
        if not payload: continue
        async with async_session() as db:
            await db.execute(pg_insert(DailyCandle).values(payload)
                             .on_conflict_do_nothing(
                                 index_elements=["ts_code", "trade_date"]))
            await db.commit()
        total += len(payload)
        if i % 40 == 0:
            print(f"  {i}/{len(elig)} · 累计 {total:,} 行 · {time.time()-t0:.0f}s", flush=True)

    print(f"\n完成: {len(elig)} 只 ETF, {total:,} 行, 耗时 {time.time()-t0:.0f}s")

if __name__ == "__main__":
    asyncio.run(main())
