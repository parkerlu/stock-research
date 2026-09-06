"""突破预警 = 收盘创 60 日新高 × 近5日内主力吸筹强档(rank>=0.95)。

⚠️ 必须配大盘择时用。裸跑组合比值只有 0.17(回撤 59.7%), 加择时到 1.21~1.91。
   原因: 突破型信号高度同步, 一起涨也一起崩; 但也正因同步, 一个开关就能整批挡住。
   择时口径 = 全市场等权指数在自身 MA10/MA20 之上(越短越好, 单调)。

⚠️ 窗口只向后看: pp.trade_date between t-5 and t, hh60 用 shift(1) 排除当日。
"""
from __future__ import annotations

import asyncio
import logging
from datetime import date

import numpy as np
import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import BreakoutSignal

log = logging.getLogger("build_breakout")
CH = 3000          # PostgreSQL 绑定参数上限 32767


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    eng = create_async_engine(settings.database_url)
    async with eng.connect() as c:
        codes = [r[0] for r in (await c.execute(text(
            "select distinct ts_code from daily_candle order by ts_code"))).fetchall()]
    log.info("全市场 %d 只", len(codes))

    out: list[dict] = []
    for i in range(0, len(codes), 600):
        batch = codes[i:i + 600]
        async with eng.connect() as c:
            d = pd.DataFrame((await c.execute(text(
                "select ts_code,trade_date,close,adj_factor from daily_candle "
                "where trade_date>=:s and ts_code = any(:cs)"),
                {"s": date(2018, 1, 1), "cs": batch})).fetchall(),
                columns=["ts_code", "trade_date", "c", "adj"])
            pp = pd.DataFrame((await c.execute(text(
                "select ts_code,trade_date,prob from pump_signal "
                "where rank_pct >= 0.95 and ts_code = any(:cs)"),
                {"cs": batch})).fetchall(), columns=["ts_code", "trade_date", "prob"])
        if d.empty or pp.empty:
            continue
        d["c"] = d["c"].astype(np.float32) * d["adj"].astype(np.float32)
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        pp["trade_date"] = pd.to_datetime(pp["trade_date"])
        d = d.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        g = d.groupby("ts_code", sort=False)
        # shift(1): 60 日高点不含当日, 否则"突破"永远成立
        d["hh60"] = g["c"].transform(lambda s: s.rolling(60).max().shift(1))
        key = d.set_index(["ts_code", "trade_date"])
        s = pp.drop_duplicates(["ts_code", "trade_date"]).set_index(["ts_code", "trade_date"])
        d["prob"] = key.join(s, how="left")["prob"].values
        # 近 5 日内吸筹强档 —— rolling(6) 只向后看
        d["pump5"] = g["prob"].transform(lambda x: x.rolling(6, min_periods=1).max())
        hit = d[(d.c > d.hh60) & d.pump5.notna()]
        for r in hit.itertuples():
            out.append({"ts_code": r.ts_code, "trade_date": r.trade_date.date(),
                        "prob": round(float(r.pump5), 5),
                        "hh60": round(float(r.hh60), 4)})
        log.info("  %d/%d  累计 %d 条", i + len(batch), len(codes), len(out))

    async with eng.begin() as c:
        await c.execute(text("delete from breakout_signal"))
        for k in range(0, len(out), CH):
            st = insert(BreakoutSignal).values(out[k:k + CH])
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["ts_code", "trade_date"]))
    log.info("突破预警写入 %d 条", len(out))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
