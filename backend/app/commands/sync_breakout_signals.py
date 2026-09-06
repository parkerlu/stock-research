"""把「突破预警」写进 strategy_signal, 供虚拟盘跑跟踪。

⚠️ 择时必须烘进信号里 —— 虚拟盘不懂大盘状态, 而这个指标离了择时不能用:
     无择时          年化  +9.99%  回撤 59.69%  比值 0.17
     中证1000>MA20   年化 +39.96%  回撤 15.24%  比值 2.62
   所以这里只在【中证1000 收盘 > 自身 MA20】的交易日输出信号。

基准为什么用中证1000: 突破预警选的是小盘股, 用沪深300/上证判断等于拿蓝筹的
脸色看小盘股死活。实测比值 中证1000 2.62 > 中证500 2.26 > 沪深300 1.92 >
上证 1.17 —— 越偏小盘越好, 单调。
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import StrategySignal

log = logging.getLogger("breakout_sig")
STRATEGY = "breakout"

SQL = """
with mflag as (
  select trade_date, close,
         avg(close) over (order by trade_date
             rows between 19 preceding and current row) ma20
  from index_daily where ts_code = '000852.SH'
)
select b.ts_code, b.trade_date
from breakout_signal b
join mflag m on m.trade_date = b.trade_date
where m.close > m.ma20
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and b.trade_date > :s" if a.since else "")
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), {"s": a.since} if a.since else {})).fetchall()
        total = (await c.execute(text("select count(*) from breakout_signal"))).scalar()
    log.info("突破信号 %d 条中, 择时放行 %d 条 (%.0f%%)",
             total, len(rows), len(rows) / max(total, 1) * 100)
    if not rows:
        await eng.dispose()
        return

    payload = [{"strategy": STRATEGY, "ts_code": r[0], "trade_date": r[1]} for r in rows]
    async with eng.begin() as c:
        for i in range(0, len(payload), 3000):
            st = pg_insert(StrategySignal).values(payload[i:i + 3000])
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["strategy", "ts_code", "trade_date"]))
        n = (await c.execute(text(
            "select count(*), min(trade_date), max(trade_date) from strategy_signal "
            "where strategy = :s"), {"s": STRATEGY})).fetchone()
    log.info("表内合计 %d 条, %s ~ %s", n[0], n[1], n[2])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
