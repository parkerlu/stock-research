"""把「拉升预警」写进 strategy_signal, 供虚拟盘跑实盘跟踪。

拉升预警 = 动力线上穿0.2 × 近5日内主力吸筹强档(rank>=0.95)。
⚠️ 窗口只向后看 —— pp.trade_date between dl.trade_date - 5 and dl.trade_date。

虚拟盘的出场规则必须和指标的论点对齐, 否则跟踪的不是这个指标:
  论点 = 10 个交易日内收盘价触及 +10%
  所以 止盈 +10% 一次清仓, 持有超 10 日超时平仓, 不做分批/移动止盈。
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

log = logging.getLogger("liftalert")
STRATEGY = "liftalert"

SQL = """
with dl as (select ts_code, trade_date from dongli_signal),
pp as (select ts_code, trade_date from pump_signal where rank_pct >= 0.95)
select distinct dl.ts_code, dl.trade_date
from dl join pp on pp.ts_code = dl.ts_code
     and pp.trade_date between dl.trade_date - 5 and dl.trade_date
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="只写该日之后(YYYY-MM-DD)")
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and dl.trade_date > :s" if a.since else "")
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), {"s": a.since} if a.since else {})).fetchall()
    log.info("拉升预警信号 %d 条", len(rows))
    if not rows:
        await eng.dispose()
        return

    payload = [{"strategy": STRATEGY, "ts_code": r[0], "trade_date": r[1]} for r in rows]
    async with eng.begin() as c:
        for i in range(0, len(payload), 3000):     # 32767 参数上限
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
