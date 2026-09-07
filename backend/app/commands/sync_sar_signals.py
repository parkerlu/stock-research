"""把「SAR预警」写进 strategy_signal, 供虚拟盘跟踪。

⚠️ 择时烘进信号里 —— 虚拟盘不看大盘状态, 而这个指标裸跑组合比值只有 0.29
   (年化 +11.57% / 回撤 40.58%)。基准用中证1000: 它选的是小盘股。

⚠️ 出场规则必须是【事件型】, 不能固定持有。2026-09-07 复核发现:
     20日固定持有   SAR +0.79%  vs 全市场基准 +1.46%   ← 跑输
     加择时后        SAR +1.27%  vs 基准 +1.51%          ← 仍跑输
   它的验证口径是「10日内触及+10%」的事件命中率, 是"冲一下就回落"的形态。
   所以账户配 +4% 走一半 / +8% 清仓 / -6% 止损, 靠周转吃到那一冲。
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

log = logging.getLogger("sar_sig")
STRATEGY = "sar"

SQL = """
with mflag as (
  select trade_date, close,
         avg(close) over (order by trade_date
             rows between 19 preceding and current row) ma20
  from index_daily where ts_code = '000852.SH'
)
select s.ts_code, s.trade_date
from sar_signal s
join mflag m on m.trade_date = s.trade_date
where m.close > m.ma20
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and s.trade_date > :s" if a.since else "")
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), {"s": a.since} if a.since else {})).fetchall()
        total = (await c.execute(text("select count(*) from sar_signal"))).scalar()
    log.info("SAR信号 %d 条中, 择时放行 %d 条 (%.0f%%)",
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
