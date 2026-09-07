"""把「买卖很准 周线版」写进 strategy_signal, 供虚拟盘跟踪。

⚠️ 这是【状态指标】不是买点: 取的是"买线>0 的每一周", 不是"刚进入那一周"。
   实测 状态 +3.45pp / 起始周 +2.51pp / 连续第2周 +2.67pp —— 价值在持续
   处于超卖这个条件本身, 不在进入的那一刻。

⚠️ 不加大盘择时: 验证时就没有, 加了就不是被验证过的那个东西了。
   (SAR/突破那两个必须加, 因为它们裸跑回撤到 40~60%。)

⚠️ week_end 存的是当周最后一个交易日, 虚拟盘 entry_mode=next_open 会在
   下一个交易日开盘买 —— 正好是下周一开盘, 与验证口径一致, 不存在穿越。

出场是固定持有 8 周(≈56 个自然日), 没有止损也没有止盈 —— 验证时就是这么
拿的。套用 +4%/+8%/-6% 那套会变成另一个策略, 数字不能沿用。
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

log = logging.getLogger("mmweek_sig")
STRATEGY = "mmweek"

SQL = "select ts_code, week_end from maimai_weekly where buy_line > 0"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and week_end > :s" if a.since else "")
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), {"s": a.since} if a.since else {})).fetchall()
        total = (await c.execute(text("select count(*) from maimai_weekly"))).scalar()
    log.info("周线 %d 条中, 买线>0 的 %d 条 (%.0f%%)",
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
