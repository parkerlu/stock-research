"""形态模型自己当日线策略 —— 每日打分 Top1%。

⚠️ 回测判据没过: 带成交约束的组合比值只有 0.24(样本外 2021-2026)。
   建这个账户不是因为它验过了, 而是【向前验】—— 回测里我已经栽过一次
   (v4 的 1.70 全来自买不到的一字涨停), 真实前瞻数据是唯一干净的检验。

⚠️ 出场规则必须与训练标签逐字一致: 标签是"三重障碍 +15%/-8%/20交易日
   下真正落袋的超额"。账户配 tier1_pct=0.15 全清 / stop_pct=0.08 /
   max_hold_days=18(×1.6≈29自然日≈20交易日)。
   规则一改, 模型优化的东西和账户执行的东西就不是一回事了。

⚠️ 分位是横截面概念: rank_pct 由 build_shape_scores 按【当日全市场】算,
   这里只做阈值筛选, 不重新排名。
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

log = logging.getLogger("shape_sig")
STRATEGY = "shape"

SQL = """
select ts_code, trade_date from shape_score where rank_pct >= :q
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--q", type=float, default=0.99, help="取当日分位以上")
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and trade_date > :s" if a.since else "")
    params = {"q": a.q}
    if a.since:
        params["s"] = a.since
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), params)).fetchall()
        total = (await c.execute(text("select count(*) from shape_score"))).scalar()
    log.info("打分 %d 条中, 分位>=%.2f 的 %d 条 (%.1f%%)",
             total, a.q, len(rows), len(rows) / max(total, 1) * 100)
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
    log.info("[%s] 表内合计 %d 条, %s ~ %s", STRATEGY, n[0], n[1], n[2])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
