"""筹码模型日线策略 —— 每日打分 Top1% 进 strategy_signal。

与形态模型(sync_shape_signals)不同, 这个是【真正当选股信号用】的:
    形态模型组合比值 0.24, 只能当排除过滤器;
    筹码模型 0.89(walk-forward + 资金池 + 真实周转, 仓位20),
    年化 +15.9% / 回撤 −17.9% / 六年 2 个负年(都在 2% 以内)。

⚠️ 出场规则必须与训练标签逐字一致 —— 标签是"次日开盘买入, 10 根K线内
   先碰 +10% 记涨 / 先碰 -8% 记跌 / 都没碰按第10日收盘"。所以账户要配:
       tier1_pct 0.10 + tier1_frac 1.0   (到 +10% 一次清仓)
       stop_pct  0.08                    (−8% 止损)
       max_hold_days 15                  (×1.6≈24自然日≈10个交易日... 见下)
   ⚠️ paper_trading 的 max_hold_days 是【自然日】, 10 个交易日约 14 自然日,
      取 15 留一天余量。规则一改, 模型优化的东西和账户执行的东西就不是一回事了。

⚠️ 为什么取 Top1% 而不是回测里选出的"验证前2%~5%":
   那两个阈值是 walk-forward 在【绝对 EV】上滚动选的, 每年都不一样, 没法固化成
   一条生产规则。Top1% 是横截面分位, 语义稳定(每天约 55 只), 而虚拟盘只有
   20 个仓位 —— 再宽也吃不下。
   ⚠️ 这意味着【生产规则与回测规则不完全一致】, 所以这个账户是向前验,
      不是回测结论的复现。

⚠️ 分位由 build_chips_scores 按当日全市场算, 这里只做阈值筛选, 不重新排名。
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

log = logging.getLogger("chips_sig")
STRATEGY = "chips"

SQL = "select ts_code, trade_date from chips_score where rank_pct >= :q"


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
        total = (await c.execute(text("select count(*) from chips_score"))).scalar()
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
