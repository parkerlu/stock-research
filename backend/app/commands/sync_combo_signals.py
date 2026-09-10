"""共振信号 —— 每日打分 Top1% 进 strategy_signal。

共振 = 筹码模型 × 裸K CNN, 两边 EV 等权平均。
目前【唯一】组合层面过线的: walk-forward + 资金池 + 真实周转, 仓位20
-> 比值 1.07 (年化 +27.7% / 回撤 −26.0%), 六年零负年。
单用: 筹码 0.89 / 裸K 0.78。

⚠️ 出场规则必须与训练标签逐字一致 —— 两个模型用的是同一个 label3_dn8:
   次日开盘买入, 10 根K线内先碰 +10% 记涨 / 先碰 −8% 记跌。所以账户配
       tier1_pct 0.10 + tier1_frac 1.0   (到 +10% 一次清仓)
       stop_pct  0.08
       max_hold_days 15                  (自然日; 10 个交易日约 14 天)
   规则一改, 模型优化的东西和账户执行的东西就不是一回事了。

⚠️ 与回测阈值不完全一致: 回测里阈值是 walk-forward 每年滚动选的绝对 EV,
   每年都变, 没法固化成生产规则。这里取当日横截面 Top1%(约 53 只),
   语义稳定。所以这个账户是【向前验】, 不是回测结论的复现。

⚠️ 分位由 build_combo_scores 按当日全市场算, 这里只筛不重排。
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

log = logging.getLogger("combo_sig")
STRATEGY = "combo_ck"      # chips × 裸K; 不叫 combo —— 那个名字已被买卖很准 v4 占用

SQL = "select ts_code, trade_date from combo_score where rank_pct >= :q"


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--q", type=float, default=0.99, help="取当日分位以上")
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL + (" and trade_date > :s" if a.since else "")
    params: dict = {"q": a.q}
    if a.since:
        params["s"] = a.since
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), params)).fetchall()
        total = (await c.execute(text("select count(*) from combo_score"))).scalar()
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
