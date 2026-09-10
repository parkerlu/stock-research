"""起爆模型信号 —— Top1% × 大盘择时，写 strategy_signal。

⚠️⚠️ 择时【必须】写进信号里, 不是可选项:
     不择时 比值 1.15 / MA10 择时 3.37 —— 择时同时提高年化(+40.8%→+51.9%)
     并砍掉一半回撤(−35.6%→−15.4%)。
     原因: 起爆票是高波动小盘股, 与中证1000 高度同步, 一个大盘开关就能整批挡住。
     这与"突破预警"当年的发现一致(裸跑 0.17, 加择时 1.91~2.62)。
     ⚠️ 而且 walk-forward 六年【每一年都选中 MA10】—— 参数选择本身是稳定的,
        不是碰巧某年好。

⚠️ 择时基准用中证1000 而不是沪深300: 起爆票是小盘股, 拿蓝筹的脸色判断小盘股
   死活是错的 —— 项目里量过, 换基准 1.91 -> 2.62。

⚠️ 流动性下限 2000 万: 92% 的信号满足, 比值 2.31。不设的话 2.64 但那部分
   超额来自根本买不进的小票。门槛与收益的完整曲线:
       不限 2.64 / >2000万 2.31 / >5000万 1.88 / >1亿 1.11
   这条曲线也是这个策略的【资金容量】—— 以 5000 万为界, 20 仓位约能容纳千万级。

⚠️ 出场规则必须与训练标签逐字一致: +30% 全清 / −8% 止损 / 20 个交易日。
   规则一改, 模型优化的东西和账户执行的东西就不是一回事了
   (本项目已因此栽过两次: 形态模型 v4 的 1.70、SAR/突破的 2.62)。
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

log = logging.getLogger("boom_sig")
STRATEGY = "boom30"
BENCH = "000852.SH"        # 中证1000
MA = 10
MIN_AMT_WAN = 2000         # 20 日均成交额下限(万元)

SQL = """
with ma as (
  select trade_date,
         close,
         avg(close) over (order by trade_date rows between :ma1 preceding and current row) as ma_n
  from index_daily where ts_code = :bench
),
ok_day as (select trade_date from ma where close > ma_n),
liq as (
  select ts_code, trade_date,
         avg(amount) over (partition by ts_code order by trade_date
                           rows between 19 preceding and current row) / 10.0 as a20
  from daily_candle
  where trade_date > (select max(trade_date) - interval '90 days' from daily_candle)
)
select b.ts_code, b.trade_date
from boom_score b
join ok_day d on d.trade_date = b.trade_date
join liq l on l.ts_code = b.ts_code and l.trade_date = b.trade_date
where b.rank_pct >= :q and l.a20 >= :amt
"""


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None)
    ap.add_argument("--q", type=float, default=0.99)
    ap.add_argument("--no-timing", action="store_true",
                    help="关掉择时 —— 只用来做对照, 生产不要开")
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    sql = SQL
    if a.no_timing:
        sql = sql.replace("join ok_day d on d.trade_date = b.trade_date", "")
    if a.since:
        sql += " and b.trade_date > :s"
    params: dict = {"q": a.q, "bench": BENCH, "ma1": MA - 1, "amt": MIN_AMT_WAN}
    if a.since:
        params["s"] = a.since
    async with eng.connect() as c:
        rows = (await c.execute(text(sql), params)).fetchall()
        total = (await c.execute(text(
            "select count(*) from boom_score where rank_pct >= :q"), {"q": a.q})).scalar()
    log.info("Top%.0f%% 共 %d 条, 过择时(%s>MA%d)与流动性(>%d万)后 %d 条 (%.0f%%)",
             (1 - a.q) * 100, total, BENCH, MA, MIN_AMT_WAN, len(rows),
             len(rows) / max(total, 1) * 100)
    if not rows:
        log.info("今日无信号 —— 大盘在 MA%d 之下时本来就该空仓", MA)
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
