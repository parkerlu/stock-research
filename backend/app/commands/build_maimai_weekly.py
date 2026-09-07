"""买卖很准 周线版 —— 周线买线 > 0 的周。

⚠️ 必须复用 maimai_henzhun.compute_lines, 不能自己重写。
   本项目为性能重写过一次日线版, 结果 648 天里 59 天买线值不一致、漏掉 42%
   信号。周线同理。

⚠️ 与日线版取法不同, 这不是笔误:
   日线版取【边沿】(买线 >0→0), 周线版取【状态】(买线 >0 的每一周)。
   实测持有8周: 每周 +3.45pp(t=5.05, 九格全正) vs 起始周 +2.51pp(九格 -0.01)。
   价值在"处于超卖状态"本身, 不在"刚进入"那一刻。

周线聚合: 自然周(周一~周五), 高=周内最高, 低=周内最低, 收=最后一个交易日收盘。
信号在周五收盘成立, 下周一开盘可操作 —— 无前视。
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
from app.models.schema import MaimaiWeekly
from app.services.tdx.indicators.maimai_henzhun import compute_lines

log = logging.getLogger("build_mm_weekly")
CH = 3000


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
                "select ts_code,trade_date,high,low,close,adj_factor from daily_candle "
                "where trade_date>=:s and ts_code = any(:cs)"),
                {"s": date(2017, 1, 1), "cs": batch})).fetchall(),
                columns=["ts_code", "trade_date", "h", "l", "c", "adj"])
        if d.empty:
            continue
        for x in "hlc":
            d[x] = d[x].astype(np.float32) * d["adj"].astype(np.float32)
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        d = d.sort_values(["ts_code", "trade_date"])
        d["wk"] = d["trade_date"].dt.to_period("W").dt.end_time.dt.normalize()
        # ⚠️ 丢掉【尚未结束的当周】。任务在周三跑时, 当周只有周一~周三三根日线,
        # 算出来的买线跟完整周不是一回事, 到周五还可能翻转 —— 这不是穿越
        # (用的数据更少), 但与回测对象不一致, 会让人按一个没定型的值下单。
        # 判据: 自然周的周日 <= 今天, 才算这一周已经走完。
        d = d[d["wk"] <= pd.Timestamp(date.today())]
        # ⚠️ week_end 存【当周最后一个交易日】而不是自然周的周日 ——
        # to_period("W").end_time 给的是周日, 永远不是交易日, 前端按日期
        # 匹配 K 线时一根都对不上(画不出来且不报错)。
        w = (d.groupby(["ts_code", "wk"])
               .agg(h=("h", "max"), l=("l", "min"), c=("c", "last"),
                    last_td=("trade_date", "max"))
               .reset_index().sort_values(["ts_code", "wk"]))
        for ts, g in w.groupby("ts_code", sort=False):
            if len(g) < 60:
                continue
            g = g.reset_index(drop=True)
            buy, _ = compute_lines(g["c"], g["h"], g["l"])
            hit = buy.values > 0
            for k in np.where(hit)[0]:
                out.append({"ts_code": ts,
                            "week_end": g["last_td"].iloc[k].date(),
                            "buy_line": round(float(buy.values[k]), 4)})
        log.info("  %d/%d  累计 %d 条", i + len(batch), len(codes), len(out))

    async with eng.begin() as c:
        await c.execute(text("delete from maimai_weekly"))
        for k in range(0, len(out), CH):
            st = insert(MaimaiWeekly).values(out[k:k + CH])
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["ts_code", "week_end"]))
    log.info("周线版写入 %d 条", len(out))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
