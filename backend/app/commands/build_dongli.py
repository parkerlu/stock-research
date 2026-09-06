"""落表: 动力线上穿 0.2 的日子(原「低点组合」的阶段底部信号)。

它自己几乎没预测力 —— 「10日涨10%」命中率 18.7%, 基准 17.45%, 按天 t=-0.70。
落表是为了给主力吸筹做最后一道收紧:

    吸筹强 单用            25.7%  (+7.93pp, t=67.35, 九格全正)
    动力线 × 吸筹强         32.6%  (+9.06pp, t= 4.79, 九格最小 +1.42pp)

⚠️ 复用 app.services.tdx.functions 里的 LLV/HHV/EMA/CROSS, 不自己重写 ——
   本项目为性能重写已有实现已经栽过一次(买卖很准 648 天里 59 天对不上)。
⚠️ 无前视: 三个窗口全是向后看的, 309 次截断重算零不一致。
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
from app.models.schema import DongliSignal
from app.services.tdx.functions import CROSS, EMA, HHV, LLV

log = logging.getLogger("build_dongli")
THR = 0.2
CH = 3000          # ⚠️ PostgreSQL 绑定参数上限 32767, 行数×列数不能超


def cross_days(g: pd.DataFrame) -> pd.DataFrame:
    var2, var33 = LLV(g["l"], 10), HHV(g["h"], 25)
    dl = EMA((g["c"] - var2) / (var33 - var2) * 4, 4)
    hit = CROSS(dl, pd.Series(THR, index=dl.index)).fillna(0).astype(bool)
    return pd.DataFrame({"trade_date": g["trade_date"][hit].values,
                         "dl_value": dl[hit].values})


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
                {"s": date(2018, 1, 1), "cs": batch})).fetchall(),
                columns=["ts_code", "trade_date", "h", "l", "c", "adj"])
        if d.empty:
            continue
        for x in "hlc":
            d[x] = d[x].astype(np.float32) * d["adj"].astype(np.float32)
        d = d.sort_values(["ts_code", "trade_date"])
        for ts, g in d.groupby("ts_code", sort=False):
            if len(g) < 60:
                continue
            r = cross_days(g.reset_index(drop=True))
            for row in r.itertuples():
                out.append({"ts_code": ts, "trade_date": row.trade_date,
                            "dl_value": round(float(row.dl_value), 4)})
        log.info("  %d/%d  累计 %d 条", i + len(batch), len(codes), len(out))

    async with eng.begin() as c:
        await c.execute(text("delete from dongli_signal"))
        for k in range(0, len(out), CH):
            chunk = out[k:k + CH]
            st = insert(DongliSignal).values(chunk)
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["ts_code", "trade_date"]))
    log.info("动力线信号写入 %d 条", len(out))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
