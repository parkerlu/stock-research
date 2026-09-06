"""SAR 预警 = SAR 由空翻多 × 近5日内主力吸筹强档(rank>=0.95)。

⚠️ 必须配大盘择时(中证1000 > 自身MA20): 裸跑组合比值 0.29(回撤 40.6%),
   加择时 1.74(年化 +34.65% / 回撤 19.96%)。

与突破预警的分工: 突破版比值更高(2.62)但胜率仅约 35%, 靠少数大赢家;
SAR 版比值 1.74 但胜率 52.5%, 连亏的串更短。九格最小 +3.93pp 是所有组合里
最高的 —— 各波动/市值格子都均匀, 不挑票。

SAR 参数 (0.02, 0.02, 0.2), 与前端 klinecharts 的 SAR(2,2,20) 一致。
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
from app.models.schema import SarSignal

log = logging.getLogger("build_sar")
CH = 3000


def sar_bull(h: np.ndarray, l: np.ndarray,
             af0: float = 0.02, step: float = 0.02, afmax: float = 0.2) -> np.ndarray:
    """标准抛物线 SAR, 返回每根是否多头(SAR 在价格下方)。"""
    n = len(h)
    bull = np.zeros(n, dtype=bool)
    if n < 3:
        return bull
    up, a, ep, sr = True, af0, h[0], l[0]
    for i in range(1, n):
        sr = sr + a * (ep - sr)
        if up:
            if l[i] < sr:
                up, sr, ep, a = False, ep, l[i], af0
            elif h[i] > ep:
                ep, a = h[i], min(a + step, afmax)
        else:
            if h[i] > sr:
                up, sr, ep, a = True, ep, h[i], af0
            elif l[i] < ep:
                ep, a = l[i], min(a + step, afmax)
        bull[i] = up
    return bull


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
                "select ts_code,trade_date,high,low,adj_factor from daily_candle "
                "where trade_date>=:s and ts_code = any(:cs)"),
                {"s": date(2018, 1, 1), "cs": batch})).fetchall(),
                columns=["ts_code", "trade_date", "h", "l", "adj"])
            pp = pd.DataFrame((await c.execute(text(
                "select ts_code,trade_date,prob from pump_signal "
                "where rank_pct >= 0.95 and ts_code = any(:cs)"),
                {"cs": batch})).fetchall(), columns=["ts_code", "trade_date", "prob"])
        if d.empty or pp.empty:
            continue
        for x in "hl":
            d[x] = d[x].astype(np.float32) * d["adj"].astype(np.float32)
        d["trade_date"] = pd.to_datetime(d["trade_date"])
        pp["trade_date"] = pd.to_datetime(pp["trade_date"])
        d = d.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
        d["flip"] = False
        for ts, g in d.groupby("ts_code", sort=False):
            if len(g) < 30:
                continue
            b = sar_bull(g["h"].values.astype(np.float64),
                         g["l"].values.astype(np.float64))
            d.loc[g.index, "flip"] = np.r_[False, b[1:] & ~b[:-1]]
        key = d.set_index(["ts_code", "trade_date"])
        s = pp.drop_duplicates(["ts_code", "trade_date"]).set_index(["ts_code", "trade_date"])
        d["prob"] = key.join(s, how="left")["prob"].values
        # 近 5 日内吸筹强档 —— rolling(6) 只向后看, 无前视
        d["pump5"] = d.groupby("ts_code")["prob"].transform(
            lambda x: x.rolling(6, min_periods=1).max())
        hit = d[d.flip & d.pump5.notna()]
        for r in hit.itertuples():
            out.append({"ts_code": r.ts_code, "trade_date": r.trade_date.date(),
                        "prob": round(float(r.pump5), 5)})
        log.info("  %d/%d  累计 %d 条", i + len(batch), len(codes), len(out))

    async with eng.begin() as c:
        await c.execute(text("delete from sar_signal"))
        for k in range(0, len(out), CH):
            st = insert(SarSignal).values(out[k:k + CH])
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["ts_code", "trade_date"]))
    log.info("SAR 预警写入 %d 条", len(out))
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
