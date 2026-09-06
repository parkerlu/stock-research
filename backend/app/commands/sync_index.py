"""拉指数日线 —— 中证1000 / 中证500 / 沪深300 / 上证指数。

用途是择时基准。中证1000 是小盘指数, 比"全市场等权"更贴近训练指标选出来的标的,
而且它可交易(有 ETF), 择时信号能直接对应到操作。

⚠️ 每日增量必须接进 scheduler —— 不接的话择时基准会停在导入那天,
   而"停更"在界面上和"大盘不好"长得一模一样(v3.5/龙虎榜都栽过这个)。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import IndexDaily

log = logging.getLogger("sync_index")

CODES = {
    "000852.SH": "中证1000",
    "000905.SH": "中证500",
    "000300.SH": "沪深300",
    "000001.SH": "上证指数",
}
CH = 3000          # PostgreSQL 绑定参数上限 32767 / 8 列


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="20180101")
    a = ap.parse_args()

    import tushare as ts
    pro = ts.pro_api(settings.tushare_token)
    eng = create_async_engine(settings.database_url)
    end = date.today().strftime("%Y%m%d")

    rows: list[dict] = []
    for code, name in CODES.items():
        df = None
        for _ in range(4):
            try:
                df = pro.index_daily(ts_code=code, start_date=a.start, end_date=end)
                break
            except Exception as e:  # noqa: BLE001
                if "超限" in str(e) or "频率" in str(e):
                    time.sleep(12)
                    continue
                log.warning("%s 拉取失败: %s", name, e)
                break
        if df is None or df.empty:
            continue
        df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
        for r in df.itertuples():
            rows.append({"ts_code": code, "trade_date": r.trade_date,
                         "open": float(r.open), "high": float(r.high),
                         "low": float(r.low), "close": float(r.close),
                         "vol": float(r.vol or 0), "amount": float(r.amount or 0)})
        log.info("  %s %s: %d 行 (%s ~ %s)", name, code, len(df),
                 df.trade_date.min(), df.trade_date.max())

    async with eng.begin() as c:
        for k in range(0, len(rows), CH):
            st = insert(IndexDaily).values(rows[k:k + CH])
            await c.execute(st.on_conflict_do_update(
                index_elements=["ts_code", "trade_date"],
                set_={"close": st.excluded.close, "open": st.excluded.open,
                      "high": st.excluded.high, "low": st.excluded.low,
                      "vol": st.excluded.vol, "amount": st.excluded.amount}))
        n = (await c.execute(text("select count(*) from index_daily"))).scalar()
    log.info("指数日线合计 %d 行", n)
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
