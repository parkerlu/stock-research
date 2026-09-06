"""拉中证1000(等指数)的多周期 K 线: 周 / 月 / 120 / 60 / 30 分钟。

数据源分两路:
  周/月    tushare index_weekly / index_monthly —— 不限速, 一次拉全
  分钟级   tushare stk_mins —— 【限速 1 次/分钟】, 且只有它能给指数分钟数据
           (baostock 对指数返回 0 行, 实测个股 72 行 / 指数 0 行)

⚠️ 分钟级必须节流 65 秒一次, 而且按【季度】切片拉 —— 单次请求返回上限 8000 行,
   30min 一天 8 根, 一个季度约 480 根, 安全。
⚠️ 断点续传: 每个切片拉完立刻入库, 中断了重跑会跳过已有的时间段。
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings
from app.models.schema import IndexBar

log = logging.getLogger("index_bars")
CH = 3000
THROTTLE = 65          # stk_mins 限速 1 次/分钟, 留 5 秒余量


async def _save(eng, rows: list[dict]) -> None:
    if not rows:
        return
    async with eng.begin() as c:
        for k in range(0, len(rows), CH):
            st = insert(IndexBar).values(rows[k:k + CH])
            await c.execute(st.on_conflict_do_nothing(
                index_elements=["ts_code", "freq", "bar_time"]))


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--code", default="000852.SH")
    ap.add_argument("--start", default="2023-01-01", help="分钟级起始(周月固定2018)")
    ap.add_argument("--skip-mins", action="store_true")
    a = ap.parse_args()

    import tushare as ts
    pro = ts.pro_api(settings.tushare_token)
    eng = create_async_engine(settings.database_url)

    # ---- 周 / 月 ----
    for fn, freq, lab in [("index_weekly", "1w", "周线"),
                          ("index_monthly", "1m", "月线")]:
        try:
            df = getattr(pro, fn)(ts_code=a.code, start_date="20180101",
                                  end_date=date.today().strftime("%Y%m%d"))
        except Exception as e:  # noqa: BLE001
            log.warning("%s 拉取失败: %s", lab, e)
            continue
        rows = [{"ts_code": a.code, "freq": freq,
                 "bar_time": datetime.strptime(r.trade_date, "%Y%m%d"),
                 "open": float(r.open), "high": float(r.high), "low": float(r.low),
                 "close": float(r.close), "vol": float(r.vol or 0)}
                for r in df.itertuples()]
        await _save(eng, rows)
        log.info("  %s: %d 根", lab, len(rows))

    if a.skip_mins:
        await eng.dispose()
        return

    # ---- 分钟级: 按季度切片 + 节流 ----
    async with eng.connect() as c:
        have = {(r[0], r[1]) for r in (await c.execute(text(
            "select freq, date_trunc('quarter', bar_time) from index_bar "
            "where ts_code = :c group by 1, 2"), {"c": a.code})).fetchall()}
    start = datetime.strptime(a.start, "%Y-%m-%d")
    quarters = pd.date_range(start, datetime.now(), freq="QS").tolist()
    total_calls = sum(1 for f in ("120min", "60min", "30min") for q in quarters
                      if (f, q) not in have)
    log.info("分钟级: %d 个请求 × 65 秒 ≈ %.0f 分钟", total_calls, total_calls * 65 / 60)

    done = 0
    for freq in ("120min", "60min", "30min"):
        for q in quarters:
            if (freq, q) in have:
                continue
            qe = (q + pd.offsets.QuarterEnd(0)).to_pydatetime()
            for attempt in range(3):
                try:
                    df = pro.stk_mins(
                        ts_code=a.code, freq=freq,
                        start_date=q.strftime("%Y-%m-%d 09:00:00"),
                        end_date=qe.strftime("%Y-%m-%d 15:30:00"))
                    break
                except Exception as e:  # noqa: BLE001
                    if "超限" in str(e) or "频率" in str(e):
                        time.sleep(THROTTLE)
                        continue
                    log.warning("  %s %s 失败: %s", freq, q.date(), str(e)[:60])
                    df = None
                    break
            else:
                df = None
            done += 1
            if df is None or df.empty:
                log.info("  [%d/%d] %s %s: 0 根", done, total_calls, freq, q.date())
            else:
                rows = [{"ts_code": a.code, "freq": freq,
                         "bar_time": pd.to_datetime(r.trade_time).to_pydatetime(),
                         "open": float(r.open), "high": float(r.high),
                         "low": float(r.low), "close": float(r.close),
                         "vol": float(r.vol or 0)} for r in df.itertuples()]
                await _save(eng, rows)
                log.info("  [%d/%d] %s %s: %d 根", done, total_calls, freq,
                         q.date(), len(rows))
            time.sleep(THROTTLE)

    async with eng.connect() as c:
        for r in (await c.execute(text(
            "select freq, count(*), min(bar_time), max(bar_time) from index_bar "
            "where ts_code = :c group by freq order by freq"), {"c": a.code})).fetchall():
            log.info("  %-8s %6d 根  %s ~ %s", r[0], r[1], r[2], r[3])
    await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
