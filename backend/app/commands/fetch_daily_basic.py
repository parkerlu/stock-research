"""拉 tushare daily_basic —— 换手率/量比/市值, 存 parquet。

⚠️ 为什么要这个(2026-09-10):
   今天量化出裸K 与筹码模型的共同瓶颈: 【方向判别力只有 ±4pt】。
   在真的会动的票里, 模型最看多的 Top10% 涨占比 52.7%, 最看空的 Bot10%
   是 44.1% —— 基准 48.4%。也就是说它 70% 的本事在"哪只票要动了",
   只有 30% 在"往哪动"。
   而现在喂给模型的量是【绝对成交量】, 它只能看出"放量了", 看不出"谁在放量":
   小盘股换手 20% 和大盘股换手 0.5% 可能是同样的成交额, 含义完全不同。
   换手率(量÷流通股本)是无量纲的, 而且带方向性线索。

⚠️ 与 cyq(筹码) 的分工: 筹码看【持仓成本结构】(谁被套、谁在赚),
   daily_basic 看【交易活跃度】(今天换手多剧烈、相对自己历史算不算异常)。
   两者互补, 都不在 OHLCV 里。

⚠️ 存 parquet 不进库: 与 cyq.parquet 同一套路。这类宽表只有训练和打分要用,
   进库会让 daily_candle 那种高频读的表变慢, 也没必要。
   ⚠️ 但生成脚本必须进仓库 —— 本项目已经因为"临时在容器里写的脚本随镜像
      重建丢失"栽过三次(SAR 回测、形态实验、缓存面板)。

用法:
    python -m app.commands.fetch_daily_basic --start 20180101   # 首次全量
    python -m app.commands.fetch_daily_basic --days 5           # 每日增量
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging
import os
import time

import pandas as pd
import tushare as ts

from app.config import settings

log = logging.getLogger("daily_basic")
OUT = "/app/data/research/daily_basic.parquet"
# 只留真正用得上的列 —— pe/pb/ps 那些估值指标先不要:
# 它们有大量 NaN(亏损股 pe 为空), 而且属于基本面, 与这条线的"交易活跃度"不是一回事
COLS = ["ts_code", "trade_date", "turnover_rate", "turnover_rate_f",
        "volume_ratio", "float_share", "free_share", "circ_mv", "total_mv"]


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=None, help="全量起点 YYYYMMDD")
    ap.add_argument("--days", type=int, default=5, help="增量: 最近几个交易日")
    a = ap.parse_args()

    pro = ts.pro_api(settings.tushare_token)
    end = dt.date.today()
    if a.start:
        start = dt.datetime.strptime(a.start, "%Y%m%d").date()
    else:
        start = end - dt.timedelta(days=a.days * 3)
    cal = pro.trade_cal(exchange="SSE", start_date=start.strftime("%Y%m%d"),
                        end_date=end.strftime("%Y%m%d"), is_open="1")
    days = sorted(cal.cal_date.tolist())
    if not a.start:
        days = days[-a.days:]

    old = pd.read_parquet(OUT) if os.path.exists(OUT) else pd.DataFrame(columns=COLS)
    have = set()
    if len(old):
        old["trade_date"] = pd.to_datetime(old["trade_date"])
        have = set(old["trade_date"].dt.strftime("%Y%m%d"))
    todo = [d for d in days if d not in have] if a.start else days
    log.info("待拉 %d 天 (已有 %d 天, %s ~ %s)", len(todo), len(have),
             min(have) if have else "-", max(have) if have else "-")

    parts, fail = [], 0
    for i, d in enumerate(todo):
        for attempt in range(4):
            try:
                df = pro.daily_basic(trade_date=d, fields=",".join(COLS))
                if df is not None and len(df):
                    parts.append(df)
                break
            except Exception as exc:                      # noqa: BLE001
                # ⚠️ tushare 限流的报错文案会变, 一律 sleep 重试, 别只匹配关键词
                if attempt == 3:
                    fail += 1
                    log.warning("%s 拉取失败: %s", d, exc)
                else:
                    time.sleep(12)
        if (i + 1) % 100 == 0:
            log.info("  %d/%d, 累计 %s 行", i + 1, len(todo),
                     f"{sum(len(x) for x in parts):,}")

    if not parts:
        log.info("没有新数据")
        return
    new = pd.concat(parts, ignore_index=True)
    new["trade_date"] = pd.to_datetime(new["trade_date"])
    allrows = (pd.concat([old, new], ignore_index=True)
               .drop_duplicates(subset=["ts_code", "trade_date"], keep="last")
               .sort_values(["ts_code", "trade_date"]).reset_index(drop=True))
    allrows.to_parquet(OUT, index=False)
    log.info("写 %s: %s 行, %s ~ %s, %d 只 (失败 %d 天)", OUT, f"{len(allrows):,}",
             allrows.trade_date.min().date(), allrows.trade_date.max().date(),
             allrows.ts_code.nunique(), fail)


if __name__ == "__main__":
    asyncio.run(main())
