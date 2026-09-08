"""数据完整性自愈 —— 独立命令, 每小时跑一次, 自己判断该不该拿、拿全没有。

⚠️ 为什么要有它(2026-09-08 用户提出: "每天的数据永远补不齐, 每次都要我问"):
   原流程是 15:30 同步一次就完事, 三个致命缺陷 ——
     1. 15:30 时收盘数据还没发全(实测当天只拿到 2301/6273)
     2. 没有完整性判据, 拿到半截也算"成功"
     3. 没有重试, 缺了就一直缺, 只能等人发现

⚠️ 完整性判据不能只看条数, 要看【哪些标的缺】:
   基准 = 最近 5 个交易日里出现过 >=3 次的标的(即"在交易的东西")。
   只比总数会误判 —— 退市的票会让历史条数虚高。

⚠️ 整批接口 pro.daily 只覆盖 A 股, 不含 ETF/北交所, 比全量少约 700 个。
   实测 09-07 整批补完 5549 只, 而正常日是 6273 只。
   所以必须【整批拿大头 + 逐只补差额】: 整批 3 秒搞定 5500 个,
   剩下几百个逐只拉也就一两分钟。只用其中一种都补不齐。

⚠️ 幂等: 已完整的日子直接跳过, 一小时跑一次也不浪费。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import logging

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from app.config import settings

log = logging.getLogger("ensure_daily")
OK_RATIO = 0.98        # 缺口小于 2% 视为完整(总有停牌/退市的零头)


async def _universe_and_have(eng, days: int):
    """返回 (在交易的标的集合, {日期: 当日已有的标的集合})。"""
    async with eng.connect() as c:
        cal = [r[0] for r in (await c.execute(text(
            "select distinct trade_date from daily_candle "
            "where trade_date >= current_date - 30 order by trade_date desc limit 6"
        ))).fetchall()]
        if not cal:
            return set(), {}
        # 最近 5 个交易日里出现 >=3 次的, 算"在交易"
        uni = {r[0] for r in (await c.execute(text("""
            select ts_code from daily_candle where trade_date = any(:ds)
            group by ts_code having count(*) >= 3
        """), {"ds": cal[:5]})).fetchall()}
        have = {}
        for d in cal[:days]:
            have[d] = {r[0] for r in (await c.execute(text(
                "select ts_code from daily_candle where trade_date = :d"),
                {"d": d})).fetchall()}
    return uni, have


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=5)
    ap.add_argument("--fix", action="store_true")
    ap.add_argument("--max-per-stock", type=int, default=800,
                    help="逐只补的上限, 超过说明整批那步出了问题, 不硬扛")
    a = ap.parse_args()

    eng = create_async_engine(settings.database_url)
    try:
        uni, have = await _universe_and_have(eng, a.days)
        if not uni:
            log.warning("库里没有近期数据"); return
        log.info("在交易的标的 %d 个 (近5个交易日出现>=3次)", len(uni))

        gaps = {}
        for d in sorted(have, reverse=True):
            miss = uni - have[d]
            ok = len(have[d]) >= len(uni) * OK_RATIO
            log.info("  %s  %5d/%d  缺 %4d  %s", d, len(have[d]), len(uni),
                     len(miss), "✅" if ok else "❌")
            if not ok:
                gaps[d] = miss

        # 今天是工作日但库里一条都没有 -> 也算缺口
        today = dt.date.today()
        if today not in have and today.weekday() < 5:
            log.info("  %s      0/%d  缺 %4d  ❌ 完全缺失", today, len(uni), len(uni))
            gaps[today] = uni

        if not gaps:
            log.info("✅ 全部完整")
            return
        if not a.fix:
            log.warning("%d 天有缺口, 加 --fix 自动补", len(gaps))
            raise SystemExit(1)

        # --- 第一步: 整批。3 秒拿到 A 股大头 ---
        from app.commands.fill_daily import fill
        log.info("整批补 %s", [str(d) for d in gaps])
        await fill([d.strftime("%Y%m%d") for d in gaps])

        # --- 第二步: 逐只补整批漏掉的(ETF/北交所等) ---
        uni2, have2 = await _universe_and_have(eng, a.days)
        from app.datasources.manager import DataSourceManager
        from app.routers.quotes import get_manager
        mgr = get_manager()
        from app.models.schema import DailyCandle
        from sqlalchemy.dialects.postgresql import insert as pg_insert
        import pandas as pd

        for d in sorted(gaps, reverse=True):
            miss = sorted(uni2 - have2.get(d, set()))
            if not miss:
                continue
            if len(miss) > a.max_per_stock:
                log.warning("  %s 仍缺 %d 个, 超过上限 %d —— 整批那步可能出了问题, 跳过",
                            d, len(miss), a.max_per_stock)
                continue
            log.info("  %s 逐只补 %d 个 ...", d, len(miss))
            got = 0
            for code in miss:
                try:
                    df = await mgr.fetch_daily(code, d - dt.timedelta(days=12), d)
                except Exception:
                    continue
                if df is None or df.empty:
                    continue
                rows = [r for r in df.to_dict("records")
                        if pd.Timestamp(r["trade_date"]).date() == d]
                if not rows:
                    continue
                async with eng.begin() as c:
                    st = pg_insert(DailyCandle).values(rows)
                    await c.execute(st.on_conflict_do_nothing(
                        index_elements=["ts_code", "trade_date"]))
                got += 1
            log.info("  %s 逐只补进 %d 个", d, got)

        # --- 复查 ---
        uni3, have3 = await _universe_and_have(eng, a.days)
        left = [str(d) for d in have3
                if len(have3[d]) < len(uni3) * OK_RATIO]
        if left:
            log.warning("仍不完整: %s (源上可能还没发全, 下一小时会再试)", left)
            raise SystemExit(1)
        log.info("✅ 全部补齐")
    finally:
        await eng.dispose()


if __name__ == "__main__":
    asyncio.run(main())
