"""Daily 15:30 scheduler — backfill all stock + ETF candles after market close."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, timedelta

import pandas as pd
import tushare as ts
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import settings
from app.datasources.tushare_provider import TuShareProvider
from app.db import async_session
from app.models.schema import DailyCandle, StockBasic

log = logging.getLogger("scheduler")

_scheduler: AsyncIOScheduler | None = None


async def _sync_stocks():
    """Backfill stale stock candles (same logic as /api/system/backfill/start)."""
    log.info("=== scheduled stock sync started ===")
    today = date.today()

    from app.services.paper_trading import _universe_ok

    async with async_session() as db:
        rows = (await db.execute(
            select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
        )).all()
        latest_rows = (await db.execute(
            select(DailyCandle.ts_code, func.max(DailyCandle.trade_date).label("latest"))
            .group_by(DailyCandle.ts_code)
        )).all()

    latest_map = {ts_code: d for ts_code, d in latest_rows}
    targets = []
    skipped = 0
    for ts_code, name, active in rows:
        if active is False:
            continue
        # 688/北交所/ST 连数据都不同步 —— 策略层本来就永远不买它们
        # (_universe_ok 白名单: 沪 60 / 深 00 / 深 30), 每晚白拉 ~900 只是
        # 纯浪费。代价: 这些票在行情页里会停留在最后一次同步的数据上。
        if not _universe_ok(ts_code, name):
            skipped += 1
            continue
        latest = latest_map.get(ts_code)
        if latest is None or latest < today:
            targets.append((ts_code, latest))
    log.info("universe 过滤跳过 %d 只 (688/北交所/ST)", skipped)

    log.info("stocks to update: %d", len(targets))
    if not targets:
        return

    provider = TuShareProvider(token=settings.tushare_token)
    inserted = 0

    for k, (ts_code, latest) in enumerate(targets):
        t_st = time.time()
        try:
            start = (latest + timedelta(days=1)) if latest else date(today.year - 7, 1, 1)
            if start <= today:
                df = await provider.fetch_daily(ts_code, start, today)
                if not df.empty:
                    df = df.copy()
                    df["source"] = "tushare"
                    records = df.to_dict("records")
                    async with async_session() as db:
                        stmt = pg_insert(DailyCandle).values(records)
                        stmt = stmt.on_conflict_do_nothing(
                            index_elements=["ts_code", "trade_date"]
                        )
                        await db.execute(stmt)
                        await db.commit()
                    inserted += len(records)
        except Exception as e:
            log.warning("stock %s failed: %s", ts_code, str(e)[:100])

        elapsed = time.time() - t_st
        if elapsed < 0.4:
            await asyncio.sleep(0.4 - elapsed)

        if (k + 1) % 100 == 0:
            log.info("  progress: %d/%d  rows=%d", k + 1, len(targets), inserted)

    log.info("=== stock sync done: %d rows inserted ===", inserted)


async def _sync_etfs():
    """Backfill ETF candles (top-50 by volume)."""
    log.info("=== scheduled ETF sync started ===")
    today = date.today()

    ts.set_token(settings.tushare_token)
    pro = ts.pro_api()

    try:
        basic = pro.fund_basic(market="E")
    except Exception as e:
        log.error("ETF basic fetch failed: %s", e)
        return

    basic["list_date_dt"] = pd.to_datetime(basic["list_date"], errors="coerce")
    cutoff = pd.Timestamp(date(today.year - 1, today.month, today.day))
    long_history = basic[basic["list_date_dt"] <= cutoff][["ts_code", "name", "list_date"]]

    daily = None
    for back in range(0, 5):
        d = (today - timedelta(days=back)).strftime("%Y%m%d")
        daily = pro.fund_daily(trade_date=d)
        if daily is not None and not daily.empty:
            break

    if daily is None or daily.empty:
        log.warning("no recent ETF daily data found")
        return

    eligible = (daily[daily["ts_code"].isin(long_history["ts_code"])]
                .sort_values("amount", ascending=False)
                .head(50)
                .merge(long_history, on="ts_code"))

    must_have = ["588060.SH"]
    for code in must_have:
        if code not in set(eligible["ts_code"]):
            extra = long_history[long_history["ts_code"] == code]
            if not extra.empty:
                eligible = pd.concat(
                    [eligible, extra.assign(amount=0)], ignore_index=True
                )

    # Upsert ETF metadata
    meta_rows = []
    for _, r in eligible.iterrows():
        meta_rows.append({
            "ts_code": r["ts_code"],
            "symbol": r["ts_code"].split(".")[0],
            "name": r.get("name") or "",
            "area": None,
            "industry": "ETF",
            "market": "ETF",
            "list_date": pd.to_datetime(r["list_date"]).date()
                          if pd.notna(r.get("list_date")) else None,
            "is_active": True,
        })
    async with async_session() as db:
        stmt = pg_insert(StockBasic).values(meta_rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["ts_code"],
            set_={
                "name": stmt.excluded.name,
                "industry": stmt.excluded.industry,
                "market": stmt.excluded.market,
                "is_active": stmt.excluded.is_active,
            },
        )
        await db.execute(stmt)
        await db.commit()

    # Fetch candles for ETFs with <100 bars
    codes = eligible["ts_code"].tolist()
    async with async_session() as db:
        bar_rows = (await db.execute(
            select(DailyCandle.ts_code, func.count())
            .where(DailyCandle.ts_code.in_(codes))
            .group_by(DailyCandle.ts_code)
        )).all()
    sync_map = {ts_code: int(n) for ts_code, n in bar_rows}
    todo = [c for c in codes if sync_map.get(c, 0) < 100]

    if not todo:
        log.info("all ETFs up to date")
        return

    start = date(today.year - 7, 1, 1)
    inserted = 0
    for i, code in enumerate(todo, 1):
        t_st = time.time()
        try:
            df = pro.fund_daily(
                ts_code=code,
                start_date=start.strftime("%Y%m%d"),
                end_date=today.strftime("%Y%m%d"),
            )
            if df is not None and not df.empty:
                df = df.copy()
                df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
                df["adj_factor"] = 1.0
                df["source"] = "tushare"
                keep = ["ts_code", "trade_date", "open", "high", "low", "close",
                        "vol", "amount", "adj_factor", "source"]
                records = df[keep].to_dict("records")
                async with async_session() as db:
                    stmt = pg_insert(DailyCandle).values(records)
                    stmt = stmt.on_conflict_do_nothing(
                        index_elements=["ts_code", "trade_date"]
                    )
                    await db.execute(stmt)
                    await db.commit()
                inserted += len(records)
        except Exception as e:
            log.warning("ETF %s failed: %s", code, str(e)[:100])
        dt = time.time() - t_st
        if dt < 0.4:
            await asyncio.sleep(0.4 - dt)

    log.info("=== ETF sync done: %d rows inserted ===", inserted)



async def _snapshot_to_pool() -> None:
    """把当天的 2 类买点写进股票池, 在网页上直接能看。只保留最近 30 天。"""
    from sqlalchemy import delete, select as sel

    from app.models.schema import ChanSignal, StockPool, StockPoolItem

    async with async_session() as db:
        day = (await db.execute(
            sel(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if not day:
            return
        codes = (await db.execute(
            sel(ChanSignal.ts_code)
            .where(ChanSignal.trade_date == day, ChanSignal.kind == "2")
        )).scalars().all()
        if not codes:
            return
        name = f"信号 {day:%m-%d}"
        pool = (await db.execute(sel(StockPool).where(StockPool.name == name))
                ).scalar_one_or_none()
        if pool is None:
            pool = StockPool(name=name,
                             description=f"{day} 收盘算出的缠论2类买点, 次日开盘可买")
            db.add(pool)
            await db.flush()
        else:
            await db.execute(delete(StockPoolItem)
                             .where(StockPoolItem.pool_id == pool.id))
        for c in sorted(set(codes)):
            db.add(StockPoolItem(pool_id=pool.id, ts_code=c))

        # 只留最近 30 个信号池, 免得列表越积越长
        olds = (await db.execute(
            sel(StockPool.id).where(StockPool.name.like("信号 %"))
            .order_by(StockPool.created_at.desc()).offset(30)
        )).scalars().all()
        if olds:
            await db.execute(delete(StockPoolItem)
                             .where(StockPoolItem.pool_id.in_(olds)))
            await db.execute(delete(StockPool).where(StockPool.id.in_(olds)))
        await db.commit()
        log.info("股票池 %s: %d 只", name, len(set(codes)))


async def _settle_paper() -> None:
    """虚拟盘逐日结算 —— 必须排在行情同步之后, 否则用的是昨天的价格。"""
    from sqlalchemy import select

    from app.models.schema import PaperAccount
    from app.services import paper_trading as pt

    async with async_session() as db:
        # 只自动推进实操盘 —— 演示盘由人手动回放, 不能被定时任务推着走
        accounts = (await db.execute(
            select(PaperAccount).where(PaperAccount.is_active.is_(True),
                                       PaperAccount.name.notlike("demo%"))
        )).scalars().all()
        if not accounts:
            return
        end = (await db.execute(
            select(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if not end:
            return
        for acct in accounts:
            # ⚠️ 还没结算过时, 起始日本身也要算进去。原来一律用 > start, 于是
            # started_on 落在交易日的账户会跳过自己的第一天 —— 实操盘定在
            # 2026-09-01 (周二, 交易日), 会从 9/2 才开始。之前没暴露是因为演示盘
            # 的起点 2022-01-01 是节假日, > 它正好得到 01-04。
            if acct.last_run_date:
                start, inclusive = acct.last_run_date, False
            else:
                start, inclusive = acct.started_on, True
            days = (await db.execute(
                select(DailyCandle.trade_date)
                .where(DailyCandle.trade_date >= start if inclusive
                       else DailyCandle.trade_date > start,
                       DailyCandle.trade_date <= end)
                .group_by(DailyCandle.trade_date)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
            for d in days:
                res = await pt.run_day(db, acct, d)
                await db.commit()
                log.info("paper[%s] %s -> 净值 %s, 动作 %d",
                         acct.name, d, res.get("equity"), len(res.get("actions", [])))


async def daily_sync_job():
    """Main scheduled job: sync stocks then ETFs, then settle paper accounts."""
    log.info("====== daily sync triggered at 15:30 ======")
    await _sync_stocks()
    await _sync_etfs()
    # 信号增量 + 虚拟盘结算。策略 tdx-dual-kdj —— 缠论已下架(三条未来函数,
    # 详见 artifact 9cd4b91f)。当前策略纯公式, 四道闸全过: 因果性撤销率 0.05%,
    # 样本内 t=22, 两个真样本外窗口 z=7.62 / 3.58 且均 12/12 跑赢随机对照。
    # ⚠️ 只做增量(since=表内最新日), 不要全历史重建 —— 见 signal_build 注释。
    try:
        from sqlalchemy import func as _f

        from app.models.schema import StrategySignal
        from app.services.signal_build import build as build_signals

        async with async_session() as _db:
            last = (await _db.execute(
                select(_f.max(StrategySignal.trade_date))
                .where(StrategySignal.strategy == PAPER_STRATEGY))).scalar_one_or_none()
        log.info("信号增量: %s", await build_signals(PAPER_STRATEGY, since=last))
    except Exception as exc:                      # noqa: BLE001
        log.exception("signal build failed: %s", exc)
    try:
        await _settle_paper()
    except Exception as exc:                      # noqa: BLE001
        # 虚拟盘出错不影响行情同步 —— run_day 对已结算日期幂等, 下次补上
        log.exception("paper settlement failed: %s", exc)

    # 训练指标(买卖很准v3 / 主力吸筹 / 低点组合v2) —— 拉当日筹码后重算评分。
    # 放在最后: 依赖当日 K 线已入库, 且失败不该影响前面任何一步。
    # ⚠️ 这里【会】重训: update_indicators 直接调 build_*(全量), 含 walk-forward
    # 逐年重训。之所以没问题, 是因为切分是 `训练年 < 打分年` —— 2026 年的信号
    # 只用 2017-2025 训出的模型打分, 不偷看。但模型必须固定随机种子, 否则
    # 每日重训会让历史信号的档位天天漂(已在 build_* 里设 random_state=42)。
    try:
        from app.commands.update_indicators import main as update_indicators
        import sys

        argv = sys.argv
        sys.argv = ["update_indicators", "--days", "10"]
        try:
            await update_indicators()
        finally:
            sys.argv = argv
    except Exception as exc:                      # noqa: BLE001
        log.exception("训练指标更新失败: %s", exc)

    log.info("====== daily sync complete ======")


def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")
    _scheduler.add_job(
        daily_sync_job,
        CronTrigger(hour=15, minute=30, timezone="Asia/Shanghai"),
        id="daily_sync",
        replace_existing=True,
    )
    _scheduler.start()
    log.info("scheduler started — daily sync at 15:30 CST")


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
