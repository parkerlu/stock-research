"""Daily 15:30 scheduler — backfill all stock + ETF candles after market close."""
from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta

import pandas as pd
import tushare as ts
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from sqlalchemy import func, select, text
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
        # ⚠️⚠️ 结算只推进到【倒数第二个】交易日 —— 2026-09-10 修正。
        #    默认 entry_mode 是 next_open: run_day(T) 用 T 日的信号, 但成交价是
        #    T+1 的开盘价。最新交易日的次日还没发生, scan_signals 因此返回 0
        #    (实测 09-09 -> 0 个, 09-08 -> 28 个), 于是当天一笔都买不进;
        #    而 run_day 对已结算日期幂等, 第二天也不会重来 ——
        #    结果是【每天的信号都被永久浪费】, 而且不报错。
        #    回放时没暴露, 是因为历史数据里 T+1 的行情早就在库里了。
        #    代价: 净值和出场记录比最新交易日滞后一天。对虚拟盘无实质影响
        #    (出场仍按 T 日的 OHLC 判定, 只是记录晚一天写)。
        last2 = (await db.execute(
            select(DailyCandle.trade_date)
            .group_by(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(2)
        )).scalars().all()
        if len(last2) < 2:
            return
        end = last2[1]
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
    # ⚠️ 先按【交易日】整批补, 再让 _sync_stocks 逐只兜底。
    #    只靠逐只会静默留缺口: 4400 只挨个调 API, 一次限流就少一批,
    #    界面上看不出来, 只是选股结果变少。2026-09-07 就这样只入库
    #    3279/5549 只, 当天所有训练指标都出不了完整信号。
    #    整批一次调用 3 秒补完一天, 逐只要半小时还跑不完。
    try:
        from app.commands.fill_daily import fill as _fill_daily
        import tushare as _ts
        from app.config import settings as _cfg
        _pro = _ts.pro_api(_cfg.tushare_token)
        _end = date.today()
        _cal = _pro.trade_cal(exchange="SSE",
                              start_date=(_end - timedelta(days=15)).strftime("%Y%m%d"),
                              end_date=_end.strftime("%Y%m%d"), is_open="1")
        _days = sorted(_cal.cal_date.tolist())[-5:]
        log.info("整批补日线 %s", _days)
        await _fill_daily(_days)
    except Exception as exc:  # noqa: BLE001
        log.warning("整批补日线失败(继续走逐只): %s", exc)

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
    # 拉升预警的信号增量。
    # ⚠️ 它依赖 dongli_signal + pump_signal, 而这两张表要到下面的 update_indicators
    # 才更新, 所以这一次写的是【昨天及以前】的信号。今天的信号由 update_indicators
    # 跑完后的那次结算之前补不上 —— 但结算已经移到最后, 且 run_day 幂等,
    # 明天推进时会带上。要彻底对齐, 应该把 sync_liftalert_signals 也并进
    # update_indicators 的信号同步段(那里已有突破/SAR/周线版/形态/筹码五路)。
    try:
        from app.commands.sync_liftalert_signals import main as sync_lift
        import sys as _sys
        _argv = _sys.argv
        _sys.argv = ["sync_liftalert_signals"]
        try:
            await sync_lift()
        finally:
            _sys.argv = _argv
    except Exception as exc:                      # noqa: BLE001
        log.exception("拉升预警信号同步失败: %s", exc)

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
    # ⚠️ SystemExit 要单独拦: argparse 报错走 sys.exit(), 而 SystemExit 继承
    #    BaseException, except Exception 拦不住, 会把整个 job 打断
    #    (2026-09-09: build_panels 读到 --days 就退出, 后面全没跑)。
    except SystemExit as exc:
        log.error("训练指标更新以退出码 %s 结束 —— 检查子命令参数", exc.code)
    except Exception as exc:                      # noqa: BLE001
        log.exception("训练指标更新失败: %s", exc)

    # ⚠️⚠️ 结算必须排在 update_indicators 【之后】—— 2026-09-10 修正。
    #    scan_signals 找的是 `trade_date == 当天` 的信号, 而当天的信号是在
    #    update_indicators 里生成的。原来结算排在它前面, 于是 run_day(T) 跑的
    #    时候 T 日信号还不存在 -> 当天不买入; 而 run_day 对已结算日期幂等,
    #    第二天也不会补 -> 【当天的信号永远用不上】。
    #    这个顺序以前之所以没暴露, 是靠一个巧合: 15:30 收盘数据还没发全时
    #    max(daily_candle) 还停在 T-1, 结算推进的是 T-1(它的信号昨天已生成)。
    #    但第一步的 fill_daily 恰恰会把当日数据补进来, 这个巧合并不可靠。
    #    移到最后之后, 时序变成确定的: 补行情 -> 算当日信号 -> 结算推进到当日。
    #    仍然满足原注释的要求(排在行情同步之后, 否则用的是昨天的价格)。
    try:
        await _settle_paper()
    except Exception as exc:                      # noqa: BLE001
        # 虚拟盘出错不影响前面任何一步 —— run_day 对已结算日期幂等, 下次补上
        log.exception("paper settlement failed: %s", exc)

    log.info("====== daily sync complete ======")


async def hourly_index_bars_job() -> None:
    """每小时补 1 片中证1000 的分钟 K 线。

    ⚠️ 为什么是每小时 1 次而不是每天 6 次: tushare 的 stk_mins 限速会动态收紧,
    实测已经到【1 次/小时】。一口气发 6 个只有第一个能成, 剩下 5 个全被拒 ——
    那样补齐 2025 至今(21 片)要 21 天; 每小时 1 次则约 21 小时。
    补齐后 todo 里只剩当前季度, 变成每小时刷新一次最新几根, 开销可以忽略。
    """
    try:
        from app.commands.sync_index_bars import main as sync_ibars
        import sys as _s

        _a = _s.argv
        _s.argv = ["sync_index_bars", "--start", "2025-01-01",
                   "--max-calls", "1", "--quiet-wk"]
        try:
            await sync_ibars()
        finally:
            _s.argv = _a
    except Exception as exc:                      # noqa: BLE001
        log.exception("指数分钟K线补片失败: %s", exc)


async def ensure_daily_job() -> None:
    """数据完整性自愈 —— 每小时跑, 自己判断该不该补。

    ⚠️ 为什么要独立成一个每小时的任务(2026-09-08 用户提出):
       收盘数据不是 15:30 就发全的, 一次性同步必然留缺口, 而缺口是【静默】的
       —— 界面上只是选股结果变少, 没人会发现。实测 09-03/09-04 各缺 899 个
       标的躺了四天。

    ⚠️ 幂等: 已完整的日子直接跳过, 一小时跑一次几乎不花开销(只查两个 count)。
    ⚠️ 只在【可能有新数据】的时段跑: 9 点前和 23 点后不折腾。
    ⚠️ 补完【数据完整】才重算指标 —— 拿残缺数据算比不算更糟, 会写出一批
       残缺信号, 看起来像"今天没信号"而不是"数据没到"。
    """
    now = datetime.now()
    if not (9 <= now.hour <= 23):
        return
    import sys as _s
    from app.commands.ensure_daily import main as ensure_main

    argv = _s.argv
    was_complete = True
    try:
        _s.argv = ["ensure_daily", "--days", "5", "--fix"]
        await ensure_main()
    except SystemExit:
        was_complete = False       # 补不齐, 下一小时再试
    except Exception as exc:       # noqa: BLE001
        log.exception("完整性自愈失败: %s", exc)
        return
    finally:
        _s.argv = argv

    # 数据完整了, 且指标还落后于日线 -> 补算一次
    try:
        async with async_session() as db:
            px = (await db.execute(text(
                "select max(trade_date) from daily_candle"))).scalar()
            sig = (await db.execute(text(
                "select max(trade_date) from breakout_signal"))).scalar()
        if was_complete and px and sig and sig < px:
            log.info("数据已完整(%s)而指标停在 %s, 触发重算", px, sig)
            from app.commands.update_indicators import main as upd
            argv = _s.argv
            try:
                _s.argv = ["update_indicators", "--days", "5"]
                await upd()
            finally:
                _s.argv = argv
    except SystemExit as exc:
        log.error("指标补算以退出码 %s 结束 —— 检查子命令参数", exc.code)
    except Exception as exc:       # noqa: BLE001
        log.exception("指标补算失败: %s", exc)



def start_scheduler():
    global _scheduler
    if _scheduler is not None:
        return
    _scheduler = AsyncIOScheduler(timezone="Asia/Shanghai")

    # ⚠️ 每小时的完整性自愈 —— 放在最前面注册, 它是其它一切的前提
    _scheduler.add_job(
        ensure_daily_job,
        CronTrigger(minute=25, timezone="Asia/Shanghai"),
        id="ensure_daily", replace_existing=True, max_instances=1,
    )
    _scheduler.add_job(
        daily_sync_job,
        CronTrigger(hour=15, minute=30, timezone="Asia/Shanghai"),
        id="daily_sync",
        replace_existing=True,
    )
    # 分钟K线补片 —— 每小时 1 次, 贴着 tushare 的 1次/小时 限速走
    _scheduler.add_job(
        hourly_index_bars_job,
        CronTrigger(minute=7, timezone="Asia/Shanghai"),
        id="index_bars_hourly",
        replace_existing=True,
    )
    _scheduler.start()
    # ⚠️ 自检: 任务必须真的注册上了才算启动成功。
    #    2026-09-09 踩过 —— ensure_daily_job 的 def 插在了 start_scheduler()
    #    函数体中间, 把它腰斩: add_job/start() 全落进了 ensure_daily_job 里,
    #    而那个函数因为注册代码在它自己体内, 永远不会被调用。于是调度器
    #    对象建了、任务一个没有、【一条错误日志都没有】, 白跑两天:
    #    15:30 收盘同步没跑、每小时自愈没跑, 当天库里只有 10 只票。
    #    光靠 "scheduler started" 这句日志是查不出来的 —— 它本身也没打印。
    jobs = {j.id for j in _scheduler.get_jobs()}
    want = {"ensure_daily", "daily_sync", "index_bars_hourly"}
    if jobs != want:
        log.error("⚠️ 定时任务注册不全: 期望 %s, 实际 %s —— 收盘同步/自愈不会跑",
                  sorted(want), sorted(jobs))
    else:
        log.info("scheduler started — %d 个任务已注册: %s",
                 len(jobs), ", ".join(sorted(jobs)))


def stop_scheduler():
    global _scheduler
    if _scheduler:
        _scheduler.shutdown(wait=False)
        _scheduler = None
