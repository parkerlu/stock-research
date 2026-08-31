"""一次性算出全市场全历史的缠论买点, 落表供虚拟盘回放使用.

为什么要预计算
--------------
虚拟盘要支持"点一下走一天"的回放。原来每推进一天都要对 4400 只票各取 400 天
K 线再跑一遍缠论 —— 生产上 75 秒/天, 交互完全不可用。

但缠论买点只依赖该票自己的历史, 与回放进度无关 —— 整段历史一次算完即可。
实测全市场一遍约 1~2 分钟, 之后每个回放日就是一次索引查询(毫秒级)。

落表的 trade_date 是**可操作日** = 分型日 + CONFIRM_LAG。
分型是"3 根合并K 的中间那根", 右邻那根出来前不知道它是分型, 直接用分型日
就是未来函数。fractal_date 一并存下来, 方便核对两者的差距。
"""
from __future__ import annotations

import logging
import time
from datetime import date

import pandas as pd
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import async_session
from app.models.schema import ChanSignal, DailyCandle, StockBasic
from app.services.chanlun import find_class1_buys, find_class2_buys
from app.services.paper_trading import CONFIRM_LAG, MIN_BARS, _universe_ok

log = logging.getLogger(__name__)


def _signals_for(df: pd.DataFrame) -> list[tuple[date, date, str]]:
    """返回 [(可操作日, 分型日, 类型)]。"""
    c = df.close.values
    h = df.high.values
    l = df.low.values
    if len(c) < MIN_BARS:
        return []
    c1 = find_class1_buys(c, h, l)
    c2 = find_class2_buys(c, h, l, {b.bar_idx for b in c1})
    dates = df.trade_date.values
    out: list[tuple[date, date, str]] = []
    n = len(c)
    for kind, buys in (("1", c1), ("2", c2)):
        for b in buys:
            j = b.bar_idx + CONFIRM_LAG
            if 0 <= j < n:
                out.append((pd.Timestamp(dates[j]).date(),
                            pd.Timestamp(dates[b.bar_idx]).date(), kind))
    return out


async def build_all(rebuild: bool = False, on_progress=None) -> dict:
    """重算全市场信号。rebuild=True 先清空, 否则只补没算过的票。"""
    t0 = time.time()
    async with async_session() as db:
        basics = (await db.execute(
            select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
        )).all()
        codes = sorted(c for c, n, a in basics
                       if a is not False and _universe_ok(c, n))
        if rebuild:
            await db.execute(delete(ChanSignal))
            await db.commit()
            done: set[str] = set()
        else:
            done = set((await db.execute(
                select(ChanSignal.ts_code).group_by(ChanSignal.ts_code)
            )).scalars().all())

    todo = [c for c in codes if c not in done]
    log.info("chan_signal: universe %d, 待算 %d", len(codes), len(todo))
    inserted = 0

    # 分批: 一次性拉全市场会压垮小内存机器 (见 portfolio_scan 的教训)
    CHUNK = 200
    for i in range(0, len(todo), CHUNK):
        batch = todo[i:i + CHUNK]
        async with async_session() as db:
            rows = (await db.execute(
                select(DailyCandle.ts_code, DailyCandle.trade_date, DailyCandle.open,
                       DailyCandle.high, DailyCandle.low, DailyCandle.close,
                       DailyCandle.adj_factor)
                .where(DailyCandle.ts_code.in_(batch))
                .order_by(DailyCandle.ts_code, DailyCandle.trade_date)
            )).all()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high",
                                             "low", "close", "adj_factor"])
            for c in ("open", "high", "low", "close", "adj_factor"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            payload: list[dict] = []
            for cd, g in df.groupby("ts_code", sort=False):
                g = g.sort_values("trade_date").reset_index(drop=True)
                # 前复权 —— 与 quote_service 同口径
                latest = g.adj_factor.iloc[-1]
                if latest and latest > 0:
                    f = (g.adj_factor / latest).values
                    for c in ("open", "high", "low", "close"):
                        g[c] = (g[c].values * f).round(4)
                for act, frac, kind in _signals_for(g):
                    payload.append({"ts_code": cd, "trade_date": act,
                                    "fractal_date": frac, "kind": kind})
            if payload:
                stmt = pg_insert(ChanSignal).values(payload)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["ts_code", "trade_date", "kind"])
                res = await db.execute(stmt)
                await db.commit()
                inserted += res.rowcount or 0
        if on_progress:
            on_progress(min(i + CHUNK, len(todo)), len(todo))

    el = time.time() - t0
    log.info("chan_signal: 完成, 新增 %d 条, 耗时 %.0fs", inserted, el)
    return {"universe": len(codes), "computed": len(todo),
            "inserted": inserted, "elapsed_sec": round(el, 1)}


async def refresh_tail(lookback_bars: int = 500, since_days: int = 15) -> dict:
    """每晚增量补新信号 —— 只算每只票最近 lookback_bars 根K, 只写最近的信号.

    为什么不能只靠 build_all(rebuild=False)
    ------------------------------------
    那个只补"表里一条都没有"的票。老票每天出的新买点它一条都不会加, 实操盘
    就永远等不到新信号。所以夜里必须跑这个。

    为什么用尾部窗口而不是全历史
    --------------------------
    全历史一遍生产上要 400 秒; 尾部 500 根只读 ~1/20 的行, 十几秒就够, 适合
    每晚跟在行情同步后面跑。

    代价要说清楚: 缠论的合并/分型是从序列头部递推的, 截断起点会让窗口最左边
    若干根的合并结果与全历史算出来的不完全一致。500 根的窗口相对 since_days
    (15 天) 有极大余量, 受影响的只是窗口最左端, 落不到我们要写的日期上。
    """
    t0 = time.time()
    async with async_session() as db:
        basics = (await db.execute(
            select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
        )).all()
        codes = sorted(c for c, n, a in basics
                       if a is not False and _universe_ok(c, n))
        end = (await db.execute(
            select(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if not end:
            return {"inserted": 0, "reason": "库里没有行情"}
        # 只写最近 since_days 个交易日的信号, 更早的由 build_all 全历史算过
        recent = (await db.execute(
            select(DailyCandle.trade_date).group_by(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(since_days)
        )).scalars().all()
        floor_date = min(recent)

    inserted = 0
    CHUNK = 200
    for i in range(0, len(codes), CHUNK):
        batch = codes[i:i + CHUNK]
        async with async_session() as db:
            # 每只票各取最后 lookback_bars 根 —— 用窗口函数一次拿完, 不逐票查
            sub = (
                select(
                    DailyCandle.ts_code, DailyCandle.trade_date, DailyCandle.open,
                    DailyCandle.high, DailyCandle.low, DailyCandle.close,
                    DailyCandle.adj_factor,
                    func.row_number().over(
                        partition_by=DailyCandle.ts_code,
                        order_by=DailyCandle.trade_date.desc()).label("rn"),
                )
                .where(DailyCandle.ts_code.in_(batch))
                .subquery()
            )
            rows = (await db.execute(
                select(sub.c.ts_code, sub.c.trade_date, sub.c.open, sub.c.high,
                       sub.c.low, sub.c.close, sub.c.adj_factor)
                .where(sub.c.rn <= lookback_bars)
                .order_by(sub.c.ts_code, sub.c.trade_date)
            )).all()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "open", "high",
                                             "low", "close", "adj_factor"])
            for c in ("open", "high", "low", "close", "adj_factor"):
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
            df = df.dropna(subset=["open", "high", "low", "close"])
            df["trade_date"] = pd.to_datetime(df["trade_date"])

            payload: list[dict] = []
            for cd, g in df.groupby("ts_code", sort=False):
                g = g.sort_values("trade_date").reset_index(drop=True)
                latest = g.adj_factor.iloc[-1]
                if latest and latest > 0:
                    f = (g.adj_factor / latest).values
                    for c in ("open", "high", "low", "close"):
                        g[c] = (g[c].values * f).round(4)
                for act, frac, kind in _signals_for(g):
                    if act >= floor_date:
                        payload.append({"ts_code": cd, "trade_date": act,
                                        "fractal_date": frac, "kind": kind})
            if payload:
                stmt = pg_insert(ChanSignal).values(payload)
                stmt = stmt.on_conflict_do_nothing(
                    index_elements=["ts_code", "trade_date", "kind"])
                res = await db.execute(stmt)
                await db.commit()
                inserted += res.rowcount or 0

    el = time.time() - t0
    log.info("chan_signal 增量: %s 起, 新增 %d 条, 耗时 %.0fs",
             floor_date, inserted, el)
    return {"since": str(floor_date), "inserted": inserted,
            "elapsed_sec": round(el, 1)}
