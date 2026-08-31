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
from sqlalchemy import delete, select
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
