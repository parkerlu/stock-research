"""每日封存"今天算出来的信号", 日后用真实前进数据量重绘漂移.

背景
----
缠论的笔是 ZigZag 式端点: 连续同类分型只保留更极端的那个。底分型可能在若干
根之后被更低的底分型顶掉, 对应的 2 类买点随之消失。所以"用全历史重算一遍"
得到的信号集, 和"当天收盘时真能算出来的"不是一回事。

截断复算能模拟这件事, 但那类脚本自己就可能写错 —— 我写过一版, 只遍历全历史
最终存在的买点再回头找首次出现时间, 等于把幸存者偏差写进了检验本身, 得出
"零漂移"的错误结论。

这里改用最笨也最可靠的办法: 当天算出什么就存什么, 只增不改; 日后拿它跟重算
结果比对。不依赖任何模拟。

顺带一个重要性质
--------------
chan_signal 的每日增量 (refresh_tail) 只 INSERT、从不删除。所以只要**以后
永远不再跑 build_all(rebuild=True)**, chan_signal 自己就会累积成因果的 ——
包括那些日后被重绘掉的信号。快照表既是度量工具, 也是这条纪律的对照物:
两者对不上, 就说明中间有人做过全历史重建。
"""
from __future__ import annotations

import logging
from datetime import date

from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import async_session
from app.models.schema import ChanSignal, DailyCandle, SignalSnapshot

log = logging.getLogger(__name__)


async def take_snapshot(day: date | None = None) -> dict:
    """把 chan_signal 里 trade_date == day 的信号原样封存。"""
    async with async_session() as db:
        if day is None:
            day = (await db.execute(
                select(DailyCandle.trade_date)
                .order_by(DailyCandle.trade_date.desc()).limit(1)
            )).scalar_one_or_none()
        if day is None:
            return {"snapshot_date": None, "inserted": 0, "reason": "库里没有行情"}

        rows = (await db.execute(
            select(ChanSignal.ts_code, ChanSignal.trade_date,
                   ChanSignal.fractal_date, ChanSignal.kind)
            .where(ChanSignal.trade_date == day)
        )).all()
        if not rows:
            return {"snapshot_date": str(day), "inserted": 0}

        codes = [r[0] for r in rows]
        closes = dict((await db.execute(
            select(DailyCandle.ts_code, DailyCandle.close)
            .where(DailyCandle.ts_code.in_(codes), DailyCandle.trade_date == day)
        )).all())
        # 20 日均额, 与选股口径一致
        sub = (
            select(DailyCandle.ts_code,
                   func.avg(DailyCandle.amount).label("a20"))
            .where(DailyCandle.ts_code.in_(codes),
                   DailyCandle.trade_date <= day,
                   DailyCandle.trade_date > day - __import__("datetime").timedelta(days=40))
            .group_by(DailyCandle.ts_code)
        )
        amts = dict((await db.execute(sub)).all())

        payload = [{
            "snapshot_date": day, "ts_code": c, "trade_date": td,
            "fractal_date": fd, "kind": k,
            "close_at_signal": float(closes[c]) if closes.get(c) is not None else None,
            "amount_20d_k": float(amts[c]) if amts.get(c) is not None else None,
        } for c, td, fd, k in rows]

        stmt = pg_insert(SignalSnapshot).values(payload)
        stmt = stmt.on_conflict_do_nothing(
            index_elements=["snapshot_date", "ts_code", "trade_date", "kind"])
        res = await db.execute(stmt)
        await db.commit()
        n = res.rowcount or 0

    log.info("信号快照 %s: 封存 %d 条", day, n)
    return {"snapshot_date": str(day), "inserted": n}


async def check_drift(snapshot_date: date | None = None) -> dict:
    """拿快照跟当前 chan_signal 比对, 标出哪些信号已经被重绘掉了。"""
    async with async_session() as db:
        if snapshot_date is None:
            snapshot_date = (await db.execute(
                select(SignalSnapshot.snapshot_date)
                .where(SignalSnapshot.still_valid.is_(None))
                .order_by(SignalSnapshot.snapshot_date).limit(1)
            )).scalar_one_or_none()
        if snapshot_date is None:
            return {"checked": 0, "reason": "没有待核对的快照"}

        snap = (await db.execute(
            select(SignalSnapshot.id, SignalSnapshot.ts_code,
                   SignalSnapshot.trade_date, SignalSnapshot.kind)
            .where(SignalSnapshot.snapshot_date == snapshot_date)
        )).all()
        if not snap:
            return {"checked": 0, "snapshot_date": str(snapshot_date)}

        alive = set((await db.execute(
            select(ChanSignal.ts_code, ChanSignal.trade_date, ChanSignal.kind)
            .where(ChanSignal.trade_date == snapshot_date)
        )).all())

        today = (await db.execute(
            select(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()

        gone = 0
        for sid, code, td, kind in snap:
            ok = (code, td, kind) in alive
            if not ok:
                gone += 1
            await db.execute(update(SignalSnapshot).where(SignalSnapshot.id == sid)
                             .values(still_valid=ok, checked_on=today))
        await db.commit()

    total = len(snap)
    log.info("信号漂移核对 %s: %d 条中 %d 条已被重绘掉 (%.1f%%)",
             snapshot_date, total, gone, 100 * gone / max(total, 1))
    return {"snapshot_date": str(snapshot_date), "checked": total,
            "repainted_away": gone,
            "repaint_pct": round(100 * gone / max(total, 1), 1)}
