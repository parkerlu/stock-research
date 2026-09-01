"""把某个策略模板的买点算成全市场缓存, 供虚拟盘逐日回放使用.

通用: 任何实现了 generate_signals(df) 的模板都能用, 不绑定具体策略。

⚠️ 只 append 不删除。全历史重建会抹掉"当时成立、后来被撤销"的信号 ——
chan-2buy 就死在这: 约一半信号被这样抹掉, 而抹掉的正是输家。
纯公式指标没有这个问题, 但纪律照旧, 免得换策略时忘记。
"""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.db import async_session
from app.models.schema import DailyCandle, StockBasic, StrategySignal
from app.services.paper_trading import _universe_ok

log = logging.getLogger(__name__)

WARMUP_BARS = 260          # 周线 KDJ 等需要足够前缀
CHUNK = 150                # 生产机内存小, 分批拉


async def build(strategy: str, since: date | None = None,
                lookback_days: int = 900) -> dict:
    """算 strategy 的买点并落表。since 非空时只写该日之后的信号(每日增量)。"""
    from app.services.strategy_templates import TEMPLATE_REGISTRY

    if strategy not in TEMPLATE_REGISTRY:
        return {"error": f"未知策略 {strategy}"}
    tpl_cls = TEMPLATE_REGISTRY[strategy]

    t0 = time.time()
    async with async_session() as db:
        basics = (await db.execute(
            select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active))).all()
        end = (await db.execute(
            select(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1))).scalar_one_or_none()
    if end is None:
        return {"error": "库里没有行情"}
    codes = sorted(c for c, n, a in basics if a is not False and _universe_ok(c, n))
    start = (since - timedelta(days=lookback_days)) if since else None

    inserted = 0
    for i in range(0, len(codes), CHUNK):
        batch = codes[i:i + CHUNK]
        async with async_session() as db:
            q = select(DailyCandle.ts_code, DailyCandle.trade_date, DailyCandle.open,
                       DailyCandle.high, DailyCandle.low, DailyCandle.close,
                       DailyCandle.vol, DailyCandle.amount, DailyCandle.adj_factor
                       ).where(DailyCandle.ts_code.in_(batch))
            if start:
                q = q.where(DailyCandle.trade_date >= start)
            rows = (await db.execute(q.order_by(DailyCandle.ts_code,
                                                DailyCandle.trade_date))).all()
            if not rows:
                continue
            df = pd.DataFrame(rows, columns=["ts_code","trade_date","open","high",
                                             "low","close","vol","amount","adj"])
            for c in ("open","high","low","close","vol","amount","adj"):
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna(subset=["open","high","low","close"])

            payload = []
            for cd, g in df.groupby("ts_code", sort=False):
                g = g.sort_values("trade_date").reset_index(drop=True)
                if len(g) < WARMUP_BARS:
                    continue
                # 前复权 —— 与 quote_service / 筛选台架同口径
                latest = g.adj.iloc[-1]
                if latest and latest > 0:
                    f = (g.adj / latest).values
                    for c in ("open","high","low","close"):
                        g[c] = (g[c].values * f).round(4)
                try:
                    sigs = tpl_cls().generate_signals(g)
                except Exception as exc:            # noqa: BLE001
                    log.warning("signal_build %s %s: %s", strategy, cd, exc)
                    continue
                for s in sigs:
                    if s.get("action") != "buy":
                        continue
                    d = s["date"]
                    d = d.date() if hasattr(d, "date") else d
                    if since and d <= since:
                        continue
                    payload.append({"strategy": strategy, "ts_code": cd, "trade_date": d})
            if payload:
                stmt = pg_insert(StrategySignal).values(payload).on_conflict_do_nothing(
                    index_elements=["strategy", "ts_code", "trade_date"])
                res = await db.execute(stmt)
                await db.commit()
                inserted += res.rowcount or 0

    el = time.time() - t0
    log.info("signal_build %s: 新增 %d 条, 耗时 %.0fs", strategy, inserted, el)
    return {"strategy": strategy, "universe": len(codes),
            "inserted": inserted, "elapsed_sec": round(el, 1)}
