"""虚拟盘 API — 账户总览 / 持仓 / 成交流水 / 净值曲线 / 逐日推进."""
from __future__ import annotations

import asyncio
import time
import uuid
from datetime import date

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete, select

from app.db import async_session
from app.models.schema import (
    DailyCandle,
    PaperAccount,
    PaperEquity,
    PaperPosition,
    PaperTrade,
)
from app.services import paper_trading as pt

router = APIRouter(prefix="/api/paper", tags=["paper"])

# 两个独立账户, 互不干扰:
#   LIVE_ACCOUNT   实操盘 —— 2026-09-01 起, 每晚数据抓取完自动推进一天, 只能向前
#   DEMO_ACCOUNT   演示盘 —— 任意历史起点, 手动"下一日"回放, 可随时重置
# 分开的原因: 演示盘会被反复 reset, 实操盘的记录必须只增不改。
LIVE_ACCOUNT = "live-2026"
DEMO_ACCOUNT = "demo"
DEFAULT_ACCOUNT = LIVE_ACCOUNT


class CreateRequest(BaseModel):
    name: str = DEFAULT_ACCOUNT
    capital: float = 100_000.0
    slots: int = 6
    start: date | None = None


class RunRequest(BaseModel):
    name: str = DEFAULT_ACCOUNT
    until: date | None = None      # 推进到哪一天, 默认库里最新交易日


_JOBS: dict[str, dict] = {}


@router.get("/config")
async def config():
    """策略配置与回测依据 —— 前端展示, 也提醒改参数就不是这个回测结果了。"""
    return {
        "default_account": DEFAULT_ACCOUNT,
        "strategy": "chan-2buy (缠论2类买点)",
        "rules": pt.DEFAULT_CONFIG,
        "backtest": {
            "period": "2015-01 ~ 2026-08 (11.6 年, 全 universe 4306 只, 34830 个买点)",
            "capital": "10 万 / 10 仓位等权",
            "cagr": 55.8, "max_drawdown": 7.4, "sharpe": 4.74, "trades": 3051,
            "note": "参数在同一批数据上精调, 样本外仅 3.7 年; +4% 即减仓对手续费与滑点敏感。"
                    "回测不预示未来 —— 这个虚拟盘正是为了用真实前瞻数据检验它。",
        },
    }


@router.post("/create")
async def create(req: CreateRequest):
    async with async_session() as db:
        if await pt.get_account(db, req.name):
            raise HTTPException(400, f"账户 {req.name} 已存在")
        acct = await pt.create_account(db, req.name, req.capital, req.slots, req.start)
        await db.commit()
        return {"id": acct.id, "name": acct.name, "capital": float(acct.initial_capital),
                "slots": acct.slots, "started_on": str(acct.started_on)}


@router.get("/status")
async def status(name: str = DEFAULT_ACCOUNT):
    """账户总览 + 当前持仓(含浮动盈亏) + 已平仓统计。"""
    async with async_session() as db:
        acct = await pt.get_account(db, name)
        if not acct:
            return {"exists": False, "name": name}

        as_of_date = acct.last_run_date or acct.started_on
        positions = (await db.execute(
            select(PaperPosition).where(PaperPosition.account_id == acct.id)
            .order_by(PaperPosition.open_date.desc())
        )).scalars().all()

        holdings, closed = [], []
        market_value = 0.0
        for p in positions:
            # ⚠️ 必须按"回放当日"取价, 不能用最新价 —— 回放 2020 年时用 2026 年
            # 的收盘价算浮盈就是未来函数, 演示会显得神准。
            last = (await db.execute(
                select(DailyCandle.close, DailyCandle.trade_date)
                .where(DailyCandle.ts_code == p.ts_code,
                       DailyCandle.trade_date <= as_of_date)
                .order_by(DailyCandle.trade_date.desc()).limit(1)
            )).first()
            px = float(last[0]) if last else float(p.open_price)
            as_of = str(last[1]) if last else None
            entry = float(p.open_price)
            if p.status == "open":
                mv = p.shares * px
                market_value += mv
                holdings.append({
                    "ts_code": p.ts_code, "name": p.name,
                    "open_date": str(p.open_date), "open_price": entry,
                    "shares": p.shares, "init_shares": p.init_shares,
                    "last_price": px, "as_of": as_of,
                    "market_value": round(mv, 2),
                    "float_pnl": round(p.shares * (px - entry), 2),
                    "float_pnl_pct": round((px / entry - 1) * 100, 2),
                    "realized_pnl": round(float(p.realized_pnl or 0), 2),
                    "stop_price": float(p.stop_price),
                    "tier1_price": round(entry * (1 + pt.DEFAULT_CONFIG["tier1_pct"]), 4),
                    "tier2_price": round(entry * (1 + pt.DEFAULT_CONFIG["tier2_pct"]), 4),
                    "tier1_done": p.tier1_done,
                    "hold_days": (as_of_date - p.open_date).days,
                })
            else:
                closed.append({
                    "ts_code": p.ts_code, "name": p.name,
                    "open_date": str(p.open_date), "close_date": str(p.close_date or ""),
                    "open_price": entry, "reason": p.close_reason,
                    "realized_pnl": round(float(p.realized_pnl or 0), 2),
                })

        cash = float(acct.cash)
        equity = cash + market_value
        init = float(acct.initial_capital)
        realized = sum(c["realized_pnl"] for c in closed) + \
                   sum(h["realized_pnl"] for h in holdings)
        return {
            "exists": True, "name": acct.name, "slots": acct.slots,
            "initial_capital": init, "cash": round(cash, 2),
            "market_value": round(market_value, 2), "equity": round(equity, 2),
            "total_pnl": round(equity - init, 2),
            "total_pnl_pct": round((equity / init - 1) * 100, 2),
            "realized_pnl": round(realized, 2),
            "float_pnl": round(sum(h["float_pnl"] for h in holdings), 2),
            "started_on": str(acct.started_on),
            "last_run_date": str(acct.last_run_date) if acct.last_run_date else None,
            "as_of": str(as_of_date),
            "n_open": len(holdings), "n_closed": len(closed),
            "holdings": holdings, "closed": closed[:50],
        }


@router.get("/trades")
async def trades(name: str = DEFAULT_ACCOUNT, limit: int = 200):
    """全部动作流水 —— 买入/分批止盈/止损/清仓, 一条不落。"""
    async with async_session() as db:
        acct = await pt.get_account(db, name)
        if not acct:
            return {"exists": False, "trades": []}
        rows = (await db.execute(
            select(PaperTrade).where(PaperTrade.account_id == acct.id)
            .order_by(PaperTrade.trade_date.desc(), PaperTrade.id.desc()).limit(limit)
        )).scalars().all()
        return {"exists": True, "trades": [{
            "date": str(t.trade_date), "ts_code": t.ts_code, "name": t.name,
            "action": t.action, "price": float(t.price), "shares": t.shares,
            "amount": float(t.amount), "fee": float(t.fee),
            "pnl": float(t.pnl) if t.pnl is not None else None,
            "pnl_pct": float(t.pnl_pct) if t.pnl_pct is not None else None,
            "note": t.note,
        } for t in rows]}


@router.get("/equity")
async def equity(name: str = DEFAULT_ACCOUNT):
    async with async_session() as db:
        acct = await pt.get_account(db, name)
        if not acct:
            return {"exists": False, "points": []}
        rows = (await db.execute(
            select(PaperEquity).where(PaperEquity.account_id == acct.id)
            .order_by(PaperEquity.trade_date)
        )).scalars().all()
        return {"exists": True, "initial": float(acct.initial_capital),
                "points": [{"date": str(r.trade_date), "equity": float(r.equity),
                            "cash": float(r.cash), "n": r.n_positions} for r in rows]}


async def _run_job(job_id: str, name: str, until: date | None):
    job = _JOBS[job_id]
    job["status"] = "running"
    t0 = time.time()
    try:
        async with async_session() as db:
            acct = await pt.get_account(db, name)
            if not acct:
                acct = await pt.create_account(db, name)
                await db.commit()
            end = until or (await db.execute(
                select(DailyCandle.trade_date).order_by(DailyCandle.trade_date.desc()).limit(1)
            )).scalar_one_or_none()
            if not end:
                job["status"] = "error"; job["error"] = "库里没有行情数据"; return
            # 同 scheduler: 首次结算时起始日本身要算进去
            if acct.last_run_date:
                start, inclusive = acct.last_run_date, False
            else:
                start, inclusive = acct.started_on, True
            days = (await db.execute(
                select(DailyCandle.trade_date)
                .where(DailyCandle.trade_date >= start if inclusive
                       else DailyCandle.trade_date > start,
                       DailyCandle.trade_date <= end)
                .group_by(DailyCandle.trade_date).order_by(DailyCandle.trade_date)
            )).scalars().all()
            job["total"] = len(days)
            for k, d in enumerate(days):
                res = await pt.run_day(db, acct, d,
                                       on_progress=lambda a, b: job.update(sub=f"{a}/{b}"))
                await db.commit()
                job["progress"] = k + 1
                job["last"] = res
                job["elapsed_sec"] = round(time.time() - t0, 1)
        job["status"] = "completed"
    except Exception as e:                     # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)[:300]
    job["elapsed_sec"] = round(time.time() - t0, 1)


@router.post("/run")
async def run(req: RunRequest, background_tasks: BackgroundTasks):
    """推进到最新交易日。已结算过的日期会跳过, 可安全重复调用。"""
    job_id = uuid.uuid4().hex
    _JOBS[job_id] = {"job_id": job_id, "status": "pending", "progress": 0,
                     "total": 0, "error": None, "elapsed_sec": 0.0}
    background_tasks.add_task(_run_job, job_id, req.name, req.until)
    return {"job_id": job_id}


@router.get("/run/{job_id}")
async def run_status(job_id: str):
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(404, "Unknown job")
    return job


# =========================================================================
# 回放控制 —— 演示盘: 设定起始日, 逐日推进看持仓与浮盈变化
# =========================================================================

class StepRequest(BaseModel):
    name: str = DEFAULT_ACCOUNT
    days: int = 1               # 一次推进几个交易日


class ResetRequest(BaseModel):
    name: str = DEFAULT_ACCOUNT
    capital: float = 100_000.0
    slots: int = 6
    start: date                 # 回放起点, 必填 —— 演示盘的意义就在这里


@router.post("/reset")
async def reset(req: ResetRequest):
    """清空并从指定日期重新开始 —— 演示用。"""
    async with async_session() as db:
        acct = await pt.get_account(db, req.name)
        if acct:
            for tbl in (PaperTrade, PaperPosition, PaperEquity):
                await db.execute(sa_delete(tbl).where(tbl.account_id == acct.id))
            await db.execute(sa_delete(PaperAccount).where(PaperAccount.id == acct.id))
            await db.commit()
        acct = await pt.create_account(db, req.name, req.capital, req.slots, req.start)
        await db.commit()
        return {"id": acct.id, "name": acct.name, "capital": float(acct.initial_capital),
                "slots": acct.slots, "started_on": str(acct.started_on)}


@router.post("/step")
async def step(req: StepRequest):
    """推进 N 个交易日并直接返回结果 —— 供"下一日"按钮同步调用。

    信号走 chan_signal 预计算表, 所以一天只要几十毫秒, 点着不卡。
    """
    async with async_session() as db:
        acct = await pt.get_account(db, req.name)
        if not acct:
            raise HTTPException(404, f"账户 {req.name} 不存在")
        cur = acct.last_run_date or acct.started_on
        days = (await db.execute(
            select(DailyCandle.trade_date).where(DailyCandle.trade_date > cur)
            .group_by(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date).limit(max(1, min(req.days, 60)))
        )).scalars().all()
        if not days:
            return {"advanced": 0, "as_of": str(cur), "done": True,
                    "message": "已到数据末端"}
        results = []
        for d in days:
            results.append(await pt.run_day(db, acct, d))
        await db.commit()
        return {"advanced": len(days), "as_of": str(days[-1]), "done": False,
                "steps": results}


@router.get("/signals")
async def signals(name: str = DEFAULT_ACCOUNT, limit: int = 20):
    """回放当日出现的候选买点 —— 展示"股票池"用, 与实际下单顺序一致。"""
    async with async_session() as db:
        acct = await pt.get_account(db, name)
        if not acct:
            return {"exists": False, "picks": []}
        day = acct.last_run_date or acct.started_on
        # ⚠️ 不能超过库里最新的交易日。实操盘 9/1 才开始, 但今晚要看的是
        # "今天(8/31)出的信号、明早要买的票" —— 拿 9/1 去查一条都查不到。
        latest = (await db.execute(
            select(DailyCandle.trade_date)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if latest and day > latest:
            day = latest
        held = set((await db.execute(
            select(PaperPosition.ts_code).where(PaperPosition.account_id == acct.id,
                                                PaperPosition.status == "open")
        )).scalars().all())
        # require_next=False: 最新信号日的"次日开盘价"当然还不存在,
        # 这里是展示明日要买什么, 不是成交, 不该拿它过滤。
        picks = await pt.scan_signals(db, day, exclude=set(), require_next=False)
        return {"exists": True, "as_of": str(day),
                "picks": [{**p, "signal_date": str(p["signal_date"]),
                           "buy_date": str(p["buy_date"]) if p.get("buy_date") else None,
                           "held": p["ts_code"] in held} for p in picks[:limit]]}
