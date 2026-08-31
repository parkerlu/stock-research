"""虚拟盘引擎 — 按验证过的 chan-2buy 最优配置逐日推进.

配置来源 (全 universe 4306 只 / 2015-2026 / 复利 + 冲击成本 / 34830 个买点):
    10 万本金 · 10 个等权仓位 (每仓 1 万) · 低流动性优先
    止损 -6% / 第一批 +4% 出半(止损上移到成本价) / 第二批 +8% 清仓
    -> 回测 年化 55.8% / 最大回撤 7.4% / 夏普 4.74 / 11.6 年 3051 笔

为什么是这套参数, 几个反直觉的实测结论:
  · 止盈位越低越好。次批 +8% 年化 47.4%, +25% 只有 28.9%, 不封顶掉到 9.4% ——
    仓位是稀缺资源, "让利润跑"会把周转率杀死 (不封顶 11.6 年仅成交 1175 笔)。
  · 止损 6% 是甜点, 3% 太紧 / 12% 太松, 两边都掉约 2 个百分点。
  · 首批止盈后必须把止损上移到成本价 —— 开/关差 8 个百分点。
  · 分仓不要做花样: 反波动率加权反而掉 8 个点, 因为小仓位买不起一手,
    实测 4753 次信号因碎股作废。等权 + 固定止损百分比本身就是风险平价 ——
    每笔最大亏损都是 1万 × 6% = 600 元 = 净值的 0.6%。
  · 10 仓位优于 5 仓位: 收益几乎相同(55.8% vs 55.6%)但回撤 7.4% vs 10.2%。

⚠️ 回测数字不是预期。参数在同一批数据上精调, OOS 仅 3.7 年; 且 +4% 就减仓
   意味着对手续费和滑点非常敏感。这个虚拟盘的意义正是在于用真实前瞻数据检验它。

逐日结算顺序 (顺序不能颠倒):
    1. 先按当日 OHLC 处理已有持仓的出场 —— 止损优先于止盈(保守)
    2. 再用当日的新信号填补空出来的仓位, 次日开盘价成交
    3. 记录当日净值快照
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import (
    DailyCandle,
    PaperAccount,
    PaperEquity,
    PaperPosition,
    PaperTrade,
    StockBasic,
)
from app.services.chanlun import find_class1_buys, find_class2_buys

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "strategy": "chan-2buy",
    "stop_pct": 0.06,        # 止损
    "tier1_pct": 0.04,       # 第一批止盈
    "tier1_frac": 0.5,       # 卖出比例
    "tier2_pct": 0.08,       # 第二批(清仓)
    "breakeven": True,       # 首批后止损上移到成本
    "min_amount_k": 5000,    # 20日均额下限(千元)
    "max_hold_days": 60,     # 保险丝
    "fee_pct": 0.0010,       # 佣金+印花税(双边合计估计)
    "lot": 100,
}
HISTORY_DAYS = 400
MIN_BARS = 130
# 与 _ChanlunBase._confirm_lag 一致 —— 改这里必须同步改策略, 否则虚拟盘和
# 回测就不是同一套东西了。
CONFIRM_LAG = 2


def _universe_ok(ts_code: str, name: str | None) -> bool:
    if ts_code.startswith(("688", "92", "8", "4")):
        return False
    return not (name and ("ST" in name or "退" in name))


@dataclass
class Bar:
    d: date
    o: float
    h: float
    l: float
    c: float


async def _load_one(db: AsyncSession, ts_code: str, end: date) -> pd.DataFrame | None:
    rows = (await db.execute(
        select(DailyCandle.trade_date, DailyCandle.open, DailyCandle.high,
               DailyCandle.low, DailyCandle.close, DailyCandle.amount,
               DailyCandle.adj_factor)
        .where(DailyCandle.ts_code == ts_code,
               DailyCandle.trade_date >= end - timedelta(days=HISTORY_DAYS),
               DailyCandle.trade_date <= end)
        .order_by(DailyCandle.trade_date)
    )).all()
    if len(rows) < MIN_BARS:
        return None
    df = pd.DataFrame(rows, columns=["trade_date", "open", "high", "low",
                                     "close", "amount", "adj_factor"])
    for c in ("open", "high", "low", "close", "amount", "adj_factor"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    df = df.dropna(subset=["open", "high", "low", "close"])
    # 前复权 — 与 quote_service 同口径
    latest = df.adj_factor.iloc[-1]
    if latest and latest > 0:
        f = (df.adj_factor / latest).values
        for c in ("open", "high", "low", "close"):
            df[c] = (df[c].values * f).round(4)
    return df.reset_index(drop=True)


def _signal_today(df: pd.DataFrame, lag: int = CONFIRM_LAG) -> bool:
    """最后一根 bar 是否是**可操作**的 chan-2buy 买点。

    ⚠️ 必须加 lag。分型是"3 根合并K 的中间那根", 右邻那根出来之前根本不知道
    它是分型 —— 直接用 find_class2_buys 返回的 bar_idx 当信号日就是偷看未来。
    实测确认滞后: 中位 1 根 / 均值 1.47 / 95分位 3 根, lag=2 覆盖 89.5%。
    与 _ChanlunBase._confirm_lag=2 保持一致, 这样虚拟盘和回测口径相同。
    """
    c = df.close.values
    h = df.high.values
    l = df.low.values
    c1 = find_class1_buys(c, h, l)
    c2 = find_class2_buys(c, h, l, {b.bar_idx for b in c1})
    last = len(c) - 1
    return any(b.bar_idx + lag == last for b in c2)


async def get_account(db: AsyncSession, name: str = "chan2-10w") -> PaperAccount | None:
    return (await db.execute(
        select(PaperAccount).where(PaperAccount.name == name)
    )).scalar_one_or_none()


async def create_account(db: AsyncSession, name: str = "chan2-10w",
                         capital: float = 100_000.0, slots: int = 10,
                         start: date | None = None) -> PaperAccount:
    acct = PaperAccount(
        name=name, initial_capital=capital, cash=capital, slots=slots,
        config=dict(DEFAULT_CONFIG), started_on=start or date.today(),
        last_run_date=None, is_active=True,
    )
    db.add(acct)
    await db.flush()
    return acct


async def run_day(db: AsyncSession, acct: PaperAccount, day: date,
                  on_progress=None) -> dict:
    """推进一个交易日。可重复调用 —— 已结算过的日期直接跳过。"""
    cfg = {**DEFAULT_CONFIG, **(acct.config or {})}
    if acct.last_run_date and day <= acct.last_run_date:
        return {"date": str(day), "skipped": "already settled"}

    actions: list[dict] = []
    cash = float(acct.cash)

    # ---------- 1. 先处理出场 ----------
    positions = (await db.execute(
        select(PaperPosition).where(PaperPosition.account_id == acct.id,
                                    PaperPosition.status == "open")
    )).scalars().all()

    for p in positions:
        bar = (await db.execute(
            select(DailyCandle.open, DailyCandle.high, DailyCandle.low, DailyCandle.close)
            .where(DailyCandle.ts_code == p.ts_code, DailyCandle.trade_date == day)
        )).first()
        if not bar:
            continue        # 停牌
        o, h, l, c = (float(x) for x in bar)
        entry = float(p.open_price)
        stop = float(p.stop_price)

        def _sell(shares: int, price: float, action: str, reason: str | None = None):
            nonlocal cash
            amt = shares * price
            fee = amt * cfg["fee_pct"]
            pnl = shares * (price - entry) - fee
            cash += amt - fee
            db.add(PaperTrade(
                account_id=acct.id, position_id=p.id, ts_code=p.ts_code, name=p.name,
                trade_date=day, action=action, price=round(price, 4), shares=shares,
                amount=round(amt, 2), fee=round(fee, 2), pnl=round(pnl, 2),
                pnl_pct=round((price / entry - 1) * 100, 4), note=reason,
            ))
            p.shares -= shares
            p.realized_pnl = float(p.realized_pnl or 0) + pnl
            actions.append({"ts_code": p.ts_code, "name": p.name, "action": action,
                            "price": round(price, 4), "shares": shares,
                            "pnl": round(pnl, 2), "pnl_pct": round((price/entry-1)*100, 2)})

        # 止损优先(同日既触止损又触止盈时按最坏处理)
        if l <= stop:
            px = o if o < stop else stop         # 跳空按开盘价
            _sell(p.shares, px, "stop", "触及止损")
            p.status = "closed"; p.close_date = day
            p.close_reason = "stop" if stop >= entry * (1 - cfg["stop_pct"]) else "breakeven"
            continue

        if not p.tier1_done and h >= entry * (1 + cfg["tier1_pct"]):
            px = max(o, entry * (1 + cfg["tier1_pct"]))
            n = int(p.init_shares * cfg["tier1_frac"] / cfg["lot"]) * cfg["lot"]
            n = min(n, p.shares)
            if n > 0:
                _sell(n, px, "tier1", f"+{cfg['tier1_pct']:.0%} 减半")
            p.tier1_done = True
            if cfg["breakeven"]:
                p.stop_price = entry          # 保本: 此后这笔不可能亏
        if p.shares > 0 and p.tier1_done and not p.tier2_done and h >= entry * (1 + cfg["tier2_pct"]):
            px = max(o, entry * (1 + cfg["tier2_pct"]))
            _sell(p.shares, px, "tier2", f"+{cfg['tier2_pct']:.0%} 清仓")
            p.tier2_done = True
            p.status = "closed"; p.close_date = day; p.close_reason = "tier2"
            continue

        if p.shares > 0 and (day - p.open_date).days >= cfg["max_hold_days"] * 1.6:
            _sell(p.shares, c, "timeout", f"持有超 {cfg['max_hold_days']} 交易日")
            p.status = "closed"; p.close_date = day; p.close_reason = "timeout"

    await db.flush()

    # ---------- 2. 再用当日信号补仓 ----------
    open_now = (await db.execute(
        select(PaperPosition).where(PaperPosition.account_id == acct.id,
                                    PaperPosition.status == "open")
    )).scalars().all()
    free = acct.slots - len(open_now)
    held = {p.ts_code for p in open_now}

    if free > 0:
        cands = await scan_signals(db, day, exclude=held, on_progress=on_progress)
        per = (cash + sum(0 for _ in open_now)) / max(free, 1)
        # 每仓目标 = 初始资金/仓位数, 但不超过可用现金
        target = float(acct.initial_capital) / acct.slots
        for cd in cands:
            if free <= 0:
                break
            price = cd["next_open"]
            if price is None or price <= 0:
                continue
            budget = min(target, cash)
            shares = int(budget // (price * cfg["lot"])) * cfg["lot"]
            if shares < cfg["lot"]:
                continue
            amt = shares * price
            fee = amt * cfg["fee_pct"]
            if amt + fee > cash:
                continue
            cash -= amt + fee
            pos = PaperPosition(
                account_id=acct.id, ts_code=cd["ts_code"], name=cd["name"],
                open_date=cd["buy_date"], open_price=round(price, 4),
                init_shares=shares, shares=shares,
                stop_price=round(price * (1 - cfg["stop_pct"]), 4),
                status="open",
            )
            db.add(pos)
            await db.flush()
            db.add(PaperTrade(
                account_id=acct.id, position_id=pos.id, ts_code=cd["ts_code"],
                name=cd["name"], trade_date=cd["buy_date"], action="buy",
                price=round(price, 4), shares=shares, amount=round(amt, 2),
                fee=round(fee, 2), note=f"chan-2buy 信号 {cd['signal_date']}",
            ))
            actions.append({"ts_code": cd["ts_code"], "name": cd["name"], "action": "buy",
                            "price": round(price, 4), "shares": shares, "pnl": None,
                            "pnl_pct": None})
            free -= 1

    # ---------- 3. 净值快照 ----------
    acct.cash = round(cash, 2)
    acct.last_run_date = day
    mv = 0.0
    finals = (await db.execute(
        select(PaperPosition).where(PaperPosition.account_id == acct.id,
                                    PaperPosition.status == "open")
    )).scalars().all()
    for p in finals:
        last = (await db.execute(
            select(DailyCandle.close).where(DailyCandle.ts_code == p.ts_code,
                                            DailyCandle.trade_date <= day)
            .order_by(DailyCandle.trade_date.desc()).limit(1)
        )).scalar_one_or_none()
        if last:
            mv += p.shares * float(last)
    db.add(PaperEquity(account_id=acct.id, trade_date=day, cash=round(cash, 2),
                       market_value=round(mv, 2), equity=round(cash + mv, 2),
                       n_positions=len(finals)))
    await db.flush()
    return {"date": str(day), "actions": actions, "cash": round(cash, 2),
            "market_value": round(mv, 2), "equity": round(cash + mv, 2),
            "positions": len(finals)}


async def scan_signals(db: AsyncSession, day: date, exclude: set[str] | None = None,
                       on_progress=None) -> list[dict]:
    """当日出 chan-2buy 买点的票, 按低流动性优先排序。

    分批扫描 —— 一次性拉全市场会压垮小内存机器 (踩过一次, 见 portfolio_scan)。
    """
    exclude = exclude or set()
    basics = (await db.execute(
        select(StockBasic.ts_code, StockBasic.name, StockBasic.is_active)
    )).all()
    names = {c: n for c, n, _ in basics}
    codes = [c for c, n, a in basics
             if a is not False and _universe_ok(c, n) and c not in exclude]

    # 次日开盘价 —— 信号日的下一个交易日
    nxt = (await db.execute(
        select(DailyCandle.trade_date).where(DailyCandle.trade_date > day)
        .order_by(DailyCandle.trade_date).limit(1)
    )).scalar_one_or_none()

    out: list[dict] = []
    CHUNK = 300
    for i in range(0, len(codes), CHUNK):
        for cd in codes[i:i + CHUNK]:
            df = await _load_one(db, cd, day)
            if df is None or df.trade_date.iloc[-1] != day:
                continue
            amt20 = df.amount.rolling(20).mean().iloc[-1]
            if np.isnan(amt20) or amt20 < DEFAULT_CONFIG["min_amount_k"]:
                continue
            c = df.close.values
            if len(c) > 1 and c[-1] > c[-2] * 1.099:      # 涨停买不进
                continue
            if not _signal_today(df):
                continue     # 已含 CONFIRM_LAG, 无前视
            nx = None
            if nxt:
                nx = (await db.execute(
                    select(DailyCandle.open).where(DailyCandle.ts_code == cd,
                                                   DailyCandle.trade_date == nxt)
                )).scalar_one_or_none()
            out.append({"ts_code": cd, "name": names.get(cd),
                        "signal_date": day, "buy_date": nxt or day,
                        "next_open": float(nx) if nx else float(c[-1]),
                        "amount_20d_wan": round(float(amt20) / 10, 1)})
        if on_progress:
            on_progress(min(i + CHUNK, len(codes)), len(codes))
    out.sort(key=lambda x: x["amount_20d_wan"])     # 低流动性优先
    return out
