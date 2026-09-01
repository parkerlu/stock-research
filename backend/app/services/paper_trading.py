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
from sqlalchemy import exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.schema import (
    ChanSignal,
    DailyCandle,
    PaperAccount,
    PaperEquity,
    PaperPosition,
    PaperTrade,
    StockBasic,
)

log = logging.getLogger(__name__)

DEFAULT_CONFIG = {
    "strategy": "chan-2buy",
    "stop_pct": 0.06,        # 止损
    "tier1_pct": 0.04,       # 第一批止盈
    "tier1_frac": 0.5,       # 卖出比例
    "tier2_pct": 0.08,       # 第二批(清仓)
    "breakeven": True,       # 首批后止损上移到成本
    # 移动止盈(可选): 设了 trail_pct 就用它取代固定的 tier2 清仓 ——
    # 涨到 trail_arm_pct 后开始跟踪, 从最高点回撤 trail_pct 才卖。
    "trail_pct": None,
    "trail_arm_pct": None,
    # tier2 只卖掉初始仓位的这个比例, 剩下的交给移动止盈。
    # 1.0 = 到 +8% 全清 (现状); 0.25 = 再卖 1/4, 留 1/4 去跑。
    # 全仓跟踪的问题是"跟踪的代价每笔都付、收益只在极少数票上兑现",
    # 留小尾巴可以把代价按比例砍掉, 而大牛股的上涨照样吃得到。
    "tier2_frac": 1.0,
    # 只在"这笔明显不一样"时才跟踪: 从买入到摸到 +8% 用了几个交易日,
    # 越快说明动能越强。None = 不做条件判断, 一律跟踪。
    "trail_max_days_to_arm": None,
    "min_amount_k": 5000,    # 20日均额下限(千元)
    "max_hold_days": 60,     # 保险丝
    # ---- 交易成本: A股是三笔独立的费, 不能揉成一个百分比 ----
    "commission_pct": 0.0003,   # 佣金 万3, 买卖双边
    "commission_min": 5.0,      # ⚠️ 佣金最低 5 元 —— 对 10 万小账户是主要成本
    "transfer_pct": 0.00001,    # 过户费 万0.1, 双边
    # ---- 入场方式 ----
    # "next_open"    : 信号日的下一个交易日开盘买 (最保守)
    # "signal_close" : 信号日(F+2)尾盘按收盘价买
    # "fractal_close": 分型次根(F+1)尾盘按收盘价买 —— 最早的合法时点。
    #   实测 250 只票 219 个 2 买信号, 首次可算出的滞后**全部恰好是 1 根K**,
    #   且首次出现后再没消失过, 所以 F+1 收盘时信号已经成立。
    #   唯一保留意见: 实盘 14:55 下单时 F+1 这根的最高/最低还没最终确定,
    #   理论上分型形态可能在最后 5 分钟被破坏。这部分无法用日线数据检验。
    "entry_mode": "next_open",
    "entry_slip_pct": 0.0,      # 尾盘抢单愿意多付的比例, 如 0.002 = 高 0.2%
    "lot": 100,
}

# 印花税: 卖出单边收取, 税率变过。2008-09-19~2023-08-27 千分之一,
# 2023-08-28 起减半到万分之五。回测跨越这个日期, 必须按成交日取值,
# 否则 2022 年的成本会被低估一半。
STAMP_CUT_DATE = date(2023, 8, 28)
STAMP_PCT_OLD = 0.0010
STAMP_PCT_NEW = 0.0005


def trade_fee(amount: float, is_sell: bool, day: date,
              cfg: dict | None = None) -> float:
    """一笔成交的真实费用 = 佣金(有下限) + 过户费 + 卖出印花税.

    单个仓位才 1 万块, 万3 佣金只有 3 元, 实际按 5 元下限收 —— 等于 0.05%;
    减半卖出的 5 千块更是等于 0.1%。把这三笔揉成一个固定百分比会算错。
    """
    cfg = cfg or DEFAULT_CONFIG
    commission = max(amount * cfg["commission_pct"], cfg["commission_min"])
    fee = commission + amount * cfg["transfer_pct"]
    if is_sell:
        fee += amount * (STAMP_PCT_OLD if day < STAMP_CUT_DATE else STAMP_PCT_NEW)
    return fee
HISTORY_DAYS = 400
MIN_BARS = 130
# 与 _ChanlunBase._confirm_lag 一致 —— 改这里必须同步改策略, 否则虚拟盘和
# 回测就不是同一套东西了。
# 信号确认滞后。缠论已下架, 保留常量供后续策略复用 —— 任何
# "分型/极值"类信号都需要右侧K线确认, 直接用极值那根当信号日就是偷看未来。
CONFIRM_LAG = 2


def _universe_ok(ts_code: str, name: str | None,
                 ignore_status: bool = False) -> bool:
    """只保留 A 股主板/中小板/创业板的正常股票.

    ⚠️ 必须用白名单, 不能只黑名单几个前缀。原来的写法只挡了 688/92/8/4,
    结果 ETF 和 LOF 全漏进来了 —— 实测混进过国债ETF、地方政府债ETF、黄金ETF、
    原油LOF, 甚至货币基金(华宝现金添益)。货币基金根本不会跌, 等于现金。

    数量上只占 1.5% 的成交, 但危害极大: 股灾时只有这些防御资产出买点, 策略
    就自动躲进去, 把最大回撤压到不真实的水平。回测里看到"熊市几乎不回撤",
    十有八九就是这个。

    代码段: 沪 60xxxx 主板 / 深 000-003xxx 主板中小板 / 深 30xxxx 创业板。
    68 科创板、5xxxxx 与 1xxxxx 基金、8/4/92 北交所, 一律排除。
    """
    code = ts_code.split(".")[0]
    if not (code.startswith("60") or code.startswith("00") or code.startswith("30")):
        return False
    # ⚠️ [审计实验 2026-09] ignore_status=True 时跳过基于**当前快照**的
    # ST/退市名称过滤。stock_basic.name 是最新名字: 2024 年才戴帽的票, 用今天
    # 的名字去过滤 2016-2023 年的历史交易 == 预先知道"它以后会变烂", 是未来
    # 函数(幸存者偏差)。审计账户用 config.universe_neutral=True 走这条分支,
    # 只保留纯代码段白名单。正常账户(demo/live-2026)行为不变。
    if ignore_status:
        return True
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
            fee = trade_fee(amt, is_sell=True, day=day, cfg=cfg)
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
        # 第二批: 固定止盈 或 移动止盈(贪婪)
        # 固定 +8% 清仓的问题是把大涨的票也在 +8% 砍掉; 移动止盈让利润继续跑,
        # 只在回撤 trail_pct 时才走。代价是每次都要还回去一段回撤。
        # ⚠️ 这里不能带 "not p.tier2_done" —— 部分减仓后 tier2_done 已置真,
        # 但剩下的尾巴还要继续跟踪, 带上就再也不抬止损了。
        if p.shares > 0 and p.tier1_done and cfg.get("trail_pct"):
            trail = float(cfg["trail_pct"])
            arm_pct = float(cfg.get("trail_arm_pct") or cfg["tier2_pct"])
            arm = entry * (1 + arm_pct)
            peak = max(float(p.peak_price or 0), h)
            if h >= arm:
                # 条件跟踪: 摸到 arm 太慢的就不给它跟踪机会, 直接按固定止盈清掉
                fast = True
                cap = cfg.get("trail_max_days_to_arm")
                if cap is not None:
                    fast = (day - p.open_date).days <= int(cap) * 1.6
                frac = float(cfg.get("tier2_frac", 1.0))
                if not fast:
                    px = max(o, arm)
                    _sell(p.shares, px, "tier2", f"+{arm_pct:.0%} 清仓(动能不足)")
                    p.tier2_done = True
                    p.status = "closed"; p.close_date = day; p.close_reason = "tier2"
                    continue
                if frac < 1.0 and not p.tier2_done:      # 只减仓一次
                    # 先落袋一部分, 剩下的尾巴才去跟踪
                    n = int(p.init_shares * frac / cfg["lot"]) * cfg["lot"]
                    n = min(n, p.shares)
                    if n > 0:
                        _sell(n, max(o, arm), "tier2", f"+{arm_pct:.0%} 减仓, 余仓跟踪")
                    p.tier2_done = True
                    if p.shares <= 0:
                        p.status = "closed"; p.close_date = day
                        p.close_reason = "tier2"
                        continue
                p.peak_price = round(peak, 4)
                new_stop = max(peak * (1 - trail), entry)
                if new_stop > float(p.stop_price):
                    p.stop_price = round(new_stop, 4)
            elif peak > float(p.peak_price or 0):
                p.peak_price = round(peak, 4)
        elif p.shares > 0 and p.tier1_done and not p.tier2_done and h >= entry * (1 + cfg["tier2_pct"]):
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
        _mode = cfg.get("entry_mode", "next_open")
        # ⚠️ next_open 模式要扫的是**前一个交易日**的信号, 因为成交发生在信号日
        # 的次日开盘 —— 也就是今天。
        #
        # 原来直接扫 day 的信号、按 day+1 的开盘价建仓, 却把这笔记进 day 的净值
        # 快照, 于是 day 收盘时账户已经持有一个用"明天开盘价"买的仓位。持有天数
        # 算出来是 -1 天, 浮动盈亏是拿"今天收盘"跟"明天开盘"比 —— 两头都不对,
        # 而且 day 收盘时根本不可能知道明天开盘价。
        #
        # 改成扫 prev、按 day 的开盘价成交后, 三种模式的成交日都等于 day, 账目
        # 和净值曲线才对得上。
        if _mode == "next_open":
            scan_day = (await db.execute(
                select(DailyCandle.trade_date).where(DailyCandle.trade_date < day)
                .group_by(DailyCandle.trade_date)
                .order_by(DailyCandle.trade_date.desc()).limit(1)
            )).scalar_one_or_none()
        else:
            scan_day = day
        cands = [] if scan_day is None else await scan_signals(
            db, scan_day, exclude=held, on_progress=on_progress,
            require_next=_mode == "next_open", mode=_mode,
            # [审计实验 2026-09] universe_neutral: 见 _universe_ok 注释;
            # signal_kind='2c' 可切到因果版(无重绘)信号表, 默认 '2' 不变
            neutral=bool(cfg.get("universe_neutral")),
            kind=str(cfg.get("signal_kind", "2")))
        # ⚠️ 每仓目标必须按"当前净值"算, 不是初始资金 —— 用初始资金就是固定
        # 金额下注, 赚到的钱永远躺在现金里不再投出去, 十年下来差好几倍。
        mv_now = 0.0
        for p in open_now:
            px = (await db.execute(
                select(DailyCandle.close)
                .where(DailyCandle.ts_code == p.ts_code, DailyCandle.trade_date <= day)
                .order_by(DailyCandle.trade_date.desc()).limit(1)
            )).scalar_one_or_none()
            if px:
                mv_now += p.shares * float(px)
        target = (cash + mv_now) / acct.slots
        for cd in cands:
            if free <= 0:
                break
            if _mode in ("signal_close", "fractal_close"):
                price = cd["signal_close"]      # 当日尾盘成交
                buy_on = cd["signal_date"]
            else:
                price = cd["next_open"]         # 信号次日开盘成交
                buy_on = cd["buy_date"]
            if buy_on != day:
                continue        # 成交日必须就是正在结算的这一天
            if price is None or price <= 0:
                continue
            price = price * (1 + cfg.get("entry_slip_pct", 0.0))
            budget = min(target, cash)
            shares = int(budget // (price * cfg["lot"])) * cfg["lot"]
            if shares < cfg["lot"]:
                continue
            amt = shares * price
            fee = trade_fee(amt, is_sell=False, day=buy_on, cfg=cfg)
            if amt + fee > cash:
                continue
            cash -= amt + fee
            pos = PaperPosition(
                account_id=acct.id, ts_code=cd["ts_code"], name=cd["name"],
                open_date=buy_on, open_price=round(price, 4),
                init_shares=shares, shares=shares,
                stop_price=round(price * (1 - cfg["stop_pct"]), 4),
                status="open",
            )
            db.add(pos)
            await db.flush()
            db.add(PaperTrade(
                account_id=acct.id, position_id=pos.id, ts_code=cd["ts_code"],
                name=cd["name"], trade_date=buy_on, action="buy",
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
                       on_progress=None, require_next: bool = True,
                       mode: str = "next_open",
                       neutral: bool = False,
                       kind: str = "2") -> list[dict]:
    """当日可操作的 chan-2buy 买点, 按低流动性优先排序.

    从 chan_signal 预计算表读 —— 现算 4400 只票要 75 秒, 回放时点一下走一天
    根本没法用。表里的 trade_date 已含 CONFIRM_LAG, 这里不用再处理 lag。
    """
    exclude = exclude or set()
    if mode == "fractal_close":
        # 要的是"分型的下一根K正好是 day"的信号。表里 trade_date = 分型 + 2 根,
        # 所以条件等价于: day 与 trade_date 之间, 该票再没有别的K线。
        # (不能直接按日期加减 —— 停牌会让自然日和K线根数对不上。)
        codes = (await db.execute(
            select(ChanSignal.ts_code).where(
                ChanSignal.kind == kind,
                ChanSignal.trade_date > day,
                ChanSignal.trade_date <= day + timedelta(days=20),
                ~exists(select(DailyCandle.ts_code).where(
                    DailyCandle.ts_code == ChanSignal.ts_code,
                    DailyCandle.trade_date > day,
                    DailyCandle.trade_date < ChanSignal.trade_date)),
            )
        )).scalars().all()
    else:
        codes = (await db.execute(
            select(ChanSignal.ts_code)
            .where(ChanSignal.trade_date == day, ChanSignal.kind == kind)
        )).scalars().all()
    codes = [c for c in codes if c not in exclude]
    if not codes:
        return []

    names = dict((await db.execute(
        select(StockBasic.ts_code, StockBasic.name)
        .where(StockBasic.ts_code.in_(codes))
    )).all())
    actives = set((await db.execute(
        select(StockBasic.ts_code).where(StockBasic.ts_code.in_(codes),
                                         StockBasic.is_active.is_(True))
    )).scalars().all())

    # 次日开盘价 —— 信号日的下一个交易日
    nxt = (await db.execute(
        select(DailyCandle.trade_date).where(DailyCandle.trade_date > day)
        .group_by(DailyCandle.trade_date)
        .order_by(DailyCandle.trade_date).limit(1)
    )).scalar_one_or_none()
    if not nxt and require_next:
        return []          # 次日开盘买 -> 没有下一个交易日就买不进

    out: list[dict] = []
    for cd in codes:
        # [审计实验 2026-09] neutral=True 时不用"当前快照"的 is_active 和
        # ST/退市名称过滤 —— 两者都是最新状态, 对历史回测是未来函数。
        # 只保留代码段白名单。正常账户 neutral=False, 行为不变。
        if not neutral and cd not in actives:
            continue
        if not _universe_ok(cd, names.get(cd), ignore_status=neutral):
            continue
        # 20 日均额 + 涨停判定, 只取最后 25 根
        rows = (await db.execute(
            select(DailyCandle.close, DailyCandle.amount)
            .where(DailyCandle.ts_code == cd, DailyCandle.trade_date <= day)
            .order_by(DailyCandle.trade_date.desc()).limit(25)
        )).all()
        if len(rows) < 21:
            continue
        closes = [float(r[0]) for r in rows]
        amts = [float(r[1]) for r in rows]
        amt20 = sum(amts[:20]) / 20
        if amt20 < DEFAULT_CONFIG["min_amount_k"]:
            continue
        if closes[0] > closes[1] * 1.099:      # 涨停买不进
            continue
        nx = (await db.execute(
            select(DailyCandle.open).where(DailyCandle.ts_code == cd,
                                           DailyCandle.trade_date == nxt)
        )).scalar_one_or_none() if nxt else None
        # ⚠️ 只有"次日开盘买"才能因次日停牌而放弃。尾盘买模式下, 下单那一刻
        # 根本不知道明天停不停牌 —— 拿它做过滤就是未来函数。
        if require_next and nx is None:
            continue
        out.append({"ts_code": cd, "name": names.get(cd),
                    "signal_date": day, "buy_date": nxt,
                    "next_open": float(nx) if nx is not None else None,
                    "signal_close": closes[0],
                    "amount_20d_wan": round(amt20 / 10, 1)})
    out.sort(key=lambda x: x["amount_20d_wan"])     # 低流动性优先
    return out
