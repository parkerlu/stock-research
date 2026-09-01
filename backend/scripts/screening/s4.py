"""S4 组合层面模拟 —— 按 10 万账户的真实条件, 不套机构容量假设。

与前面"单笔超额"的区别: 这里有 10 个仓位的硬约束。信号多于空仓位时只能挑
一部分, 仓位被占住时再好的信号也进不来。单笔超额换算成年化的那个乘法
(1.0176^32) 假设仓位永不空闲, 这里要把它证伪或证实。

费用按 A 股实际三笔: 佣金万3(最低5元)、印花税卖出单边(按成交日切税率)、
过户费万0.1。10 万 / 10 仓位 = 每笔 1 万, 佣金 5 元下限是主要成本。
"""
from __future__ import annotations
import asyncio, json, math, sys, time, numpy as np, pandas as pd
from datetime import date
from sqlalchemy import select
from app.db import async_session
from app.models.schema import DailyCandle
from app.services.paper_trading import DEFAULT_CONFIG as CFG, trade_fee
import screen as S

STOP, T1, T2 = CFG["stop_pct"], CFG["tier1_pct"], CFG["tier2_pct"]
MAXH, LOT = int(CFG["max_hold_days"] * 1.6), CFG["lot"]
SLOTS, CAP = 10, 100_000.0
MIN_AMT_K = CFG["min_amount_k"]          # 20日均额下限(千元)


async def build(codes, d0, d1, tmpl):
    """预先算好每只票的K线与信号日, 之后按天推进。"""
    from app.services.strategy_templates import TEMPLATE_REGISTRY as T
    data = {}
    for k, code in enumerate(codes):
        async with async_session() as db:
            rows = (await db.execute(
                select(DailyCandle.trade_date, DailyCandle.open, DailyCandle.high,
                       DailyCandle.low, DailyCandle.close, DailyCandle.vol,
                       DailyCandle.amount, DailyCandle.adj_factor)
                .where(DailyCandle.ts_code == code,
                       DailyCandle.trade_date >= date(d0.year-2, d0.month, 1),
                       DailyCandle.trade_date <= d1)
                .order_by(DailyCandle.trade_date))).all()
        if len(rows) < 300: continue
        df = pd.DataFrame(rows, columns=["trade_date","open","high","low","close","vol","amount","adj"])
        for x in ("open","high","low","close","vol","amount","adj"):
            df[x] = pd.to_numeric(df[x], errors="coerce")
        df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
        if len(df) < 300: continue
        f = (df.adj / df.adj.iloc[-1]).values
        for x in ("open","high","low","close"):
            df[x] = df[x].values * f
        sig = set()
        if tmpl:
            try:
                for sg in T[tmpl]().generate_signals(df):
                    if sg.get("action") == "buy":
                        dd = sg["date"]; sig.add(dd.date() if hasattr(dd,"date") else dd)
            except Exception as e:
                if not hasattr(build, "_warned"):
                    build._warned = True
                    print(f"    ⚠ 信号生成失败 {code}: {type(e).__name__}: {e}", flush=True)
        dts = [pd.Timestamp(x).date() for x in df.trade_date.values]
        data[code] = dict(d=dts, idx={t:i for i,t in enumerate(dts)},
                          o=df.open.values, h=df.high.values, l=df.low.values,
                          c=df.close.values, amt=df.amount.values, sig=sig)
        if (k+1) % 150 == 0: print(f"    载入 {k+1}/{len(codes)}", flush=True)
    return data


def simulate(data, days, rng=None):
    """逐日撮合。rng 非空 = 随机入场对照(忽略真实信号, 每天随机挑候选)。"""
    cash, pos = CAP, {}                  # code -> dict
    peak = eq = CAP
    mdd = 0.0
    trades = 0
    for di, day in enumerate(days):
        # --- 出场 ---
        for code in list(pos):
            p = pos[code]; s = data[code]
            i = s["idx"].get(day)
            if i is None: continue
            o,h,l = s["o"][i], s["h"][i], s["l"][i]
            ent, stop = p["entry"], p["stop"]
            if l <= stop:
                px = o if o < stop else stop
                cash += p["sh"]*px - trade_fee(p["sh"]*px, True, day)
                del pos[code]; trades += 1; continue
            if not p["half"] and h >= ent*(1+T1):
                px = max(o, ent*(1+T1)); n = (p["sh"]//2//LOT)*LOT
                if n > 0:
                    cash += n*px - trade_fee(n*px, True, day); p["sh"] -= n
                p["half"] = True; p["stop"] = ent
            if p["half"] and h >= ent*(1+T2):
                px = max(o, ent*(1+T2))
                cash += p["sh"]*px - trade_fee(p["sh"]*px, True, day)
                del pos[code]; trades += 1; continue
            if di - p["di"] >= MAXH:
                px = s["c"][i]
                cash += p["sh"]*px - trade_fee(p["sh"]*px, True, day)
                del pos[code]; trades += 1
        # --- 入场: 用**前一日**的信号, 按今日开盘价成交 ---
        free = SLOTS - len(pos)
        if free > 0 and di > 0:
            prev = days[di-1]
            cands = []
            for code, s in data.items():
                if code in pos: continue
                j = s["idx"].get(prev); i = s["idx"].get(day)
                if j is None or i is None or j < 20: continue
                hit = (rng.random() < 0.004) if rng is not None else (prev in s["sig"])
                if not hit: continue
                a20 = float(np.nanmean(s["amt"][j-19:j+1]))
                if not np.isfinite(a20) or a20 < MIN_AMT_K: continue
                if s["c"][j] > s["c"][j-1]*1.099: continue     # 涨停买不进
                cands.append((a20, code, i))
            cands.sort()                                        # 低流动性优先
            # ⚠️ 仓位大小要用**昨收**估净值。用当日收盘价 = 开盘下单时already
            # 知道今天怎么收 —— 这正是我在缠论审计里挑出来的那类盘中穿越。
            mv = sum(p["sh"]*data[c]["c"][data[c]["idx"][prev]]
                     for c,p in pos.items() if prev in data[c]["idx"])
            target = (cash + mv) / SLOTS
            for a20, code, i in cands:
                if free <= 0: break
                px = data[code]["o"][i]
                if not np.isfinite(px) or px <= 0: continue
                sh = int(min(target, cash) // (px*LOT)) * LOT
                if sh < LOT: continue
                cost = sh*px + trade_fee(sh*px, False, day)
                if cost > cash: continue
                cash -= cost
                pos[code] = dict(entry=px, sh=sh, stop=px*(1-STOP), half=False, di=di)
                free -= 1
        # --- 净值 ---
        mv = 0.0
        for code, p in pos.items():
            s = data[code]; i = s["idx"].get(day)
            if i is not None: mv += p["sh"]*s["c"][i]
        eq = cash + mv
        peak = max(peak, eq); mdd = max(mdd, (peak-eq)/peak)
    return eq, mdd, trades


async def main():
    tmpl = sys.argv[1]
    # 第三个窗口: 2022-01~2024-08。对 tdx-dual-kdj (纯公式, 无训练) 而言,
    # 这段既没参与样本内选参(2016-2021), 也不是 S3/S4 用的那段(2024-09 之后)
    # —— 是真正独立的第三次检验。
    if len(sys.argv) > 2 and sys.argv[2] == "mid":
        d0, d1 = date(2022,1,1), date(2024,8,31)
    else:
        d0, d1 = date(2024,9,1), date(2026,8,31)
    all_codes = await S.load_codes()
    rng0 = np.random.default_rng(4242)
    codes = sorted(rng0.choice(all_codes, min(900, len(all_codes)), replace=False))
    print(f"S4 组合模拟 · {tmpl} · {d0}~{d1} · {len(codes)} 只 · {SLOTS} 仓位 · {CAP:,.0f} 本金", flush=True)
    data = await build(codes, d0, d1, tmpl)
    days = sorted({d for s in data.values() for d in s["d"] if d0 <= d <= d1})
    print(f"  {len(data)} 只有效 · {len(days)} 个交易日", flush=True)
    yrs = len(days)/252

    eq, mdd, tr = simulate(data, days)
    print(f"\n{tmpl:<18} 期末 {eq:>12,.0f}  {eq/CAP:>6.2f}x  年化 {100*((eq/CAP)**(1/yrs)-1):>7.2f}%"
          f"  回撤 {100*mdd:>5.1f}%  {tr} 笔")
    # 3 个种子的极差就有 37.7pp —— 噪声比策略差异还大, 必须多跑几个把分布量准
    cagrs, mdds = [], []
    for sd in range(12):
        e2, m2, t2 = simulate(data, days, rng=np.random.default_rng(sd+1))
        cg = 100*((e2/CAP)**(1/yrs)-1)
        cagrs.append(cg); mdds.append(100*m2)
        print(f"  随机对照 seed{sd:<2}  {e2:>12,.0f}  {e2/CAP:>6.2f}x  年化 {cg:>7.2f}%"
              f"  回撤 {100*m2:>5.1f}%  {t2} 笔", flush=True)
    a = np.array(cagrs)
    strat = 100*((eq/CAP)**(1/yrs)-1)
    z = (strat - a.mean()) / a.std(ddof=1)
    beat = int((a < strat).sum())
    print(f"\n随机对照 {len(a)} 个种子: 均值 {a.mean():.2f}%  标准差 {a.std(ddof=1):.2f}pp"
          f"  区间 [{a.min():.2f}%, {a.max():.2f}%]  平均回撤 {np.mean(mdds):.1f}%")
    print(f"策略 {strat:.2f}%  =>  超出随机均值 {strat-a.mean():+.2f}pp  z = {z:.2f}"
          f"  ({beat}/{len(a)} 个种子被跑赢)")
    print(f"判定: {'显著跑赢' if z > 2.5 and beat == len(a) else '与随机不可区分' if z < 2.0 else '边缘'}")

if __name__ == "__main__":
    asyncio.run(main())
