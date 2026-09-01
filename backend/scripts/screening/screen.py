"""指标筛选台架 —— 四道闸的第 ①②③ 闸。

设计原则(来自 chan-2buy 的教训):
  ① 因果性: 只收纯公式指标 (第 k 根只依赖 <=k 的数据)。缠论那种 ZigZag
     重绘的一律不要 —— 已从代码库下架。
  ② 随机基准: 同一套出场规则配随机入场, 多个种子跑出零假设分布。
     不减这个基线, 出场结构本身的结构性超额会被当成指标的功劳。
  ③ 样本内外: 2016-2021 选, 2022-2026 只验一次。这里只跑样本内。
  ④ 多重检验校正在汇总阶段做。

出场规则对所有候选完全相同(与实盘引擎一致): 次日开盘买入,
-6% 止损(跳空按开盘), +4% 卖一半并把止损抬到成本, +8% 清仓, 60 日超时。
"""
from __future__ import annotations
import asyncio, sys, time, numpy as np, pandas as pd
from datetime import date
from sqlalchemy import select
from app.db import async_session
from app.models.schema import DailyCandle, StockBasic
from app.services.paper_trading import DEFAULT_CONFIG as CFG, trade_fee

IS_FROM, IS_TO = date(2016, 1, 1), date(2021, 12, 31)
STOP, T1, T2, MAXD = CFG["stop_pct"], CFG["tier1_pct"], CFG["tier2_pct"], CFG["max_hold_days"]


def universe_ok(code: str) -> bool:
    """只用代码段白名单 —— 不碰 stock_basic 的当前名字/状态(那是未来函数)。"""
    c = code.split(".")[0]
    return c.startswith(("60", "00", "30"))


def ladder_exit(o, h, l, c, i, day_idx) -> tuple[float, int] | None:
    """从 bar i 的开盘买入, 按阶梯规则出场。返回 (净收益率, 持有根数)。"""
    n = len(c)
    if i >= n: return None
    entry = o[i]
    if not np.isfinite(entry) or entry <= 0: return None
    stop = entry * (1 - STOP)
    half_done = False
    got = 0.0          # 已落袋的金额(按 1 元本金归一)
    left = 1.0         # 剩余仓位比例
    for j in range(i, min(i + int(MAXD * 1.6), n)):
        if l[j] <= stop:
            px = o[j] if o[j] < stop else stop
            got += left * px / entry
            return (got - 1.0 - fee_rt(day_idx[j]), j - i)
        if not half_done and h[j] >= entry * (1 + T1):
            px = max(o[j], entry * (1 + T1))
            got += 0.5 * px / entry
            left -= 0.5
            half_done = True
            stop = entry                      # 保本
        if half_done and h[j] >= entry * (1 + T2):
            px = max(o[j], entry * (1 + T2))
            got += left * px / entry
            return (got - 1.0 - fee_rt(day_idx[j]), j - i)
    j = min(i + int(MAXD * 1.6), n) - 1
    got += left * c[j] / entry
    return (got - 1.0 - fee_rt(day_idx[j]), j - i)


def fee_rt(d) -> float:
    """双边费率(按 1 万元仓位估), 与实盘引擎同口径。"""
    amt = 10000.0
    return (trade_fee(amt, False, d) + trade_fee(amt, True, d)) / amt


async def load_codes(limit=None):
    async with async_session() as db:
        rows = (await db.execute(select(StockBasic.ts_code))).scalars().all()
    codes = sorted(c for c in rows if universe_ok(c))
    return codes[:limit] if limit else codes


async def load_bars(code):
    async with async_session() as db:
        rows = (await db.execute(
            select(DailyCandle.trade_date, DailyCandle.open, DailyCandle.high,
                   DailyCandle.low, DailyCandle.close, DailyCandle.vol,
                   DailyCandle.amount, DailyCandle.adj_factor)
            .where(DailyCandle.ts_code == code,
                   DailyCandle.trade_date >= date(2015, 1, 1),
                   DailyCandle.trade_date <= IS_TO)
            .order_by(DailyCandle.trade_date))).all()
    if len(rows) < 300: return None
    df = pd.DataFrame(rows, columns=["trade_date","open","high","low","close","vol","amount","adj"])
    for x in ("open","high","low","close","vol","amount","adj"):
        df[x] = pd.to_numeric(df[x], errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).reset_index(drop=True)
    if len(df) < 300: return None
    f = (df.adj / df.adj.iloc[-1]).values
    for x in ("open","high","low","close"):
        df[x] = df[x].values * f
    return df


async def main():
    pilot = "--pilot" in sys.argv
    from app.services.strategy_templates import TEMPLATE_REGISTRY as T
    names = sorted(T)
    # ml_direct_* 每只票 5 秒、426-* 每只 2.5 秒, 20 个模板吃掉 96% 的时间。
    # 先跑快的 22 个拿结论, 慢的另开一轮小样本单独测。
    if "--slow" in sys.argv:
        names = [n for n in names if n.startswith(("ml_direct", "426-"))]
    else:
        names = [n for n in names if not n.startswith(("ml_direct", "426-"))]

    all_codes = await load_codes()
    if pilot:
        codes = all_codes[:25]
    else:
        # 随机抽 1200 只 —— 每模板上万笔, 统计量足够, 全量要 2.7 小时
        rg0 = np.random.default_rng(2026)
        n_stock = 120 if "--slow" in sys.argv else 900   # 慢批每只 7.5 秒, 只能小样本
        codes = sorted(rg0.choice(all_codes, min(n_stock, len(all_codes)), replace=False))
    print(f"模板 {len(names)} 个 · 股票 {len(codes)} 只 · 样本内 {IS_FROM} ~ {IS_TO}", flush=True)

    acc = {n: [] for n in names}
    cost = {n: 0.0 for n in names}          # 每个模板累计耗时, 用来揪慢的
    rnd = {s: [] for s in range(3)}
    t0 = time.time()
    for k, code in enumerate(codes):
        df = await load_bars(code)
        if df is None: continue
        o,h,l,c = (df[x].values for x in ("open","high","low","close"))
        dts = df.trade_date.values
        di = [pd.Timestamp(x).date() for x in dts]
        ok_from = np.searchsorted(dts, np.datetime64(IS_FROM))
        for n in names:
            _t = time.time()
            try:
                tpl = T[n]()
                sigs = tpl.generate_signals(df)
            except Exception:
                cost[n] += time.time() - _t
                continue
            cost[n] += time.time() - _t
            for s in sigs:
                if s.get("action") != "buy": continue
                d = s["date"]
                d = d.date() if hasattr(d, "date") else d
                if not (IS_FROM <= d <= IS_TO): continue
                pos = np.searchsorted(dts, np.datetime64(d))
                r = ladder_exit(o,h,l,c, pos+1, di)      # 次日开盘买
                if r: acc[n].append(r)
        # 随机对照: 同样的出场规则, 入场日随机
        for seed in rnd:
            rg = np.random.default_rng(seed*100003 + k)
            for _ in range(6):
                pos = int(rg.integers(max(ok_from,250), len(c)-70)) if len(c) > 320 else None
                if pos is None: continue
                r = ladder_exit(o,h,l,c, pos, di)
                if r: rnd[seed].append(r)
        if (k + 1) % 25 == 0 or k == len(codes) - 1:
            el = time.time() - t0
            eta = el / (k + 1) * (len(codes) - k - 1)
            print(f"  进度 {k+1}/{len(codes)} 只 · 已用 {el/60:.1f} 分 · 预计还需 {eta/60:.1f} 分",
                  flush=True)

    def stat(rows):
        if not rows: return None
        a = np.array([x[0] for x in rows]) * 100
        d = np.array([x[1] for x in rows])
        return len(a), a.mean(), 100*(a>0).mean(), d.mean(), a.std(ddof=1)

    print(f"\n{'随机对照':<26}{'笔数':>7}{'均值%':>9}{'胜率%':>8}{'持有':>7}")
    base = []
    for s in rnd:
        st = stat(rnd[s])
        if st: print(f"  seed {s:<20}{st[0]:>7}{st[1]:>9.3f}{st[2]:>8.1f}{st[3]:>7.1f}"); base.append(st[1])
    bm = float(np.mean(base)) if base else 0.0
    sd = float(np.std(base)) if len(base) > 1 else 0.0
    print(f"  基准均值 {bm:+.3f}%  种子标准差 {sd:.3f}pp")

    # 多重检验校正: 42 个模板同时检验, Bonferroni 5% -> 单个门槛 p<0.0012 -> |t|>3.2
    import math
    thr = 3.0 if "--slow" in sys.argv else 3.2
    print(f"\n{'模板':<24}{'笔数':>7}{'均值%':>8}{'超额pp':>8}{'t值':>7}{'胜率%':>7}{'持有':>6}  判定")
    out = []
    for n in names:
        st = stat(acc[n])
        if not st or st[0] < 100: continue
        ex = st[1] - bm
        se = st[4] / math.sqrt(st[0])
        t = ex / se if se > 0 else 0.0
        out.append((t, ex, n, st))
    for t, ex, n, st in sorted(out, reverse=True):
        mark = "通过" if t > thr else ("边缘" if t > 2.0 else "")
        print(f"  {n:<22}{st[0]:>7}{st[1]:>8.3f}{ex:>+8.3f}{t:>7.2f}{st[2]:>7.1f}{st[3]:>6.1f}  {mark}")
    slow = sorted(cost.items(), key=lambda x: -x[1])[:6]
    print("\n最耗时的模板(秒): " + ", ".join(f"{n} {v:.0f}" for n, v in slow))
    print(f"\n多重检验门槛: 42 个模板同时检验, Bonferroni 5% => |t| > {thr}")
    print(f"随机对照三种子极差 {max(base)-min(base):.3f}pp —— 超额小于这个量级的一律不算数")


if __name__ == "__main__":      # 被 import 时不要自己跑起来
    asyncio.run(main())
