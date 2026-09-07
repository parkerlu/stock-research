"""形态模型 · 第3步: 组合回测。

⚠️ 信号层过关 ≠ 组合能用 —— 这个项目已经在这上面栽过好几次
   (突破预警信号层各项都好, 裸跑组合比值 0.17)。判据是【年化 > 回撤】。

口径与 scripts/backtest_portfolio.py 一致:
   T日收盘出信号 → T+1开盘买 → 持有H日 → 收盘卖
   成本: 佣金双边万2.5 + 印花税卖出万5 + 过户费双边万0.1 + 单边滑点千1
"""
from __future__ import annotations

import argparse
import asyncio

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

DIR = "/app/data/research/shape"
COMM, STAMP, TRANSFER, SLIP = 0.00025, 0.0005, 0.00001, 0.001


async def load(start):
    """⚠️ 700万行里 ts_code 的 Python 字符串和 trade_date 的 date 对象
    加起来能占好几个G(实测容器 OOM)。取回来立刻转成分类码 + 整数日期索引,
    内存从 ~4G 降到 ~200M。"""
    async with engine.connect() as c:
        cal = [r[0] for r in (await c.execute(text(
            "select distinct trade_date from daily_candle where trade_date >= :s "
            "order by trade_date"), {"s": start})).fetchall()]
        didx = {d: i for i, d in enumerate(cal)}
        # ⚠️ 不能 fetchall 700万行: Row 对象在转换之前就把内存吃光(OOM 137)。
        #    按年分批取, 每批立刻压成 numpy, 峰值降到几百兆。
        chunks_c, chunks_d, chunks_v = [], [], []
        y0, y1 = cal[0].year, cal[-1].year
        for y in range(y0, y1 + 1):
            rows = (await c.execute(text(
                "select ts_code,trade_date,open*adj_factor,high*adj_factor,"
                "low*adj_factor,close*adj_factor,amount,"
                "lag(close*adj_factor) over (partition by ts_code order by trade_date) "
                "from (select * from daily_candle where trade_date >= :a0) t "
                "where trade_date >= :a and trade_date < :b"),
                {"a0": start,
                 "a": max(start, __import__("datetime").date(y, 1, 1)),
                 "b": __import__("datetime").date(y + 1, 1, 1)})).fetchall()
            if not rows:
                continue
            cc = np.empty(len(rows), dtype=object)
            dd = np.empty(len(rows), dtype=np.int32)
            vv = np.empty((len(rows), 6), dtype=np.float32)
            for k, r in enumerate(rows):
                cc[k] = r[0]; dd[k] = didx[r[1]]
                vv[k] = (r[2], r[3], r[4], r[5], r[6],
                         r[7] if r[7] is not None else np.nan)
            del rows
            chunks_c.append(cc); chunks_d.append(dd); chunks_v.append(vv)
        codes = np.concatenate(chunks_c)
        di = np.concatenate(chunks_d)
        vals = np.concatenate(chunks_v)
        del chunks_c, chunks_d, chunks_v
        tim = pd.DataFrame((await c.execute(text("""
            select trade_date, close > avg(close) over (order by trade_date
                     rows between 19 preceding and current row) ok
            from index_daily where ts_code='000852.SH' and trade_date >= :s
        """), {"s": start})).fetchall(), columns=["trade_date", "ok"])

    px = pd.DataFrame(vals, columns=["o", "h", "l", "c", "amt", "pc"])
    px["ts_code"] = pd.Categorical(codes)
    px["di"] = di
    return px, cal, tim


def _limit(code: str) -> float:
    """涨跌停幅度。创业板/科创板 20%, 其余 10%(ST 已在别处排除)。"""
    return 0.20 if code[:3] in ("300", "301", "688") else 0.10


def run(sig, px, cal, tim, hold, maxpos, regime, min_amt_k,
        stop=None, tp=None, strict=False, trail=None, arm=None,
        cap=1_000_000.0):
    """⚠️ 价格不要存成 {(code,date): tuple} 的字典 —— 750万条 Python 元组
    直接把容器撑爆(实测 OOM, 退出码137)。按股票存 numpy 数组, 用
    searchsorted 定位, 内存降到几十兆。"""
    didx = {d: i for i, d in enumerate(cal)}
    px = px.sort_values(["ts_code", "di"])
    arr = {}
    for code, g in px.groupby("ts_code", sort=False, observed=True):
        arr[code] = (g["di"].to_numpy(np.int32),
                     g[["o", "h", "l", "c", "pc"]].to_numpy(np.float32),
                     g["amt"].to_numpy(np.float32))

    def bar(code, i):
        a = arr.get(code)
        if a is None:
            return None
        k = np.searchsorted(a[0], i)
        if k >= len(a[0]) or a[0][k] != i:
            return None
        return a[1][k]

    def amt_of(code, i):
        a = arr.get(code)
        if a is None:
            return 0.0
        k = np.searchsorted(a[0], i)
        return float(a[2][k]) if k < len(a[0]) and a[0][k] == i else 0.0

    ok_day = dict(zip(tim["trade_date"], tim["ok"])) if regime else {}
    byday = {}
    for r in sig.itertuples():
        byday.setdefault(didx.get(r.trade_date, -1), []).append((r.score, r.ts_code))

    cash, holds, trades, eq = cap, [], [], []
    for i, d in enumerate(cal):
        keep = []
        for hh in holds:
            b = bar(hh["ts_code"], i)
            if b is None:
                keep.append(hh); continue
            o_, hi_, lo_, c_ = float(b[0]), float(b[1]), float(b[2]), float(b[3])
            pc_ = float(b[4]) if np.isfinite(b[4]) else c_
            # ⚠️ 一字跌停当天卖不掉 —— 挂单排不上, 只能继续持有到下一天。
            #    不加这条会把最坏的那些笔按止损价结算, 系统性高估。
            if strict and (hi_ - lo_) < 1e-6 and lo_ <= pc_ * (1 - _limit(hh["ts_code"]) + 0.005):
                keep.append(hh); continue
            px_out = None
            # ⚠️ 出场规则必须与【训练标签的定义】逐字一致, 否则模型优化的东西
            #    和回测执行的东西不是一回事 —— 这个项目已经栽过两次。
            #    顺序: 止损优先于止盈(同日都触及时按最坏处理), 再到期。
            if stop is not None and lo_ <= hh["stop_px"]:
                px_out = o_ if o_ < hh["stop_px"] else hh["stop_px"]
            elif tp is not None and hi_ >= hh["tp_px"]:
                px_out = max(o_, hh["tp_px"])
            elif i >= hh["exit_i"]:
                px_out = c_
            # 移动止盈: 涨到 arm 之后开始跟踪, 从最高点回撤 trail 才走。
            # ⚠️ 与固定止盈的取舍: 固定止盈把大涨的票也在同一位置砍掉,
            #    移动止盈让利润跑但每笔都要还回去一段回撤。
            if px_out is None and trail is not None:
                pk = max(hh.get("peak", 0.0), hi_)
                hh["peak"] = pk
                if pk >= hh["arm_px"]:
                    ns = pk * (1 - trail)
                    if ns > hh["stop_px"]:
                        hh["stop_px"] = ns
            if px_out is None:
                keep.append(hh); continue
            p = px_out * (1 - SLIP)
            amt = hh["sh"] * p
            cash += amt - amt * (COMM + STAMP + TRANSFER)
            trades.append((p * hh["sh"] - hh["cost"]) / hh["cost"])
        holds = keep
        if i > 0 and len(holds) < maxpos:
            prev = cal[i - 1]
            if not regime or ok_day.get(prev, False):
                held = {h["ts_code"] for h in holds}
                for _, code in sorted(byday.get(i - 1, []), reverse=True):
                    if len(holds) >= maxpos:
                        break
                    if code in held:
                        continue
                    bb = bar(code, i)
                    if bb is None:
                        continue
                    if amt_of(code, i - 1) < min_amt_k:
                        continue          # 成交额下限, 千元
                    # ⚠️ 只有【一字涨停】(全天封死, 最低价也在涨停位)才真的
                    #    买不到。开盘涨停但盘中打开过的, 挂涨停价能成交。
                    #    早先按"开盘即涨停就跳过"算, 过严。
                    pc0 = float(bb[4]) if np.isfinite(bb[4]) else float(bb[0])
                    lim_up = pc0 * (1 + _limit(code) - 0.005)
                    if strict and float(bb[2]) >= lim_up:
                        continue
                    p = float(bb[0]) * (1 + SLIP)
                    equity = cash + sum(x["sh"] * x["last"] for x in holds)
                    sh = int(min(equity / maxpos, cash) / p / 100) * 100
                    if sh < 100:
                        continue
                    cost = sh * p
                    fee = cost * (COMM + TRANSFER)
                    if cost + fee > cash:
                        continue
                    cash -= cost + fee
                    holds.append({"ts_code": code, "sh": sh, "cost": cost + fee,
                                  "exit_i": i + hold, "last": p,
                                  "stop_px": p * (1 - stop) if stop else 0.0,
                                  "tp_px": p * (1 + tp) if tp else 1e18,
                                  "arm_px": p * (1 + (arm or 0.10)),
                                  "peak": 0.0})
                    held.add(code)
        for x in holds:
            bb = bar(x["ts_code"], i)
            if bb is not None:
                x["last"] = float(bb[3])
        eq.append((d, cash + sum(x["sh"] * x["last"] for x in holds)))

    e = pd.DataFrame(eq, columns=["d", "eq"])
    e["r"] = e["eq"].pct_change().fillna(0.0)
    yrs = (e["d"].iloc[-1] - e["d"].iloc[0]).days / 365.25
    cagr = (e["eq"].iloc[-1] / cap) ** (1 / yrs) - 1
    dd = -(e["eq"] / e["eq"].cummax() - 1).min()
    t = np.array(trades)
    return {"年化": cagr, "回撤": dd, "比值": cagr / max(dd, 1e-9),
            "笔数": len(t), "胜率": float((t > 0).mean()) if len(t) else 0, "eq": e}


def hedged(e: pd.DataFrame, idx: pd.DataFrame, cost_bp: float = 2.0) -> dict:
    """对冲版: 每天减掉中证1000的当日收益(等于用指数期货/ETF做空等额)。
    ⚠️ 这【不是】择时 —— 仓位天天满、对牛熊无感, 只是把 beta 剥掉。
       模型本身依旧没有 regime 记忆, 审计结论不受影响。
    cost_bp: 对冲每日摩擦(展期/跟踪误差), 单位基点。"""
    j = e.merge(idx, on="d", how="left")
    j["mr"] = j["mr"].fillna(0.0)
    j["hr"] = j["r"] - j["mr"] - cost_bp / 10000.0
    nav = (1 + j["hr"]).cumprod()
    yrs = (j["d"].iloc[-1] - j["d"].iloc[0]).days / 365.25
    cagr = nav.iloc[-1] ** (1 / yrs) - 1
    dd = -(nav / nav.cummax() - 1).min()
    return {"年化": cagr, "回撤": dd, "比值": cagr / max(dd, 1e-9), "nav": nav, "d": j["d"]}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--top", type=float, default=0.99, help="取当日分位以上")
    ap.add_argument("--maxpos", type=int, default=10)
    ap.add_argument("--min-amt-k", type=float, default=20000)
    ap.add_argument("--rules", default="none:none,0.12:0.06,0.15:0.08,0.08:0.05",
                    help="止盈:止损 组合, 逗号分隔")
    ap.add_argument("--sigfile", default=None)
    ap.add_argument("--strict", action="store_true", help="涨跌停不可成交")
    ap.add_argument("--trail", type=float, default=None, help="移动止盈回撤比例")
    ap.add_argument("--arm", type=float, default=None, help="涨到多少才开始跟踪")
    a = ap.parse_args()

    sig = pd.read_parquet(a.sigfile or f"{DIR}/oos_top_v2_h{a.h}.parquet")
    sig = sig[sig.rk >= a.top]
    sig["trade_date"] = pd.to_datetime(sig["trade_date"]).dt.date
    start = min(sig["trade_date"])
    px, cal, tim = await load(start)
    async with engine.connect() as c:
        ir = pd.DataFrame((await c.execute(text(
            "select trade_date, close from index_daily where ts_code='000852.SH' "
            "and trade_date >= :s order by trade_date"), {"s": start})).fetchall(),
            columns=["d", "close"])
    ir["close"] = ir["close"].astype(float)
    idx = pd.DataFrame({"d": ir["d"], "mr": ir["close"].pct_change().fillna(0.0)})

    print(f"\n形态模型 h={a.h}  分位>={a.top}  {len(sig):,} 个信号  "
          f"{a.maxpos} 仓位  样本外 {start} ~ {max(sig['trade_date'])}\n")
    print(f"{'止盈/止损':<12}{'年化':>9}{'回撤':>9}{'比值':>8}{'胜率':>8}{'笔数':>8}")
    for rule in a.rules.split(","):
        tps, sls = rule.split(":")
        tp = None if tps == "none" else float(tps)
        stop = None if sls == "none" else float(sls)
        r = run(sig, px, cal, tim, a.h, a.maxpos, False, a.min_amt_k, stop, tp,
                a.strict, a.trail, a.arm)
        print(f"{rule:<12}{r['年化']:>9.2%}{r['回撤']:>9.2%}{r['比值']:>8.2f}"
              f"{r['胜率']:>8.1%}{r['笔数']:>8d}")
        e = r["eq"].copy()
        e["y"] = pd.to_datetime(e["d"]).dt.year
        yr = e.groupby("y")["eq"].agg(["first", "last"])
        print("  逐年 " + "  ".join(f"{y}:{(v['last']/v['first']-1):+.0%}"
                                    for y, v in yr.iterrows()))
        hg = hedged(r["eq"], idx)
        print(f"  └ 对冲中证1000后: 年化 {hg['年化']:>7.2%}  回撤 {hg['回撤']:>6.2%}"
              f"  比值 {hg['比值']:.2f}")
        hn = pd.DataFrame({"d": hg["d"], "nav": hg["nav"]})
        hn["y"] = pd.to_datetime(hn["d"]).dt.year
        hy = hn.groupby("y")["nav"].agg(["first", "last"])
        print("    逐年 " + "  ".join(f"{y}:{(v['last']/v['first']-1):+.0%}"
                                      for y, v in hy.iterrows()))
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
