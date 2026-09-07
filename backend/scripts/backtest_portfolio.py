"""组合回测(固定持有) —— 判断一个信号能不能变成策略。

用法:
    docker exec stock-backend-1 python -m scripts.backtest_portfolio sar
    docker exec stock-backend-1 python -m scripts.backtest_portfolio mmweek --hold 40
    docker exec stock-backend-1 python -m scripts.backtest_portfolio breakout   # 复现 2.62

⚠️ 这个脚本存在的理由: SAR 的 1.74 当初是在容器里写的临时脚本跑的, 重建镜像
   就没了, 想复现只能重写。研究脚本一律进仓库。

⚠️ 信号层过关 ≠ 组合能用。突破预警信号层各项都好, 裸跑组合比值只有 0.17,
   加了择时才到 2.62。判据是【年化 > 回撤】。

口径:
  · T 日收盘出信号 → T+1 开盘买入(不可能更早) → 持有 HOLD 个交易日 → 收盘卖出
  · 等权 MAXPOS 个仓位, 满仓则当日多余信号丢弃(按信号强度排序取前几个)
  · 成本: 佣金双边万2.5 + 印花税卖出万5 + 过户费双边万0.1 + 单边滑点千1
  · 择时: 中证1000 收盘 > 自身 MA20 才建仓(小盘基准, 见 sync_breakout_signals)
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date as _date

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

COMM, STAMP, TRANSFER, SLIP = 0.00025, 0.0005, 0.00001, 0.001

# 各信号源 -> (SQL, 默认持有天数)。SQL 必须给出 ts_code / trade_date / strength,
# 且 trade_date 是【信号已经成立】的那一天(收盘后可知)。
SOURCES = {
    "sar": ("""
        select ts_code, trade_date, prob::float strength from sar_signal
    """, 20),
    "breakout": ("""
        select ts_code, trade_date, prob::float strength from breakout_signal
    """, 20),
    "liftalert": ("""
        select ts_code, trade_date, dl_value::float strength from dongli_signal
    """, 10),
    # 周线版是【状态】: 买线>0 的每一周都算, 不只是起始周。
    # week_end 存的是当周最后一个交易日, 所以下一个交易日开盘买 = 下周一开盘。
    "mmweek": ("""
        select ts_code, week_end trade_date, buy_line::float strength
        from maimai_weekly where buy_line > 0 and week_end >= :s
    """, 40),
}


async def load(src: str, start):
    sql, _ = SOURCES[src]
    async with engine.connect() as c:
        sig = pd.DataFrame(
            (await c.execute(text(sql if ":s" in sql else
                                  (sql + " and trade_date >= :s" if "where" in sql
                                   else sql + " where trade_date >= :s")),
                             {"s": start})).fetchall(),
            columns=["ts_code", "trade_date", "strength"])
        codes = list(sig["ts_code"].unique())
        px = pd.DataFrame((await c.execute(text(
            "select ts_code,trade_date,open,close,adj_factor from daily_candle "
            "where trade_date >= :s and ts_code = any(:cs)"),
            {"s": start, "cs": codes})).fetchall(),
            columns=["ts_code", "trade_date", "open", "close", "adj"])
        cal = [r[0] for r in (await c.execute(text(
            "select distinct trade_date from daily_candle where trade_date >= :s "
            "order by trade_date"), {"s": start})).fetchall()]
        # 择时: 中证1000 在自身 MA20 之上
        tim = pd.DataFrame((await c.execute(text("""
            select trade_date, close > avg(close) over (order by trade_date
                     rows between 19 preceding and current row) ok
            from index_daily where ts_code = '000852.SH' and trade_date >= :s
        """), {"s": start})).fetchall(), columns=["trade_date", "ok"])
    return sig, px, cal, tim


def run(sig, px, cal, tim, hold: int, maxpos: int, regime: bool, cap=1_000_000.0):
    px = px.copy()
    for col in ("open", "close", "adj"):
        px[col] = px[col].astype(float)
    # 前复权: 用复权因子还原真实涨跌, 否则除权日会算出假亏损
    px["o"] = px["open"] * px["adj"]
    px["c"] = px["close"] * px["adj"]
    book = {(r.ts_code, r.trade_date): (r.o, r.c) for r in px.itertuples()}
    idx = {d: i for i, d in enumerate(cal)}
    ok_day = dict(zip(tim["trade_date"], tim["ok"])) if regime else {}

    byday: dict[object, list] = {}
    for r in sig.itertuples():
        byday.setdefault(r.trade_date, []).append((r.strength, r.ts_code))

    cash, holds, trades, eq = cap, [], [], []
    for i, d in enumerate(cal):
        # --- 到期卖出 ---
        keep = []
        for h in holds:
            if i >= h["exit_i"] and (h["ts_code"], d) in book:
                p = book[(h["ts_code"], d)][1] * (1 - SLIP)
                amt = h["sh"] * p
                cash += amt - amt * (COMM + STAMP + TRANSFER)
                trades.append((p * h["sh"] - h["cost"]) / h["cost"])
            else:
                keep.append(h)
        holds = keep
        # --- 买入: 用【昨天】的信号, 今天开盘成交 ---
        if i > 0 and len(holds) < maxpos:
            prev = cal[i - 1]
            if not regime or ok_day.get(prev, False):
                held = {h["ts_code"] for h in holds}
                for _, code in sorted(byday.get(prev, []), reverse=True):
                    if len(holds) >= maxpos:
                        break
                    if code in held or (code, d) not in book:
                        continue
                    p = book[(code, d)][0] * (1 + SLIP)
                    equity = cash + sum(hh["sh"] * book.get((hh["ts_code"], d),
                                        (0, hh["last"]))[1] for hh in holds)
                    budget = min(equity / maxpos, cash)
                    sh = int(budget / p / 100) * 100
                    if sh < 100:
                        continue
                    cost = sh * p
                    fee = cost * (COMM + TRANSFER)
                    if cost + fee > cash:
                        continue
                    cash -= cost + fee
                    holds.append({"ts_code": code, "sh": sh, "cost": cost + fee,
                                  "exit_i": i + hold, "last": p})
                    held.add(code)
        for h in holds:
            if (h["ts_code"], d) in book:
                h["last"] = book[(h["ts_code"], d)][1]
        eq.append((d, cash + sum(h["sh"] * h["last"] for h in holds), len(holds)))

    e = pd.DataFrame(eq, columns=["d", "eq", "n"])
    yrs = (e["d"].iloc[-1] - e["d"].iloc[0]).days / 365.25
    cagr = (e["eq"].iloc[-1] / cap) ** (1 / yrs) - 1
    dd = (e["eq"] / e["eq"].cummax() - 1).min()
    t = np.array(trades)
    return {"年化": cagr, "回撤": -dd, "比值": cagr / max(-dd, 1e-9),
            "笔数": len(t), "胜率": float((t > 0).mean()) if len(t) else 0.0,
            "年数": yrs, "eq": e}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=list(SOURCES))
    ap.add_argument("--hold", type=int, default=None)
    ap.add_argument("--maxpos", type=int, default=8)
    ap.add_argument("--exclude", default=None,
                    help="排除名单 parquet(ts_code, trade_date), 形态模型 Bot20%")
    ap.add_argument("--start", default="2019-01-01")
    a = ap.parse_args()
    start = _date.fromisoformat(a.start)
    hold = a.hold or SOURCES[a.source][1]

    sig, px, cal, tim = await load(a.source, start)
    n0 = len(sig)
    if a.exclude:
        # ⚠️ 排除名单按【信号当天】匹配 —— 模型那天嫌弃这只票, 就不买。
        #    不能用"之后某天"的名单, 那是穿越。
        ex = pd.read_parquet(a.exclude)
        ex["trade_date"] = pd.to_datetime(ex["trade_date"]).dt.date
        ex = set(zip(ex["ts_code"], ex["trade_date"]))
        keep = [not ((r.ts_code, r.trade_date) in ex) for r in sig.itertuples()]
        sig = sig[keep]
        print(f"  排除后 {len(sig):,} / {n0:,} 条 (剔掉 {(1-len(sig)/max(n0,1))*100:.1f}%)")
    print(f"\n{a.source}: 信号 {len(sig)} 条 / {sig['ts_code'].nunique()} 只票 / "
          f"持有 {hold} 交易日 / {a.maxpos} 仓位\n")
    print(f"{'择时':<10}{'年化':>9}{'回撤':>9}{'比值':>8}{'胜率':>8}{'笔数':>8}")
    for regime in (False, True):
        r = run(sig, px, cal, tim, hold, a.maxpos, regime)
        print(f"{'中证1000>MA20' if regime else '裸跑':<10}"
              f"{r['年化']:>9.2%}{r['回撤']:>9.2%}{r['比值']:>8.2f}"
              f"{r['胜率']:>8.1%}{r['笔数']:>8d}")
        # 逐年
        e = r["eq"].copy()
        e["y"] = pd.to_datetime(e["d"]).dt.year
        yr = e.groupby("y")["eq"].agg(["first", "last"])
        s = "  ".join(f"{y}:{(v['last']/v['first']-1):+.0%}" for y, v in yr.iterrows())
        print(f"  逐年 {s}")
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
