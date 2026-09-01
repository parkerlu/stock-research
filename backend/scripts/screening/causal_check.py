"""S1 因果性检验 —— 信号是否会被后来的K线追溯修改?

原理: 一个因果指标, 在 bar k 时算出的信号集, 必须等于用全历史算完后
截取到 k 为止的那一段。若不等, 说明它用到了 k 之后的信息 —— 这正是缠论
ZigZag 重绘的特征 (当时成立的买点, 后来被更低的低点撤销)。

做法: 每只票取若干随机切点, 用 df[:cut] 重算, 与全历史结果的对应前缀比对。
  · 撤销(phantom): 截断时有、全历史没有 —— 实盘会照买, 回测里却不存在
  · 迟现(missing): 截断时没有、全历史有 —— 回测能看到, 实盘当时看不到
两者都是未来函数, 方向不同。
"""
from __future__ import annotations
import asyncio, sys, time, numpy as np, pandas as pd
from screen import load_bars, load_codes
from app.services.strategy_templates import TEMPLATE_REGISTRY as T

# 快批里通过 |t|>3.2 的 16 个
PASSED = ["tdx-dual-kdj","rev-55","rev-60","rev-50","mm-55","mm-50","mm-30","mm-40",
          "tdx-macd-pit","rev-40","rev-30","mmhz","mm-broad-50","mm-pure-50",
          "mm-broad-40","mm-pure-40"]
# ⚠️ 切点必须密集且覆盖"周内每一天"。
# 教训: 原来 6 个随机切点, 信号落进残缺周的概率太低 —— tdx-dual-kdj 的周线
# 穿越(周一就用到周五收盘)只测出 0.05% 不一致, 我把低检验力误当成了干净。
# 多周期策略请另跑 week_leak.py, 它专门在每周第 1~4 天切。
NCUT, NSTOCK = 24, 120

def buy_dates(tpl, df):
    try:
        out = set()
        for s in tpl.generate_signals(df):
            if s.get("action") == "buy":
                d = s["date"]
                out.add(d.date() if hasattr(d, "date") else d)
        return out
    except Exception:
        return None

async def main():
    codes = await load_codes()
    rng = np.random.default_rng(9)
    codes = list(rng.choice(codes, NSTOCK, replace=False))
    stat = {n: [0, 0, 0] for n in PASSED}      # [实时信号数, 撤销数, 迟现数]
    t0 = time.time()
    done = 0
    for code in codes:
        df = await load_bars(code)
        if df is None or len(df) < 400: continue
        done += 1
        dates = [pd.Timestamp(x).date() for x in df.trade_date.values]
        cuts = sorted(rng.choice(range(300, len(df)), min(NCUT, len(df)-300), replace=False))
        for n in PASSED:
            tpl = T[n]()
            full = buy_dates(tpl, df)
            if full is None: continue
            for cut in cuts:
                part = buy_dates(tpl, df.iloc[:cut].reset_index(drop=True))
                if part is None: continue
                edge = dates[cut-1]
                full_prefix = {d for d in full if d <= edge}
                stat[n][0] += len(part)
                stat[n][1] += len(part - full_prefix)      # 截断有、全历史无 = 被撤销
                stat[n][2] += len(full_prefix - part)      # 全历史有、截断无 = 迟现
        if done % 30 == 0:
            print(f"  {done}/{NSTOCK} 只 · {time.time()-t0:.0f}s", flush=True)

    print(f"\n{'模板':<20}{'实时信号':>9}{'撤销':>7}{'撤销率':>8}{'迟现':>7}{'迟现率':>8}  判定")
    for n in PASSED:
        tot, ph, ms = stat[n]
        if tot == 0:
            print(f"  {n:<18}{'—':>9}"); continue
        pr, mr = 100*ph/tot, 100*ms/tot
        bad = pr > 2.0 or mr > 2.0
        print(f"  {n:<18}{tot:>9}{ph:>7}{pr:>7.2f}%{ms:>7}{mr:>7.2f}%  "
              f"{'淘汰 未来函数' if bad else '通过'}")
    print(f"\n淘汰线: 撤销率或迟现率 > 2%。缠论 chan-2buy 在同口径下撤销率约 44~50%。")

if __name__ == "__main__":
    asyncio.run(main())
