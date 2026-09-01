"""S3 样本外验证 —— 2022-01-01 ~ 2026-08-31, 只跑一次。

纪律 (写在这里是为了让违规变得显眼):
  · 候选名单在跑之前就定死, 由 S1/S2 决出, 本文件不再挑选。
  · 参数一律沿用样本内那一套, 不做任何调整。
  · 结果不好就是不好, 不许回头改参数再跑第二遍 —— 那等于把样本外变成样本内,
    正是 chan-2buy 229 倍的成因之一。
  · 随机对照与候选同口径、同期间、同股票池, 一起跑。
"""
from __future__ import annotations
import asyncio, json, math, sys, time, numpy as np, pandas as pd
from datetime import date
from sqlalchemy import select
from app.db import async_session
from app.models.schema import DailyCandle
import screen as S

# ⚠️ 切分必须避开模型训练期。
# mm-* 和 rev-* 会加载训练好的 XGBoost (maimai_filter_multi.json /
# reversal_filter_multi.json), 训练截止日 2024-09-01 (见 scripts/
# train_maimai_filter.py:229 与 train_reversal_filter.py:262)。
# ml_direct_* 用 GroupKFold 按**股票**分组交叉验证, 训练集与验证集覆盖同一
# 时段, 模型文件是 2026-04 生成的, 等于见过全部历史。
# 所以真正干净的样本外只有 2024-09-01 之后, 而 ml_direct_* 连这个都不干净。
OOS_FROM, OOS_TO = date(2024, 9, 1), date(2026, 8, 31)


async def load_bars_oos(code):
    async with async_session() as db:
        rows = (await db.execute(
            select(DailyCandle.trade_date, DailyCandle.open, DailyCandle.high,
                   DailyCandle.low, DailyCandle.close, DailyCandle.vol,
                   DailyCandle.amount, DailyCandle.adj_factor)
            .where(DailyCandle.ts_code == code,
                   DailyCandle.trade_date >= date(2023, 6, 1),
                   DailyCandle.trade_date <= OOS_TO)
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
    cands = json.loads(sys.argv[1]) if len(sys.argv) > 1 else []
    if not cands:
        print("用法: python oos.py '[\"tdx-dual-kdj\", ...]'"); return
    from app.services.strategy_templates import TEMPLATE_REGISTRY as T
    all_codes = await S.load_codes()
    rng = np.random.default_rng(4242)          # 与样本内不同的抽样种子
    codes = sorted(rng.choice(all_codes, min(900, len(all_codes)), replace=False))
    print(f"样本外 {OOS_FROM} ~ {OOS_TO} · 候选 {len(cands)} 个 · 股票 {len(codes)} 只", flush=True)

    acc = {n: [] for n in cands}
    rnd = {s: [] for s in range(3)}
    t0 = time.time()
    for k, code in enumerate(codes):
        df = await load_bars_oos(code)
        if df is None: continue
        o,h,l,c = (df[x].values for x in ("open","high","low","close"))
        dts = df.trade_date.values
        di = [pd.Timestamp(x).date() for x in dts]
        for n in cands:
            try: sigs = T[n]().generate_signals(df)
            except Exception: continue
            for sg in sigs:
                if sg.get("action") != "buy": continue
                d = sg["date"]; d = d.date() if hasattr(d, "date") else d
                if not (OOS_FROM <= d <= OOS_TO): continue
                pos = np.searchsorted(dts, np.datetime64(d))
                r = S.ladder_exit(o,h,l,c, pos+1, di)
                if r: acc[n].append(r)
        for seed in rnd:
            rg = np.random.default_rng(seed*100003 + k + 777)
            for _ in range(6):
                if len(c) <= 320: continue
                pos = int(rg.integers(260, len(c)-70))
                r = S.ladder_exit(o,h,l,c, pos, di)
                if r: rnd[seed].append(r)
        if (k+1) % 100 == 0:
            print(f"  {k+1}/{len(codes)} · {(time.time()-t0)/60:.1f} 分", flush=True)

    def stat(rows):
        if not rows: return None
        a = np.array([x[0] for x in rows])*100; d = np.array([x[1] for x in rows])
        return len(a), a.mean(), 100*(a>0).mean(), d.mean(), a.std(ddof=1)

    base = [stat(rnd[s])[1] for s in rnd if stat(rnd[s])]
    bm = float(np.mean(base)); spread = max(base)-min(base)
    print(f"\n随机对照 样本外基准 {bm:+.3f}%  三种子极差 {spread:.3f}pp")
    print(f"\n{'候选':<20}{'笔数':>8}{'均值%':>8}{'超额pp':>9}{'t值':>7}{'胜率%':>7}{'持有':>6}  判定")
    for n in cands:
        st = stat(acc[n])
        if not st: print(f"  {n:<18}{'无信号':>8}"); continue
        ex = st[1]-bm; se = st[4]/math.sqrt(st[0]); t = ex/se if se>0 else 0
        keep = ex > spread and t > 3.0
        print(f"  {n:<18}{st[0]:>8}{st[1]:>8.3f}{ex:>+9.3f}{t:>7.2f}{st[2]:>7.1f}{st[3]:>6.1f}  "
              f"{'存活' if keep else '淘汰'}")

if __name__ == "__main__":
    asyncio.run(main())
