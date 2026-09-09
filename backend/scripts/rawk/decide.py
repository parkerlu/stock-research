"""买入决策 —— "买入就要赚钱"的评估方式。

不按"每天买最高的 5%"(那是研究用的排序指标), 而是:
    每笔期望(净) = up·P(涨) − dn·P(跌) − 手续费
    只在 期望(净) >= 阈值 时买, 没有合格的就不买。
阈值在 2020 验证集上定(几个候选), 然后原封不动搬到 2021+ 测试集看:
    每天几笔 / 识别率 / 跌率 / 每笔净期望 / 逐年 / 简化资金曲线。

⚠️ 阈值绝不能在测试集上挑 —— 这就是之前 Top5% +0.47 变 +0.02 的那个坑。
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DIR = "/root/data"
COST = 0.003          # 佣金+印花税+滑点, 单边买卖合计的保守估计


def stats(df: pd.DataFrame, m: np.ndarray, up: float, dn: float) -> str:
    g = df[m]
    if len(g) == 0:
        return "无交易"
    nd = df["day"].nunique()
    return (f"{len(g) / nd:5.1f}笔/日  识别{(g.y == 1).mean() * 100:5.1f}%  "
            f"跌{(g.y == 2).mean() * 100:5.1f}%  净期望{(g.r.mean() - COST) * 100:+.2f}%")


def equity(df: pd.DataFrame, m: np.ndarray, hold: int, maxpos: int) -> tuple[float, float]:
    """简化资金曲线: 每天最多买 maxpos 笔(按期望排), 持有 hold 天, 资金分 hold 份轮动。
    返回 (年化%, 最大回撤%)。"""
    g = df[m].sort_values(["day", "ev"], ascending=[True, False]).groupby("day").head(maxpos)
    daily = g.groupby("day")["r"].mean().sub(COST)            # 该日建仓那一份的收益
    days = np.sort(df["day"].unique())
    dr = pd.Series(0.0, index=days)
    dr.loc[daily.index] = daily.to_numpy() / hold             # 摊到 hold 天里 (近似)
    # 每天 1/hold 的资金到期换仓; 没有合格票的那份资金空仓 —— 只买合格的
    eq = (1 + dr).cumprod()
    yrs = len(days) / 243
    ann = (eq.iloc[-1] ** (1 / yrs) - 1) * 100
    mdd = ((eq / eq.cummax()) - 1).min() * 100
    return float(ann), float(mdd)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--label", default="label3_dn8.npz")
    ap.add_argument("--maxpos", type=int, default=20)
    a = ap.parse_args()
    o = np.load(f"{DIR}/oos_v2{a.tag}.npz")
    l3 = np.load(f"{DIR}/{a.label}")
    up, dn, hold = float(l3["up"]), float(l3["dn"]), int(l3["hold"])
    y3, ret3 = l3["y3"], l3["ret3"]

    va = pd.DataFrame({"day": o["va_day"], "pu": o["va_p_up"], "pd": o["va_p_dn"],
                       "y": y3[o["va_idx"]], "r": ret3[o["va_idx"]]})
    te = pd.DataFrame({"day": o["lab_day"], "pu": o["p_up"], "pd": o["p_dn"],
                       "y": y3[o["te_idx"]], "r": ret3[o["te_idx"]]})
    for d in (va, te):
        d["ev"] = up * d.pu - dn * d.pd - COST
    print(f"标签 +{up:.0%}/-{dn:.0%}/{hold}日  手续费 {COST:.2%}  "
          f"测试基准: 识别{(te.y == 1).mean() * 100:.1f}% 跌{(te.y == 2).mean() * 100:.1f}% "
          f"净期望{(te.r.mean() - COST) * 100:+.2f}%")

    print("\n== 规则A: 按【净期望】买 (up·P涨 − dn·P跌 − 费) ==")
    print("阈值(2020定)     | 2020验证                                    | 2021+测试                                   | 年化/回撤")
    for q in (0.90, 0.95, 0.98, 0.99, 0.995):
        th = float(va["ev"].quantile(q))
        ann, mdd = equity(te, (te.ev >= th).to_numpy(), hold, a.maxpos)
        print(f"ev>={th * 100:+.2f}% (验证前{(1 - q) * 100:.1f}%) | {stats(va, (va.ev >= th).to_numpy(), up, dn)} | "
              f"{stats(te, (te.ev >= th).to_numpy(), up, dn)} | {ann:+.1f}% / {mdd:.1f}%  比值 {ann / abs(mdd) if mdd else 0:.2f}")

    print("\n== 规则B: 只按【P(涨)】买 (用户定义的识别率) ==")
    for q in (0.90, 0.95, 0.98, 0.99, 0.995):
        th = float(va["pu"].quantile(q))
        m_va, m_te = (va.pu >= th).to_numpy(), (te.pu >= th).to_numpy()
        ann, mdd = equity(te, m_te, hold, a.maxpos)
        print(f"P涨>={th * 100:5.1f}% (验证前{(1 - q) * 100:.1f}%) | {stats(va, m_va, up, dn)} | "
              f"{stats(te, m_te, up, dn)} | {ann:+.1f}% / {mdd:.1f}%")

    th = float(va["ev"].quantile(0.98))
    g = te[te.ev >= th]
    yr = pd.to_datetime(g["day"], unit="D").dt.year
    print(f"\n规则A 验证前2% 阈值 逐年: 净期望% "
          f"{ {int(k): round(float(v), 2) for k, v in g.groupby(yr)['r'].mean().sub(COST).mul(100).items()} }")
    print(f"   逐年识别率% { {int(k): round(float(v), 1) for k, v in g.groupby(yr)['y'].apply(lambda x: (x == 1).mean() * 100).items()} }")


if __name__ == "__main__":
    main()
