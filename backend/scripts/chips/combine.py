"""筹码 × 裸K 共振 —— 把两个【信息源完全独立】的模型合起来。

为什么值得试（本项目已经应验过一次）：
    主力吸筹单独用 H=20 胜率只有 43.5%(低于基准 49%), 但叠到买卖很准 v3 上
    把 50.9% 抬到 58.3%。文档的结论是"共振需要的是足够的信号量 + 独立的
    误报模式, 而不是单个指标的精度"。
    这里两个模型一个看 200~400 根K线的形态、一个看持仓成本结构, 独立性比
    当年那对更强。

⚠️ 组合方式故意只用【无参数】的两种, 不做加权扫描:
    昨晚刚量化过"参数越多 walk-forward 越差"(0.78 -> 0.38)。引入一个权重
    就等于多一个要在选参窗口上拟合的东西。
      - avg  : 两边 EV 的等权平均
      - rank : 两边【当日横截面分位】的等权平均(对量纲不敏感, 更稳)
      - min  : 取两边分位的较小值 —— 真正的"共振", 要求两个模型都看好

⚠️ 样本对齐: 筹码只覆盖 81%(2018 起), 裸K 覆盖 100%。交集才有意义,
    直接按 te_idx 求交, 不能假设两边行数一样。

用法: python combine.py --a _chips_v1 --b _cls512_nbar400 --how rank --tag _combo
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

DIR = "/app/data/research/rawk"
COST = 0.003


def load(dirname: str, tag: str, up: float, dn: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = np.load(f"{dirname}/oos_v2{tag}.npz")
    te = pd.DataFrame({"idx": o["te_idx"], "day": o["lab_day"],
                       "ev": up * o["p_up"] - dn * o["p_dn"] - COST})
    va = pd.DataFrame({"idx": o["va_idx"], "day": o["va_day"],
                       "ev": up * o["va_p_up"] - dn * o["va_p_dn"] - COST})
    return va, te


def merge(x: pd.DataFrame, y: pd.DataFrame, how: str) -> pd.DataFrame:
    m = x.merge(y, on=["idx", "day"], suffixes=("_a", "_b"))
    if how == "avg":
        m["ev"] = (m.ev_a + m.ev_b) / 2
    else:
        ra = m.groupby("day")["ev_a"].rank(pct=True)
        rb = m.groupby("day")["ev_b"].rank(pct=True)
        # 分位平均/取小之后再映回一个 EV 量纲: 用两边 EV 的均值做尺度,
        # 这样下游 decide/pool 的阈值语义不变
        w = np.minimum(ra, rb) if how == "min" else (ra + rb) / 2
        m["ev"] = w * (m.ev_a.std() + m.ev_b.std()) / 2 + (m.ev_a.mean() + m.ev_b.mean()) / 2
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=DIR)
    ap.add_argument("--a", default="_chips_v1")
    ap.add_argument("--b", default="_cls512_nbar400")
    ap.add_argument("--how", default="rank", choices=["avg", "rank", "min"])
    ap.add_argument("--tag", default="_combo")
    ap.add_argument("--label", default="label3_dn8.npz")
    a = ap.parse_args()

    l3 = np.load(f"{a.dir}/{a.label}")
    up, dn = float(l3["up"]), float(l3["dn"])
    y3, ret3 = l3["y3"], l3["ret3"]

    va_a, te_a = load(a.dir, a.a, up, dn)
    va_b, te_b = load(a.dir, a.b, up, dn)
    print(f"{a.a}: 验证 {len(va_a):,} 测试 {len(te_a):,}")
    print(f"{a.b}: 验证 {len(va_b):,} 测试 {len(te_b):,}")

    va = merge(va_a, va_b, a.how)
    te = merge(te_a, te_b, a.how)
    print(f"交集:  验证 {len(va):,} 测试 {len(te):,}  (组合方式 {a.how})")

    # 两个模型的分数相关性 —— 太高就说明它们看的是同一件事, 共振没意义
    rho = te[["ev_a", "ev_b"]].corr().iloc[0, 1]
    rk = te.groupby("day")[["ev_a", "ev_b"]].rank(pct=True)
    rho_rk = rk.corr().iloc[0, 1]
    print(f"两模型分数相关: 原值 {rho:+.3f}  当日分位 {rho_rk:+.3f}"
          f"  {'← 独立性好' if abs(rho_rk) < 0.3 else '← 偏高, 共振价值有限'}")

    for nm, df in (("单用 " + a.a, te.assign(ev=te.ev_a)),
                   ("单用 " + a.b, te.assign(ev=te.ev_b)),
                   ("共振 " + a.how, te)):
        r = df.assign(y=y3[df.idx.to_numpy()], r=ret3[df.idx.to_numpy()])
        top = r[r.groupby("day")["ev"].rank(pct=True) >= 0.95]
        print(f"  {nm:<24} Top5% 期望{top.r.mean() * 100:+.2f}%  "
              f"识别{(top.y == 1).mean() * 100:5.1f}%  跌{(top.y == 2).mean() * 100:5.1f}%")

    out = f"{a.dir}/oos_v2{a.tag}.npz"
    # 存回 oos 格式: p_up/p_dn 反解不出来, 直接把 ev 拆成"等价的" p_up
    # (pool/decide 只用 up*p_up − dn*p_dn − COST, 这里让 p_dn=0, p_up=(ev+COST)/up)
    np.savez(out,
             va_day=va["day"].to_numpy(), va_idx=va["idx"].to_numpy(),
             va_p_up=((va.ev + COST) / up).to_numpy(np.float32),
             va_p_dn=np.zeros(len(va), np.float32),
             lab_day=te["day"].to_numpy(), te_idx=te["idx"].to_numpy(),
             p_up=((te.ev + COST) / up).to_numpy(np.float32),
             p_dn=np.zeros(len(te), np.float32))
    print(f"存 {out}")


if __name__ == "__main__":
    main()
