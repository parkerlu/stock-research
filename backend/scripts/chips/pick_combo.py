"""在【验证集】上选组合方式, 测试集只碰一次 —— 给 1.07 一个诚实的复核。

⚠️ 为什么必须做这一步:
   2026-09-10 我试了 avg / rank / min 三种组合, 在【测试集】上比出 avg 最好
   (1.07 vs 0.51 vs 0.49), 然后报了 1.07。这在性质上等同于"用测试集挑轮次"
   —— 正是本项目最贵的那条教训(裸K Top5% 报 +0.47, 换诚实流程后 +0.02)。
   缓解证据当时给了两条(avg 是零参数的自然选择、rank/min 变差可解释),
   但那是事后辩护, 不是验证。

   这里改成: 在 2020 验证集上按同一判据选出组合方式, 然后【只在测试集上
   跑这一种】。如果选出来的还是 avg, 1.07 就站得住; 如果选出别的, 说明
   当初那个 1.07 确实是挑出来的。

⚠️ 判据用验证集上的资金池比值, 与最终评估口径一致 —— 不能验证集用 Top5%、
   测试集用组合比值, 那是换了尺子。
"""
from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd

DIR = "/app/data/research/rawk"
COST = 0.003
HOWS = ("avg", "rank", "min")


def load(tag: str, up: float, dn: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    o = np.load(f"{DIR}/oos_v2{tag}.npz")
    va = pd.DataFrame({"idx": o["va_idx"], "day": o["va_day"],
                       "ev": up * o["va_p_up"] - dn * o["va_p_dn"] - COST})
    te = pd.DataFrame({"idx": o["te_idx"], "day": o["lab_day"],
                       "ev": up * o["p_up"] - dn * o["p_dn"] - COST})
    return va, te


def combine(x: pd.DataFrame, y: pd.DataFrame, how: str) -> pd.DataFrame:
    m = x.merge(y, on=["idx", "day"], suffixes=("_a", "_b"))
    if how == "avg":
        m["ev"] = (m.ev_a + m.ev_b) / 2
    else:
        ra = m.groupby("day")["ev_a"].rank(pct=True)
        rb = m.groupby("day")["ev_b"].rank(pct=True)
        w = np.minimum(ra, rb) if how == "min" else (ra + rb) / 2
        m["ev"] = w * (m.ev_a.std() + m.ev_b.std()) / 2 + (m.ev_a.mean() + m.ev_b.mean()) / 2
    return m


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="_chips_v1")
    ap.add_argument("--b", default="_cls512_nbar400")
    ap.add_argument("--label", default="label3_dn8.npz")
    ap.add_argument("--slots", type=int, default=20)
    a = ap.parse_args()

    l3 = np.load(f"{DIR}/{a.label}")
    up, dn = float(l3["up"]), float(l3["dn"])
    y3, ret3 = l3["y3"], l3["ret3"]
    days = np.load(f"{DIR}/holddays_dn8.npz")["days"]
    sys.path.insert(0, "/app/scripts/rawk")
    from pool import run                                    # noqa: E402

    va_a, te_a = load(a.a, up, dn)
    va_b, te_b = load(a.b, up, dn)

    print("== 只看 2020 验证集(测试集一次都不碰) ==")
    scores = {}
    for how in HOWS:
        v = combine(va_a, va_b, how)
        v = v.assign(y=y3[v.idx.to_numpy()], r=ret3[v.idx.to_numpy()],
                     idxpos=v.idx.to_numpy()).sort_values("day")
        best = None
        for q in (0.90, 0.95, 0.98, 0.99):
            thr = float(v["ev"].quantile(q))
            r = run(v, days, a.slots, thr)
            if r and r["n"] >= 60 and (best is None or r["ratio"] > best[0]):
                best = (r["ratio"], q, r)
        if best:
            scores[how] = best
            _, q, r = best
            print(f"  {how:<5} 最好档 前{(1 - q) * 100:4.1f}%  {r['n']:4d}笔  "
                  f"年化{r['ann']:+6.1f}% 回撤{r['mdd']:6.1f}%  比值 {r['ratio']:5.2f}")
    if not scores:
        print("验证集上没有任何一种组合凑够 60 笔")
        return

    pick = max(scores, key=lambda k: scores[k][0])
    print(f"\n验证集选出: 【{pick}】")

    print("\n== 搬到 2021+ 测试集(只跑选中的这一种, walk-forward 滚动选阈值) ==")
    t = combine(te_a, te_b, pick)
    out = f"{DIR}/oos_v2_combo_picked.npz"
    np.savez(out,
             va_day=combine(va_a, va_b, pick)["day"].to_numpy(),
             va_idx=combine(va_a, va_b, pick)["idx"].to_numpy(),
             va_p_up=((combine(va_a, va_b, pick).ev + COST) / up).to_numpy(np.float32),
             va_p_dn=np.zeros(len(combine(va_a, va_b, pick)), np.float32),
             lab_day=t["day"].to_numpy(), te_idx=t["idx"].to_numpy(),
             p_up=((t.ev + COST) / up).to_numpy(np.float32),
             p_dn=np.zeros(len(t), np.float32))
    subprocess.run([sys.executable, "/app/scripts/rawk/pool.py", "_combo_picked",
                    "--dir", DIR, "--walk", "--no-cap"], check=False)


if __name__ == "__main__":
    main()
