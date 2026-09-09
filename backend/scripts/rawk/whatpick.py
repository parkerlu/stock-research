"""模型在看什么: 把 Top5%(按EV) 选出的票, 在若干【人能读懂】的维度上与全市场对比。

目的不是调参, 是判断它学到的是可解释的形态还是噪声 —— 如果各维度都与全市场
无差别, 那"选股能力"就只是统计噪声; 如果集中在某类形态上, 至少可以人工核对,
也能和项目里已有的指标(主力吸筹/动力线/突破)对照看是不是同一族东西。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

D = "/root/data"
TAG = "_cls512_nbar400"
N = 200_000


def main() -> None:
    d = np.load(f"{D}/panel.npz")
    ohlcv = d["ohlcv"]
    c, h, l, v = ohlcv[:, 3], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 4]

    o_ = np.load(f"{D}/oos_v2{TAG}.npz")
    l3 = np.load(f"{D}/label3_dn8.npz")
    up, dn = float(l3["up"]), float(l3["dn"])
    ev = up * o_["p_up"] - dn * o_["p_dn"] - 0.003

    df = pd.DataFrame({"ev": ev, "day": o_["lab_day"], "pos": o_["te_idx"]})
    df["rk"] = df.groupby("day")["ev"].rank(pct=True)
    rng = np.random.default_rng(0)
    sel_all = df.loc[df.rk >= 0.95, "pos"].to_numpy()
    sel = rng.choice(sel_all, size=min(N, len(sel_all)), replace=False)
    base = rng.choice(df["pos"].to_numpy(), size=min(N, len(df)), replace=False)

    def roll_max(pos: np.ndarray, arr: np.ndarray, k: int) -> np.ndarray:
        # 向量化的滚动极值: 用累积法避免 20 万次 python 循环
        out = np.empty(len(pos), np.float64)
        for i, p in enumerate(pos):
            s = max(p - k, 0)
            out[i] = arr[s:p + 1].max()
        return out

    def roll_min(pos: np.ndarray, arr: np.ndarray, k: int) -> np.ndarray:
        out = np.empty(len(pos), np.float64)
        for i, p in enumerate(pos):
            s = max(p - k, 0)
            out[i] = arr[s:p + 1].min()
        return out

    def roll_mean(pos: np.ndarray, arr: np.ndarray, k: int) -> np.ndarray:
        out = np.empty(len(pos), np.float64)
        for i, p in enumerate(pos):
            s = max(p - k, 0)
            out[i] = arr[s:p + 1].mean()
        return out

    def feat(pos: np.ndarray) -> dict[str, np.ndarray]:
        def ret(k: int) -> np.ndarray:
            p0 = np.maximum(pos - k, 0)
            return (c[pos] / np.maximum(c[p0], 1e-6) - 1) * 100

        hi = roll_max(pos, h, 60)
        lo = roll_min(pos, l, 60)
        v5 = roll_mean(pos, v, 5)
        v60 = roll_mean(pos, v, 60)
        return {
            "近5日涨幅%": ret(5),
            "近20日涨幅%": ret(20),
            "近60日涨幅%": ret(60),
            "60日区间位置%": (c[pos] - lo) / np.maximum(hi - lo, 1e-6) * 100,
            "量比(5日/60日)": v5 / np.maximum(v60, 1e-6),
            "当日振幅%": (h[pos] - l[pos]) / np.maximum(c[pos], 1e-6) * 100,
        }

    fs, fb = feat(sel), feat(base)
    print(f"{TAG}  Top5%(按EV) n={len(sel):,}   全市场对照 n={len(base):,}\n")
    print(f"{'维度':<20}{'选出的(中位)':>14}{'全市场(中位)':>14}{'差':>10}")
    for k in fs:
        a, b = float(np.median(fs[k])), float(np.median(fb[k]))
        print(f"{k:<20}{a:>14.2f}{b:>14.2f}{a - b:>+10.2f}")


if __name__ == "__main__":
    main()
