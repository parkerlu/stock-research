"""三分类标签 —— 用户定义的题目:

    看完今天的K, 明天开盘买入, 之后 10 个交易日内:
        先碰到 +10%  -> 涨   (1)
        先碰到 -5%   -> 跌   (2)
        都没碰到     -> 平   (0)

为什么要换掉回归 y_0.15_0.08:
    回归拟合的是"期望收益", 而 +15%/-8% 不对称 -> 高波动股更容易先撞 -8% ->
    "低波动" 成了最省力的答案。实测 h=512 模型 95% 的排序能力 = "-波动率"。
    三分类让模型直接回答"谁会涨到 +10%", 低波动股在这个问题下没有捷径
    (它们几乎全是"平")。

⚠️ 入场价 = 次日开盘, 不是当日收盘 —— 当日收盘你还没看到这根K。
⚠️ 次日一字涨停(开=高=低 且 >= 前收*1.095)买不到, 标 -9 剔除。
⚠️ 同一天既碰 +10% 又碰 -5%(大阴大阳都有可能): 按"跌"算, 保守。
⚠️ 未来不足 10 根(股票末尾)或跨股票: 剔除。

输出 label3.npz, 与 panel.npz 的样本顺序一一对应:
    y3    int8   0/1/2, -9 = 剔除
    ret3  f32    该出场规则下的真实收益(+0.10 / -0.05 / 第10日收盘/入场-1)
    vol20 f32    过去 20 根对数收益标准差 —— 用来事后检查"是不是又在学波动率"
"""
from __future__ import annotations

import argparse
import logging

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk.label3")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/root/data")
    ap.add_argument("--up", type=float, default=0.10)
    ap.add_argument("--dn", type=float, default=0.05)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--out", default="label3.npz")
    a = ap.parse_args()

    d = np.load(f"{a.dir}/panel.npz")
    ohlcv, cid, idx = d["ohlcv"], d["cid"], d["idx"].astype(np.int64)
    o, h, l, c = ohlcv[:, 0], ohlcv[:, 1], ohlcv[:, 2], ohlcv[:, 3]
    n_bar, n = len(o), len(idx)
    H = a.hold
    log.info("样本 %s, K线 %s", f"{n:,}", f"{n_bar:,}")

    # 未来 H 根必须存在且属于同一只票
    ok = (idx + H < n_bar)
    ok &= cid[np.minimum(idx + H, n_bar - 1)] == cid[idx]
    # 次日一字涨停买不到
    j = np.minimum(idx + 1, n_bar - 1)
    yizi = (o[j] == h[j]) & (o[j] == l[j]) & (o[j] >= c[idx] * 1.095)
    ok &= ~yizi
    entry = o[j]
    ok &= entry > 0
    log.info("剔除: 未来不足/跨票 %s, 次日一字板 %s", f"{(~(idx + H < n_bar) | (cid[np.minimum(idx + H, n_bar - 1)] != cid[idx])).sum():,}", f"{yizi.sum():,}")

    up_first = np.full(n, H + 1, np.int16)      # 第几天首次碰到 +up
    dn_first = np.full(n, H + 1, np.int16)
    up_lv, dn_lv = entry * (1 + a.up), entry * (1 - a.dn)
    for k in range(1, H + 1):                   # k=1 是入场当天(开盘买, 当天高低算数)
        jj = np.minimum(idx + k, n_bar - 1)
        hit_u = (h[jj] >= up_lv) & (up_first > H)
        hit_d = (l[jj] <= dn_lv) & (dn_first > H)
        up_first[hit_u] = k
        dn_first[hit_d] = k
    y3 = np.zeros(n, np.int8)
    y3[(up_first <= H) & (up_first < dn_first)] = 1
    y3[(dn_first <= H) & (dn_first <= up_first)] = 2    # 同日: 保守按跌
    last = c[np.minimum(idx + H, n_bar - 1)]
    ret3 = np.where(y3 == 1, a.up, np.where(y3 == 2, -a.dn, last / entry - 1)).astype(np.float32)
    y3[~ok] = -9
    ret3[~ok] = np.nan

    # 过去 20 根波动率(事后检查用)
    lr = np.zeros(n, np.float32)
    acc = np.zeros((n, 20), np.float32)
    for k in range(20):
        p1 = np.maximum(idx - k, 0); p0 = np.maximum(idx - k - 1, 0)
        acc[:, k] = np.log(np.maximum(c[p1], 1e-6) / np.maximum(c[p0], 1e-6))
    vol20 = acc.std(1)
    del acc

    v = y3[ok]
    log.info("有效 %s (%.1f%%) | 涨 %.1f%%  跌 %.1f%%  平 %.1f%%",
             f"{ok.sum():,}", ok.mean() * 100,
             (v == 1).mean() * 100, (v == 2).mean() * 100, (v == 0).mean() * 100)
    np.savez(f"{a.dir}/{a.out}", y3=y3, ret3=ret3, vol20=vol20,
             up=a.up, dn=a.dn, hold=H)
    log.info("存 %s/%s", a.dir, a.out)


if __name__ == "__main__":
    main()
