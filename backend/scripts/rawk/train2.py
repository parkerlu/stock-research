"""裸K模型 v2 —— 只改方法, 不改输入。

输入仍是【裸K + 量 + 一条MA + 真实大盘K线】, 与 v1 完全相同。
改的是三个方法缺陷(2026-09-08 从 v1 的日志里诊断出来的):

⚠️ 缺陷一: 用测试集挑轮次。
   v1 逐轮在 +0.12 ~ +0.47 之间震荡, 而我每次报的是最高那轮 ——
   这是【用测试集做模型选择】, 是泄漏。真实水平接近震荡均值(约+0.30)。
   也因此 "h=128 比 h=48 差" 这个对比从一开始就不公平: h=128 只跑了6轮,
   还没机会挑出它的幸运轮。
   → 改成三段: 训练 ≤2019 / 验证 2020 / 测试 ≥2021。
     用【验证集】挑轮次, 测试集只在最后碰一次。

⚠️ 缺陷二: 有效样本量只有名义的 1/20。
   标签是未来20日, 相邻两天的标签重叠95% —— 400万名义样本里只有约20万
   是独立的。这解释了"样本从80万加到400万没用"和"容量一放大就过拟合"。
   → 净化采样: 同一只票的训练样本间隔 >= H 天, 标签互不重叠。

⚠️ 缺陷三: 训练目标与使用方式不一致。
   用 Huber 拟合数值, 但实际只用【当日排名】。
   → 主指标改用 RankIC(与用法对齐, 也与文献可比), 并保留分层价差。

另外加了多种子集成 —— 直接消灭那个震荡, 比挑幸运轮诚实。
"""
from __future__ import annotations

import argparse
import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from rawk.train import DEV, NBAR, Net, make_batch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk2")
DIR = "/root/data"


def purged_sample(idx_pos: np.ndarray, cid: np.ndarray, day: np.ndarray,
                  h: int, rng) -> np.ndarray:
    """净化采样: 同一只票内, 相邻被选样本的间隔 >= h 天。

    ⚠️ 不做这一步的话, 名义样本量是有效样本量的 h 倍, 容量/数据量的对比
       全部失真 —— 加的"新数据"其实是同一段行情的重复标注。
    """
    order = np.lexsort((day[idx_pos], cid[idx_pos]))
    s = idx_pos[order]
    c, d = cid[s], day[s]
    keep = np.zeros(len(s), bool)
    last_c, last_d = -1, -10**9
    # 每只票随机起一个相位, 避免总是取每月同一天
    phase = rng.integers(0, h)
    for i in range(len(s)):
        if c[i] != last_c:
            last_c, last_d = c[i], d[i] - h + phase
        if d[i] - last_d >= h:
            keep[i] = True
            last_d = d[i]
    return s[keep]


def rank_ic(day: np.ndarray, score: np.ndarray, y: np.ndarray) -> float:
    """逐日 RankIC 的均值 —— 与"每天在全市场里排名"这个用法对齐。"""
    d = pd.DataFrame({"day": day, "s": score, "y": y})
    g = d.groupby("day")
    ic = g.apply(lambda x: x["s"].corr(x["y"], method="spearman")
                 if len(x) > 20 else np.nan, include_groups=False)
    return float(np.nanmean(ic.to_numpy()))


@torch.no_grad()
def predict(net, ohlcv, idx, day_all, mkt, day_min, bs=4096):
    net.eval()
    out = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), bs):
        x = make_batch(ohlcv, idx[i:i + bs], day=day_all, mkt=mkt, day_min=day_min)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
            o = net(torch.as_tensor(x, device=DEV))
        out[i:i + bs] = o.float().cpu().numpy()
    net.train()
    return out


def report(tag, day, score, y):
    d = pd.DataFrame({"day": day, "y": y, "s": score})
    d["rk"] = d.groupby("day")["s"].rank(pct=True)
    t1 = d.loc[d.rk >= 0.99, "y"].mean() * 100
    t5 = d.loc[d.rk >= 0.95, "y"].mean() * 100
    b20 = d.loc[d.rk < 0.20, "y"].mean() * 100
    ic = rank_ic(day, score, y)
    yr = pd.to_datetime(d["day"], unit="D").dt.year
    ys = d[d.rk >= 0.95].groupby(yr[d.rk >= 0.95])["y"].mean().mul(100)
    log.info("%s  RankIC %+.4f | Top1%% %+.2f Top5%% %+.2f Bot20%% %+.2f | 负年 %d",
             tag, ic, t1, t5, b20, int((ys < 0).sum()))
    log.info("      逐年 %s", {int(k): round(float(v), 2) for k, v in ys.items()})
    return ic


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--h", type=int, default=48)
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--seeds", type=int, default=5, help="集成几个种子")
    ap.add_argument("--hold", type=int, default=20, help="标签持有期, 用于净化间隔")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()

    d = np.load(f"{DIR}/panel.npz")
    ohlcv, idx, y, lday = d["ohlcv"], d["idx"], d["y"], d["lab_day"]
    day_all, mkt, day_min = d["day"], d["mkt"], int(d["day_min"])
    n_ch = 6 + mkt.shape[0] * 4
    yr = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()

    # ⚠️ 三段划分: 验证集只用来挑轮次, 测试集只在最后碰一次
    tr_all = np.flatnonzero(yr <= 2019)
    va_all = np.flatnonzero(yr == 2020)
    te_all = np.flatnonzero(yr >= 2021)
    rng = np.random.default_rng(0)
    tr = purged_sample(tr_all, d["lab_cid"], lday, a.hold, rng)
    va = purged_sample(va_all, d["lab_cid"], lday, a.hold, rng)
    log.info("训练 %s(净化前 %s) / 验证 %s / 测试 %s | 通道 %d",
             f"{len(tr):,}", f"{len(tr_all):,}", f"{len(va):,}", f"{len(te_all):,}", n_ch)
    log.info("⚠️ 净化后训练样本降到 %.1f%% —— 这才是【有效】样本量",
             len(tr) / max(len(tr_all), 1) * 100)

    te_scores = []
    for si in range(a.seeds):
        torch.manual_seed(si); np.random.seed(si)
        net = Net(ch=n_ch, h=a.h, n_rep=a.rep).to(DEV)
        opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(DEV == "cuda"))
        lossf = nn.HuberLoss(delta=0.05)
        best_ic, best_state, bad = -9e9, None, 0
        r2 = np.random.default_rng(si)
        for ep in range(1, a.epochs + 1):
            t0 = time.time()
            perm = r2.permutation(len(tr))
            for i in range(0, len(tr), a.bs):
                b = tr[perm[i:i + a.bs]]
                x = torch.as_tensor(make_batch(ohlcv, idx[b], day=day_all,
                                               mkt=mkt, day_min=day_min), device=DEV)
                t = torch.as_tensor(y[b], device=DEV)
                with torch.autocast("cuda", dtype=torch.float16, enabled=(DEV == "cuda")):
                    loss = lossf(net(x), t)
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            sv = predict(net, ohlcv, idx[va], day_all, mkt, day_min)
            ic = rank_ic(lday[va], sv, y[va])
            log.info("  种子%d ep%-2d 验证RankIC %+.4f (%.0fs)", si, ep, ic, time.time() - t0)
            # ⚠️ 只看验证集挑最好的一轮, 测试集这时候一次都不碰
            if ic > best_ic:
                best_ic, bad = ic, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= 3:
                    log.info("  种子%d 连续3轮没提升, 提前停", si); break
        net.load_state_dict(best_state)
        st = predict(net, ohlcv, idx[te_all], day_all, mkt, day_min)
        te_scores.append(st)
        report(f"种子{si} 单模型(验证最优 RankIC {best_ic:+.4f})", lday[te_all], st, y[te_all])
        torch.save(net.state_dict(), f"{DIR}/cnn2{a.tag}_s{si}.pt")

    # ⚠️ 集成: 把各种子的【当日排名】平均, 而不是原始分数平均 ——
    #    不同种子的分数尺度不同, 直接平均会被尺度大的那个主导。
    log.info("=" * 70)
    ranks = []
    for s in te_scores:
        ranks.append(pd.DataFrame({"d": lday[te_all], "s": s})
                     .groupby("d")["s"].rank(pct=True).to_numpy())
    ens = np.mean(ranks, axis=0).astype(np.float32)
    report(f"★ {a.seeds}种子集成", lday[te_all], ens, y[te_all])
    np.savez(f"{DIR}/oos_v2{a.tag}.npz", score=ens, lab_day=lday[te_all],
             lab_cid=d["lab_cid"][te_all])
    log.info("样本外分数已存")


if __name__ == "__main__":
    main()
