"""在 Kronos 表示上训一个小头, 用我们自己的标签 —— 并与裸K模型同口径对比。

⚠️ 判据必须和裸K模型完全一致, 否则比较没意义:
      样本外 Top1%/Top5%/Bot20% 超额、逐年负年数、去记忆审计相关系数
   最终还要落到组合回测(带涨跌停成交约束), 那一步在本地跑。

⚠️ 标签是三重障碍超额(+15%/-8%/20交易日, 且次日一字涨停样本已剔除)。
   用超额而非绝对收益 -> 牛市普涨不给分, 这是去 regime 的最后一道防线。
"""
from __future__ import annotations

import argparse
import glob
import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kronos.head")


class Head(nn.Module):
    """故意做小 —— 表示已经由 Kronos 提好, 头再大只会过拟合。"""

    def __init__(self, n_in: int, h: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(n_in), nn.Linear(n_in, h), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(h, h // 2), nn.GELU(), nn.Linear(h // 2, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def report(tag, sc, y, day):
    d = pd.DataFrame({"day": day, "y": y, "s": sc})
    d["rk"] = d.groupby("day")["s"].rank(pct=True)
    parts = []
    for lo, hi, nm in ((0.99, 1.01, "Top1%"), (0.95, 1.01, "Top5%"),
                       (0.0, 0.20, "Bot20%")):
        g = d[(d.rk >= lo) & (d.rk < hi)]
        parts.append(f"{nm} {g['y'].mean() * 100:+.2f}")
    dl = d.groupby("day").agg(a=("s", "mean"), m=("y", "mean"))
    corr = dl["a"].corr(dl["m"])
    yr = pd.to_datetime(d["day"], unit="D").dt.year
    ys = d[d.rk >= 0.95].groupby(yr[d.rk >= 0.95])["y"].mean().mul(100)
    log.info("%s  %s | 审计 %+.3f | 负年 %d", tag, "  ".join(parts), corr,
             int((ys < 0).sum()))
    log.info("      逐年 %s", {int(k): round(float(v), 2) for k, v in ys.items()})
    return d


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb", default="kronos_emb")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--bs", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # ⚠️ 不能 np.concatenate 全部分片: 7.4G 特征 + 拼大盘 = 11G, 再标准化翻倍,
    #    29G 的机器直接 OOM(实测内核 killed, rss 30.6G)。
    #    改成: 先建一个 memmap 大矩阵, 逐片写进去, 全程不出现第二份拷贝。
    import os as _os
    m = np.load(f"{a.emb}/meta.npz")
    y, lday, lcid, is_tr = m["y"], m["lab_day"], m["lab_cid"], m["is_train"]
    day_min = int(m["day_min"])
    MK = np.load(f"{a.emb}/mkt_emb.npy")          # (n_idx, span, D)
    files = sorted(glob.glob(f"{a.emb}/emb_*.npy"))
    d0 = np.load(files[0], mmap_mode="r")
    D_stock, D_mkt = d0.shape[1], MK.shape[2]
    n_tot = len(y)
    D = D_stock + MK.shape[0] * D_mkt
    log.info("特征维 %d (个股 %d + 大盘 %d×%d), 样本 %s",
             D, D_stock, MK.shape[0], D_mkt, f"{n_tot:,}")

    path = f"{a.emb}/X.mm"
    X = np.memmap(path, dtype=np.float32, mode="w+", shape=(n_tot, D))
    di = np.clip(lday - day_min, 0, MK.shape[1] - 1)
    off = 0
    for f in files:
        part = np.load(f, mmap_mode="r")
        n = part.shape[0]
        X[off:off + n, :D_stock] = part
        sl = di[off:off + n]
        for k in range(MK.shape[0]):
            c0 = D_stock + k * D_mkt
            X[off:off + n, c0:c0 + D_mkt] = MK[k][sl]
        off += n
        del part
    assert off == n_tot, f"分片行数 {off} 与标签 {n_tot} 不符"
    del MK
    X.flush()

    # ⚠️ 三段划分, 与裸K模型同一套流程 —— 否则这轮 Kronos 的数字又是
    #    "用测试集挑轮次"挑出来的, 和之前一样虚高, 对比毫无意义。
    #    2026-09-08 发现: 同一网络同一数据, 挑轮次 vs 验证集挑,
    #    Top5% 从 +0.47 掉到 +0.02, 差一个数量级。
    yr_all = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()
    tr = np.flatnonzero(is_tr & (yr_all <= 2019))
    va = np.flatnonzero(is_tr & (yr_all == 2020))
    te = np.flatnonzero(~is_tr)
    if len(va) == 0:            # 训练段没有2020, 从训练集尾部切一段当验证
        tr_sorted = tr[np.argsort(lday[tr])]
        cut = int(len(tr_sorted) * 0.85)
        tr, va = tr_sorted[:cut], tr_sorted[cut:]
    log.info("训练 %s / 验证 %s / 测试 %s", f"{len(tr):,}", f"{len(va):,}", f"{len(te):,}")
    # ⚠️ 标准化只能用训练集的统计量; 抽样估计即可, 不必全量(省内存)
    smp = tr[np.random.default_rng(0).choice(len(tr), min(200_000, len(tr)),
                                             replace=False)]
    mu = np.asarray(X[smp]).mean(0)
    sd = np.asarray(X[smp]).std(0) + 1e-6
    log.info("训练 %s / 测试 %s", f"{len(tr):,}", f"{len(te):,}")

    net = Head(D).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.05)
    rng = np.random.default_rng(42)

    def _rank_ic(dd, sc, yy):
        x = pd.DataFrame({"d": dd, "s": sc, "y": yy})
        g = x.groupby("d").apply(
            lambda z: z["s"].corr(z["y"], method="spearman") if len(z) > 20 else np.nan,
            include_groups=False)
        return float(np.nanmean(g.to_numpy()))

    def _score(rows):
        net.eval()
        with torch.no_grad():
            o = np.concatenate([
                net(torch.as_tensor(((np.asarray(X[rows[i:i + 65536]]) - mu) / sd
                                     ).astype(np.float32), device=dev)).cpu().numpy()
                for i in range(0, len(rows), 65536)])
        net.train()
        return o

    best_ic, best_state, bad = -9e9, None, 0
    for ep in range(1, a.epochs + 1):
        rng.shuffle(tr)
        net.train()
        for i in range(0, len(tr), a.bs):
            b = tr[i:i + a.bs]
            xb = (np.asarray(X[b]) - mu) / sd
            loss = lossf(net(torch.as_tensor(xb, device=dev)),
                         torch.as_tensor(y[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        ic = _rank_ic(lday[va], _score(va), y[va])
        log.info("ep%-2d 验证RankIC %+.4f", ep, ic)
        # ⚠️ 只看验证集; 测试集在训练全程一次都不碰
        if ic > best_ic:
            best_ic, bad = ic, 0
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 3:
                log.info("连续3轮没提升, 提前停"); break
    net.load_state_dict(best_state)
    sc = _score(te)
    log.info("验证最优 RankIC %+.4f", best_ic)
    log.info("测试 RankIC %+.4f", _rank_ic(lday[te], sc, y[te]))
    report("★最终(验证集选出)", sc, y[te], lday[te])

    np.savez("kronos_scores.npz", score=sc, lab_day=lday[te], lab_cid=lcid[te])
    log.info("分数已存 kronos_scores.npz —— 拿回本地跑组合回测")


if __name__ == "__main__":
    main()
