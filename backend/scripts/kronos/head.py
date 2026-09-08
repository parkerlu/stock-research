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
    E = np.concatenate([np.load(p) for p in sorted(glob.glob(f"{a.emb}/emb_*.npy"))])
    m = np.load(f"{a.emb}/meta.npz")
    y, lday, lcid, is_tr = m["y"], m["lab_day"], m["lab_cid"], m["is_train"]
    day_min = int(m["day_min"])
    MK = np.load(f"{a.emb}/mkt_emb.npy")          # (n_idx, span, D)
    assert len(E) == len(y), f"特征 {len(E)} 与标签 {len(y)} 不匹配"

    # 拼上大盘表示 —— 与裸K模型的信息对等
    di = np.clip(lday - day_min, 0, MK.shape[1] - 1)
    X = np.concatenate([E] + [MK[k][di] for k in range(MK.shape[0])], axis=1)
    log.info("特征维 %d (个股 %d + 大盘 %d×%d)", X.shape[1], E.shape[1],
             MK.shape[0], MK.shape[2])
    del E, MK

    tr, te = np.flatnonzero(is_tr), np.flatnonzero(~is_tr)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6      # ⚠️ 只能用训练集的统计量
    X = ((X - mu) / sd).astype(np.float32)

    net = Head(X.shape[1]).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.05)
    rng = np.random.default_rng(42)

    for ep in range(1, a.epochs + 1):
        rng.shuffle(tr)
        net.train()
        for i in range(0, len(tr), a.bs):
            b = tr[i:i + a.bs]
            loss = lossf(net(torch.as_tensor(X[b], device=dev)),
                         torch.as_tensor(y[b], device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        with torch.no_grad():
            sc = np.concatenate([
                net(torch.as_tensor(X[te[i:i + 65536]], device=dev)).cpu().numpy()
                for i in range(0, len(te), 65536)])
        report(f"ep{ep}", sc, y[te], lday[te])

    np.savez("kronos_scores.npz", score=sc, lab_day=lday[te], lab_cid=lcid[te])
    log.info("分数已存 kronos_scores.npz —— 拿回本地跑组合回测")


if __name__ == "__main__":
    main()
