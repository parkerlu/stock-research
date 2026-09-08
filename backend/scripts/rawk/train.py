"""裸K序列模型 —— 给 200 根原始K线, 让它自己找规律。

与之前所有版本的区别: 不喂手工特征。之前 40 个特征全是我拍的, 模型只能在
我的框框里找; 这里只给原始 OHLCV + 一条 MA20(用户指定"最多加一条均线")。

⚠️ 归一化只能做无量纲的:
      价格 -> log(p / 窗口最后一根收盘)   纯形状, 抹掉绝对价位与时代烙印
      量   -> log1p(v / 窗口内均量)       抹掉规模(绝对量含市值信息)
⚠️ 标签是【超额】(三重障碍 y_0.15_0.08), 牛市普涨不给分 —— 去 regime 靠这个。
⚠️ 训完必须跑去记忆审计: 每日均分 vs 当日市场超额, |corr| 要 < 0.10。

⚠️ 没有 GPU(10 核 CPU)。所以: 网络做小、训练集抽样、窗口现切不物化
   (全量展开 200根×6通道 = 57GB)。
"""
from __future__ import annotations

import argparse
import logging
import os
import time

import numpy as np
import torch
import torch.nn as nn

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk")
DIR = "/app/data/research/rawk"
NBAR = 200
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_batch(ohlcv, idx, nbar=NBAR, day=None, mkt=None, day_min=0):
    """现切窗口并归一化。idx 是每个样本"当前这根"在扁平数组里的下标。"""
    # (B, nbar) 的取数下标
    off = np.arange(-nbar + 1, 1, dtype=np.int64)
    rows = idx[:, None] + off[None, :]
    w = ohlcv[rows]                                  # (B, nbar, 5)
    c_last = w[:, -1, 3:4]                           # 最后一根收盘
    c_last = np.maximum(c_last, 1e-6)
    px = np.log(np.maximum(w[:, :, :4], 1e-6) / c_last[:, :, None])   # (B,nbar,4)
    v = w[:, :, 4]
    vm = np.maximum(v.mean(1, keepdims=True), 1e-6)
    vol = np.log1p(v / vm)[:, :, None]
    # MA20(用户指定最多一条均线) —— 同样以最后一根收盘归一化
    cser = w[:, :, 3]
    ma = np.copy(cser)
    cs = np.cumsum(cser, axis=1)
    ma[:, 20:] = (cs[:, 20:] - cs[:, :-20]) / 20.0
    ma[:, :20] = cs[:, :20] / np.arange(1, 21, dtype=np.float32)
    ma = np.log(np.maximum(ma, 1e-6) / c_last)[:, :, None]
    chans = [px, vol, ma]
    # ---- 真实大盘K线(中证1000 + 沪深300), 同样只给形状 ----
    # ⚠️ 用【各自窗口最后一根收盘】归一化, 与个股同一口径; 这样个股与大盘
    #    的相对强弱由模型自己去算, 我不替它算。
    if mkt is not None:
        dd = day[rows] - day_min                     # (B, nbar) 每根对应的日索引
        dd = np.clip(dd, 0, mkt.shape[2] - 1)
        for k in range(mkt.shape[0]):
            mk = mkt[k][:, dd]                       # (4, B, nbar)
            mk = np.transpose(mk, (1, 2, 0))         # (B, nbar, 4)
            last = np.maximum(mk[:, -1:, 3:4], 1e-6)
            chans.append(np.log(np.maximum(mk, 1e-6) / last))
    x = np.concatenate(chans, axis=2)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0
                         ).transpose(0, 2, 1).astype(np.float32)   # (B,C,nbar)


class Net(nn.Module):
    """膨胀卷积 —— 膨胀率 1,2,4,8,16,32 叠起来感受野覆盖 200 根。
    网络故意做小: 金融数据信噪比极低, 大网络只会把噪声背下来。"""

    def __init__(self, ch=6, h=48):
        super().__init__()
        self.inp = nn.Conv1d(ch, h, 5, padding=2)
        blocks = []
        for d in (1, 2, 4, 8, 16, 32):
            blocks.append(nn.Sequential(
                nn.Conv1d(h, h, 3, padding=d, dilation=d),
                nn.GroupNorm(4, h), nn.GELU()))
        self.blocks = nn.ModuleList(blocks)
        self.head = nn.Sequential(nn.Linear(h * 2, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x):
        z = torch.relu(self.inp(x))
        for b in self.blocks:
            z = z + b(z)                      # 残差
        # 全局平均 + 末端取值(末端携带"最近发生了什么")
        z = torch.cat([z.mean(-1), z[:, :, -1]], dim=1)
        return self.head(z).squeeze(-1)


def evaluate(net, ohlcv, idx, y, lday, bs=4096, day=None, mkt=None, day_min=0):
    net.eval()
    sc = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            x = make_batch(ohlcv, idx[i:i + bs], day=day, mkt=mkt, day_min=day_min)
            sc[i:i + bs] = net(torch.as_tensor(x, device=DEV)).cpu().numpy()
    net.train()
    import pandas as pd
    d = pd.DataFrame({"day": lday, "y": y, "s": sc})
    d["rk"] = d.groupby("day")["s"].rank(pct=True)
    out = {}
    for lo, hi, nm in ((0.99, 1.01, "Top1%"), (0.95, 1.01, "Top5%"),
                       (0.0, 0.20, "Bot20%")):
        g = d[(d.rk >= lo) & (d.rk < hi)]
        out[nm] = float(g["y"].mean()) * 100
    dl = d.groupby("day").agg(avg=("s", "mean"), m=("y", "mean"))
    out["corr"] = float(dl["avg"].corr(dl["m"]))
    out["yearly"] = d[d.rk >= 0.95].assign(
        yr=(pd.to_datetime(d.loc[d.rk >= 0.95, "day"], unit="D").dt.year)
    ).groupby("yr")["y"].mean().mul(100).round(2).to_dict()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=512)
    ap.add_argument("--n-train", type=int, default=800_000)
    ap.add_argument("--n-eval", type=int, default=600_000)
    ap.add_argument("--lr", type=float, default=1e-3)
    a = ap.parse_args()

    d = np.load(f"{DIR}/panel.npz")
    ohlcv, idx, y, lday = d["ohlcv"], d["idx"], d["y"], d["lab_day"]
    day_all, mkt, day_min = d["day"], d["mkt"], int(d["day_min"])
    n_ch = 6 + mkt.shape[0] * 4
    log.info("通道数 %d = 个股6(OHLC/量/MA20) + 大盘%d条×4", n_ch, mkt.shape[0])
    import pandas as pd
    yr = pd.to_datetime(pd.Series(lday), unit="D").dt.year.to_numpy()
    tr_m, te_m = yr <= 2020, yr >= 2021
    rng = np.random.default_rng(42)
    tr_i = rng.choice(np.flatnonzero(tr_m), min(a.n_train, tr_m.sum()), replace=False)
    te_all = np.flatnonzero(te_m)
    te_i = rng.choice(te_all, min(a.n_eval, len(te_all)), replace=False)
    log.info("设备 %s | 训练 %s / 测试 %s (样本外 2021-2026)",
             DEV, f"{len(tr_i):,}", f"{len(te_i):,}")

    net = Net(ch=n_ch).to(DEV)
    n_par = sum(p.numel() for p in net.parameters())
    log.info("参数量 %s", f"{n_par:,}")
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    lossf = nn.HuberLoss(delta=0.05)     # 标签有厚尾, Huber 比 MSE 稳

    for ep in range(1, a.epochs + 1):
        rng.shuffle(tr_i)
        t0, tot, nb = time.time(), 0.0, 0
        for i in range(0, len(tr_i), a.bs):
            b = tr_i[i:i + a.bs]
            x = torch.as_tensor(make_batch(ohlcv, idx[b], day=day_all, mkt=mkt,
                                           day_min=day_min), device=DEV)
            t = torch.as_tensor(y[b], device=DEV)
            loss = lossf(net(x), t)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            tot += float(loss.detach()); nb += 1
            if nb % 400 == 0:
                log.info("  ep%d %5d/%d 批 损失 %.5f (%.0f 批/分)",
                         ep, nb, len(tr_i) // a.bs, tot / nb,
                         nb / max(time.time() - t0, 1) * 60)
        r = evaluate(net, ohlcv, idx[te_i], y[te_i], lday[te_i],
                     day=day_all, mkt=mkt, day_min=day_min)
        log.info("ep%d 样本外 Top1%% %+.2f Top5%% %+.2f Bot20%% %+.2f | 审计 %+.3f",
                 ep, r["Top1%"], r["Top5%"], r["Bot20%"], r["corr"])
        log.info("     逐年 %s", r["yearly"])
        torch.save(net.state_dict(), f"{DIR}/cnn.pt")


if __name__ == "__main__":
    main()
