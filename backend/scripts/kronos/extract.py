"""用 Kronos 预训练表示提特征 —— GPU 机器上跑。

⚠️ 不走它的生成接口。Kronos 是自回归的, 预测20天要采样20次;
   我们只要它的【表示】, 一次前向取隐状态就够, 便宜 20 倍。
   而且生成出来的是绝对价格预测, 天然带 regime(牛市里预测所有票都涨) ——
   那正是本项目从头到尾要避免的东西。

⚠️ 大盘也要提。Kronos 只吃单个标的, 不给大盘它就和我们裸K第一版一样瞎
   (实测只给个股 Top5% +0.17, 补上真实大盘K线后 +0.47)。指数只有 2839 天,
   提特征几乎不花钱, 但信息上必须对等, 否则这个对比不公平。

⚠️ 预处理必须与 Kronos 自己的一致(见 KronosPredictor.predict):
   6 列 open/high/low/close/volume/amount, 按【窗口自身】z-score, 截断 ±5。
⚠️ 抢占式实例可能被回收 —— 每批存盘, 重跑时自动跳过已完成的分片。
"""
from __future__ import annotations

import argparse
import logging
import os

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("kronos.extract")
NBAR = 200


def build_x(ohlcv: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """(B, NBAR, 6) 的 Kronos 输入: o,h,l,c,v,amount, 按窗口 z-score + 截断。"""
    w = ohlcv[rows]                                    # (B, NBAR, 5)
    amt = w[:, :, 4] * w[:, :, :4].mean(axis=2)        # amount ≈ 量 × 均价
    x = np.concatenate([w, amt[:, :, None]], axis=2).astype(np.float32)
    mu = x.mean(axis=1, keepdims=True)
    sd = x.std(axis=1, keepdims=True)
    x = (x - mu) / (sd + 1e-5)
    return np.clip(x, -5.0, 5.0)


def build_stamp(days: np.ndarray) -> np.ndarray:
    """(B, NBAR, 5) 时间特征 minute/hour/weekday/day/month。
    ⚠️ 没有【年】—— 这是 Kronos 自己的设计, 编码的是季节性不是牛熊, 正合我们要求。"""
    ts = pd.to_datetime(days.reshape(-1), unit="D")
    f = np.stack([np.zeros(len(ts)), np.zeros(len(ts)), ts.weekday.values,
                  ts.day.values, ts.month.values], axis=1).astype(np.float32)
    return f.reshape(days.shape[0], days.shape[1], 5)


@torch.no_grad()
def embed(model, tokenizer, x: np.ndarray, stamp: np.ndarray, dev: str) -> np.ndarray:
    """取 transformer 最后一层的隐状态。复刻 Kronos.forward 的前半段,
    跳过 s1/s2 两个预测头 —— 我们要表示, 不要它的预测。"""
    xt = torch.as_tensor(x, device=dev)
    st = torch.as_tensor(stamp, device=dev)
    z = tokenizer.encode(xt, half=True)
    s1, s2 = (z[0], z[1]) if isinstance(z, (list, tuple)) else (z[..., 0], z[..., 1])
    h = model.embedding([s1, s2])
    h = h + model.time_emb(st)
    for layer in model.transformer:
        h = layer(h)
    h = model.norm(h)
    # 末端 + 全局均值: 末端携带"最近发生了什么", 均值携带整段形态
    return torch.cat([h[:, -1], h.mean(1)], dim=1).float().cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="kaggle_pack.npz")
    ap.add_argument("--out", default="kronos_emb")
    ap.add_argument("--model", default="NeoQuasar/Kronos-small",
                    help="mini(410万) / small(2474万) / base(1.02亿)")
    ap.add_argument("--tok", default="NeoQuasar/Kronos-Tokenizer-base")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--shard", type=int, default=200_000, help="每片样本数, 断点续跑单位")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    import sys
    sys.path.append("Kronos")           # git clone 下来的仓库
    from model import Kronos, KronosTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    log.info("设备 %s | 模型 %s", dev, a.model)
    tokenizer = KronosTokenizer.from_pretrained(a.tok).to(dev).eval()
    model = Kronos.from_pretrained(a.model).to(dev).eval()

    d = np.load(a.pack)
    ohlcv, day_all = d["ohlcv"], d["day"]
    idx, y, lday, lcid = d["idx"], d["y"], d["lab_day"], d["lab_cid"]
    is_tr = d["is_train"]
    mkt, day_min = d["mkt"], int(d["day_min"])
    off = np.arange(-NBAR + 1, 1, dtype=np.int64)

    # ---- 大盘特征: 每个指数每天一条, 先全算好, 之后按日期查表 ----
    log.info("提大盘特征 ...")
    mk_emb = []
    for k in range(mkt.shape[0]):
        cl = mkt[k]                                   # (4, span) o,h,l,c
        span = cl.shape[1]
        starts = np.arange(NBAR - 1, span)
        rows = starts[:, None] + off[None, :]
        w = np.transpose(cl[:, rows], (1, 2, 0))      # (T, NBAR, 4)
        v = np.ones((*w.shape[:2], 1), np.float32)    # 指数无量, 填 1
        x = np.concatenate([w, v, v], axis=2).astype(np.float32)
        mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
        x = np.clip((x - mu) / (sd + 1e-5), -5, 5)
        stp = build_stamp(np.tile(np.arange(span)[rows] + day_min, (1, 1)))
        out = []
        for i in range(0, len(x), a.bs):
            out.append(embed(model, tokenizer, x[i:i + a.bs], stp[i:i + a.bs], dev))
        e = np.concatenate(out)
        full = np.zeros((span, e.shape[1]), np.float32)
        full[starts] = e
        mk_emb.append(full)
        log.info("  指数%d: %s 天", k, f"{len(starts):,}")
    np.save(f"{a.out}/mkt_emb.npy", np.stack(mk_emb))

    # ---- 个股特征, 分片存盘(抢占式被回收也不怕) ----
    n = len(idx)
    log.info("提个股特征: %s 条, 每片 %s", f"{n:,}", f"{a.shard:,}")
    for s0 in range(0, n, a.shard):
        p = f"{a.out}/emb_{s0:08d}.npy"
        if os.path.exists(p):
            log.info("  跳过已完成 %s", p); continue
        s1_ = min(s0 + a.shard, n)
        rows = idx[s0:s1_, None] + off[None, :]
        out = []
        for i in range(0, s1_ - s0, a.bs):
            r = rows[i:i + a.bs]
            out.append(embed(model, tokenizer, build_x(ohlcv, r),
                             build_stamp(day_all[r]), dev))
        np.save(p, np.concatenate(out))
        log.info("  %s / %s", f"{s1_:,}", f"{n:,}")

    np.savez(f"{a.out}/meta.npz", y=y, lab_day=lday, lab_cid=lcid,
             is_train=is_tr, day_min=day_min)
    log.info("完成 -> %s/", a.out)


if __name__ == "__main__":
    main()
