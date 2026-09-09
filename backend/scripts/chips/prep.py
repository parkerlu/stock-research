"""筹码特征 —— 换信息源的第一步。

背景（2026-09-10）：裸K 六个实验(nbar 100/200/400/800、h1024、nomkt)全部收敛到
组合比值 0.38~0.78，且裸K(+0.47) 并没有超过手工40特征版(+0.45) —— 两种完全不同
的表示落在同一水平，更像是【日线 OHLCV 这个信息源本身的上限】。所以换信息源。

⚠️ 为什么是筹码而不是别的：项目里唯一被独立验证过携带【增量】信息的就是它 ——
   主力吸筹用获利盘族做到样本外 Q10/Q1=4.81、按天 t=77.4，而且过了
   "价格位置 × 量能"双重控制的 3×3 九宫格(九格全正、最小 1.63)，
   控制前后几乎不衰减(2.16 -> 2.09)。即它带的是价量里【没有】的信息。

⚠️ 只做获利盘族。文档已证伪: 筹码集中度 Q10/Q1=1.10、低位筹码占比 1.01, 都无效 ——
   "吸筹不一定让筹码变集中, 但一定会改变获利盘结构"。这里仍算集中度, 但只作为
   【对照特征】, 用来事后确认它确实没用(而不是默默把它当成有用的)。

⚠️ 题目与裸K 完全一致: 直接复用 label3_dn8.npz。不一致就没法比。

⚠️ 时间对齐: cyq_perf 是当日盘后发布, T 日筹码 T 日收盘后才知道, 而买入在 T+1
   开盘 —— 因果上成立。绝不能用 T+1 的筹码。这里所有滚动窗口都只向后看。

⚠️ cid 映射: panel.npz 的 cid 来自 shape/features_v2.parquet 的 ts_code
   Categorical 类别(排序后唯一值)。必须用同一个映射, 否则整张表对错位而【不报错】。

输出 chips_feat.npz: X(样本 × 特征) + names + ok(该样本筹码是否齐全)。
样本顺序与 panel.npz 的 idx 严格一致。
"""
from __future__ import annotations

import argparse
import gc
import logging

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from scripts.chips.features import CYQ_COLS, FEAT_NAMES, bad_rows, chip_feats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("chips.prep")

DIR = "/app/data/research"
MAPSRC = f"{DIR}/shape/features_v2.parquet"


def shift_within(arr: np.ndarray, cid: np.ndarray, k: int) -> np.ndarray:
    """按股票分组向后取 k 根。跨股票的位置填 nan。

    ⚠️ 不能直接 np.roll —— 那会让 A 股票的头部取到 B 股票的尾部, 而且不报错。
    """
    out = np.full(len(arr), np.nan, np.float32)
    if k <= 0:
        return arr.astype(np.float32)
    out[k:] = arr[:-k]
    bad = np.zeros(len(arr), bool)
    bad[:k] = True
    bad[k:] = cid[k:] != cid[:-k]
    out[bad] = np.nan
    return out


def build(out: str) -> None:
    d = np.load(f"{DIR}/rawk/panel.npz")
    cid, day, idx = d["cid"], d["day"], d["idx"].astype(np.int64)
    close = d["ohlcv"][:, 3].astype(np.float32)
    n_bar = len(close)
    log.info("面板: K线 %s, 样本 %s, 股票 %d",
             f"{n_bar:,}", f"{len(idx):,}", len(np.unique(cid)))

    codes = pq.read_table(MAPSRC, columns=["ts_code"]).to_pandas()["ts_code"]
    code_index = pd.Index(pd.Categorical(codes).categories)
    cmap = {c: i for i, c in enumerate(code_index)}
    del codes
    gc.collect()

    cyq = pd.read_parquet(f"{DIR}/cyq.parquet")
    cyq["cid"] = cyq["ts_code"].map(cmap)
    cyq = cyq.dropna(subset=["cid"])
    cyq["cid"] = cyq["cid"].astype(np.int32)
    cyq["day"] = ((pd.to_datetime(cyq["trade_date"]) - pd.Timestamp("1970-01-01"))
                  .dt.days.astype(np.int32))
    log.info("筹码: %s 行, 映射到 %d 只(未匹配 %d 只)",
             f"{len(cyq):,}", cyq.cid.nunique(), len(cmap) - cyq.cid.nunique())

    # ---- 用 (cid, day) 复合键把筹码贴到面板的每一根K线上 ----
    # ⚠️ day 最大约 20700, 用 *100000 不会串位(prep.py 里同一套写法)
    key_px = cid.astype(np.int64) * 100000 + day
    key_cy = cyq["cid"].to_numpy(np.int64) * 100000 + cyq["day"].to_numpy(np.int64)
    order = np.argsort(key_cy)
    key_cy = key_cy[order]
    pos = np.searchsorted(key_cy, key_px)
    pos_c = np.clip(pos, 0, len(key_cy) - 1)
    hit = key_cy[pos_c] == key_px
    log.info("面板K线里有筹码的: %s (%.1f%%) —— 筹码 2018 起, 之前的行必然缺",
             f"{hit.sum():,}", hit.mean() * 100)

    cols = ["winner_rate", "cost_5pct", "cost_15pct", "cost_50pct",
            "cost_85pct", "cost_95pct", "weight_avg", "his_low", "his_high"]
    raw = {}
    for c in cols:
        v = np.full(n_bar, np.nan, np.float32)
        src = cyq[c].to_numpy(np.float32)[order]
        v[hit] = src[pos_c[hit]]
        raw[c] = v
    del cyq
    gc.collect()

    # ⚠️ 特征算法只留一份, 见 features.chip_feats —— 训练和生产必须完全同一套。
    #    以前本项目"同一份东西抄两遍"栽过五次, 而特征算不一致的后果是模型拿到
    #    另一个分布, 结果变差【且不报错】。
    long = pd.DataFrame({"gid": cid, "close": close})
    for c in CYQ_COLS:
        long[c] = raw[c]
    F = chip_feats(long, gid="gid")
    bad = bad_rows(long).to_numpy()
    feats = {n: F[n].to_numpy() for n in FEAT_NAMES}

    names = list(feats)
    X = np.empty((len(idx), len(names)), np.float32)
    for j, nm in enumerate(names):
        X[:, j] = feats[nm][idx]
    # ⚠️ 清洗: 成本价 <=0 的行(停牌/数据错误)会让比值爆掉 —— 实测"上半区宽度"
    #    均值 153.965 而中位只有 0.132。树模型对单调变换不敏感, 但极端值会挤占
    #    分裂点、也会让特征统计失去可读性。先剔无效行, 再按 p0.1/p99.9 截断。
    bad_price = bad[idx]
    ok = np.isfinite(X).all(1) & ~bad_price
    log.info("剔除成本价异常 %s 行", f"{int(bad_price.sum()):,}")
    for j in range(X.shape[1]):
        v = X[ok, j]
        lo, hi = np.percentile(v, [0.1, 99.9])
        X[:, j] = np.clip(X[:, j], lo, hi)
    log.info("特征 %d 个, 样本 %s, 筹码齐全的 %s (%.1f%%)",
             len(names), f"{len(idx):,}", f"{ok.sum():,}", ok.mean() * 100)
    for j, nm in enumerate(names):
        v = X[ok, j]
        log.info("  %-16s 均值 %+9.3f  中位 %+9.3f  p1 %+9.3f  p99 %+9.3f",
                 nm, v.mean(), np.median(v), np.percentile(v, 1), np.percentile(v, 99))

    np.savez(f"{DIR}/chips/{out}", X=X, names=np.array(names), ok=ok)
    log.info("存 %s/chips/%s", DIR, out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="chips_feat.npz")
    a = ap.parse_args()
    build(a.out)


if __name__ == "__main__":
    main()
