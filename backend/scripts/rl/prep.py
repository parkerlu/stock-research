"""为 RL 造面板: 特征(已横截面分位) + 【每日】超额收益, 按股票切成连续序列。

⚠️ 时序对齐(错一格就是前视):
       t 时刻观察到的是【收盘后】的特征 feats[t]
       据此决定仓位, 赚到的是【下一日】的收益
   所以 exret[t] = 第 t+1 日的超额收益。

⚠️ 超额 = 个股当日收益 − 当日全市场等权收益。用超额而非原始收益, 是为了
   让"空仓=0分、牛市满仓也≈0分", 从源头堵死策略学成"牛市就买"。

⚠️ 内存: 逐年读、立刻转 float32、按股票切片后只留 npz。整表拼会 OOM。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gc
import logging

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rl.prep")
DIR = "/app/data/research/shape"
OUT = "/app/data/research/rl"
META = {"day", "cid", "fwd"}
EPOCH = dt.date(1970, 1, 1)


async def daily_excess(y0: int, y1: int) -> pd.DataFrame:
    """全市场每日超额收益。按年分批取 —— fetchall 千万行会 OOM。"""
    parts = []
    async with engine.connect() as c:
        for y in range(y0, y1 + 1):
            rows = (await c.execute(text(
                "select ts_code, trade_date, close*adj_factor from daily_candle "
                "where trade_date >= :a and trade_date < :b"),
                {"a": dt.date(y, 1, 1), "b": dt.date(y + 1, 1, 1)})).fetchall()
            if not rows:
                continue
            d = pd.DataFrame({
                "ts_code": [r[0] for r in rows],
                "day": np.array([(r[1] - EPOCH).days for r in rows], np.int32),
                "c": np.array([float(r[2]) for r in rows], np.float32)})
            del rows
            parts.append(d)
    px = pd.concat(parts, ignore_index=True)
    del parts
    px = px.sort_values(["ts_code", "day"])
    px["r"] = px.groupby("ts_code", sort=False)["c"].pct_change().astype(np.float32)
    mkt = px.groupby("day")["r"].transform("mean")
    px["ex"] = (px["r"] - mkt).astype(np.float32)
    # ⚠️ 要的是"在 t 日收盘决策、赚到的下一个【交易日】的收益"。
    #    用自然日 day+1 去查会把周五和假期前一天整行丢掉(每周只剩4天,
    #    策略等于从没见过周五)。这里按每只票向前移一格, 天然就是下个交易日。
    px["ex_next"] = px.groupby("ts_code", sort=False)["ex"].shift(-1).astype(np.float32)
    return px[["ts_code", "day", "ex_next"]].dropna()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=2016)
    ap.add_argument("--y1", type=int, default=2026)
    a = ap.parse_args()
    import os
    os.makedirs(OUT, exist_ok=True)

    log.info("读全市场每日超额 ...")
    ex = await daily_excess(a.y0, a.y1)
    log.info("  %s 行", f"{len(ex):,}")

    # cid -> ts_code 的映射与训练时一致(pd.Categorical 类别 = 排序后唯一值)
    import pyarrow.parquet as pq
    codes = pq.read_table(f"{DIR}/features_v2.parquet",
                          columns=["ts_code"]).to_pandas()["ts_code"]
    code_index = pd.Index(pd.Categorical(codes).categories)
    del codes; gc.collect()
    cmap = {c: i for i, c in enumerate(code_index)}
    ex["cid"] = ex["ts_code"].map(cmap).astype("float32")
    ex = ex.dropna(subset=["cid"])
    ex["cid"] = ex["cid"].astype(np.int32)
    ex_key = ex["cid"].to_numpy(np.int64) * 100000 + ex["day"].to_numpy()
    ex_val = ex["ex_next"].to_numpy(np.float32)
    order = np.argsort(ex_key)
    ex_key, ex_val = ex_key[order], ex_val[order]
    del ex, order; gc.collect()

    cols = None
    for tag, years in (("train", list(range(a.y0, 2021))),
                       ("test", list(range(2021, a.y1 + 1)))):
        F, R, C, D = [], [], [], []
        for y in years:
            d = pd.read_parquet(f"{DIR}/_yr/{y}.parquet")
            if cols is None:
                cols = [c for c in d.columns if c not in META and not c.startswith("y_")]
            cid = d["cid"].to_numpy(np.int32)
            day = d["day"].to_numpy(np.int64)
            # ex 表里存的已经是"下一个交易日的超额", 所以这里按当日 key 对齐
            k = cid.astype(np.int64) * 100000 + day
            pos = np.minimum(np.searchsorted(ex_key, k), len(ex_key) - 1)
            hit = ex_key[pos] == k
            F.append(d[cols].to_numpy(np.float32)[hit])
            R.append(ex_val[pos][hit])
            C.append(cid[hit]); D.append(day[hit])
            del d; gc.collect()
        feats = np.concatenate(F); ret = np.concatenate(R)
        # ⚠️ 缺失填 0.5 —— 特征是当日横截面【分位】, 0.5 = "排在正中间",
        #    是信息量最小的取值。XGBoost 能原生处理 NaN, 神经网络不行:
        #    一个 NaN 会顺着网络污染整批 logits(实测直接训不动)。
        #    实测训练集 16.5% 的行至少有一个 NaN(新股凑不满 120/250 日窗口)。
        n_nan = int(np.isnan(feats).sum())
        if n_nan:
            log.info("  填补 %s 个 NaN -> 0.5 (%.2f%% 的行受影响)",
                     f"{n_nan:,}", np.isnan(feats).any(1).mean() * 100)
            np.nan_to_num(feats, copy=False, nan=0.5)
        cid = np.concatenate(C); day = np.concatenate(D)
        del F, R, C, D; gc.collect()
        # 按 (cid, day) 排好, 便于切连续序列
        o = np.lexsort((day, cid))
        np.savez(f"{OUT}/{tag}.npz", feats=feats[o], ret=ret[o],
                 cid=cid[o], day=day[o], cols=np.array(cols))
        log.info("%s: %s 行 × %d 特征 -> %s.npz", tag, f"{len(o):,}", len(cols), tag)
        del feats, ret, cid, day, o; gc.collect()
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
