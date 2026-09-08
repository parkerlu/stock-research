"""裸K数据准备 —— 按股票存原始 OHLCV 序列, 窗口在取数时现切。

与之前所有版本的根本区别:
    之前   我手工造 40 个特征喂给模型 —— 模型只能在我拍的框框里找规律
    这里   直接给 200 根原始K线, 让它自己找

⚠️ 不能物化窗口: 1180万样本 × 200根 × 6通道 = 57GB。按股票存原始序列只要
   283MB, 取样时现切一段就行。

⚠️ 归一化必须做, 但只能做【无量纲】的那种:
       价格 -> log(p / 最后一根收盘)   纯形状, 抹掉绝对价位
       成交量 -> log1p(v / 窗口内均量)  抹掉规模
   绝对价格有时代烙印(2016年的10块和2026年的10块不是一回事), 留着就等于
   给了模型一个日期线索。

⚠️ 标签仍用三重障碍 y_0.15_0.08(+15%/-8%/20交易日下真正落袋的超额,
   且次日一字涨停的样本已剔除)。标签是超额 -> 牛市普涨不给分。
"""
from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import gc
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("rawk.prep")
OUT = "/app/data/research/rawk"
LAB_DIR = "/app/data/research/shape/_yr7"      # 含 y_0.15_0.08
EPOCH = dt.date(1970, 1, 1)
NBAR = 200


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=2015)   # 多取一年做 200 根的预热
    ap.add_argument("--y1", type=int, default=2026)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    # ---- 1. 原始 OHLCV, 按年分批(fetchall 千万行会 OOM) ----
    log.info("取原始K线 ...")
    parts = []
    async with engine.connect() as c:
        for y in range(a.y0, a.y1 + 1):
            rows = (await c.execute(text(
                "select ts_code, trade_date, open*adj_factor, high*adj_factor, "
                "low*adj_factor, close*adj_factor, vol from daily_candle "
                "where trade_date >= :a and trade_date < :b"),
                {"a": dt.date(y, 1, 1), "b": dt.date(y + 1, 1, 1)})).fetchall()
            if not rows:
                continue
            parts.append(pd.DataFrame({
                "ts_code": [r[0] for r in rows],
                "day": np.array([(r[1] - EPOCH).days for r in rows], np.int32),
                "o": np.array([float(r[2]) for r in rows], np.float32),
                "h": np.array([float(r[3]) for r in rows], np.float32),
                "l": np.array([float(r[4]) for r in rows], np.float32),
                "c": np.array([float(r[5]) for r in rows], np.float32),
                "v": np.array([float(r[6]) for r in rows], np.float32)}))
            del rows
    px = pd.concat(parts, ignore_index=True); del parts; gc.collect()
    px = px.sort_values(["ts_code", "day"]).reset_index(drop=True)
    log.info("  %s 行 / %d 只", f"{len(px):,}", px["ts_code"].nunique())

    # cid 与之前各版本保持一致(pd.Categorical 类别 = 排序后唯一值)
    import pyarrow.parquet as pq
    codes = pq.read_table("/app/data/research/shape/features_v2.parquet",
                          columns=["ts_code"]).to_pandas()["ts_code"]
    code_index = pd.Index(pd.Categorical(codes).categories)
    del codes; gc.collect()
    cmap = {c: i for i, c in enumerate(code_index)}
    px["cid"] = px["ts_code"].map(cmap).astype("float32")
    px = px.dropna(subset=["cid"])
    px["cid"] = px["cid"].astype(np.int32)

    # ---- 2. 存成扁平数组 + 每只票的起止下标 ----
    ohlcv = px[["o", "h", "l", "c", "v"]].to_numpy(np.float32)
    cid = px["cid"].to_numpy(np.int32)
    day = px["day"].to_numpy(np.int32)
    del px; gc.collect()
    bounds = np.searchsorted(cid, np.arange(cid.max() + 2))   # 每个 cid 的起点

    # ---- 3. 标签: 三重障碍 y_0.15_0.08 ----
    labs = []
    for y in range(2016, a.y1 + 1):
        p = f"{LAB_DIR}/{y}.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p, columns=["cid", "day", "y_0.15_0.08"])
        labs.append(d)
        del d
    lab = pd.concat(labs, ignore_index=True); del labs; gc.collect()
    log.info("  标签 %s 行", f"{len(lab):,}")

    # 标签行 -> 在扁平数组里的下标; 且必须有 >=NBAR 根历史
    key_px = cid.astype(np.int64) * 100000 + day
    key_lb = lab["cid"].to_numpy(np.int64) * 100000 + lab["day"].to_numpy(np.int64)
    pos = np.minimum(np.searchsorted(key_px, key_lb), len(key_px) - 1)
    hit = key_px[pos] == key_lb
    pos = pos[hit]
    yv = lab["y_0.15_0.08"].to_numpy(np.float32)[hit]
    dayv = lab["day"].to_numpy(np.int32)[hit]
    cidv = lab["cid"].to_numpy(np.int32)[hit]
    del lab, key_px, key_lb, hit; gc.collect()
    # ⚠️ 窗口不能跨股票: 起点必须落在同一只票内
    enough = pos - NBAR + 1 >= bounds[cidv]
    pos, yv, dayv, cidv = pos[enough], yv[enough], dayv[enough], cidv[enough]
    log.info("  可用样本 %s (要求 >=%d 根历史)", f"{len(pos):,}", NBAR)

    # ---- 4. 真实大盘K线 —— 让模型能自己算"这只票相对大盘怎么样" ----
    # ⚠️ 裸K版之前的缺陷: CNN 孤立地看每只票, 不知道当天全市场什么情况。
    #    同一根放量长阳, 在"全市场都涨"和"只有它涨"下含义完全相反。
    # ⚠️ 必须用【真实指数】, 不能自己拿成分股拼等权指数 —— 等权每日再平衡
    #    会凭空造出巨额虚假收益(实测拼出来 11 年涨 39 倍), 那不是市场真实走势。
    # ⚠️ 这确实让模型看得到 regime。防线仍是标签: 标签是超额收益,
    #    "大盘在涨"对预测超额毫无用处。训完必须看审计相关系数。
    IDX = ["000852.SH", "000300.SH"]      # 中证1000(小盘, 贴近本universe) + 沪深300(大盘)
    async with engine.connect() as c:
        irows = (await c.execute(text(
            "select ts_code, trade_date, open, high, low, close from index_daily "
            "where ts_code = any(:cs) order by ts_code, trade_date"),
            {"cs": IDX})).fetchall()
    day_min = int(day.min())
    span = int(day.max()) - day_min + 2
    # (指数数, 4通道, 天数) —— 非交易日用前值填, 保证任意 day 都查得到
    mkt = np.zeros((len(IDX), 4, span), np.float32)
    for k, code in enumerate(IDX):
        sub = [r for r in irows if r[0] == code]
        d_ = np.array([(r[1] - EPOCH).days for r in sub]) - day_min
        v_ = np.array([[float(r[2]), float(r[3]), float(r[4]), float(r[5])]
                       for r in sub], np.float32)
        ok = (d_ >= 0) & (d_ < span)
        tmp = np.full((span, 4), np.nan, np.float32)
        tmp[d_[ok]] = v_[ok]
        mkt[k] = pd.DataFrame(tmp).ffill().bfill().to_numpy(np.float32).T
        log.info("  %s: %d 天, 收盘 %.0f -> %.0f", code, ok.sum(),
                 mkt[k, 3, 0], mkt[k, 3, -1])
    del irows; gc.collect()

    np.savez(f"{OUT}/panel.npz", ohlcv=ohlcv, cid=cid, day=day,
             mkt=mkt, day_min=np.int64(day_min),
             idx=pos.astype(np.int64), y=yv, lab_day=dayv, lab_cid=cidv)
    log.info("存 %s/panel.npz  (%.0f MB)", OUT,
             os.path.getsize(f"{OUT}/panel.npz") / 1e6)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
