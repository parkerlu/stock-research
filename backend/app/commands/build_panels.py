"""重建 panel.npz / cyq_panel.npz —— 买卖很准/主力吸筹/低点组合三个指标的输入。

⚠️ 这个文件是补写的。原来【没有任何脚本生成这两个面板】—— 当初在容器里临时
   写的,随镜像重建丢了。后果是三个指标读着 2026-09-04 的缓存面板,每天照常
   "重算"却永远出不了新信号, 而且不报错。2026-09-08 才被用户发现。

   同一个病本项目犯过三次(SAR 回测脚本、Kronos 前的形态实验脚本)。
   **凡是生产链依赖的中间产物, 生成脚本必须进仓库。**

⚠️ 面板是 (T 天 × N 只) 的稠密矩阵, 停牌日填 NaN —— 下游按 np.nanmean 等
   处理。不能用 0 填, 0 会被当成真实价格。
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
log = logging.getLogger("build_panels")
OUT = "/app/data/research"
COLS = ["open", "high", "low", "close", "vol", "amount"]
# ⚠️ 少一个字段, 下游就报 "xxx is not a file in the archive" 然后整个指标
#    重算失败。把 cyq.parquet 里的数值列【全部】存进去, 不要手工挑。
CYQ_COLS = ["his_low", "his_high", "cost_5pct", "cost_15pct", "cost_50pct",
            "cost_85pct", "cost_95pct", "weight_avg", "winner_rate"]


async def _axes() -> tuple[np.ndarray, np.ndarray]:
    """先拿两条轴。⚠️ 用 distinct 单列查, 比把 1180 万行拉回来再 unique 快几个量级。"""
    async with engine.connect() as c:
        ds = [r[0] for r in (await c.execute(text(
            "select distinct trade_date from daily_candle order by trade_date"))).fetchall()]
        cs = [r[0] for r in (await c.execute(text(
            "select distinct ts_code from daily_candle order by ts_code"))).fetchall()]
    return np.array(ds), np.array(cs)


async def _fill(dates, codes, mats: dict, cols: list[str], table: str,
                extra: str = "") -> None:
    """按 ts_code 分批填矩阵。

    ⚠️ 两个坑都在这一行里:
      1. 不能按 trade_date 范围查 —— 索引是 (ts_code, trade_date), 按日期
         范围会全表扫描, 11 年扫 11 遍(实测卡死 10 分钟没出来)。
      2. 不能把全部行拉回来拼成一张 DataFrame —— 1180 万行的 ts_code 字符串
         就要吃掉一两个 G(本项目为此 OOM 过五次)。
      按 ts_code 分批走主键索引, 每批立刻写进预分配矩阵, 峰值只有一批。
    """
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}
    sel = ", ".join(cols)
    B = 400
    for k in range(0, len(codes), B):
        batch = list(codes[k:k + B])
        async with engine.connect() as c:
            rows = (await c.execute(text(
                f"select ts_code, trade_date, {sel}{extra} from {table} "
                "where ts_code = any(:cs)"), {"cs": batch})).fetchall()
        if not rows:
            continue
        ri = np.fromiter((di.get(r[1], -1) for r in rows), np.int32, len(rows))
        cc = np.fromiter((ci.get(r[0], -1) for r in rows), np.int32, len(rows))
        ok = (ri >= 0) & (cc >= 0)
        vals = np.array([[float(v) if v is not None else np.nan for v in r[2:]]
                         for r in rows], np.float32)
        del rows
        for j, name in enumerate(cols):
            mats[name][ri[ok], cc[ok]] = vals[ok, j]
        if (k // B) % 4 == 0:
            log.info("  %d/%d 只", min(k + B, len(codes)), len(codes))
        del vals
        gc.collect()


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--y0", type=int, default=2015)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)

    log.info("取两条轴 ...")
    dates, codes = await _axes()
    dates = dates[dates >= dt.date(a.y0, 1, 1)]
    log.info("  面板 %d 天 × %d 只, %s ~ %s", len(dates), len(codes),
             dates[0], dates[-1])

    # ⚠️ 存【前复权】价: 下游直接拿来算收益, 不复权的话除权日会算出假暴跌。
    #    前复权 = 原价 × (当日factor / 该股最新factor), 所以要把 adj_factor 也填进来。
    mats = {c: np.full((len(dates), len(codes)), np.nan, np.float32) for c in COLS}
    mats["adj_factor"] = np.full((len(dates), len(codes)), np.nan, np.float32)
    log.info("填日线 ...")
    await _fill(dates, codes, mats, COLS + ["adj_factor"], "daily_candle")

    af = mats.pop("adj_factor")
    # 每只票的最新有效 factor
    last = np.full(af.shape[1], np.nan, np.float32)
    for j in range(af.shape[1]):
        col = af[:, j]
        v = col[np.isfinite(col)]
        if len(v):
            last[j] = v[-1]
    f = af / np.where(np.isfinite(last) & (last != 0), last, np.nan)
    f = np.where(np.isfinite(f), f, 1.0)
    for c in ("open", "high", "low", "close"):
        mats[c] = (mats[c] * f).astype(np.float32)
    del af, f
    gc.collect()

    np.savez(f"{OUT}/panel.npz", dates=dates.astype("datetime64[D]"),
             codes=codes, **mats)
    log.info("panel.npz 已写 (%.0f MB)", os.path.getsize(f"{OUT}/panel.npz") / 1e6)
    del mats
    gc.collect()

    # ⚠️ 筹码【不在数据库】, 在 cyq.parquet(见 update_indicators.fetch_cyq_recent)。
    #    我第一版写成查 cyq_chips 表, 而那张表根本不存在 —— 会静默产出空面板。
    log.info("取筹码 ...")
    cyp = f"{OUT}/cyq.parquet"
    if not os.path.exists(cyp):
        log.warning("没有 cyq.parquet, 跳过筹码面板")
    else:
        cy = pd.read_parquet(cyp, columns=["ts_code", "trade_date", *CYQ_COLS])
        cy["trade_date"] = pd.to_datetime(cy["trade_date"]).dt.date
        log.info("  %s 行, 最新 %s", f"{len(cy):,}", cy["trade_date"].max())
        # ⚠️⚠️ 隐式耦合, 改动前务必看这里:
        #    build_pump / build_didian 把日线切到 >= 2018-01-01 (m0 掩码),
        #    却【不切】筹码面板 —— 也就是说它们默认 cyq_panel 本来就是
        #    2018+ 的轴。所以这里必须按同一个起点建, 否则会报
        #    "operands could not be broadcast together" 而整个指标重算失败。
        #    (build_maimai 的 START 是 2014-06-01, 但它不读筹码, 不受影响。)
        CYQ_START = dt.date(2018, 1, 1)
        cdates = dates[dates >= CYQ_START]
        log.info("  筹码面板轴: %d 天 (>= %s, 与 build_pump/didian 的切法一致)",
                 len(cdates), CYQ_START)
        di = {d: i for i, d in enumerate(cdates)}
        ci = {c: i for i, c in enumerate(codes)}
        ri = cy["trade_date"].map(di).to_numpy()
        cc = cy["ts_code"].map(ci).to_numpy()
        ok = pd.notna(ri) & pd.notna(cc)
        ri, cc = ri[ok].astype(int), cc[ok].astype(int)
        cm = {}
        for v in CYQ_COLS:
            m = np.full((len(cdates), len(codes)), np.nan, np.float32)
            m[ri, cc] = pd.to_numeric(cy.loc[ok, v], errors="coerce").to_numpy(np.float32)
            cm[v] = m
        del cy
        gc.collect()
        # 顺带存一份日期轴, 以后出问题能一眼看出是不是错位了
        np.savez(f"{OUT}/cyq_panel.npz", dates=cdates.astype("datetime64[D]"), **cm)
        log.info("cyq_panel.npz 已写 (%.0f MB)",
                 os.path.getsize(f"{OUT}/cyq_panel.npz") / 1e6)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
