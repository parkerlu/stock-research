"""形态模型 · 第1步: 造特征 —— 全部是【无量纲的形状描述】。

设计约束(用户 2026-09-07 提的, 也是这个项目最该守的一条):
    模型不许对"现在是牛市还是熊市"有记忆。
    否则它会学成"牛市全买、熊市不动", 回测很漂亮, 实盘等于裸做多。

三层保证, 每层都是【构造上】杜绝, 不是训完再检查:

  1. 特征里没有任何能指示时间或身份的东西 ——
     没有绝对价格(价格有时代烙印: 2016年的10块和2026年的10块不是一回事)、
     没有日期、没有股票代码、没有行业、没有指数、没有市值。
     全部是比值: 位置占比、影线占比、量比、以ATR为单位的距离。

  2. 每个特征在【当日横截面】内转成分位 (见 rank_normalize.py)。
     这一步之后, "今天全市场普涨"和"今天全市场普跌"在特征空间里
     完全同形 —— 因为分位只描述"这只票在今天所有票里排第几"。

  3. 标签是【超额】: 未来H日收益 减去 当日全市场等权均值 (见 build_labels)。
     牛市里所有票都涨, 减掉均值后"牛市"这个信息就归零了。

⚠️ 所有滚动窗口一律用 shift(1) 之后的数据算, 或只用到 t 时刻已知的量。
   涉及未来的只有标签。
"""
from __future__ import annotations

import argparse
import asyncio
from datetime import date as _date
import logging
import os

import numpy as np
import pandas as pd
from sqlalchemy import text

from app.db import engine

log = logging.getLogger("shape.feat")
OUT_DIR = "/app/data/research/shape"   # ⚠️ 必须放挂载卷, /app 下别处重建镜像就没了

# 特征窗口。最长 60, 所以每只票前 60 根不可用。
WINDOWS = (5, 10, 20, 60)


def _feats(g: pd.DataFrame) -> pd.DataFrame:
    """单只股票的形状特征。g 已按 trade_date 升序, 且是前复权价。"""
    o, h, l, c = g["o"].values, g["h"].values, g["l"].values, g["c"].values
    v, amt = g["v"].values, g["amt"].values
    n = len(c)
    out = {}

    prev_c = np.concatenate([[np.nan], c[:-1]])
    r1 = c / prev_c - 1

    # 波动率(以 t 时刻已知的收益算), 作为后面所有"距离"的标度单位
    s = pd.Series(r1)
    vol20 = s.rolling(20).std().values
    vol60 = s.rolling(60).std().values
    denom = np.where(np.isfinite(vol20) & (vol20 > 1e-6), vol20, np.nan)

    # --- 1. 归一化收益: 除以自身波动率 -> 去掉"这只票天生波动大"的影响 ---
    for w in WINDOWS:
        rw = c / np.concatenate([[np.nan] * w, c[:-w]]) - 1
        out[f"ret{w}_z"] = rw / (denom * np.sqrt(w))

    # --- 2. 收盘在N日区间中的位置: 0=最低 1=最高。纯位置, 与价位无关 ---
    for w in WINDOWS:
        hi = pd.Series(h).rolling(w).max().values
        lo = pd.Series(l).rolling(w).min().values
        rng = hi - lo
        out[f"pos{w}"] = np.where(rng > 1e-9, (c - lo) / rng, 0.5)

    # --- 3. K线自身形状: 实体/上影/下影 占全幅的比例 ---
    span = np.where((h - l) > 1e-9, h - l, np.nan)
    body = (c - o) / span
    upper = (h - np.maximum(o, c)) / span
    lower = (np.minimum(o, c) - l) / span
    out["body"] = body
    out["upper"] = upper
    out["lower"] = lower
    for w in (5, 20):
        out[f"body_ma{w}"] = pd.Series(body).rolling(w).mean().values
        out[f"upper_ma{w}"] = pd.Series(upper).rolling(w).mean().values
        out[f"lower_ma{w}"] = pd.Series(lower).rolling(w).mean().values

    # --- 4. 量: 只用比值, 不用绝对量(绝对量含市值信息) ---
    vma20 = pd.Series(v).rolling(20).mean().values
    out["vol_ratio"] = np.log1p(v / np.where(vma20 > 0, vma20, np.nan))
    ama20 = pd.Series(amt).rolling(20).mean().values
    out["amt_ratio"] = np.log1p(amt / np.where(ama20 > 0, ama20, np.nan))
    out["vol_trend"] = (pd.Series(v).rolling(5).mean().values /
                        np.where(vma20 > 0, vma20, np.nan))

    # --- 5. 波动结构: 短波动/长波动, 描述"是在收敛还是发散" ---
    vol5 = s.rolling(5).std().values
    out["vol_5_20"] = vol5 / np.where(vol20 > 1e-6, vol20, np.nan)
    out["vol_20_60"] = vol20 / np.where(vol60 > 1e-6, vol60, np.nan)

    # --- 6. 跳空(以波动率为单位) ---
    out["gap_z"] = (o / prev_c - 1) / denom

    # --- 7. 距均线的距离, 单位是 ATR -> 完全无量纲 ---
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    atr = pd.Series(tr).rolling(20).mean().values
    atr_s = np.where(atr > 1e-9, atr, np.nan)
    for w in WINDOWS:
        ma = pd.Series(c).rolling(w).mean().values
        out[f"dist_ma{w}"] = (c - ma) / atr_s

    # --- 8. 连涨/连跌: 形态的"节奏", 与价位无关 ---
    up = (r1 > 0).astype(np.int8)
    streak = np.zeros(n, dtype=np.float32)
    run = 0
    for i in range(n):
        if not np.isfinite(r1[i]):
            run = 0
        elif up[i]:
            run = run + 1 if run > 0 else 1
        else:
            run = run - 1 if run < 0 else -1
        streak[i] = run
    out["streak"] = streak

    # --- 9. 近20日里涨停/跌停的根数(比例) —— A股特有的形态信息 ---
    lim = (np.abs(r1) > 0.095).astype(np.float32)
    out["limit_rate20"] = pd.Series(lim).rolling(20).mean().values

    d = pd.DataFrame(out, index=g.index).astype(np.float32)
    d["ts_code"] = g["ts_code"].values
    d["trade_date"] = g["trade_date"].values
    d["c"] = c.astype(np.float32)
    return d


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2016-01-01")
    a = ap.parse_args()
    start = _date.fromisoformat(a.start)
    os.makedirs(OUT_DIR, exist_ok=True)

    async with engine.connect() as conn:
        codes = [r[0] for r in (await conn.execute(text(
            "select distinct ts_code from daily_candle "
            "where trade_date >= :s order by ts_code"), {"s": start})).fetchall()]
    log.info("全市场 %d 只", len(codes))

    parts = []
    for i in range(0, len(codes), 400):
        batch = codes[i:i + 400]
        async with engine.connect() as conn:
            rows = (await conn.execute(text("""
                select ts_code, trade_date,
                       open*adj_factor, high*adj_factor,
                       low*adj_factor, close*adj_factor, vol, amount
                from daily_candle
                where ts_code = any(:cs) and trade_date >= :s
                order by ts_code, trade_date
            """), {"cs": batch, "s": start})).fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "o", "h",
                                         "l", "c", "v", "amt"])
        for col in ("o", "h", "l", "c", "v", "amt"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = df.dropna(subset=["o", "h", "l", "c"])
        # ⚠️ 不用 groupby.apply: pandas 3.0 里分组键不再传进函数, 会 KeyError。
        #    显式遍历分组, 顺带也好控制内存。
        for _, g in df.groupby("ts_code", sort=False):
            if len(g) > 60:
                parts.append(_feats(g.reset_index(drop=True)))
        log.info("  %d/%d", min(i + 400, len(codes)), len(codes))

    feat = pd.concat(parts, ignore_index=True)
    feat = feat.dropna(subset=["pos60"])          # 前60根不可用
    p = f"{OUT_DIR}/features.parquet"
    feat.to_parquet(p, index=False)
    log.info("特征 %s 行 × %s 列 -> %s", f"{len(feat):,}", feat.shape[1], p)
    log.info("特征名: %s", [c for c in feat.columns
                            if c not in ("ts_code", "trade_date", "c")])
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
