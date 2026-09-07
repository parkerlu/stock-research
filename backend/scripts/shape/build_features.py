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


def _feats(g: pd.DataFrame, mkt: np.ndarray | None = None) -> pd.DataFrame:
    """单只股票的形状特征。g 已按 trade_date 升序, 且是前复权价。
    mkt: 与 g 逐行对齐的【当日全市场等权收益】, 只用于算 beta/corr(比值, 无量纲)。
    传 None(生产打分路径)时 corr_mkt60/beta60 置 NaN —— 生产模型不用这两列。"""
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

    # ================= 量能机制 (2026-09-07 新增) =================
    # 全部是比值/分位/相关, 无量纲 —— 不引入市值和绝对活跃度。
    vs = pd.Series(v.astype(np.float64))
    vma5 = vs.rolling(5).mean().values
    vma60 = vs.rolling(60).mean().values
    up1 = r1 > 0

    # 1. 量在自身过去120日中的分位: 现在是活跃还是冷清
    out["vol_pct120"] = vs.rolling(120).rank(pct=True).values

    # 2/3. 放量上涨 与 缩量下跌 的占比 —— 量价配合是否健康。
    #      缩量下跌是"回调不恐慌"的标志, 与放量下跌完全两回事。
    big = v > np.where(vma20 > 0, vma20, np.inf)
    out["up_vol_rate20"] = pd.Series((up1 & big).astype(np.float32)).rolling(20).mean().values
    out["dn_shrink_rate20"] = pd.Series(((~up1) & (~big)).astype(np.float32)).rolling(20).mean().values

    # 4. 量价相关: 涨放量/跌缩量 -> 正相关
    dv = pd.Series(np.concatenate([[np.nan], np.diff(v.astype(np.float64))]))
    out["pv_corr20"] = pd.Series(r1).rolling(20).corr(dv).values

    # 5. OBV 斜率, 用均量归一化 -> 去掉规模
    obv = np.nancumsum(np.where(np.isfinite(r1), np.sign(r1) * v, 0.0))
    obv_s = pd.Series(obv)
    out["obv_slope20"] = ((obv_s - obv_s.shift(20)) /
                          np.where(vma20 > 0, vma20 * 20, np.nan)).values

    # 6. 主动买盘代理: 收盘落在当日区间的哪个位置, 用量加权
    clv = np.where((h - l) > 1e-9, ((c - l) - (h - c)) / (h - l), 0.0)
    for w in (5, 20):
        num = pd.Series(clv * v).rolling(w).sum().values
        den = vs.rolling(w).sum().values
        out[f"mfi{w}"] = num / np.where(den > 0, den, np.nan)

    # 7. 量堆: 短期均量 / 长期均量, >1 = 在积聚
    out["vol_pile"] = vma5 / np.where(vma60 > 0, vma60, np.nan)

    # 8. 地量: 近5日最低量 / 60日均量, 越小说明抛压越枯竭
    out["vol_dry"] = vs.rolling(5).min().values / np.where(vma60 > 0, vma60, np.nan)

    # 9. 突破放量: 是否创20日新高, 以及当日量比(两者相乘 -> 只有突破日非零)
    hh20 = pd.Series(h).rolling(20).max().shift(1).values
    brk = (c > hh20).astype(np.float32)
    out["brk_vol"] = brk * (v / np.where(vma20 > 0, vma20, np.nan))

    # 10. 量能收敛: 量的短期波动 / 长期波动
    out["vol_std_5_20"] = (vs.rolling(5).std().values /
                           np.where(vs.rolling(20).std().values > 1e-9,
                                    vs.rolling(20).std().values, np.nan))
    # ==============================================================

    # --- 9. 近20日里涨停/跌停的根数(比例) —— A股特有的形态信息 ---
    lim = (np.abs(r1) > 0.095).astype(np.float32)
    out["limit_rate20"] = pd.Series(lim).rolling(20).mean().values

    # ================= v3 新增 (2026-09-07): 四组新特征 =================
    # 全部仍是比值/分位/以波动率为单位的距离 —— 无绝对价格/日期/身份。
    from numpy.lib.stride_tricks import sliding_window_view as _swv

    # ---- L 组: 长周期形态 (120/250日) ----
    vol250 = s.rolling(250).std().values
    for w in (120, 250):
        if n > w:
            rw = c / np.concatenate([[np.nan] * w, c[:-w]]) - 1
            out[f"ret{w}_z"] = rw / (denom * np.sqrt(w))
        else:
            out[f"ret{w}_z"] = np.full(n, np.nan, dtype=np.float32)
        hi_w = pd.Series(h).rolling(w).max().values
        lo_w = pd.Series(l).rolling(w).min().values
        rng_w = hi_w - lo_w
        out[f"pos{w}"] = np.where(rng_w > 1e-9, (c - lo_w) / rng_w, 0.5)
        ma_w = pd.Series(c).rolling(w).mean().values
        out[f"dist_ma{w}"] = (c - ma_w) / atr_s
    # 距 250日最高/最低点已过多少天(占窗口比例): 0=今天刚创, 1=一年前
    dhi = np.full(n, np.nan, dtype=np.float32)
    dlo = np.full(n, np.nan, dtype=np.float32)
    if n >= 250:
        Wh = _swv(h, 250)
        Wl = _swv(l, 250)
        dhi[249:] = (249 - Wh.argmax(1)) / 250.0
        dlo[249:] = (249 - Wl.argmin(1)) / 250.0
    out["dhi250"] = dhi
    out["dlo250"] = dlo
    out["vol_60_250"] = vol60 / np.where(vol250 > 1e-6, vol250, np.nan)

    # ---- P 组: 路径/形状 ----
    absr = pd.Series(np.abs(r1))
    for w in (20, 60):
        rw_abs = np.abs(c / np.concatenate([[np.nan] * w, c[:-w]]) - 1)
        sm = absr.rolling(w).sum().values
        out[f"er{w}"] = rw_abs / np.where(sm > 1e-9, sm, np.nan)   # 趋势效率
        out[f"up_rate{w}"] = pd.Series(up1.astype(np.float32)).rolling(w).mean().values
    out["skew60"] = s.rolling(60).skew().values
    out["spike20"] = absr.rolling(20).max().values / denom
    out["clv_std20"] = pd.Series(clv).rolling(20).std().values
    rngc = pd.Series((h - l) / np.where(c > 1e-9, c, np.nan))
    a20 = rngc.rolling(20).mean().values
    out["amp_5_20"] = rngc.rolling(5).mean().values / np.where(a20 > 1e-9, a20, np.nan)
    gap_abs = np.abs(o / prev_c - 1)
    out["gapfreq20"] = pd.Series((gap_abs > vol20).astype(np.float32)).rolling(20).mean().values

    # ---- D 组: 量价背离 / 流动性 ----
    cs_path = pd.Series(np.nancumsum(np.where(np.isfinite(r1), r1, 0.0)))
    out["pv_div20"] = cs_path.rolling(20).corr(obv_s).values     # 价格路径 vs OBV路径
    ama60 = pd.Series(amt).rolling(60).mean().values
    rel_amt = amt / np.where(ama60 > 0, ama60, np.nan)
    out["illiq20"] = pd.Series(np.abs(r1) / np.where(rel_amt > 1e-9, rel_amt, np.nan)
                               ).rolling(20).mean().values       # 无量纲 Amihud
    vsum20 = vs.rolling(20).sum().values
    out["vol_conc20"] = vs.rolling(20).max().values / np.where(vsum20 > 0, vsum20, np.nan)
    vt = out["vol_trend"]
    vt5 = np.concatenate([[np.nan] * 5, vt[:-5]])
    out["vol_accel"] = vt / np.where(vt5 > 1e-9, vt5, np.nan)    # 换手加速度

    # ---- M 组: 与等权市场的相关结构 (不引入指数水平, 只有相关/比值) ----
    if mkt is not None:
        m_s = pd.Series(mkt)
        out["corr_mkt60"] = s.rolling(60).corr(m_s).values
        mvar = m_s.rolling(60).var().values
        out["beta60"] = s.rolling(60).cov(m_s).values / np.where(mvar > 1e-10, mvar, np.nan)
    else:
        out["corr_mkt60"] = np.full(n, np.nan, dtype=np.float32)
        out["beta60"] = np.full(n, np.nan, dtype=np.float32)
    # ====================================================================

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
    # v3: 250日窗口需要暖机 —— 往前多取 550 个日历日, 但只输出 start 之后的行
    import datetime as _dtm
    warm = start - _dtm.timedelta(days=550)
    v3_dir = f"{OUT_DIR}/features_v3"
    os.makedirs(v3_dir, exist_ok=True)
    for f_ in os.listdir(v3_dir):                 # 重跑先清空旧 part
        os.remove(os.path.join(v3_dir, f_))

    async with engine.connect() as conn:
        codes = [r[0] for r in (await conn.execute(text(
            "select distinct ts_code from daily_candle "
            "where trade_date >= :s order by ts_code"), {"s": start})).fetchall()]
        # 全市场等权日收益(供 beta/corr 用), 服务器端算好, 无前视
        mrows = (await conn.execute(text("""
            select trade_date, avg(r) from (
              select trade_date,
                     close*adj_factor / nullif(lag(close*adj_factor) over
                       (partition by ts_code order by trade_date), 0) - 1 as r
              from daily_candle where trade_date >= :w
            ) s where r is not null and r > -0.5 and r < 0.5
            group by trade_date order by trade_date
        """), {"w": warm})).fetchall()
    mkt_map = pd.Series([float(r[1]) for r in mrows],
                        index=[r[0] for r in mrows], dtype="float64")
    log.info("全市场 %d 只, 市场收益序列 %d 天 (%s~%s)",
             len(codes), len(mkt_map), mkt_map.index.min(), mkt_map.index.max())

    total = 0
    ncols = None
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
            """), {"cs": batch, "s": warm})).fetchall()
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["ts_code", "trade_date", "o", "h",
                                         "l", "c", "v", "amt"])
        del rows
        for col in ("o", "h", "l", "c", "v", "amt"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
        df = df.dropna(subset=["o", "h", "l", "c"])
        # ⚠️ 不用 groupby.apply: pandas 3.0 里分组键不再传进函数, 会 KeyError。
        #    显式遍历分组, 顺带也好控制内存。
        parts = []
        for _, g in df.groupby("ts_code", sort=False):
            if len(g) > 60:
                g = g.reset_index(drop=True)
                mkt = mkt_map.reindex(g["trade_date"]).to_numpy("float64")
                parts.append(_feats(g, mkt))
        del df
        if not parts:
            continue
        feat = pd.concat(parts, ignore_index=True)
        del parts
        feat = feat.dropna(subset=["pos60"])            # 前60根不可用
        feat = feat[feat["trade_date"] >= start]        # 暖机段不输出
        feat.to_parquet(f"{v3_dir}/part-{i // 400:04d}.parquet", index=False)
        total += len(feat)
        ncols = feat.shape[1]
        if i // 400 == 0:
            log.info("特征名: %s", [c for c in feat.columns
                                    if c not in ("ts_code", "trade_date", "c")])
        del feat
        log.info("  %d/%d  累计 %s 行", min(i + 400, len(codes)), len(codes),
                 f"{total:,}")

    log.info("完成: %s 行 × %s 列 -> %s/", f"{total:,}", ncols, v3_dir)
    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
