"""形态模型 v7 · 特征扩充实验 —— 标签不动(y_0.15_0.08 三重障碍+可成交), 只加特征。

问题: v5 样本外分层单调但 alpha 太小(Top5% +0.45pp), 组合比值 0.24。
假设: 瓶颈在特征信息量。本脚本在【同一份样本、同一个标签】上对比六组特征:

    base      v2 的 40 个特征(在 v3 数据上重测, 作为公平基线)
    base+L    + 长周期形态 9 个 (120/250日: ret_z/pos/dist_ma/距新高新低天数/波动比)
    base+P    + 路径形状 9 个 (趋势效率/上涨日占比/偏度/尖峰/日内位置离散/振幅收敛/跳空频率)
    base+DM   + 量价背离与流动性 4 个 + 市场相关结构 2 个 (beta/corr, 无指数水平)
    allprice  + 上面全部 (64 个纯价量特征)
    all+S     + 信号表 3 个 (pump prob / maimai buy score / breakout prob)
              ⚠️ S 组覆盖 2017/2019 起且逐年变密, 有"日历痕迹"风险, 单列一组观察

判据(用户定的, 不许改):
    样本外(2021-2026) Top5% 超额 + 逐年负年数(逐年均匀比总量重要)
    去记忆审计 |corr| < 0.10

⚠️ 内存铁律与 v5 相同(8G 容器, OOM 过五次): ts_code 转整数码、逐列读 parquet、
   不整表 merge、按年落盘、datetime 用 datetime64[D] 转 int(pandas 3.0 的
   astype("int64") 陷阱)。
"""
from __future__ import annotations

import argparse
import gc
import logging
import os

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger("shape.v7")
DIR = "/app/data/research/shape"
FEAT_DIR = f"{DIR}/features_v3"
META = ("ts_code", "trade_date", "c")
TAG = "y_0.15_0.08"

GROUP_L = ["ret120_z", "ret250_z", "pos120", "pos250", "dist_ma120",
           "dist_ma250", "dhi250", "dlo250", "vol_60_250"]
GROUP_P = ["er20", "er60", "up_rate20", "up_rate60", "skew60", "spike20",
           "clv_std20", "amp_5_20", "gapfreq20"]
GROUP_D = ["pv_div20", "illiq20", "vol_conc20", "vol_accel"]
GROUP_M = ["corr_mkt60", "beta60"]
GROUP_S = ["sig_pump", "sig_maimai", "sig_brk"]


def barrier_labels(px: pd.DataFrame, h: int, u: float, d_: float) -> pd.DataFrame:
    """与 train_v5 完全一致: 从次日开盘可成交价起算的三重障碍落袋超额,
    次日一字涨停样本剔除。只算一组障碍 (u, d_)。"""
    px = px.sort_values(["cid", "day"]).reset_index(drop=True)
    px["r"] = px.groupby("cid", sort=False)["c"].pct_change().astype(np.float32)
    mkt = px.groupby("day")["r"].transform("mean")
    px["e"] = (px["r"] - mkt).astype(np.float32)
    mkt_by_day = px.groupby("day")["r"].mean()

    from numpy.lib.stride_tricks import sliding_window_view as swv
    outs = []
    for _, g in px.groupby("cid", sort=False, observed=True):
        e = np.nan_to_num(g["e"].to_numpy(np.float32))
        c = g["c"].to_numpy(np.float32)
        o = g["o"].to_numpy(np.float32)
        lo = g["lo"].to_numpy(np.float32)
        lim = g["lim"].to_numpy(np.float32)
        n = len(e)
        if n <= h + 2:
            continue
        cs = np.concatenate([[0.0], np.cumsum(e)]).astype(np.float32)
        m = n - h - 1
        if m <= 0:
            continue
        t = np.arange(m)
        up_lim = c[t] * (1.0 + lim[t + 1] - 0.005)
        fillable = lo[t + 1] < up_lim
        gap = o[t + 1] / np.maximum(c[t], 1e-6) - 1.0
        gap_ex = gap - mkt_by_day.reindex(g["day"].to_numpy()[t + 1]).to_numpy(np.float32)
        gap_ex = np.nan_to_num(gap_ex)
        W = swv(cs[2:], h)[:m] - cs[1:1 + m, None] - gap_ex[:, None]
        up = W >= u
        dn = W <= -d_
        iu = np.where(up.any(1), up.argmax(1), h + 1)
        idn = np.where(dn.any(1), dn.argmax(1), h + 1)
        realized = np.where(idn <= iu, -d_,
                   np.where(iu < idn, u, W[:, -1])).astype(np.float32)
        both = (iu > h) & (idn > h)
        realized[both] = W[both, -1]
        outs.append(pd.DataFrame({
            "cid": g["cid"].to_numpy()[t], "day": g["day"].to_numpy()[t],
            "fwd": W[:, -1].astype(np.float32), "fillable": fillable,
            TAG: realized}))
    d = pd.concat(outs, ignore_index=True)
    keep = d["fillable"].to_numpy()
    log.info("  可成交样本 %s / %s (剔除一字涨停 %.2f%%)",
             f"{int(keep.sum()):,}", f"{len(d):,}", (1 - keep.mean()) * 100)
    return d[keep].drop(columns=["fillable"]).reset_index(drop=True)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--tr1", default="2020-12-31")
    ap.add_argument("--sample", type=int, default=3_000_000)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    part_files = sorted(os.path.join(FEAT_DIR, f) for f in os.listdir(FEAT_DIR)
                        if f.endswith(".parquet"))
    assert part_files, "features_v3 为空, 先跑 build_features"
    schema = pq.ParquetFile(part_files[0]).schema_arrow.names

    def read_cols(cs: list[str]) -> pd.DataFrame:
        """按固定文件顺序逐 part 读 → 行序在多次调用之间严格一致。"""
        return pd.concat([pq.read_table(f, columns=cs).to_pandas()
                          for f in part_files], ignore_index=True)

    price_cols = [c for c in schema if c not in META]
    new_price = GROUP_L + GROUP_P + GROUP_D + GROUP_M
    base_cols = [c for c in price_cols if c not in new_price]
    log.info("特征: base %d + 新价量 %d + 信号 %d", len(base_cols),
             len(new_price), len(GROUP_S))
    assert len(base_cols) == 40, f"base 应为 40, 实际 {len(base_cols)}"

    keys = read_cols(["ts_code", "trade_date", "c"])
    cat = pd.Categorical(keys["ts_code"])
    code_index = pd.Index(cat.categories)
    cid = cat.codes.astype(np.int32)
    dates = pd.to_datetime(keys["trade_date"])
    # ⚠️ pandas 3.0: astype("int64") 会返回全 0, 必须走 datetime64[D]
    day = dates.to_numpy().astype("datetime64[D]").astype(np.int32)
    assert day.min() != day.max(), "day 退化成常数"
    log.info("  %s 行, %s ~ %s", f"{len(cid):,}", dates.min().date(),
             dates.max().date())
    closes = keys["c"].to_numpy(np.float32)
    del keys, cat, dates
    gc.collect()
    n = len(cid)

    # ---------- 次日开盘/最低 (判可成交), 按年分批取 ----------
    import asyncio as _aio
    import datetime as _dt
    from sqlalchemy import text as _text
    from app.db import engine as _eng

    cmap = {c: i for i, c in enumerate(code_index)}
    EPOCH = _dt.date(1970, 1, 1)

    async def _fetch_all():
        """⚠️ OHLC 与信号表必须在【同一个 asyncio.run】里取 —— 引擎连接池里的
        连接绑定事件循环, 第二次 asyncio.run 复用旧连接会炸 'different loop'。"""
        ks, os_, ls_ = [], [], []
        async with _eng.connect() as conn:
            for yy in range(2016, 2027):
                rr = (await conn.execute(_text(
                    "select ts_code, trade_date, open*adj_factor, low*adj_factor "
                    "from daily_candle where trade_date >= :a and trade_date < :b"),
                    {"a": _dt.date(yy, 1, 1), "b": _dt.date(yy + 1, 1, 1)})).fetchall()
                if not rr:
                    continue
                ks.append(np.array([cmap[r[0]] * 100000 + (r[1] - EPOCH).days
                                    for r in rr if r[0] in cmap], dtype=np.int64))
                os_.append(np.array([float(r[2]) for r in rr if r[0] in cmap], np.float32))
                ls_.append(np.array([float(r[3]) for r in rr if r[0] in cmap], np.float32))
                del rr
            qs = {
                "sig_pump": "select ts_code, trade_date, prob from pump_signal "
                            "where trade_date >= :a and trade_date < :b",
                "sig_maimai": "select ts_code, trade_date, score from maimai_signal "
                              "where side = 'buy' and trade_date >= :a and trade_date < :b",
                "sig_brk": "select ts_code, trade_date, prob from breakout_signal "
                           "where trade_date >= :a and trade_date < :b",
            }
            sig = {}
            for name, q in qs.items():
                sks, svs = [], []
                for yy in range(2016, 2027):
                    rr = (await conn.execute(_text(q),
                          {"a": _dt.date(yy, 1, 1), "b": _dt.date(yy + 1, 1, 1)})).fetchall()
                    if not rr:
                        continue
                    sks.append(np.array([cmap[r[0]] * 100000 + (r[1] - EPOCH).days
                                         for r in rr if r[0] in cmap], dtype=np.int64))
                    svs.append(np.array([float(r[2]) for r in rr if r[0] in cmap],
                                        np.float32))
                    del rr
                sig[name] = (np.concatenate(sks), np.concatenate(svs))
        return np.concatenate(ks), np.concatenate(os_), np.concatenate(ls_), sig

    kk, oo, ll, sig_raw = _aio.run(_fetch_all())
    key_f = cid.astype(np.int64) * 100000 + day
    srt = np.argsort(kk)
    ks_ = kk[srt]
    pos_ = np.minimum(np.searchsorted(ks_, key_f), len(ks_) - 1)
    hit = ks_[pos_] == key_f
    opens = np.where(hit, oo[srt][pos_], np.nan).astype(np.float32)
    lows = np.where(hit, ll[srt][pos_], np.nan).astype(np.float32)
    del kk, oo, ll, srt, ks_, pos_, hit
    gc.collect()

    lims = np.array([0.20 if code_index[i][:3] in ("300", "301", "688") else 0.10
                     for i in range(len(code_index))], dtype=np.float32)[cid]
    log.info("三重障碍标签 h=%d (0.15, 0.08) ...", a.h)
    lab = barrier_labels(pd.DataFrame({"cid": cid, "day": day, "c": closes,
                                       "o": opens, "lo": lows, "lim": lims}),
                         a.h, 0.15, 0.08)
    del closes, opens, lows, lims
    gc.collect()

    # ---------- 标签对齐到特征行 ----------
    key_all = cid.astype(np.int64) * 100000 + day
    order = np.argsort(key_all)
    key_sorted = key_all[order]
    lab_key = lab["cid"].to_numpy(np.int64) * 100000 + lab["day"].to_numpy()
    pos = np.searchsorted(key_sorted, lab_key)
    ok = (pos < len(key_sorted)) & (key_sorted[np.minimum(pos, len(key_sorted) - 1)] == lab_key)
    rows = order[pos[ok]]
    y_lab = lab[TAG].to_numpy(np.float32)[ok]
    fwd = lab["fwd"].to_numpy(np.float32)[ok]
    del lab, order, key_sorted, lab_key, pos
    gc.collect()
    log.info("  标签对齐 %s 行", f"{len(rows):,}")

    # ---------- 信号表特征: 复合键对齐到全量行, NaN=当天无信号 ----------
    sig_full = {}
    for name, (sk, sv) in sig_raw.items():
        srt2 = np.argsort(sk)
        sk_s = sk[srt2]
        p2 = np.minimum(np.searchsorted(sk_s, key_f), len(sk_s) - 1)
        hit2 = sk_s[p2] == key_f
        sig_full[name] = np.where(hit2, sv[srt2][p2], np.nan).astype(np.float32)
        log.info("  %s 覆盖 %.2f%% 行", name, hit2.mean() * 100)
        del srt2, sk_s, p2, hit2
    del sig_raw, key_f
    gc.collect()

    # ---------- 横截面分位 → 只保留有标签的行 ----------
    all_cols = price_cols + GROUP_S
    log.info("横截面分位(逐列, %d 列) ...", len(all_cols))
    R = np.empty((len(rows), len(all_cols)), dtype=np.float32)
    daysr = pd.Series(day)
    for j, c in enumerate(all_cols):
        if c in GROUP_S:
            v = sig_full[c]
        else:
            v = read_cols([c])[c].to_numpy(np.float32)
        R[:, j] = pd.Series(v).groupby(daysr, sort=False).rank(pct=True) \
                    .to_numpy(np.float32)[rows]
        del v
        if j % 10 == 0:
            log.info("    %d/%d", j, len(all_cols))
            gc.collect()
    del sig_full, daysr
    gc.collect()

    # ---------- 按年落盘 ----------
    tmp = f"{DIR}/_yr7"
    os.makedirs(tmp, exist_ok=True)
    yr_all = (pd.to_datetime(pd.Series(day[rows]), unit="D").dt.year).to_numpy()
    for yv in np.unique(yr_all):
        sel = yr_all == yv
        d = pd.DataFrame(R[sel], columns=all_cols)
        d["day"] = day[rows[sel]]
        d["cid"] = cid[rows[sel]]
        d[TAG] = y_lab[sel]
        d["fwd"] = fwd[sel]
        d.to_parquet(f"{tmp}/{yv}.parquet", index=False)
        del d
    years = sorted(int(v) for v in np.unique(yr_all))
    del R, rows, y_lab, fwd, yr_all, cid, day
    gc.collect()
    log.info("  按年落盘完成: %s", years)

    tr_years = [v for v in years if v <= pd.Timestamp(a.tr1).year]
    te_years = [v for v in years if v > pd.Timestamp(a.tr1).year]
    per = max(1, a.sample // max(len(tr_years), 1))
    trs = []
    for v in tr_years:
        d = pd.read_parquet(f"{tmp}/{v}.parquet")
        trs.append(d.sample(min(per, len(d)), random_state=42))
        del d
    tr = pd.concat(trs, ignore_index=True)
    del trs
    gc.collect()
    log.info("训练 %s 行 (%s) / 测试年 %s", f"{len(tr):,}", tr_years, te_years)

    CONFIGS = [
        ("base", base_cols),
        ("base+L", base_cols + GROUP_L),
        ("base+P", base_cols + GROUP_P),
        ("base+DM", base_cols + GROUP_D + GROUP_M),
        ("allprice", base_cols + new_price),
        ("all+S", base_cols + new_price + GROUP_S),
    ]
    tr_y = tr[TAG].to_numpy(np.float32)
    log.info("=" * 78)
    for name, cc in CONFIGS:
        m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=50, reg_lambda=2.0,
                             tree_method="hist", n_jobs=8, random_state=42)
        m.fit(tr[cc], tr_y)
        m.save_model(f"{DIR}/shape_v7_{name.replace('+', '_')}.json")
        imp = dict(zip(cc, m.feature_importances_))
        new_share = sum(v for k, v in imp.items() if k not in base_cols)
        top_new = sorted(((k, v) for k, v in imp.items() if k not in base_cols),
                         key=lambda x: -x[1])[:3]

        acc = {k: [0, 0.0] for k in ("Top1%", "Top5%", "Bot20%")}
        yearly, dl = [], []
        for v in te_years:
            d = pd.read_parquet(f"{tmp}/{v}.parquet")
            sc = np.empty(len(d), np.float32)
            for i in range(0, len(d), 500_000):
                sc[i:i + 500_000] = m.predict(d[cc].iloc[i:i + 500_000]).astype(np.float32)
            d = d[["day", "fwd", TAG]].assign(score=sc)
            d["rk"] = d.groupby("day")["score"].rank(pct=True)
            for k, (lo, hi) in {"Top1%": (0.99, 1.01), "Top5%": (0.95, 1.01),
                                "Bot20%": (0.0, 0.20)}.items():
                g = d[(d.rk >= lo) & (d.rk < hi)]
                acc[k][0] += len(g)
                acc[k][1] += float(g[TAG].sum())
            g5 = d[d.rk >= 0.95]
            yearly.append(float(g5[TAG].mean()) * 100)
            dl.append(d.groupby("day").agg(avg=("score", "mean"),
                                           m=("fwd", "mean")).reset_index())
            del d, sc
            gc.collect()
        jd = pd.concat(dl, ignore_index=True)
        rcorr = jd["avg"].corr(jd["m"])
        t1 = acc["Top1%"][1] / max(acc["Top1%"][0], 1) * 100
        t5 = acc["Top5%"][1] / max(acc["Top5%"][0], 1) * 100
        b20 = acc["Bot20%"][1] / max(acc["Bot20%"][0], 1) * 100
        log.info("%-9s Top1%% %+5.2f  Top5%% %+5.2f  Bot20%% %+5.2f | 负年 %d "
                 "最小 %+5.2f | 审计 %+.3f | 新特征占比 %4.1f%% | top新: %s",
                 name, t1, t5, b20, sum(1 for x in yearly if x < 0),
                 min(yearly), rcorr, new_share * 100,
                 " ".join(f"{k}:{v:.3f}" for k, v in top_new))
        log.info("%-9s   逐年: %s", "",
                 "  ".join(f"{y}:{v:+.2f}" for y, v in zip(te_years, yearly)))
        del jd, dl, m
        gc.collect()
    log.info("=" * 78)


if __name__ == "__main__":
    main()
