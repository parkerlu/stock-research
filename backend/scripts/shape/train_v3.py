"""形态模型 v2 · 路径型奖励 + 量能。

与 v1 的唯一区别在【标签】—— v1 用终点收益, 奖励"冲一下"; 结果模型学成
纯动量, 组合比值 0.12, 且 alpha 全集中在 2021/2026 两年。

v2 的奖励要求"买进去 + 待得住":

    对每个 (票, t), 看 t+1..t+H 每天的累计超额 cum_s
      A = mean(cum_s)          净值曲线整体在水上多高 -> 奖励涨得快并保持
      B = -min(0, min cum_s)   最深水下多少           -> 惩罚挖坑
      奖励 = A - λ·B

    先跌30%再涨回原点: v1 得 0(终点相同), v2 重罚
    第1天涨10%后横盘:  v1 得 +10%,        v2 高分
    横盘19天末日涨10%: v1 得 +10%,        v2 低分

⚠️ 去记忆三层与 v1 完全一致, 不动:
   特征无量纲/无身份 · 每日横截面转分位 · 超额(减当日全市场均值)。
⚠️ 新增一条硬判据: 逐年必须均匀。v1 的 alpha 全在两年里, 那不算过。
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger("shape.v2")
DIR = "/app/data/research/shape"
META = ("ts_code", "trade_date", "c")


def cs_rank_inplace(df: pd.DataFrame, cols: list[str]) -> None:
    """⚠️ 必须原地改。另建一个 DataFrame 会让 1170万行×40列 的内存翻倍,
    容器直接 OOM(实测)。逐列排序后写回原列, 峰值只多一列。"""
    g = df.groupby("trade_date", sort=False)
    for i, c in enumerate(cols):
        df[c] = g[c].rank(pct=True).astype(np.float32)
        if i % 10 == 0:
            log.info("    分位 %d/%d", i, len(cols))


def barrier_labels(px: pd.DataFrame, h: int, bars: list[tuple]) -> pd.DataFrame:
    """三重障碍标签 —— 标签 = 在给定出场规则下【真正落袋】的超额收益。

    ⚠️ 这是 2026-09-07 第三版。前两版的教训:
       v1 用终点收益 -> 模型学成纯动量, 组合 0.12
       v2 用 A - λ·MAE(只罚亏不奖赚) -> 模型去挑不动的票, Top5% 掉到 +0.12
       v2.5 加 MFE(奖励窗口内最高点) -> 更差, Top5% 掉到 -1.05
             原因: MFE 是【吃不到】的利润。固定持有到期按收盘卖, 那个高点
             早还回去了; 重奖 MFE 等于挑"冲上去再砸下来"的彩票票。

    结论: 奖励必须和出场规则对齐, 只奖励【可兑现】的利润。
       上障碍 +u 先碰 -> 实得 +u      (止盈落袋)
       下障碍 -d 先碰 -> 实得 -d      (止损)
       都没碰       -> 实得 cum_h    (到期收盘)
    回测可以逐字执行同一套规则, 所以标签里的收益是真能拿到的。

    cum 用【当日超额】累计(减全市场当日等权均值), 去 regime 保证不变。
    """
    px = px.sort_values(["cid", "day"]).reset_index(drop=True)
    px["r"] = px.groupby("cid", sort=False)["c"].pct_change().astype(np.float32)
    mkt = px.groupby("day")["r"].transform("mean")
    px["e"] = (px["r"] - mkt).astype(np.float32)

    from numpy.lib.stride_tricks import sliding_window_view as swv
    outs = []
    for _, g in px.groupby("cid", sort=False, observed=True):
        e = np.nan_to_num(g["e"].to_numpy(np.float32))
        n = len(e)
        if n <= h + 1:
            continue
        cs = np.concatenate([[0.0], np.cumsum(e)]).astype(np.float32)
        # W[t, s] = 从 t 买入后第 s+1 步的累计超额, s=0..h-1
        W = swv(cs[2:], h)[:max(n - h, 0)] - cs[1:1 + max(n - h, 0), None]
        m = W.shape[0]
        rec = {"cid": g["cid"].to_numpy()[:m], "day": g["day"].to_numpy()[:m],
               "fwd": W[:, -1].astype(np.float32)}
        for u, d in bars:
            up = W >= u
            dn = W <= -d
            iu = np.where(up.any(1), up.argmax(1), h + 1)
            idn = np.where(dn.any(1), dn.argmax(1), h + 1)
            # 同一根同时触及两边时按最坏处理(先止损)
            realized = np.where(idn <= iu, -d,
                       np.where(iu < idn, u, W[:, -1])).astype(np.float32)
            both = (iu > h) & (idn > h)
            realized[both] = W[both, -1]
            rec[f"y_{u:.2f}_{d:.2f}"] = realized
        outs.append(pd.DataFrame(rec))
    return pd.concat(outs, ignore_index=True)


def audit_regime(te: pd.DataFrame) -> None:
    daily = te.groupby("trade_date")["score"].mean().rename("avg")
    mkt = te.groupby("trade_date")["fwd"].mean().rename("m")
    j = pd.concat([daily, mkt], axis=1).dropna()
    r = j["avg"].corr(j["m"])
    log.info("【去记忆审计】每日均分 vs 当日市场超额  相关 %+.4f   均分std %.5f",
             r, j["avg"].std())
    log.info("               判定: %s",
             "通过" if abs(r) < 0.10 else "⚠️ 失败 —— 仍在押方向")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20)
    ap.add_argument("--lam", type=float, default=1.0, help="挖坑惩罚权重")
    ap.add_argument("--tr1", default="2020-12-31")
    ap.add_argument("--sample", type=int, default=3_000_000)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    import gc
    path = f"{DIR}/features_v2.parquet"
    schema = pq.ParquetFile(path).schema_arrow.names
    cols = [c for c in schema if c not in META]
    log.info("读键列 ... (%d 个特征, 含量能 11 个)", len(cols))

    # ⚠️ 内存是这个脚本最大的敌人。三条铁律(前面各撞过一次 OOM):
    #    1) ts_code 不要以字符串形态留在内存 —— 1180万个 str 对象 1~2G
    #    2) 不要反复往宽表上 df[c]=... 赋值 —— 每次触发 BlockManager 整表
    #       复制(2G), 跑到第30列必死
    #    3) 不要整表 merge —— 同样是整表复制
    #    做法: 逐列从 parquet 读, 排完名写进预分配的 float32 矩阵。
    keys = pq.read_table(path, columns=["ts_code", "trade_date", "c"]).to_pandas()
    cat = pd.Categorical(keys["ts_code"])
    code_index = pd.Index(cat.categories)
    cid = cat.codes.astype(np.int32)
    dates = pd.to_datetime(keys["trade_date"])
    # ⚠️ 不能用 dates.astype("int64")//86400e9 —— pandas 3.0 上返回全 0,
    #    会让"当日市场均值"变成"全样本均值", 标签直接作废且不报错。
    day = dates.to_numpy().astype("datetime64[D]").astype(np.int32)
    assert day.min() != day.max(), "day 退化成常数, 日期换算又错了"
    log.info("  日期索引 %d ~ %d (%s ~ %s)", day.min(), day.max(),
             dates.min().date(), dates.max().date())
    closes = keys["c"].to_numpy(np.float32)
    del keys, cat
    gc.collect()
    n = len(cid)
    log.info("  %s 行", f"{n:,}")

    log.info("三重障碍标签 h=%d, 障碍 %s ...", a.h, BARS)
    a_h, a_tr1, a_sample = a.h, a.tr1, a.sample
    BARS = [(0.08, 0.05), (0.12, 0.06), (0.15, 0.08), (0.20, 0.10)]
    lab = barrier_labels(pd.DataFrame({"cid": cid, "day": day, "c": closes}),
                         a.h, BARS)
    del closes
    gc.collect()
    log.info("  标签 %s 行", f"{len(lab):,}")

    log.info("横截面分位标准化(逐列) ...")
    R = np.empty((n, len(cols)), dtype=np.float32)
    daysr = pd.Series(day)
    for j, c in enumerate(cols):
        v = pq.read_table(path, columns=[c]).to_pandas()[c].to_numpy(np.float32)
        R[:, j] = pd.Series(v).groupby(daysr, sort=False).rank(pct=True).to_numpy(np.float32)
        del v
        if j % 10 == 0:
            log.info("    %d/%d", j, len(cols))
            gc.collect()

    # 标签按 (cid, day) 对齐到 R 的行 —— 用一维复合键做 searchsorted, 不 merge
    key_all = cid.astype(np.int64) * 100000 + day
    order = np.argsort(key_all)
    key_sorted = key_all[order]
    lab_key = lab["cid"].to_numpy(np.int64) * 100000 + lab["day"].to_numpy()
    pos = np.searchsorted(key_sorted, lab_key)
    ok = (pos < len(key_sorted)) & (key_sorted[np.minimum(pos, len(key_sorted) - 1)] == lab_key)
    rows = order[pos[ok]]
    ycols = [c for c in lab.columns if c.startswith("y_")]
    YS = {c: lab[c].to_numpy(np.float32)[ok] for c in ycols}
    fwd = lab["fwd"].to_numpy(np.float32)[ok]
    del lab, key_all, order, key_sorted, lab_key, pos
    gc.collect()

    # ⚠️ 到这里为止内存已经很紧。绝不能把 1170万×44 的整表留在内存里做
    #    切分/预测 —— 前面为此 OOM 了四次。改成【按年落盘, 逐年处理】,
    #    任何时刻只驻留一年(约130万行)。
    import os as _os
    tmp = f"{DIR}/_yr"
    _os.makedirs(tmp, exist_ok=True)
    yr_all = (pd.to_datetime(pd.Series(day[rows]), unit="D").dt.year).to_numpy()
    for yv in np.unique(yr_all):
        sel = yr_all == yv
        d = pd.DataFrame(R[rows[sel]], columns=cols)
        d["day"] = day[rows[sel]]
        d["cid"] = cid[rows[sel]]
        for c, arr in YS.items():
            d[c] = arr[sel]
        d["fwd"] = fwd[sel]
        d.to_parquet(f"{tmp}/{yv}.parquet", index=False)
        del d
    years = sorted(int(v) for v in np.unique(yr_all))
    del R, rows, YS, fwd, yr_all
    gc.collect()
    log.info("  按年落盘完成: %s", years)

    tr_years = [v for v in years if v <= pd.Timestamp(a_tr1).year]
    te_years = [v for v in years if v > pd.Timestamp(a_tr1).year]
    per = max(1, a_sample // max(len(tr_years), 1))

    # 训练/测试数据只读一次, 反复用于各个权重组合
    trs = []
    for v in tr_years:
        d = pd.read_parquet(f"{tmp}/{v}.parquet")
        trs.append(d.sample(min(per, len(d)), random_state=42))
        del d
    tr = pd.concat(trs, ignore_index=True); del trs; gc.collect()
    log.info("训练 %s 行 (%s) / 测试年 %s", f"{len(tr):,}", tr_years, te_years)

    # 权重组合。w_mfe = 吃到大利润的奖励, lam = 挖坑的惩罚。
    #   (0, 0, 1)   等价 v1 终点收益的近似(只看曲线高度)
    #   (0, 1, 0)   v2 的老配置: 只罚亏不奖赚 -> 模型挑不动的票
    #   带 MFE 的几档才是这次要验的
    CONFIGS = ycols_global
    log.info("=" * 78)
    ycols_global = [c for c in tr.columns if c.startswith("y_")]
    log.info("扫障碍组合: 标签 = 该出场规则下真正落袋的超额")
    results = []
    for tag in CONFIGS:
        tr_y = tr[tag].to_numpy(np.float32)
        m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             min_child_weight=50, reg_lambda=2.0,
                             tree_method="hist", n_jobs=8, random_state=42)
        m.fit(tr[cols], tr_y)
        m.save_model(f"{DIR}/shape_v4_{tag}_h{a_h}.json")
        VOLF = {"vol_pct120", "up_vol_rate20", "dn_shrink_rate20", "pv_corr20",
                "obv_slope20", "mfi5", "mfi20", "vol_pile", "vol_dry", "brk_vol",
                "vol_std_5_20", "vol_ratio", "amt_ratio", "vol_trend"}
        volshare = sum(v for k, v in zip(cols, m.feature_importances_) if k in VOLF)
        top1 = sorted(zip(cols, m.feature_importances_), key=lambda x: -x[1])[0]

        acc = {k: [0, 0.0] for k in ("Top1%", "Top5%", "Bot20%")}
        yearly, dl = [], []
        tops = []
        for v in te_years:
            d = pd.read_parquet(f"{tmp}/{v}.parquet")
            sc = np.empty(len(d), np.float32)
            for i in range(0, len(d), 500_000):
                sc[i:i+500_000] = m.predict(d[cols].iloc[i:i+500_000]).astype(np.float32)
            d = d[["day", "cid", "fwd", tag]].assign(score=sc)
            d["rk"] = d.groupby("day")["score"].rank(pct=True)
            for k, (lo, hi) in {"Top1%": (0.99, 1.01), "Top5%": (0.95, 1.01),
                                "Bot20%": (0.0, 0.20)}.items():
                g = d[(d.rk >= lo) & (d.rk < hi)]
                acc[k][0] += len(g); acc[k][1] += float(g[tag].sum())
            g5 = d[d.rk >= 0.95]
            yearly.append(float(g5[tag].mean()) * 100)
            dl.append(d.groupby("day").agg(avg=("score","mean"), m=("fwd","mean")).reset_index())
            tops.append(d[d.rk >= 0.99][["day", "cid", "score", "rk"]])
            del d, sc
            gc.collect()
        jd = pd.concat(dl, ignore_index=True)
        rcorr = jd["avg"].corr(jd["m"])
        t1 = acc["Top1%"][1] / max(acc["Top1%"][0], 1) * 100
        t5 = acc["Top5%"][1] / max(acc["Top5%"][0], 1) * 100
        b20 = acc["Bot20%"][1] / max(acc["Bot20%"][0], 1) * 100
        log.info("%-22s Top1%% %+5.2f  Top5%% %+5.2f  Bot20%% %+5.2f | "
                 "逐年最小 %+5.2f 负年 %d | 审计 %+.3f | 量能 %.0f%% | 首特征 %s",
                 tag, t1, t5, b20, min(yearly), sum(1 for x in yearly if x < 0),
                 rcorr, volshare * 100, top1[0])
        log.info("%-22s   逐年: %s", "", "  ".join(f"{y}:{v:+.2f}"
                 for y, v in zip(te_years, yearly)))
        results.append((tag, t5, min(yearly)))
        out = pd.concat(tops, ignore_index=True)
        out["ts_code"] = code_index[out["cid"].to_numpy()]
        out["trade_date"] = pd.to_datetime(out["day"], unit="D")
        out[["ts_code", "trade_date", "score", "rk"]].to_parquet(
            f"{DIR}/oos_v4_{tag}_h{a_h}.parquet", index=False)
        del out, tops, dl, jd
        gc.collect()

    log.info("=" * 78)
    best = max(results, key=lambda x: x[1])
    log.info("Top5%% 最高: %s (%+.2fpp, 逐年最小 %+.2fpp)", best[0], best[1], best[2])


if __name__ == "__main__":
    main()
