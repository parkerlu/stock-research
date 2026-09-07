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


def path_reward(px: pd.DataFrame, h: int, lam: float) -> pd.DataFrame:
    """路径型奖励。返回 ts_code/trade_date/y/fwd。

    做法: 先把每只票的日收益转成【当日超额】(减去全市场当日等权均值),
    再在 t+1..t+h 上累计, 对累计曲线取 均值 与 最小值。
    这样"牛市普涨"在第一步就被减掉了 —— 与 v1 同一个去 regime 保证。
    """
    px = px.sort_values(["cid", "day"]).reset_index(drop=True)
    px["r"] = px.groupby("cid", sort=False)["c"].pct_change().astype(np.float32)
    mkt = px.groupby("day")["r"].transform("mean")
    px["e"] = (px["r"] - mkt).astype(np.float32)      # 当日超额

    # 每只票单独在时间轴上滚: cum_s = e[t+1]+...+e[t+s]
    outs = []
    for code, g in px.groupby("cid", sort=False, observed=True):
        e = g["e"].to_numpy(np.float32)
        n = len(e)
        if n <= h + 1:
            continue
        # 累计和的前缀, 便于取任意窗口
        cs = np.concatenate([[0.0], np.nancumsum(np.nan_to_num(e))]).astype(np.float32)
        # 对每个 t: 窗口内第 s 步的累计 = cs[t+1+s] - cs[t+1]
        A = np.full(n, np.nan, np.float32)
        B = np.full(n, np.nan, np.float32)
        for s in range(1, h + 1):
            pass  # 占位, 下面用向量化
        # 向量化: 构造 (n, h) 的累计矩阵会太大, 改用滑动最小/均值
        base = cs[1:n + 1]                      # cs[t+1]
        # 各步累计: cs[t+1+s] - cs[t+1], s=1..h
        idx = np.arange(n)
        valid = idx + h < n
        # 均值: (sum_{s=1..h} cs[t+1+s]) / h - cs[t+1]
        cs_pad = np.concatenate([cs, np.full(h + 2, cs[-1], np.float32)])
        win_sum = np.convolve(cs_pad, np.ones(h, np.float32), "valid")
        # win_sum[k] = cs[k] + ... + cs[k+h-1]; 需要 k = t+2
        A_all = win_sum[2:2 + n] / h - base
        # 最小: 滑动最小 of cs[t+2..t+1+h] 减 base
        m = pd.Series(cs_pad).rolling(h).min().to_numpy(np.float32)
        # m[k] = min(cs[k-h+1..k]); 需要 min(cs[t+2..t+h+1]) -> k = t+h+1
        min_all = m[h + 1:h + 1 + n] - base
        A[valid] = A_all[valid]
        B[valid] = -np.minimum(0.0, min_all[valid])
        fwd = np.full(n, np.nan, np.float32)
        fwd[valid] = (cs[np.minimum(idx + 1 + h, n)] - base)[valid]
        outs.append(pd.DataFrame({
            "cid": g["cid"].to_numpy(),
            "day": g["day"].to_numpy(),
            "A": A, "B": B, "fwd": fwd}))
    d = pd.concat(outs, ignore_index=True).dropna(subset=["A"])
    d["y"] = (d["A"] - lam * d["B"]).astype(np.float32)
    return d[["cid", "day", "y", "fwd"]]


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

    log.info("路径型奖励 h=%d λ=%.1f ...", a.h, a.lam)
    a_h, a_tr1, a_sample = a.h, a.tr1, a.sample
    lab = path_reward(pd.DataFrame({"cid": cid, "day": day, "c": closes}),
                      a.h, a.lam)
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
    y = lab["y"].to_numpy(np.float32)[ok]
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
        d["y"] = y[sel]
        d["fwd"] = fwd[sel]
        d.to_parquet(f"{tmp}/{yv}.parquet", index=False)
        del d
    years = sorted(int(v) for v in np.unique(yr_all))
    del R, rows, y, fwd, yr_all
    gc.collect()
    log.info("  按年落盘完成: %s", years)

    tr_years = [v for v in years if v <= pd.Timestamp(a_tr1).year]
    te_years = [v for v in years if v > pd.Timestamp(a_tr1).year]
    per = max(1, a_sample // max(len(tr_years), 1))
    trs = []
    for v in tr_years:
        d = pd.read_parquet(f"{tmp}/{v}.parquet")
        trs.append(d.sample(min(per, len(d)), random_state=42))
        del d
    tr = pd.concat(trs, ignore_index=True)
    del trs
    gc.collect()
    log.info("训练 %s 行 (%s) / 测试年 %s", f"{len(tr):,}", tr_years, te_years)

    m = xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8,
                         min_child_weight=50, reg_lambda=2.0,
                         tree_method="hist", n_jobs=8, random_state=42)
    m.fit(tr[cols], tr["y"])
    m.save_model(f"{DIR}/shape_v2_h{a_h}.json")
    del tr
    gc.collect()

    imp = sorted(zip(cols, m.feature_importances_), key=lambda x: -x[1])[:14]
    log.info("重要性 Top14: %s", ", ".join(f"{k}={v:.3f}" for k, v in imp))
    VOLF = {"vol_pct120", "up_vol_rate20", "dn_shrink_rate20", "pv_corr20",
            "obv_slope20", "mfi5", "mfi20", "vol_pile", "vol_dry", "brk_vol",
            "vol_std_5_20", "vol_ratio", "amt_ratio", "vol_trend"}
    log.info("量能类合计重要性: %.1f%%",
             sum(v for k, v in zip(cols, m.feature_importances_) if k in VOLF) * 100)

    # ---- 逐年评估 ----
    tiers = {"Top1%": (0.99, 1.01), "Top5%": (0.95, 1.01), "Top10%": (0.90, 1.01),
             "Top20%": (0.80, 1.01), "Bot20%": (0.0, 0.20)}
    acc = {k: [0, 0.0, 0] for k in tiers}      # n, sum(fwd), n_win
    yearly, daily_rows, tops = [], [], []
    for v in te_years:
        d = pd.read_parquet(f"{tmp}/{v}.parquet")
        sc = np.empty(len(d), np.float32)
        for i in range(0, len(d), 500_000):
            sc[i:i + 500_000] = m.predict(d[cols].iloc[i:i + 500_000]).astype(np.float32)
        d = d[["day", "cid", "fwd"]].assign(score=sc)
        d["rk"] = d.groupby("day")["score"].rank(pct=True)
        for k, (lo, hi) in tiers.items():
            g = d[(d.rk >= lo) & (d.rk < hi)]
            acc[k][0] += len(g); acc[k][1] += float(g["fwd"].sum())
            acc[k][2] += int((g["fwd"] > 0).sum())
        g5 = d[d.rk >= 0.95]
        yearly.append((v, float(g5["fwd"].mean()) * 100, len(g5)))
        daily_rows.append(d.groupby("day").agg(avg=("score", "mean"),
                                               m=("fwd", "mean")).reset_index())
        tops.append(d[d.rk >= 0.90][["day", "cid", "score", "rk"]])
        del d, sc
        gc.collect()

    dl = pd.concat(daily_rows, ignore_index=True)
    r = dl["avg"].corr(dl["m"])
    log.info("【去记忆审计】每日均分 vs 当日市场超额 相关 %+.4f  均分std %.5f",
             r, dl["avg"].std())
    log.info("               判定: %s", "通过" if abs(r) < 0.10 else "⚠️ 失败")

    log.info("--- 样本外分层 (fwd = 持有期超额, pp) ---")
    for k in tiers:
        n, sm, w = acc[k]
        log.info("  %-8s n=%9s  超额 %+6.2fpp  胜率 %.1f%%",
                 k, f"{n:,}", sm / max(n, 1) * 100, w / max(n, 1) * 100)

    log.info("--- 逐年 Top5% 超额 (硬判据: 必须均匀) ---")
    for v, mv, n in yearly:
        log.info("  %d  %+6.2fpp  n=%s", v, mv, f"{n:,}")
    vs = [x[1] for x in yearly]
    log.info("  最小 %+.2fpp / 均值 %+.2fpp / 负年数 %d",
             min(vs), float(np.mean(vs)), sum(1 for x in vs if x < 0))

    out = pd.concat(tops, ignore_index=True)
    out["ts_code"] = code_index[out["cid"].to_numpy()]
    out["trade_date"] = pd.to_datetime(out["day"], unit="D")
    out[["ts_code", "trade_date", "score", "rk"]].to_parquet(
        f"{DIR}/oos_top_v2_h{a_h}.parquet", index=False)
    log.info("样本外 Top10% 已存 %s 行", f"{len(out):,}")


if __name__ == "__main__":
    main()
