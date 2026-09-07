"""形态模型 · 第2步: 横截面标准化 + 训练 + 【去记忆审计】。

训练窗口 2016-2020 (5年), 测试窗口 2021-2026 全部样本外。
这个划分本身就是对"有没有 regime 记忆"的考验:
    训练期含 2018 熊市 + 2019-2020 牛市
    测试期含 2021 结构牛、2022 熊、2024 急跌、2025-26 轮动
如果模型学的是"牛市全买", 换到测试期的不同 regime 上就会垮。

⚠️ 两处必须守住, 违反任何一处这个模型就没有意义:

  A. 每个特征在【当日横截面】内转分位。
     做完之后, 一个"全市场普涨"的日子和"全市场普跌"的日子,
     特征分布完全一样 —— 分位只说"这只票在今天排第几", 不说今天是涨是跌。

  B. 标签是【超额】: 未来H日收益 − 当日全市场等权均值。
     牛市里人人都涨, 减掉均值后剩下的才是"这只票比别人强多少"。
     直接用原始收益训练, 模型第一件事就是学会认牛市。

审计(train 完自动跑, 见 audit_regime):
     把模型每天的平均打分, 与当日大盘涨跌做相关。
     若显著非零, 说明它仍在押注方向 —— 那就是失败, 必须回头改。
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd
import xgboost as xgb

log = logging.getLogger("shape.train")
DIR = "/app/data/research/shape"
META = ("ts_code", "trade_date", "c")


def cs_rank(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """每个特征在当日横截面内转成 [0,1] 分位 —— 去 regime 的第一层。"""
    g = df.groupby("trade_date", sort=False)
    out = df[["ts_code", "trade_date"]].copy()
    for c in cols:
        out[c] = g[c].rank(pct=True).astype(np.float32)
    return out


def build_labels(px: pd.DataFrame, h: int) -> pd.DataFrame:
    """标签 = 未来h日收益 − 当日全市场等权均值(超额)。"""
    px = px.sort_values(["ts_code", "trade_date"])
    fwd = px.groupby("ts_code", sort=False)["c"].shift(-h) / px["c"] - 1
    px = px.assign(fwd=fwd.astype(np.float32)).dropna(subset=["fwd"])
    mkt = px.groupby("trade_date")["fwd"].transform("mean")
    px["y"] = (px["fwd"] - mkt).astype(np.float32)
    return px[["ts_code", "trade_date", "fwd", "y"]]


def audit_regime(df: pd.DataFrame, mkt: pd.DataFrame) -> None:
    """去记忆审计 —— 模型每日均分 vs 当日大盘涨跌, 相关性应接近 0。"""
    daily = df.groupby("trade_date")["score"].mean().rename("avg_score")
    j = pd.concat([daily, mkt.set_index("trade_date")["m"]], axis=1).dropna()
    r_same = j["avg_score"].corr(j["m"])
    r_next = j["avg_score"].corr(j["m"].shift(-1))
    log.info("【去记忆审计】模型每日均分 vs 当日大盘涨跌  相关 %+.4f", r_same)
    log.info("               模型每日均分 vs 次日大盘涨跌  相关 %+.4f", r_next)
    log.info("               每日均分的标准差 %.5f (越小说明越不押方向)",
             j["avg_score"].std())
    verdict = "通过 —— 没有方向记忆" if abs(r_same) < 0.10 else "⚠️ 失败 —— 仍在押大盘方向"
    log.info("               判定: %s", verdict)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                        datefmt="%H:%M:%S")
    ap = argparse.ArgumentParser()
    ap.add_argument("--h", type=int, default=20, help="持有期(交易日)")
    ap.add_argument("--tr0", default="2016-01-01")
    ap.add_argument("--tr1", default="2020-12-31")
    ap.add_argument("--sample", type=int, default=3_000_000)
    a = ap.parse_args()

    log.info("读特征 ...")
    feat = pd.read_parquet(f"{DIR}/features.parquet")
    cols = [c for c in feat.columns if c not in META]
    feat["trade_date"] = pd.to_datetime(feat["trade_date"])
    log.info("  %s 行, %d 个特征", f"{len(feat):,}", len(cols))

    log.info("标签: 未来%d日超额收益 ...", a.h)
    lab = build_labels(feat[["ts_code", "trade_date", "c"]], a.h)

    log.info("横截面分位标准化 ...")
    X = cs_rank(feat[["ts_code", "trade_date"] + cols], cols)
    del feat

    df = X.merge(lab, on=["ts_code", "trade_date"], how="inner")
    del X, lab
    log.info("  合并后 %s 行", f"{len(df):,}")

    tr0, tr1 = pd.Timestamp(a.tr0), pd.Timestamp(a.tr1)
    tr = df[(df.trade_date >= tr0) & (df.trade_date <= tr1)]
    te = df[df.trade_date > tr1]
    if len(tr) > a.sample:
        tr = tr.sample(a.sample, random_state=42)   # 随机抽样, 固定种子
    log.info("训练 %s 行 (%s ~ %s) / 测试 %s 行 (%s ~ %s)",
             f"{len(tr):,}", tr.trade_date.min().date(), tr.trade_date.max().date(),
             f"{len(te):,}", te.trade_date.min().date(), te.trade_date.max().date())

    m = xgb.XGBRegressor(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=50, reg_lambda=2.0,
        tree_method="hist", n_jobs=8,
        random_state=42,      # ⚠️ 固定种子: 不固定的话同一历史信号的分数天天漂
    )
    m.fit(tr[cols], tr["y"])
    m.save_model(f"{DIR}/shape_h{a.h}.json")
    log.info("模型已存 %s/shape_h%d.json", DIR, a.h)

    imp = sorted(zip(cols, m.feature_importances_), key=lambda x: -x[1])[:12]
    log.info("特征重要性 Top12: %s", ", ".join(f"{k}={v:.3f}" for k, v in imp))

    te = te.assign(score=m.predict(te[cols]).astype(np.float32))
    te["rk"] = te.groupby("trade_date")["score"].rank(pct=True)

    # 当日大盘 = 全市场等权未来1日? 不 —— 用同期已实现的市场收益做审计基准
    mkt = (te.groupby("trade_date")["fwd"].mean().reset_index()
             .rename(columns={"fwd": "m"}))
    audit_regime(te, mkt)

    log.info("--- 样本外分层 (超额, 单位 pp) ---")
    for lo, hi, nm in ((0.99, 1.01, "Top1%"), (0.95, 1.01, "Top5%"),
                       (0.90, 1.01, "Top10%"), (0.80, 1.01, "Top20%"),
                       (0.0, 0.20, "Bot20%")):
        s = te[(te.rk >= lo) & (te.rk < hi)]
        log.info("  %-8s n=%9s  超额 %+6.2fpp  胜率 %.1f%%",
                 nm, f"{len(s):,}", s["y"].mean() * 100, (s["y"] > 0).mean() * 100)

    log.info("--- 逐年 Top5% 超额 ---")
    te["yr"] = te.trade_date.dt.year
    for y, g in te[te.rk >= 0.95].groupby("yr"):
        log.info("  %d  %+6.2fpp  n=%s", y, g["y"].mean() * 100, f"{len(g):,}")

    te[te.rk >= 0.90][["ts_code", "trade_date", "score", "rk"]].to_parquet(
        f"{DIR}/oos_top_h{a.h}.parquet", index=False)
    log.info("样本外 Top10% 已存, 供组合回测")


if __name__ == "__main__":
    main()
