"""诊断: 模型是欠拟合、过拟合, 还是特征信息量不够?

三者治法完全相反, 不分清就是瞎调:
  过拟合   训练集分层远强于测试集   -> 加正则/减容量/加数据
  欠拟合   两边都弱, 且加容量能改善 -> 加轮数/加深度 (用户的假设)
  信息不够 两边都弱, 加容量也不动   -> 换特征, 训练轮数再多也没用

判据用【分层价差】: Top10% 的标签均值 减 Bot10% 的标签均值。
比单看 Top 更稳 —— 它同时考察模型有没有把差的排到下面去。
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("diag")
DIR = "/app/data/research/shape"
# ⚠️ _yr 里的标签列会被最近一次训练覆盖 —— 别写死列名, 自动挑第一个 y_*。
#    (v5 是 y_0.15_0.08 三重障碍, v6 换成了 y_s*_a*_t* 移动止盈。)
TAG = None
META = {"day", "cid", "fwd"}


def spread(d: pd.DataFrame, m, cols, tag) -> tuple[float, float, float]:
    sc = np.empty(len(d), np.float32)
    for i in range(0, len(d), 500_000):
        sc[i:i + 500_000] = m.predict(d[cols].iloc[i:i + 500_000]).astype(np.float32)
    x = d[["day", tag]].assign(score=sc)
    x["rk"] = x.groupby("day")["score"].rank(pct=True)
    top = x.loc[x.rk >= 0.90, tag].mean() * 100
    bot = x.loc[x.rk < 0.10, tag].mean() * 100
    return top, bot, top - bot


def main() -> None:
    global TAG
    tr_years, te_years = [2016, 2017, 2018, 2019, 2020], [2021, 2022, 2023, 2024, 2025, 2026]
    cols = None
    trs = []
    for y in tr_years:
        d = pd.read_parquet(f"{DIR}/_yr/{y}.parquet")
        if cols is None:
            cols = [c for c in d.columns if c not in META and not c.startswith("y_")]
            TAG = next(c for c in d.columns if c.startswith("y_"))
            log.info("用标签列: %s", TAG)
        trs.append(d.sample(min(600_000, len(d)), random_state=42))
        del d
    tr = pd.concat(trs, ignore_index=True); del trs

    log.info("训练 %s 行, %d 特征", f"{len(tr):,}", len(cols))
    log.info("%-28s %8s %8s %8s %8s", "配置", "训练价差", "测试价差", "训练Top", "测试Top")

    # 从小到大加容量。若测试价差随容量单调上升 -> 欠拟合, 用户假设成立。
    CFGS = [
        ("轮数100 深度4",  dict(n_estimators=100,  max_depth=4)),
        ("轮数400 深度6",  dict(n_estimators=400,  max_depth=6)),   # 现行
        ("轮数1200 深度6", dict(n_estimators=1200, max_depth=6)),
        ("轮数1200 深度8", dict(n_estimators=1200, max_depth=8)),
        ("轮数3000 深度8", dict(n_estimators=3000, max_depth=8)),
    ]
    for name, kw in CFGS:
        m = xgb.XGBRegressor(learning_rate=0.05, subsample=0.8,
                             colsample_bytree=0.8, min_child_weight=50,
                             reg_lambda=2.0, tree_method="hist", n_jobs=8,
                             random_state=42, **kw)
        m.fit(tr[cols], tr[TAG])
        tr_top, _, tr_sp = spread(tr, m, cols, TAG)
        te_tops, te_sps = [], []
        for y in te_years:
            d = pd.read_parquet(f"{DIR}/_yr/{y}.parquet")
            t, b, s = spread(d, m, cols, TAG)
            te_tops.append(t); te_sps.append(s)
            del d
        log.info("%-28s %+8.2f %+8.2f %+8.2f %+8.2f", name, tr_sp,
                 float(np.mean(te_sps)), tr_top, float(np.mean(te_tops)))


if __name__ == "__main__":
    main()
