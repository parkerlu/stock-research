"""筹码特征的【唯一】实现 —— 训练(prep.py) 与生产(build_chips_scores.py) 都调这里。

⚠️ 为什么必须只留一份: 本项目已经因为"同一份清单/同一套算法抄了两遍"栽过五次
   (板块条漏两页、v4 切换漏两页、v3.5 完全没接数据、SAR/突破/拉升/周线版只接了
   行情页、实时页训练菜单停在4个)。特征更严重 —— 训练和生产算得不一样,
   模型拿到的是另一个分布, 结果只会变差【而不会报错】。

⚠️ 输入约定(两边都必须满足, 否则结果错而不报错):
     - 按 (股票, 日期) 升序排列
     - gid 列标识股票分组边界(训练用 cid, 生产用 ts_code)
     - close 是【后复权价】(close*adj_factor)。训练侧 panel.npz 存的就是这个,
       生产侧查库时也要乘 adj_factor —— 涨幅跨除权日时只有复权价才是真实收益率
     - cyq 各列是【原始价】, 所以只做表内比值, 绝不与 close 相除
       (2026-09-10 第一版就是这么错的: "现价/低位成本"中位 3.602, 算出来的
        其实是 adj_factor 本身, 而且不报错)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 生产端查库/训练端取列都按这个清单来, 少一列下游就 KeyError 而不是静默出错
CYQ_COLS = ["winner_rate", "cost_5pct", "cost_15pct", "cost_50pct",
            "cost_85pct", "cost_95pct", "weight_avg", "his_low", "his_high"]

FEAT_NAMES = [
    "获利盘", "获利盘变化5", "获利盘变化10", "获利盘变化20",
    "获利盘背离10", "获利盘背离20",
    "均价在成本区间位置", "中位/均价", "上半区宽度", "下半区宽度",
    "均价在历史区间位置", "筹码集中度(对照)", "筹码宽度(对照)",
]


def chip_feats(df: pd.DataFrame, gid: str = "gid") -> pd.DataFrame:
    """算 13 个筹码特征。df 需含 gid / close / CYQ_COLS，按 (gid, 日期) 升序。

    返回与 df 同索引的特征表，列顺序 = FEAT_NAMES。
    """
    g = df.groupby(gid, sort=False)
    wr = df["winner_rate"]
    close = df["close"]
    out = pd.DataFrame(index=df.index)

    # --- 获利盘族: 训练里重要性前 6 名合计 60.0% ---
    out["获利盘"] = wr
    for k in (5, 10, 20):
        out[f"获利盘变化{k}"] = wr - g["winner_rate"].shift(k)
    # 获利盘背离 = 获利盘变化 − 价格涨幅。价格没动而获利盘上升 = 有人在低位接浮筹。
    # 文档: Q10/Q1=2.16, 过了价格位置×量能的九宫格控制后仍有 2.09
    for k in (10, 20):
        px_chg = (close / g["close"].shift(k) - 1) * 100
        out[f"获利盘背离{k}"] = out[f"获利盘变化{k}"] - px_chg

    # --- 成本结构: 只用 cyq 表内部比值(见模块 docstring 的复权口径警告) ---
    c5, c15 = df["cost_5pct"], df["cost_15pct"]
    c50, c85, c95 = df["cost_50pct"], df["cost_85pct"], df["cost_95pct"]
    wa = df["weight_avg"]
    span = (c95 - c5).clip(lower=1e-6)
    c50s = c50.clip(lower=1e-6)
    out["均价在成本区间位置"] = (wa - c5) / span
    out["中位/均价"] = c50 / wa.clip(lower=1e-6)
    out["上半区宽度"] = (c95 - c50) / c50s
    out["下半区宽度"] = (c50 - c5) / c50s
    span_h = (df["his_high"] - df["his_low"]).clip(lower=1e-6)
    out["均价在历史区间位置"] = (wa - df["his_low"]) / span_h

    # --- 对照特征: 文档已证伪(集中度 Q10/Q1=1.10)。留着是为了每次训练都能
    #     确认它确实排在末尾, 而不是默默把它当成有用的 ---
    out["筹码集中度(对照)"] = (c85 - c15) / c50s
    out["筹码宽度(对照)"] = span / c50s

    return out[FEAT_NAMES].astype(np.float32)


def bad_rows(df: pd.DataFrame) -> pd.Series:
    """成本价异常的行(停牌/数据错误)。这些行的比值会爆掉 —— 实测未剔时
    "上半区宽度"均值 153.965 而中位只有 0.132。"""
    return (~np.isfinite(df["cost_50pct"]) | (df["cost_50pct"] <= 0)
            | ~np.isfinite(df["weight_avg"]) | (df["weight_avg"] <= 0))
