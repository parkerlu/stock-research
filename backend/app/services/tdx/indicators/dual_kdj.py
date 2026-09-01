"""日周 KDJ 共振 (tdx-dual-kdj) —— 虚拟盘当前使用的策略, 画在图上便于复核.

信号条件 (与 strategy_templates/tdx_classics.py::_entry_dual_kdj 完全一致):
    日线 K 上穿 D  AND  周线 K > 周线 D  AND  周线 K < 50

⚠️ 这里必须复用策略那份实现, 不能照着公式重写一遍 —— 图上画的和实际买的
必须是同一个东西。之前缠论就吃过亏: 图表指标每次重算整个窗口, 只画"事后
仍然成立"的信号, 看起来比实盘准得多。本策略是纯公式(实测撤销率 0.05%),
不存在那个问题, 但同源仍是硬要求。

四道闸检验结果 (2026-09):
    因果性  截断复算撤销率 0.05% (缠论同口径 44~50%)
    样本内  2016-2021, 超额 +1.206pp/笔, t=22.15
    样本外  2024-09 起(纯公式无训练, 未参与任何调参), 超额 +1.127pp, t=12.56
    组合层  熊市 z=7.62 / 反弹 z=3.58, 两窗口均 12/12 跑赢随机对照
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.tdx.indicators.base import (
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)

name = "dual_kdj"
label = "日周 KDJ 共振"
pane = "sub"


def compute(df: pd.DataFrame) -> IndicatorResult:
    # 复用策略实现算指标序列, 保证图与实盘同源
    from app.services.strategy_templates.tdx_classics import (
        _compute_pack,
        _entry_dual_kdj,
    )

    ts_col = df["timestamp"].astype("int64")
    close = df["close"].to_numpy(dtype="float64")
    high = df["high"].to_numpy(dtype="float64")
    low = df["low"].to_numpy(dtype="float64")
    vol = df["vol"].to_numpy(dtype="float64") if "vol" in df else np.ones_like(close)
    ind = _compute_pack(close, high, low, vol)
    # ⚠️ 用 numpy 位置掩码, 不要 ts_col[bool_series] —— 后者在某些 index 下
    # 会退化成标签索引, 结果拿到的是行号而不是时间戳(实测踩过)。
    fire = np.asarray(_entry_dual_kdj(close, ind), dtype=bool)

    k = pd.Series(ind["k"], index=df.index)
    d = pd.Series(ind["d"], index=df.index)
    wk = pd.Series(ind["wk"], index=df.index)
    wd = pd.Series(ind["wd"], index=df.index)

    ts_arr = ts_col.to_numpy()
    markers = [
        IndicatorMarker(timestamp=int(ts), value=6, color="#FF3333", icon="triangle_up")
        for ts in ts_arr[fire]
    ]

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(0, 100),
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("日K", series_to_json(k), "#FF8800", thickness=2),
            IndicatorLine("日D", series_to_json(d), "#3399FF", thickness=1),
            IndicatorLine("周K", series_to_json(wk), "#FF3333", thickness=2),
            IndicatorLine("周D", series_to_json(wd), "#9966FF", thickness=1),
        ],
        markers=markers,
    )
