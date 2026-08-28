"""
提前预知主力拉升 — 通达信原始公式移植.

公式原文见 `zhuli_lasheng` 模块 docstring（用户导出的"指标代码"就是这一支）。
本模块与 `zhuli_lasheng` 共用同一份 `compute_zijin_ruchang()`，差别只在显示:

    提前预知主力拉升   STICK 柱状, a1..a5 小写, 红→黄渐变   ← 源码原样
    主力拉升           细线,      A1..A5 大写               ← 按截图

⚠️ 用户只导出了本指标的源码，并说明"另一个主力拉升应该是差不多的"，因此
`主力拉升` 目前复用同一套计算。若之后拿到它自己的源码，只需在
`zhuli_lasheng.compute()` 里换成它自己的 `compute_*` 即可，扇形和渲染不用动。

渲染: 源码里 资金入场 与 a1..a5 全部标了 STICK（柱状），linethick2 / linethick5。
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.indicators.base import IndicatorResult
from app.services.tdx.indicators.zhuli_lasheng import (
    TDX_FAN_COLORS,
    TDX_MAIN_COLOR,
    build_result,
    compute_zijin_ruchang,
)


name = "zhuli_lasheng_tiqian"
label = "提前预知主力拉升"
pane = "sub"
min_bars = 60  # 同 zhuli_lasheng — MA(CLOSE,58) 守卫


def compute(df: pd.DataFrame) -> IndicatorResult:
    return build_result(
        df,
        compute_zijin_ruchang(df),
        result_name=name,
        result_label=label,
        fan_colors=TDX_FAN_COLORS,
        main_color=TDX_MAIN_COLOR,
        fan_prefix="a",   # 源码是小写 a1..a5
        render="bar",     # 源码标了 STICK
    )
