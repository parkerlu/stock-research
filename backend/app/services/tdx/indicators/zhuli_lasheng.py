"""
主力拉升 / 提前预知主力拉升 — 通达信原始公式移植.

公式原文（用户从同花顺"指标代码"导出，逐字移植）
------------------------------------------------
    Var1:=EMA(HHV(HIGH,500),21);  Var2:=EMA(HHV(HIGH,250),21);  Var3:=EMA(HHV(HIGH,90),21);
    Var4:=EMA(LLV(LOW,500),21);   Var5:=EMA(LLV(LOW,250),21);   Var6:=EMA(LLV(LOW,90),21);
    Var7:=EMA((Var4*0.96+Var5*0.96+Var6*0.96+Var1*0.558+Var2*0.558+Var3*0.558)/6,21);
    Var8:=EMA((Var4*1.25+Var5*1.23+Var6*1.2 +Var1*0.55 +Var2*0.55 +Var3*0.65)/6,21);
    Var9:=EMA((Var4*1.3 +Var5*1.3 +Var6*1.3 +Var1*0.68 +Var2*0.68 +Var3*0.68)/6,21);
    VarA:=EMA((Var7*3+Var8*2+Var9)/6*1.738,21);
    VarB:=REF(LOW,1);
    VarC:=SMA(ABS(LOW-VarB),3,1)/SMA(MAX(LOW-VarB,0),3,1)*100;
    VarD:=EMA(IF(CLOSE*1.35<=VarA,VarC*10,VarC/10),3);
    VarE:=LLV(LOW,30);
    VarF:=HHV(VarD,30);
    Var10:=IF(MA(CLOSE,58),1,0);
    资金入场: EMA(IF(LOW<=VarE,(VarD+VarF*2)/2,0),3)/618*Var10;
    资金入场: IF(资金入场>0,资金入场,0),STICK,linethick2,COLOR0000ff;
    今量: 资金入场;
    a1:IF(资金入场>0,今量*1.2,0),STICK,linethick5,COLOR0000ff;
    a2:IF(资金入场>0,今量*0.8,0),STICK,linethick5,COLOR0066ff;
    a3:IF(资金入场>0,今量*0.6,0),STICK,linethick5,COLOR0099ff;
    a4:IF(资金入场>0,今量*0.4,0),STICK,linethick5,COLOR00ccff;
    a5:IF(资金入场>0,今量*0.2,0),STICK,linethick5,COLOR00ffff;

在算什么
--------
VarA 是一条由 90/250/500 日高低点通道加权、再多层 EMA(21) 平滑出来的
"长期估值中枢"。`CLOSE*1.35<=VarA` 表示股价比这条中枢还低 35% 以上 —— 深度
超跌；此时 VarC 被放大 10 倍，否则缩小到 1/10。

VarC 本身是把 RSI 的分子分母倒过来: SMA(|ΔLOW|)/SMA(MAX(ΔLOW,0))，低点连续
下挫、几乎没有反弹时分母趋近 0，值急剧放大 —— 衡量下跌的"单边程度"。

最后只在 `LOW<=LLV(LOW,30)`（当日创 30 日新低）那一刻取 (VarD+VarF*2)/2 输出，
EMA(3) 平滑后除以 618。所以整条线大部分时间为 0，只有"深度超跌 + 创 30 日新低"
同时成立时才冒出柱子 —— 这就是"主力资金入场"的判定。

⚠️ 已知偏差
-----------
`Var10:=IF(MA(CLOSE,58),1,0)` 在通达信里是"MA58 有效即为 1"的取数守卫，
本移植按此语义实现（前 57 根输出 0）。

长周期 HHV/LLV(500) 使用**部分窗口**（min_periods=1）而非严格 500 根，与通达信
"数据不足时用现有数据计算"的惯例一致；对于历史 ≥500 根的股票，从第 500 根起
两种算法结果完全相同，只影响上市初期的少数几根。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from app.services.tdx.functions import ABS, EMA, MA, MAX, REF, SMA
from app.services.tdx.indicators.base import (
    IndicatorLine,
    IndicatorResult,
    series_to_json,
)


name = "zhuli_lasheng"
label = "主力拉升（资金入场）"
pane = "sub"
# MA(CLOSE,58) 是硬门槛（Var10 守卫）；长周期 HHV/LLV 用部分窗口，故不要求 500 根。
min_bars = 60


# 扇形档位 — 源码逐字: a1..a5 = 今量 × 1.2 / 0.8 / 0.6 / 0.4 / 0.2
FAN_MULTS: list[float] = [1.2, 0.8, 0.6, 0.4, 0.2]

# 源码 COLOR 值（提前预知主力拉升那支）。通达信是 BBGGRR 顺序，转成 RGB:
#   COLOR0000ff -> #FF0000   COLOR0066ff -> #FF6600   COLOR0099ff -> #FF9900
#   COLOR00ccff -> #FFCC00   COLOR00ffff -> #FFFF00
TDX_FAN_COLORS = ["#FF0000", "#FF6600", "#FF9900", "#FFCC00", "#FFFF00"]
TDX_MAIN_COLOR = "#FF0000"

# 主力拉升自己的配色 — 按截图: A1 绿 / A2 蓝 / A3 黑 / A4 深灰 / A5 浅灰，
# 资金入场红。(只导出了提前预知的源码，主力拉升的颜色从截图取。)
ZL_FAN_COLORS = ["#00B050", "#3C8EFF", "#202020", "#808080", "#C0C0C0"]
ZL_MAIN_COLOR = "#FF0000"


def _hhv(s: pd.Series, n: int) -> pd.Series:
    """HHV，数据不足时用现有数据（通达信惯例，见模块 docstring）。"""
    return s.rolling(window=n, min_periods=1).max()


def _llv(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(window=n, min_periods=1).min()


def compute_zijin_ruchang(df: pd.DataFrame) -> pd.Series:
    """资金入场 / 今量 —— 通达信原始公式逐行移植。"""
    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    var1 = EMA(_hhv(high, 500), 21)
    var2 = EMA(_hhv(high, 250), 21)
    var3 = EMA(_hhv(high, 90), 21)
    var4 = EMA(_llv(low, 500), 21)
    var5 = EMA(_llv(low, 250), 21)
    var6 = EMA(_llv(low, 90), 21)

    var7 = EMA(
        (var4 * 0.96 + var5 * 0.96 + var6 * 0.96
         + var1 * 0.558 + var2 * 0.558 + var3 * 0.558) / 6, 21
    )
    var8 = EMA(
        (var4 * 1.25 + var5 * 1.23 + var6 * 1.2
         + var1 * 0.55 + var2 * 0.55 + var3 * 0.65) / 6, 21
    )
    var9 = EMA(
        (var4 * 1.3 + var5 * 1.3 + var6 * 1.3
         + var1 * 0.68 + var2 * 0.68 + var3 * 0.68) / 6, 21
    )
    var_a = EMA((var7 * 3 + var8 * 2 + var9) / 6 * 1.738, 21)

    var_b = REF(low, 1)
    d_low = (low - var_b).fillna(0.0)          # 首根无前值 -> 0
    numer = SMA(ABS(d_low), 3, 1)
    denom = SMA(MAX(d_low, 0.0), 3, 1)
    # 通达信约定 x/0 = 0（低点连续下挫、毫无反弹时分母为 0）
    var_c = (numer / denom.replace(0.0, np.nan) * 100).fillna(0.0)

    var_d = EMA(pd.Series(np.where((close * 1.35).values <= var_a.values,
                                   var_c.values * 10, var_c.values / 10),
                          index=df.index), 3)
    var_e = _llv(low, 30)
    var_f = _hhv(var_d, 30)

    # Var10:=IF(MA(CLOSE,58),1,0) —— MA58 有效(非 NaN 且非 0)即为 1
    ma58 = MA(close, 58)
    var_10 = ((ma58.notna()) & (ma58 != 0)).astype(float)

    gate = low.values <= var_e.values
    raw = pd.Series(np.where(gate, (var_d.values + var_f.values * 2) / 2, 0.0),
                    index=df.index)
    zijin = EMA(raw, 3) / 618 * var_10

    # 资金入场: IF(资金入场>0,资金入场,0)
    return zijin.where(zijin > 0, 0.0).fillna(0.0)


def build_result(
    df: pd.DataFrame,
    zijin: pd.Series,
    *,
    result_name: str,
    result_label: str,
    fan_colors: list[str],
    main_color: str,
    fan_prefix: str = "a",
    render: str = "line",
) -> IndicatorResult:
    """把 资金入场 展开成扇形并组装 IndicatorResult。

    源码里 a1..a5 都带 `IF(资金入场>0, …, 0)` 守卫，等价于直接乘（资金入场
    已被钳到 ≥0），这里按乘法实现，结果一致。
    """
    ts_col = df["timestamp"].astype("int64")

    lines = [
        IndicatorLine(
            f"{fan_prefix}{i + 1}",
            series_to_json(zijin * mult),
            fan_colors[i],
            thickness=1,
            render=render,
        )
        for i, mult in enumerate(FAN_MULTS)
    ]
    # 资金入场 (= 今量) 是唯一的真实数据线，放最前面 —— 图例按此顺序显示。
    # 柱状模式下会被更高的 a1 (×1.2) 盖住，但图例数值照常显示；a1..a5 由高到低
    # 依次叠画，形成 5 段分层色带（即截图里的渐变柱）。
    lines.insert(
        0,
        IndicatorLine(
            "资金入场", series_to_json(zijin), main_color,
            thickness=2, render=render,
        ),
    )

    return IndicatorResult(
        name=result_name,
        label=result_label,
        pane=pane,
        y_axis_range=None,
        timestamps=ts_col.tolist(),
        lines=lines,
        hlines=[],
        bands=[],
        markers=[],
        # 注意: warnings 在前端是 alert() 弹窗，只用于真正的失败
        # (无 K 线数据 / 数据不足)，不要放说明性文字。
    )


def compute(df: pd.DataFrame) -> IndicatorResult:
    return build_result(
        df,
        compute_zijin_ruchang(df),
        result_name=name,
        result_label=label,
        fan_colors=ZL_FAN_COLORS,
        main_color=ZL_MAIN_COLOR,
        fan_prefix="A",   # 主力拉升截图里是大写 A1..A5
        render="line",    # 截图里是细线，非柱状
    )
