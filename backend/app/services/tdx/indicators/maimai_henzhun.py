"""
买卖很准 (Maimai Henzhun) — "Very Accurate Buy/Sell" indicator, merged 2-line version.

Port of the TDX formula that combines price-channel breakouts with a DMI /
ADX strength filter. Original TDX source draws five step lines; this version
merges them into ONE buy line and ONE sell line (2026-08-10, per user request)
so the pane stays readable.

Original TDX logic:

    你  := CLOSE
    老  := (LOW + HIGH + CLOSE) / 3            # typical price
    板  := MA(老, 5)
    好  := HHV(板, 10)                         # sell threshold
    买卖 := LLV(板, 10)                         # buy threshold

    急卖奇准 = IF(HHV((你<好),  5),  100, 50)   # close < sell-threshold sometime in last 5
    短卖奇准 = IF(HHV((你<好),  10), 100, 50)
    急买奇准 = IF(LLV((你<买卖), 5),   50,  0)   # close < buy-threshold every bar in last 5
    短买奇准 = IF(LLV((你<买卖), 10),  50,  0)

    # DMI / ADX section
    TD   = SUM(TR, 5)
    DMP  = SUM(IF(HD>0 AND HD>LD, HD, 0), 5)
    DMM  = SUM(IF(LD>0 AND LD>HD, LD, 0), 5)
    神偷线    = DMP * 100 / TD
    辅助线    = DMM * 100 / TD
    动向趋势线 = MA(ABS(辅助线 - 神偷线) / (辅助线 + 神偷线) * 100, 3)

    准备现金 = IF(动向趋势线>88 AND 神偷线<5.8, 80, 0)

Merge rules (this version):

    买 = MAX(急买奇准, 短买奇准, 准备现金)      # 0 / 50 / 80
        高度 80 = 准备现金 (ADX 极端段) 也在场; 50 = 仅急买/短买
    卖 = MIN(急卖奇准, 短卖奇准)                # 50 / 100, 保留原版取值
        50 = 急卖<100 OR 短卖<100 → 近期从未跌破卖出阈值 = 强势上涨进行中
        (急卖 HHV5 触发必含于 短卖 HHV10, 故 MIN 数学上等于 急卖 —
         写成 MIN 只为对齐"OR 合并"的语义)

Signal rules (用户定义, 2026-08-11 修正卖出方向):

    买入信号 = 买[t-1] > 0 AND 买[t] == 0        # 从非0回落到0
        含义: 连续超卖状态刚刚结束 → 反转确认
    卖出信号 = 卖[t-1] < 100 AND 卖[t] == 100    # 从<100回到100
        含义: 强势段被打破 (重新跌破卖出阈值) → 止盈离场
        (⚠️ 不是强势段开始时卖 —— 早先版本把信号放在段首, 方向反了)

    (原版的 买点/卖点 是"条件成立期间恒为 1", 在图上几乎连成一片;
     改成边沿触发后每段状态只标一次, 信号可数、可回测。)
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import ABS, HHV, IF, LLV, MA, MAX, MIN, REF, SUM
from app.services.tdx.indicators.base import (
    IndicatorHLine,
    IndicatorLine,
    IndicatorMarker,
    IndicatorResult,
    series_to_json,
)


name = "maimai_henzhun"
label = "买卖很准"
pane = "sub"
min_bars = 15  # HHV(板, 10) over MA(5) needs ~15 bars to stabilize


def compute_lines(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """合并版 买/卖 线 — 指标绘制与选股策略 (strategy_templates/maimai_zhun.py)
    共用的唯一实现。返回 (buy_line 0/50/80, sell_line 50/100)。"""
    # Core price-channel setup
    you = close
    lao = (low + high + close) / 3
    ban = MA(lao, 5)
    hao = HHV(ban, 10)        # sell threshold
    maimai = LLV(ban, 10)     # buy threshold

    # Boolean conditions as 0/1 series
    sell_cond = (you < hao).astype(float)
    buy_cond = (you < maimai).astype(float)

    # HHV(bool, n) = 1 if any bar in last n is true (window-OR)
    # LLV(bool, n) = 1 if all bars in last n are true (window-AND)
    jibuy = IF(LLV(buy_cond, 5) > 0, 50.0, 0.0)        # 急买奇准
    duanbuy = IF(LLV(buy_cond, 10) > 0, 50.0, 0.0)     # 短买奇准
    jimai = IF(HHV(sell_cond, 5) > 0, 100.0, 50.0)     # 急卖奇准
    duanmai = IF(HHV(sell_cond, 10) > 0, 100.0, 50.0)  # 短卖奇准

    # DMI / ADX section
    prev_close = REF(close, 1)
    prev_high = REF(high, 1)
    prev_low = REF(low, 1)

    tr = MAX(MAX(high - low, ABS(high - prev_close)), ABS(low - prev_close))
    td = SUM(tr, 5)

    hd = high - prev_high
    ld = prev_low - low

    dmp = SUM(IF((hd > 0) & (hd > ld), hd, 0.0), 5)
    dmm = SUM(IF((ld > 0) & (ld > hd), ld, 0.0), 5)

    # Guard against divide-by-zero
    td_safe = td.replace(0, pd.NA)
    shentou = dmp * 100 / td_safe     # 神偷线
    fuzhu = dmm * 100 / td_safe       # 辅助线

    denom = (fuzhu + shentou).replace(0, pd.NA)
    dongxiang_raw = ABS(fuzhu - shentou) / denom * 100
    dongxiang = MA(dongxiang_raw, 3)  # 动向趋势线

    zhunbei = IF((dongxiang > 88) & (shentou < 5.8), 80.0, 0.0)  # 准备现金

    # ---- Merged lines ----
    buy_line = MAX(MAX(jibuy, duanbuy), zhunbei).fillna(0.0)     # 0 / 50 / 80
    sell_line = MIN(jimai, duanmai).fillna(100.0)                # 50 / 100
    return buy_line, sell_line


def compute_edges(
    close: pd.Series, high: pd.Series, low: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """边沿信号 (布尔): 买 = 买线 非0→0, 卖 = 卖线 <100→100。"""
    buy_line, sell_line = compute_lines(close, high, low)
    buy_fire = (buy_line.shift(1) > 0) & (buy_line == 0)
    sell_fire = (sell_line.shift(1) < 100) & (sell_line == 100)
    return buy_fire.fillna(False), sell_fire.fillna(False)


def compute(df: pd.DataFrame) -> IndicatorResult:
    close, high, low = df["close"], df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")

    buy_line, sell_line = compute_lines(close, high, low)
    buy_fire = (buy_line.shift(1) > 0) & (buy_line == 0)         # 非0 -> 0
    sell_fire = (sell_line.shift(1) < 100) & (sell_line == 100)  # <100 -> 100

    markers = [
        # 买入 — red up-triangle just above baseline
        *(IndicatorMarker(timestamp=ts, value=8, color="#FF3333", icon="triangle_up")
          for ts in ts_col[buy_fire.fillna(False)].tolist()),
        # 卖出 — green down-triangle near the top
        *(IndicatorMarker(timestamp=ts, value=92, color="#00CC00", icon="triangle_down")
          for ts in ts_col[sell_fire.fillna(False)].tolist()),
    ]

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(0, 100),
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("买", series_to_json(buy_line), "#FF3333", thickness=2),
            IndicatorLine("卖", series_to_json(sell_line), "#00CC00", thickness=2),
        ],
        hlines=[
            IndicatorHLine("顶", 100, "#55AA77", dashed=True),
            IndicatorHLine("准备现金档 80", 80, "#FF00FF", dashed=True),
            IndicatorHLine("买档 50", 50, "#888888", dashed=True),
            IndicatorHLine("底", 0, "#55AA77", dashed=True),
        ],
        markers=markers,
    )
