"""
买卖很准 (Maimai Henzhun) — "Very Accurate Buy/Sell" indicator.

Port of the TDX formula that combines price-channel breakouts with a DMI /
ADX strength filter. Displayed as five step-function lines on a 0..100
pane, plus 买点/卖点 markers when the conditions fire.

Logic summary:

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

    买点 = IF(LLV((你<买卖), 10), 1, 0)         # binary buy marker
    卖点 = IF(HHV((你<好),   10), 1, 0)         # binary sell marker
"""

from __future__ import annotations

import pandas as pd

from app.services.tdx.functions import ABS, HHV, IF, LLV, MA, MAX, REF, SUM
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


def compute(df: pd.DataFrame) -> IndicatorResult:
    close, high, low = df["close"], df["high"], df["low"]
    ts_col = df["timestamp"].astype("int64")

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
    jimai = IF(HHV(sell_cond, 5) > 0, 100.0, 50.0)    # 急卖奇准
    duanmai = IF(HHV(sell_cond, 10) > 0, 100.0, 50.0)  # 短卖奇准
    jibuy = IF(LLV(buy_cond, 5) > 0, 50.0, 0.0)        # 急买奇准
    duanbuy = IF(LLV(buy_cond, 10) > 0, 50.0, 0.0)     # 短买奇准

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

    # Binary entry/exit markers
    buy_signal = (LLV(buy_cond, 10) > 0).fillna(False)
    sell_signal = (HHV(sell_cond, 10) > 0).fillna(False)
    zhunbei_signal = (zhunbei == 80).fillna(False)

    markers = []
    # 买点 markers — red up-triangle at y=10 (just above baseline)
    for ts in ts_col[buy_signal].tolist():
        markers.append(IndicatorMarker(
            timestamp=ts, value=10,
            color="#FF3333", icon="triangle_up",
        ))
    # 准备现金 "始" markers — magenta down-triangle at y=80
    for ts in ts_col[zhunbei_signal].tolist():
        markers.append(IndicatorMarker(
            timestamp=ts, value=80,
            color="#FF00FF", icon="triangle_down",
        ))

    return IndicatorResult(
        name=name,
        label=label,
        pane=pane,
        y_axis_range=(0, 100),
        timestamps=ts_col.tolist(),
        lines=[
            IndicatorLine("急卖奇准", series_to_json(jimai), "#0088FF", thickness=1),
            IndicatorLine("短卖奇准", series_to_json(duanmai), "#00FF00", thickness=1),
            IndicatorLine("急买奇准", series_to_json(jibuy), "#FF3333", thickness=1),
            IndicatorLine("短买奇准", series_to_json(duanbuy), "#FFFFFF", thickness=1),
            IndicatorLine("准备现金", series_to_json(zhunbei), "#FF00FF", thickness=3),
        ],
        hlines=[
            IndicatorHLine("顶", 100, "#55AA77", dashed=True),
            IndicatorHLine("中", 50, "#FFFFFF", dashed=True),
            IndicatorHLine("底", 0, "#55AA77", dashed=True),
        ],
        markers=markers,
    )
