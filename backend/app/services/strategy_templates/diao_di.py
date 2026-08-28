"""抄底逃顶 (Diao-Di Tao-Ding) — 移植自 Pine Script Momentum Line Indicator.

原始 Pine 公式:
    lowestLow_10  = LLV(low, 10)
    highestHigh_25 = HHV(high, 25)
    momentum = EMA((close - lowestLow_10) / (highestHigh_25 - lowestLow_10) * 4, 4)

关键水平:
    0.2  bottom (深底)        →  入场候选 (低频/高准)
    0.5  attention (注意)     →  入场候选 (中频)
    1.75 strong/weak boundary →  仅作图
    3.2  stage sell           →  退出
    3.5  sell-all             →  退出

Buy-only 项目里 4 类 Pine 信号映射:
    B (lime)        : momentum ↑ 0.5      → entry candidate
    TriB (teal)     : momentum ↑ 0.2      → entry candidate (deep V)
    S (red)         : momentum ↓ 3.5      → exit
    TriS (orange)   : momentum ↓ 3.2/3.5  → exit (early stage)

在原 Pine 之上叠加的信号质量过滤:
    1. 深 V 过滤        过去 20 bar 内动量线必须触及过 ≤ 0.2 (确认真反转)
    2. 趋势过滤        close > SMA60 × 0.95 (容许小幅在均线下方)
    3. 量能确认        vol > VMA20 × 1.05
    4. 排除涨停板第一天买入
    5. ATR 贪婪 trail 退出 + momentum 下穿 3.2/3.5 强制平仓
    6. 可选 ml_score 阈值 (dt-40 / dt-50)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .base import StrategyTemplate


def _shift1(a):
    return np.concatenate(([a[0]], a[:-1]))


def _cross_above(s, level):
    s = np.asarray(s, dtype=float); p = _shift1(s)
    return (p <= level) & (s > level)


def _cross_below(s, level):
    s = np.asarray(s, dtype=float); p = _shift1(s)
    return (p >= level) & (s < level)


def compute_momentum_line(close, low, high, llv_n: int = 10, hhv_n: int = 25,
                          ema_n: int = 4) -> np.ndarray:
    """Pine-faithful: EMA((close - LLV(low,10)) / (HHV(high,25) - LLV(low,10)) × 4, 4)."""
    close = np.asarray(close, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    ll = pd.Series(low).rolling(llv_n, min_periods=1).min().values
    hh = pd.Series(high).rolling(hhv_n, min_periods=1).max().values
    rng = hh - ll
    raw = np.where(rng > 0, (close - ll) / np.where(rng > 0, rng, 1.0) * 4.0, 0.0)
    ema = pd.Series(raw).ewm(span=ema_n, adjust=False).mean().values
    return ema


def _compute_filter_pack(close, high, low, vol):
    """Trend (SMA60) + 量能 (VMA20) + ATR(14)."""
    n = len(close)
    sc = pd.Series(close); sv = pd.Series(vol)
    sma60 = sc.rolling(60, min_periods=10).mean().values
    vma20 = sv.rolling(20, min_periods=5).mean().values
    prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr14 = pd.Series(tr).ewm(alpha=1 / 14.0, adjust=False).mean().values
    return {"sma60": sma60, "vma20": vma20, "atr14": atr14}


def detect_bullish_divergence(close, open_, high, low, vol, momentum,
                              window: int = 30, gap: int = 10,
                              require_above_sma20: bool = True) -> np.ndarray:
    """底背驰 + 反弹日确认.

    在 bar i:
      1) 之前 (i-1) 已创出过 window 内最低 close — 即 i-1 是当时的窗口低
      2) 在前段 [i-window, i-gap] 区间另有一个低点 j
      3) close[i-1] < close[j]      (价格在 bar i-1 创出新低)
      4) momentum[i-1] > momentum[j] (动量未跟随新低)
      5) bar i 是反弹日: close[i] > close[i-1] 且 close[i] > open[i] (收阳)
      6) bar i 量能 > VMA20
    （把信号点定在反弹日，而非新低日，避免下跌中被打脸）
    """
    n = len(close)
    div = np.zeros(n, dtype=bool)
    if n < window + 2:
        return div
    vma20 = pd.Series(vol).rolling(20, min_periods=5).mean().values
    sma20 = pd.Series(close).rolling(20, min_periods=5).mean().values
    for i in range(window + 1, n):
        prev = i - 1
        win_close = close[prev - window + 1:prev + 1]
        if close[prev] > win_close.min() + 1e-9:
            continue
        prev_end = prev - gap
        prev_start = prev - window + 1
        if prev_end <= prev_start:
            continue
        prev_slice = close[prev_start:prev_end]
        j_rel = int(np.argmin(prev_slice))
        j = prev_start + j_rel
        if prev - j < gap:
            continue
        if close[prev] >= close[j] - 1e-9:
            continue
        if momentum[prev] <= momentum[j]:
            continue
        # 反弹日确认
        if close[i] <= close[prev]:
            continue
        if close[i] <= open_[i]:
            continue
        if not np.isnan(vma20[i]) and vol[i] < vma20[i]:
            continue
        # 反弹日收盘价必须上穿 SMA20 (短期趋势翻转确认，避免接飞刀)
        if require_above_sma20 and not np.isnan(sma20[i]) and close[i] <= sma20[i]:
            continue
        div[i] = True
    return div


def detect_double_dip(close, open_, vol, momentum, lookback: int = 80,
                      first_thr: float = 0.2,
                      second_thr: float = 0.3,
                      bounce_thr: float = 1.5,
                      min_gap: int = 15,
                      price_tol: float = 0.93) -> np.ndarray:
    """双次偏离 (W 底): 第一次深底 → 显著反弹 → 第二次浅底 → 反弹日确认.

    在 bar i:
      1) 上穿 0.5 (动量回升触发)
      2) 过去 `lookback` bar 内: 早段有 momentum ≤ 0.2 的"第一坑"
      3) 第一坑后的反弹峰 momentum ≥ 1.5 (真正离开底部，非震荡)
      4) 反弹后又出现 momentum ≤ 0.3 的"第二坑"
      5) 第二坑距第一坑 ≥ 15 bar (避免短期双重测试)
      6) 第二坑收盘价 ≥ 第一坑 × 0.93 (W 形非破位)
      7) bar i 收阳 (close > open) + 量能 > VMA20
    """
    n = len(close)
    sig = np.zeros(n, dtype=bool)
    if n < lookback + 2:
        return sig
    cross_up = _cross_above(momentum, 0.5)
    vma20 = pd.Series(vol).rolling(20, min_periods=5).mean().values
    for i in range(lookback, n):
        if not cross_up[i]:
            continue
        win_mom = momentum[i - lookback:i]
        win_close = close[i - lookback:i]
        first = np.where(win_mom <= first_thr)[0]
        if len(first) == 0:
            continue
        first_idx = int(first[0])
        after_first = win_mom[first_idx + 1:]
        if len(after_first) == 0 or after_first.max() < bounce_thr:
            continue
        bounce_peak_idx = first_idx + 1 + int(after_first.argmax())
        if bounce_peak_idx >= lookback - 2:
            continue
        after_peak_mom = win_mom[bounce_peak_idx + 1:]
        recent = np.where(after_peak_mom <= second_thr)[0]
        if len(recent) == 0:
            continue
        second_idx = bounce_peak_idx + 1 + int(recent[-1])
        if second_idx - first_idx < min_gap:
            continue
        if win_close[second_idx] < win_close[first_idx] * price_tol:
            continue
        # 反弹日确认
        if close[i] <= open_[i]:
            continue
        if not np.isnan(vma20[i]) and vol[i] < vma20[i]:
            continue
        sig[i] = True
    return sig


def dt_entry_mask(close, high, low, vol, momentum, open_=None, score=None,
                  ml_thr: float | None = None,
                  modes: tuple[str, ...] = ("deep_v",),
                  trend_tol: float = 0.95,
                  vol_mult: float = 1.05) -> np.ndarray:
    """Entry mask — 任选 'deep_v' / 'div' / 'w' 模式 (OR 融合).

      - deep_v: 动量 ↑ 0.5 且过去 20 bar 触及过 ≤ 0.2  (信号自带量能/趋势 filter)
      - div:    底背驰 (价格新低 + 动量未新低 + 反弹日 + 放量)  ← 自带量/反弹过滤
      - w:      W 双底 (深坑 + 反弹 ≥ 1.5 + 浅坑 + ≥15bar 间距 + 反弹日 + 放量)

    deep_v 走外部 trend_ok / vol_ok / not_limit_up 过滤；
    div / w 在内部已做反弹日 + 量能确认，外部只做趋势 + 涨停。
    """
    n = len(close)
    if open_ is None:
        open_ = close  # fallback：若调用方没传 open，反弹日条件退化
    pack = _compute_filter_pack(close, high, low, vol)

    raw = np.zeros(n, dtype=bool)
    if "deep_v" in modes:
        cross_attention = _cross_above(momentum, 0.5)
        cross_bottom = _cross_above(momentum, 0.2)
        was_bottomed = (
            pd.Series((momentum <= 0.2).astype(float))
            .rolling(20, min_periods=1).max().values > 0
        )
        deep_v_sig = (cross_attention & was_bottomed) | cross_bottom
        # deep_v 需要外部的量能确认
        vma20 = pack["vma20"]
        vol_ok = np.where(np.isnan(vma20), True, vol > vma20 * vol_mult)
        deep_v_sig = deep_v_sig & vol_ok
        raw = raw | deep_v_sig
    if "div" in modes:
        raw = raw | detect_bullish_divergence(close, open_, high, low, vol, momentum)
    if "w" in modes:
        raw = raw | detect_double_dip(close, open_, vol, momentum)

    # 趋势 + 涨停过滤 (所有模式都过)
    sma60 = pack["sma60"]
    trend_ok = np.where(np.isnan(sma60), True, close > sma60 * trend_tol)
    prev_c = _shift1(close)
    not_limit_up = ~(close > prev_c * 1.099)

    mask = raw & trend_ok & not_limit_up
    if ml_thr is not None and score is not None:
        mask = mask & (np.asarray(score, dtype=float) >= ml_thr)
    return mask.astype(bool)


# =========================================================================
# Exit walker — ATR greedy trail + force-exit on momentum cross-below 3.2/3.5
# =========================================================================

def _walk_diao_di(i: int, close, high, low, momentum, p, n: int):
    entry = close[i]
    hard_stop = entry * 0.90
    atr_mult = p.get("atr_mult", 2.0)
    activation = p.get("trail_activation", 0.06)
    time_stop = p.get("time_stop", 60)
    peak = high[i]
    activated = False
    alpha = 1.0 / 14.0
    atr = max(high[i] - low[i], 1e-9)
    end = min(i + time_stop, n - 1)
    for k in range(i + 1, end + 1):
        prev_c = close[k - 1] if k - 1 >= 0 else close[k]
        tr = max(high[k] - low[k], abs(high[k] - prev_c), abs(low[k] - prev_c))
        atr = (1 - alpha) * atr + alpha * tr
        peak = max(peak, high[k])
        if peak / entry - 1 >= activation:
            activated = True

        # Hard stop
        if low[k] <= hard_stop:
            return k, hard_stop

        # ATR trail
        if activated and atr > 0:
            stop = peak - atr_mult * atr
            if low[k] <= stop:
                return k, max(stop, hard_stop)

        # Force exit on momentum cross-below 3.2 or 3.5
        if k > 0:
            if (momentum[k - 1] >= 3.5 and momentum[k] < 3.5) or \
               (momentum[k - 1] >= 3.2 and momentum[k] < 3.2):
                return k, close[k]

    return end, close[end]


# =========================================================================
# Strategy classes
# =========================================================================

class _DiaoDiBase(StrategyTemplate):
    template_id = "dt-base"
    _entry_name = ""
    _params: dict = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 60}
    _ml_thr: float | None = None
    _modes: tuple[str, ...] = ("deep_v", "div", "w")
    _trend_tol: float = 0.95
    _vol_mult: float = 1.05

    def __init__(self, ts_code: str | None = None):
        self.ts_code = ts_code

    @staticmethod
    def parameter_candidates() -> list[dict]:
        return [{}]

    @property
    def name(self) -> str:
        return self._entry_name

    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        if len(df) < 100:
            return []
        close = df["close"].astype(float).values
        open_ = df["open"].astype(float).values
        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        vol = df["vol"].astype(float).values
        dates = df["trade_date"]

        momentum = compute_momentum_line(close, low, high)

        score = None
        if self._ml_thr is not None and "ml_score" in df.columns:
            score = df["ml_score"].astype(float).values

        sig_mask = dt_entry_mask(
            close, high, low, vol, momentum, open_=open_, score=score,
            ml_thr=self._ml_thr, modes=self._modes,
            trend_tol=self._trend_tol, vol_mult=self._vol_mult,
        )

        signals: list[dict] = []
        in_pos = False
        exit_until = -1
        n = len(close)
        for i in range(60, n - 1):
            if in_pos:
                if i >= exit_until:
                    in_pos = False
                continue
            if not sig_mask[i]:
                continue
            exit_idx, _ = _walk_diao_di(i, close, high, low, momentum, self._params, n)
            signals.append({"date": dates.iloc[i], "action": "buy"})
            signals.append({"date": dates.iloc[exit_idx], "action": "sell"})
            in_pos = True
            exit_until = exit_idx
        return signals


class DiaoDiDiv(_DiaoDiBase):
    template_id = "dt-div"
    _entry_name = "抄底逃顶 底背驰"
    _params = {"atr_mult": 2.5, "trail_activation": 0.08, "time_stop": 90}
    _ml_thr = None
    _modes = ("div",)
    _trend_tol = 0.95   # 背驰也得在 SMA60 附近发生（确认下跌趋缓）
    _vol_mult = 1.0


class DiaoDiW(_DiaoDiBase):
    template_id = "dt-w"
    _entry_name = "抄底逃顶 W 双底"
    _params = {"atr_mult": 2.2, "trail_activation": 0.07, "time_stop": 75}
    _ml_thr = None
    _modes = ("w",)
    _trend_tol = 0.95
    _vol_mult = 1.0


class DiaoDiPure(_DiaoDiBase):
    template_id = "dt-pure"
    _entry_name = "抄底逃顶 深V反转"
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 60}
    _ml_thr = None
    _modes = ("deep_v",)


class DiaoDi20(_DiaoDiBase):
    template_id = "dt-20"
    _entry_name = "抄底逃顶 深V + ML 0.20"
    _params = {"atr_mult": 2.0, "trail_activation": 0.06, "time_stop": 60}
    _ml_thr = 0.20
    _modes = ("deep_v",)


class DiaoDi30(_DiaoDiBase):
    template_id = "dt-30"
    _entry_name = "抄底逃顶 深V + ML 0.30"
    _params = {"atr_mult": 2.2, "trail_activation": 0.07, "time_stop": 75}
    _ml_thr = 0.30
    _modes = ("deep_v",)
