"""缠论 (Chan Theory) — single-level (daily) implementation.

Pipeline:
  raw OHLC bars
    ↓ inclusion processing (含包关系)
  merged bars
    ↓ fractal detection (分型)
  fractals (top / bottom alternating)
    ↓ stroke validation (笔识别 — 至少 5 根原始 K + 关键点过滤)
  strokes
    ↓ MACD area divergence (背驰)
  signals: 1类 / 2类 / 3类 买点

Reference: 缠中说禅 blog 教你炒股票 series; community implementations.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# =========================================================================
# Step 1: Inclusion processing (含包关系)
# =========================================================================

@dataclass
class MergedBar:
    """One bar after inclusion-merging adjacent K-lines."""
    idx: int          # index of the LAST raw bar that contributes to this merged bar
    high: float
    low: float
    direction: int    # +1 = formed by up trend, -1 = down trend, 0 = first bar


def merge_inclusion(highs: np.ndarray, lows: np.ndarray) -> list[MergedBar]:
    """Apply inclusion-resolution rules.

    Rule:
      A contains B iff (A.high >= B.high) AND (A.low <= B.low). Merge into:
        - if trend was up:    high = max(A.high, B.high), low = max(A.low, B.low)
        - if trend was down:  high = min(A.high, B.high), low = min(A.low, B.low)
      Trend at the moment of merge = sign of (A.high - prev_merged.high).
    """
    if len(highs) == 0:
        return []
    bars: list[MergedBar] = [MergedBar(0, float(highs[0]), float(lows[0]), 0)]
    direction = 0  # set after first non-equal pair
    for i in range(1, len(highs)):
        h, lo = float(highs[i]), float(lows[i])
        last = bars[-1]
        # Inclusion check
        if (last.high >= h and last.low <= lo) or (h >= last.high and lo <= last.low):
            if direction >= 0:   # default to up if undefined
                merged = MergedBar(
                    idx=i,
                    high=max(last.high, h),
                    low=max(last.low, lo),
                    direction=last.direction,
                )
            else:
                merged = MergedBar(
                    idx=i,
                    high=min(last.high, h),
                    low=min(last.low, lo),
                    direction=last.direction,
                )
            bars[-1] = merged
        else:
            # Update direction based on movement
            if h > last.high:
                direction = 1
            elif h < last.high:
                direction = -1
            bars.append(MergedBar(i, h, lo, direction))
    return bars


# =========================================================================
# Step 2: Fractal detection (分型)
# =========================================================================

@dataclass
class Fractal:
    bar_idx: int     # original bar index (in input arrays)
    merged_idx: int  # index in merged-bars list
    type: str        # "top" or "bottom"
    price: float


def find_fractals(bars: list[MergedBar]) -> list[Fractal]:
    """3 consecutive merged bars form a fractal.

    top:    bars[i].high is strictly highest of i-1..i+1, low strictly highest
    bottom: bars[i].low strictly lowest of i-1..i+1, high strictly lowest
    """
    out: list[Fractal] = []
    for i in range(1, len(bars) - 1):
        a, b, c = bars[i - 1], bars[i], bars[i + 1]
        if b.high > a.high and b.high > c.high and b.low > a.low and b.low > c.low:
            out.append(Fractal(b.idx, i, "top", b.high))
        elif b.low < a.low and b.low < c.low and b.high < a.high and b.high < c.high:
            out.append(Fractal(b.idx, i, "bottom", b.low))
    return out


# =========================================================================
# Step 3: Stroke validation (笔识别)
# =========================================================================

@dataclass
class Stroke:
    start: Fractal
    end: Fractal
    direction: str   # "up" (bottom→top) or "down" (top→bottom)
    bars_between: int


def find_strokes(fractals: list[Fractal], min_bars: int = 5) -> list[Stroke]:
    """Connect alternating fractals into strokes.

    Rules (simplified per common 缠论 conventions):
      - alternating top/bottom
      - at least `min_bars` raw bars between fractal endpoints (4 merged bars)
      - if two same-type fractals appear consecutively, keep the more extreme one
    """
    if not fractals:
        return []

    cleaned: list[Fractal] = []
    for f in fractals:
        if not cleaned:
            cleaned.append(f); continue
        prev = cleaned[-1]
        if prev.type == f.type:
            # keep the more extreme
            if f.type == "top" and f.price > prev.price:
                cleaned[-1] = f
            elif f.type == "bottom" and f.price < prev.price:
                cleaned[-1] = f
            # else: keep prev
        else:
            # opposite type — only accept if min_bars apart
            if f.bar_idx - prev.bar_idx >= min_bars:
                cleaned.append(f)
            # else: ignore (too close)

    strokes: list[Stroke] = []
    for i in range(1, len(cleaned)):
        a, b = cleaned[i - 1], cleaned[i]
        direction = "up" if (a.type == "bottom" and b.type == "top") else "down"
        strokes.append(Stroke(a, b, direction, b.bar_idx - a.bar_idx))
    return strokes


# =========================================================================
# Step 4: MACD area divergence (背驰)
# =========================================================================

def macd_arrays(close: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sc = pd.Series(close)
    ema12 = sc.ewm(span=12, adjust=False).mean().values
    ema26 = sc.ewm(span=26, adjust=False).mean().values
    diff = ema12 - ema26
    dea = pd.Series(diff).ewm(span=9, adjust=False).mean().values
    hist = (diff - dea) * 2
    return diff, dea, hist


def macd_negative_area(hist: np.ndarray, start: int, end: int) -> float:
    """Sum of |hist[i]| where hist[i] < 0, between start..end inclusive."""
    if end < start:
        return 0.0
    seg = hist[start:end + 1]
    return float(-seg[seg < 0].sum())


def macd_positive_area(hist: np.ndarray, start: int, end: int) -> float:
    if end < start:
        return 0.0
    seg = hist[start:end + 1]
    return float(seg[seg > 0].sum())


# =========================================================================
# Step 5: Class-1 buy point (1类买点)
# =========================================================================

@dataclass
class BuyPoint:
    bar_idx: int       # entry bar
    type: str          # "1" / "2" / "3"
    price: float
    note: str          # short explanation


def find_class1_buys(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> list[BuyPoint]:
    """1类买点: latest down-stroke makes new price low BUT MACD negative
    area is smaller than the previous down-stroke's negative area
    (底背驰 / bullish divergence).

    Entry = bar of the bottom fractal that closed the down stroke.
    """
    bars = merge_inclusion(high, low)
    fractals = find_fractals(bars)
    strokes = find_strokes(fractals, min_bars=5)
    if len(strokes) < 3:
        return []
    _, _, hist = macd_arrays(close)

    out: list[BuyPoint] = []
    # Look at down-stroke pairs: previous down vs current down
    down_strokes = [(i, s) for i, s in enumerate(strokes) if s.direction == "down"]
    for k in range(1, len(down_strokes)):
        prev_i, prev = down_strokes[k - 1]
        cur_i, cur = down_strokes[k]
        # Conditions
        # 1. Current bottom is at or below previous bottom (new low)
        if cur.end.price >= prev.end.price * 1.001:
            continue
        # 2. MACD negative area shrinks
        prev_area = macd_negative_area(hist, prev.start.bar_idx, prev.end.bar_idx)
        cur_area = macd_negative_area(hist, cur.start.bar_idx, cur.end.bar_idx)
        if prev_area <= 0 or cur_area >= prev_area * 0.95:
            continue
        # 3. Strokes are not too far apart in time (within 90 raw bars)
        if cur.start.bar_idx - prev.end.bar_idx > 90:
            continue

        out.append(BuyPoint(
            bar_idx=cur.end.bar_idx,
            type="1",
            price=cur.end.price,
            note=f"1类买点：底背驰 (前段面积 {prev_area:.2f}, 本段 {cur_area:.2f})",
        ))
    return out


def find_class2_buys(close: np.ndarray, high: np.ndarray, low: np.ndarray,
                     class1_idx_set: set[int]) -> list[BuyPoint]:
    """2类买点: 1类后第一次反弹高点（顶分型）后的回踩低点（底分型），
    且新低高于 1 类买点的低点。"""
    bars = merge_inclusion(high, low)
    fractals = find_fractals(bars)
    strokes = find_strokes(fractals, min_bars=5)

    out: list[BuyPoint] = []
    for i in range(2, len(strokes)):
        s_prev = strokes[i - 2]
        s_mid = strokes[i - 1]
        s_now = strokes[i]
        # Pattern: down (prev = 1类), up, down, looking for the bottom of s_now
        if not (s_prev.direction == "down" and s_mid.direction == "up"
                and s_now.direction == "down"):
            continue
        # 1类 buy must have happened at s_prev's end
        if s_prev.end.bar_idx not in class1_idx_set:
            continue
        # New low MUST be above the 1类 low
        if s_now.end.price <= s_prev.end.price:
            continue
        out.append(BuyPoint(
            bar_idx=s_now.end.bar_idx,
            type="2",
            price=s_now.end.price,
            note=f"2类买点：1类后回踩不破底（前1类 {s_prev.end.price:.2f}, 本次 {s_now.end.price:.2f}）",
        ))
    return out


def find_all_buys(close: np.ndarray, high: np.ndarray, low: np.ndarray) -> list[BuyPoint]:
    """Return all class-1 + class-2 buy points (sorted by bar_idx)."""
    c1 = find_class1_buys(close, high, low)
    c1_idx = {b.bar_idx for b in c1}
    c2 = find_class2_buys(close, high, low, c1_idx)
    out = c1 + c2
    out.sort(key=lambda b: b.bar_idx)
    return out


# =========================================================================
# Diagnostics — useful for debugging in script form
# =========================================================================

def diagnose(close, high, low) -> dict:
    bars = merge_inclusion(high, low)
    fractals = find_fractals(bars)
    strokes = find_strokes(fractals, min_bars=5)
    buys = find_all_buys(close, high, low)
    return {
        "n_raw_bars": len(close),
        "n_merged_bars": len(bars),
        "n_fractals": len(fractals),
        "n_strokes": len(strokes),
        "n_class1": sum(1 for b in buys if b.type == "1"),
        "n_class2": sum(1 for b in buys if b.type == "2"),
    }
