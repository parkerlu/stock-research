"""
TDX (通达信) indicator function library.

Implements the numerical primitives used in Tongda Xin formulas, based on
pandas/numpy. Inspired by MyTT (https://github.com/mpquant/MyTT) but written
independently to avoid GPL propagation.

All functions accept pandas Series and return pandas Series of the same length.
Leading bars that lack enough history return NaN.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def MA(series: pd.Series, n: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window=n, min_periods=n).mean()


def EMA(series: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with alpha = 2/(n+1)."""
    return series.ewm(span=n, adjust=False).mean()


def SMA(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """
    TDX-style SMA (not a standard SMA).

    Recurrence: Y_today = (m * X_today + (n - m) * Y_yesterday) / n
    Equivalent to EMA with alpha = m / n.
    """
    alpha = m / n
    return series.ewm(alpha=alpha, adjust=False).mean()


def HHV(series: pd.Series, n: int) -> pd.Series:
    """Highest value over the trailing n bars (inclusive of current)."""
    return series.rolling(window=n, min_periods=n).max()


def LLV(series: pd.Series, n: int) -> pd.Series:
    """Lowest value over the trailing n bars (inclusive of current)."""
    return series.rolling(window=n, min_periods=n).min()


def REF(series: pd.Series, n: int) -> pd.Series:
    """Reference the value n bars ago."""
    return series.shift(n)


def CROSS(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    Returns 1 where `a` crosses above `b`, else 0.

    A cross happens when a_prev <= b_prev and a_today > b_today.
    """
    a_prev = a.shift(1)
    b_prev = b.shift(1)
    crossed = (a > b) & (a_prev <= b_prev)
    return crossed.astype(int)


def IF(cond: pd.Series, yes, no) -> pd.Series:
    """Vectorized ternary. `yes` and `no` can be Series or scalars."""
    return pd.Series(np.where(cond, yes, no), index=cond.index)


def ABS(series: pd.Series) -> pd.Series:
    return series.abs()


def MAX(a: pd.Series, b) -> pd.Series:
    """Elementwise max. `b` can be Series or scalar."""
    if isinstance(b, pd.Series):
        return pd.Series(np.maximum(a.values, b.values), index=a.index)
    return pd.Series(np.maximum(a.values, b), index=a.index)


def MIN(a: pd.Series, b) -> pd.Series:
    """Elementwise min. `b` can be Series or scalar."""
    if isinstance(b, pd.Series):
        return pd.Series(np.minimum(a.values, b.values), index=a.index)
    return pd.Series(np.minimum(a.values, b), index=a.index)


def SUM(series: pd.Series, n: int) -> pd.Series:
    """Trailing n-bar rolling sum."""
    return series.rolling(window=n, min_periods=n).sum()


def COUNT(cond: pd.Series, n: int) -> pd.Series:
    """Number of bars in the trailing n bars where `cond` is true."""
    return cond.astype(int).rolling(window=n, min_periods=n).sum()


def EVERY(cond: pd.Series, n: int) -> pd.Series:
    """1 if `cond` is true in every of the trailing n bars, else 0."""
    return (COUNT(cond, n) == n).astype(int)


def EXIST(cond: pd.Series, n: int) -> pd.Series:
    """1 if `cond` is true at least once in the trailing n bars, else 0."""
    return (COUNT(cond, n) > 0).astype(int)
