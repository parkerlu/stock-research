import numpy as np
import pandas as pd
import pytest

from app.services.tdx.functions import (
    ABS,
    COUNT,
    CROSS,
    EMA,
    EVERY,
    EXIST,
    HHV,
    IF,
    LLV,
    MA,
    MAX,
    MIN,
    REF,
    SMA,
    SUM,
)


def s(values):
    return pd.Series(values, dtype=float)


def test_ma_basic():
    result = MA(s([1, 2, 3, 4, 5]), 3)
    # First two bars NaN, then (1+2+3)/3=2, (2+3+4)/3=3, (3+4+5)/3=4
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 2.0
    assert result.iloc[3] == 3.0
    assert result.iloc[4] == 4.0


def test_ema_matches_pandas_ewm():
    series = s([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    # alpha = 2/(3+1) = 0.5
    expected = series.ewm(span=3, adjust=False).mean()
    result = EMA(series, 3)
    assert np.allclose(result.values, expected.values)


def test_sma_china_style():
    # TDX SMA recurrence with n=3, m=1 means alpha = 1/3.
    # Y[0] = X[0] = 1
    # Y[1] = (1*2 + 2*1)/3 = 4/3
    # Y[2] = (1*3 + 2*4/3)/3 = (3 + 8/3)/3 = 17/9
    result = SMA(s([1, 2, 3]), 3, 1)
    assert abs(result.iloc[0] - 1.0) < 1e-9
    assert abs(result.iloc[1] - 4 / 3) < 1e-9
    assert abs(result.iloc[2] - 17 / 9) < 1e-9


def test_sma_alpha_equals_m_over_n():
    series = s([10, 11, 12, 13, 14, 15])
    ours = SMA(series, 5, 2)  # alpha = 2/5 = 0.4
    expected = series.ewm(alpha=0.4, adjust=False).mean()
    assert np.allclose(ours.values, expected.values)


def test_hhv_rolling_max():
    result = HHV(s([5, 3, 8, 2, 9, 1, 7]), 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 8.0  # max(5,3,8)
    assert result.iloc[3] == 8.0  # max(3,8,2)
    assert result.iloc[4] == 9.0  # max(8,2,9)
    assert result.iloc[5] == 9.0  # max(2,9,1)
    assert result.iloc[6] == 9.0  # max(9,1,7)


def test_llv_rolling_min():
    result = LLV(s([5, 3, 8, 2, 9, 1, 7]), 3)
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 3.0
    assert result.iloc[3] == 2.0
    assert result.iloc[4] == 2.0
    assert result.iloc[5] == 1.0
    assert result.iloc[6] == 1.0


def test_ref_shifts_back():
    result = REF(s([10, 20, 30, 40]), 2)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 10.0
    assert result.iloc[3] == 20.0


def test_cross_golden_cross():
    # a starts below b then crosses above
    a = s([1, 2, 3, 5, 6])
    b = s([4, 4, 4, 4, 4])
    result = CROSS(a, b)
    assert result.iloc[0] == 0
    assert result.iloc[1] == 0
    assert result.iloc[2] == 0  # a=3 < b=4
    assert result.iloc[3] == 1  # a_prev=3 <= 4 and a=5 > 4 ← cross
    assert result.iloc[4] == 0  # already above


def test_cross_no_cross_when_equal_both_sides():
    a = s([3, 4, 4])
    b = s([4, 4, 4])
    result = CROSS(a, b)
    assert (result.values == [0, 0, 0]).all()


def test_if_ternary():
    cond = pd.Series([True, False, True, False])
    result = IF(cond, 1, 0)
    assert result.tolist() == [1, 0, 1, 0]


def test_if_with_series_branches():
    cond = pd.Series([True, False, True])
    yes = s([10, 20, 30])
    no = s([1, 2, 3])
    result = IF(cond, yes, no)
    assert result.tolist() == [10.0, 2.0, 30.0]


def test_abs():
    result = ABS(s([-1, 2, -3, 4]))
    assert result.tolist() == [1.0, 2.0, 3.0, 4.0]


def test_max_scalar():
    result = MAX(s([1, 5, 3]), 4)
    assert result.tolist() == [4.0, 5.0, 4.0]


def test_max_series():
    result = MAX(s([1, 5, 3]), s([4, 2, 6]))
    assert result.tolist() == [4.0, 5.0, 6.0]


def test_min_scalar():
    result = MIN(s([1, 5, 3]), 2)
    assert result.tolist() == [1.0, 2.0, 2.0]


def test_sum_rolling():
    result = SUM(s([1, 2, 3, 4, 5]), 3)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 6.0
    assert result.iloc[3] == 9.0
    assert result.iloc[4] == 12.0


def test_count_condition():
    cond = pd.Series([True, False, True, True, False])
    result = COUNT(cond, 3)
    assert result.iloc[2] == 2.0  # T,F,T → 2
    assert result.iloc[3] == 2.0  # F,T,T → 2
    assert result.iloc[4] == 2.0  # T,T,F → 2


def test_every():
    cond = pd.Series([True, True, True, False, True])
    result = EVERY(cond, 3)
    assert result.iloc[2] == 1  # all three true
    assert result.iloc[3] == 0  # F breaks it
    assert result.iloc[4] == 0


def test_exist():
    cond = pd.Series([False, False, True, False, False, False])
    result = EXIST(cond, 3)
    assert result.iloc[2] == 1  # found True
    assert result.iloc[3] == 1
    assert result.iloc[4] == 1
    assert result.iloc[5] == 0  # True fell off window
