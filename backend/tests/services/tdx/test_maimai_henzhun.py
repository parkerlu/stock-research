import numpy as np
import pandas as pd

from app.services.tdx.indicators import maimai_henzhun


def _make_ohlcv(n: int, seed: int = 77) -> pd.DataFrame:
    rng = np.random.default_rng(seed=seed)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    open_ = close + rng.normal(0, 0.5, n)
    timestamps = [1_700_000_000_000 + i * 86400000 for i in range(n)]
    return pd.DataFrame({
        "timestamp": timestamps,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    })


def test_compute_structure():
    df = _make_ohlcv(100)
    result = maimai_henzhun.compute(df)

    assert result.name == "maimai_henzhun"
    assert result.label == "买卖很准"
    assert result.y_axis_range == (0, 100)
    # 5 lines: 急卖奇准, 短卖奇准, 急买奇准, 短买奇准, 准备现金
    line_names = [l.name for l in result.lines]
    assert line_names == ["急卖奇准", "短卖奇准", "急买奇准", "短买奇准", "准备现金"]
    # 3 reference hlines (top/mid/bottom)
    assert len(result.hlines) == 3


def test_sell_lines_are_100_or_50():
    df = _make_ohlcv(100)
    result = maimai_henzhun.compute(df)
    jimai = next(l for l in result.lines if l.name == "急卖奇准")
    duanmai = next(l for l in result.lines if l.name == "短卖奇准")
    for line in (jimai, duanmai):
        non_null = [v for v in line.values if v is not None]
        assert all(v in (50.0, 100.0) for v in non_null), f"unexpected values in {line.name}"


def test_buy_lines_are_0_or_50():
    df = _make_ohlcv(100)
    result = maimai_henzhun.compute(df)
    jibuy = next(l for l in result.lines if l.name == "急买奇准")
    duanbuy = next(l for l in result.lines if l.name == "短买奇准")
    for line in (jibuy, duanbuy):
        non_null = [v for v in line.values if v is not None]
        assert all(v in (0.0, 50.0) for v in non_null), f"unexpected values in {line.name}"


def test_zhunbei_xianjin_is_0_or_80():
    df = _make_ohlcv(200)
    result = maimai_henzhun.compute(df)
    zhunbei = next(l for l in result.lines if l.name == "准备现金")
    non_null = [v for v in zhunbei.values if v is not None]
    # Some values may be NaN near the start due to divide-by-zero guards
    for v in non_null:
        assert v in (0.0, 80.0), f"unexpected 准备现金 value: {v}"


def test_markers_structure_valid():
    df = _make_ohlcv(100)
    result = maimai_henzhun.compute(df)
    # All markers must have a valid icon and reference a real bar timestamp
    valid_ts = set(df["timestamp"].astype("int64").tolist())
    for m in result.markers:
        assert m.icon in ("triangle_up", "triangle_down", "dot")
        assert m.timestamp in valid_ts


def test_min_bars():
    assert maimai_henzhun.min_bars == 15
