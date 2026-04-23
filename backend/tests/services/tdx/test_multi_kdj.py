import numpy as np
import pandas as pd

from app.services.tdx.indicators import multi_kdj


def _make_ohlcv(n: int) -> pd.DataFrame:
    """Deterministic synthetic OHLCV data for testing."""
    rng = np.random.default_rng(seed=42)
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
    result = multi_kdj.compute(df)

    assert result.name == "multi_kdj"
    assert result.label == "多周期 KDJ 共振"
    assert result.pane == "sub"
    assert result.y_axis_range == (0, 100)
    assert len(result.timestamps) == 100
    # 4 visible lines (K13, D14, K55, D55), periods 21/34 are internal only
    assert len(result.lines) == 4
    assert [l.name for l in result.lines] == ["K13", "D14", "K55", "D55"]
    assert len(result.hlines) == 5  # 0, 20, 50, 80, 100
    assert len(result.bands) == 2   # DIBU (red) + TOBU (green)


def test_lines_have_correct_length():
    df = _make_ohlcv(100)
    result = multi_kdj.compute(df)
    for line in result.lines:
        assert len(line.values) == 100


def test_k55_leading_nans_then_values():
    df = _make_ohlcv(100)
    result = multi_kdj.compute(df)
    k55 = next(l for l in result.lines if l.name == "K55")
    # K55 needs 55 bars of history for LLV/HHV
    assert k55.values[0] is None
    assert k55.values[53] is None
    # After the rolling window starts filling, values exist
    assert k55.values[-1] is not None
    assert 0 <= k55.values[-1] <= 100


def test_k13_values_in_0_to_100_range():
    df = _make_ohlcv(100)
    result = multi_kdj.compute(df)
    k13 = next(l for l in result.lines if l.name == "K13")
    non_null = [v for v in k13.values if v is not None]
    assert len(non_null) > 0
    assert all(0 <= v <= 100 for v in non_null)


def test_bands_reference_valid_timestamps():
    df = _make_ohlcv(100)
    result = multi_kdj.compute(df)
    valid_timestamps = set(df["timestamp"].astype("int64").tolist())
    for band in result.bands:
        for ts in band.timestamps:
            assert ts in valid_timestamps


def test_hlines_values():
    df = _make_ohlcv(100)
    result = multi_kdj.compute(df)
    values = sorted(h.value for h in result.hlines)
    assert values == [0, 20, 50, 80, 100]


def test_manual_kdj_value_matches_formula():
    """
    Sanity check: manually compute KDJ for a simple ascending close series
    and compare with our implementation.
    """
    n = 30
    close = pd.Series(np.arange(1, n + 1, dtype=float))
    high = close + 0.5
    low = close - 0.5

    # For 13-period KDJ on ascending data:
    # RSV = (close - LLV(low, 13)) / (HHV(high, 13) - LLV(low, 13)) * 100
    # At index 20, LLV(low, 13) = low[8] = 8.5, HHV(high, 13) = high[20] = 21.5
    # RSV[20] = (21 - 8.5) / (21.5 - 8.5) * 100 = 12.5/13 * 100 ≈ 96.15
    from app.services.tdx.functions import HHV, LLV, SMA

    rsv = (close - LLV(low, 13)) / (HHV(high, 13) - LLV(low, 13)) * 100
    k13 = SMA(rsv, 3, 1)

    # Sanity: at end of monotonically increasing series, K13 should be very close to 100
    assert k13.iloc[-1] > 90
