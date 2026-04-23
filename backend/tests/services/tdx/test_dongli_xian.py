import numpy as np
import pandas as pd

from app.services.tdx.indicators import dongli_xian


def _make_ohlcv(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed=123)
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
    result = dongli_xian.compute(df)

    assert result.name == "dongli_xian"
    assert result.label == "动力线（阶段指标）"
    assert result.pane == "sub"
    assert result.y_axis_range == (0, 100)
    assert len(result.timestamps) == 100
    assert len(result.lines) == 1
    assert result.lines[0].name == "动力线"
    # 4 reference lines: 0.2, 0.5, 3.2, 3.5
    assert len(result.hlines) == 4
    assert sorted(h.value for h in result.hlines) == [0.2, 0.5, 3.2, 3.5]
    # 4 band types: 阶段底部, 阶段关注, 清仓, 短线卖出
    assert len(result.bands) == 4


def test_dongli_line_in_expected_range():
    df = _make_ohlcv(100)
    result = dongli_xian.compute(df)
    dongli = result.lines[0].values
    non_null = [v for v in dongli if v is not None]
    assert len(non_null) > 0
    # EMA((close - LLV) / (HHV - LLV) * 4, 4) — raw value is in [0, 4],
    # EMA preserves that range
    assert all(0 <= v <= 4 for v in non_null), f"out-of-range values: {non_null}"


def test_bands_reference_valid_timestamps():
    df = _make_ohlcv(100)
    result = dongli_xian.compute(df)
    valid_timestamps = set(df["timestamp"].astype("int64").tolist())
    for band in result.bands:
        for ts in band.timestamps:
            assert ts in valid_timestamps


def test_stage_bottom_markers_match_bands():
    """DRAWICON(阶段底部, ...) must have the same timestamps as the red band."""
    df = _make_ohlcv(100)
    result = dongli_xian.compute(df)
    red_band = next(b for b in result.bands if b.color == "#FF0000")
    marker_timestamps = sorted(m.timestamp for m in result.markers)
    assert sorted(red_band.timestamps) == marker_timestamps


def test_min_bars_enforced():
    # The indicator needs at least 25 bars for HHV(high, 25) to start producing
    assert dongli_xian.min_bars == 25
