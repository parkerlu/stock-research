import numpy as np
import pandas as pd

from app.services.tdx.indicators import didian_zuhe


def _make_ohlcv(n: int, seed: int = 42) -> pd.DataFrame:
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
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)

    assert result.name == "didian_zuhe"
    assert result.label == "低点组合"
    assert result.pane == "sub"
    assert result.y_axis_range == (0, 100)
    assert len(result.timestamps) == 120


def test_lines_match_spec():
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)

    line_names = [l.name for l in result.lines]
    assert line_names == ["K13", "D14", "K55", "D55", "动力线"]


def test_kdj_lines_in_0_100_range():
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)
    for line in result.lines[:4]:  # K13, D14, K55, D55
        non_null = [v for v in line.values if v is not None]
        assert non_null, f"{line.name} is all null"
        assert all(0 <= v <= 100 for v in non_null), \
            f"{line.name} out of [0,100]: {[v for v in non_null if not 0 <= v <= 100]}"


def test_dongli_line_in_0_4_range():
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)
    dongli = next(l for l in result.lines if l.name == "动力线")
    non_null = [v for v in dongli.values if v is not None]
    assert non_null
    assert all(0 <= v <= 4 for v in non_null), \
        f"动力线 out of [0,4]: {[v for v in non_null if not 0 <= v <= 4]}"


def test_seven_hlines():
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)
    # 5 KDJ refs (0/20/50/80/100) + 2 动力线 thresholds (0.2/3.5)
    assert len(result.hlines) == 7
    values = sorted(h.value for h in result.hlines)
    assert values == [0, 0.2, 3.5, 20, 50, 80, 100]


def test_six_bands():
    df = _make_ohlcv(120)
    result = didian_zuhe.compute(df)
    # DIBU, TOBU, 阶段底部, 阶段关注, 清仓, 短线卖出
    assert len(result.bands) == 6


def test_bands_reference_valid_timestamps():
    df = _make_ohlcv(200)
    result = didian_zuhe.compute(df)
    valid = set(df["timestamp"].astype("int64").tolist())
    for band in result.bands:
        for ts in band.timestamps:
            assert ts in valid


def test_stage_bottom_markers_match_band():
    """Markers fire at the same timestamps as the 阶段底部 band (y1=0.2)."""
    df = _make_ohlcv(200)
    result = didian_zuhe.compute(df)
    stage_bottom_band = next(b for b in result.bands if b.y1 == 0.2)
    marker_ts = sorted(m.timestamp for m in result.markers)
    assert sorted(stage_bottom_band.timestamps) == marker_ts


def test_min_bars_is_55():
    assert didian_zuhe.min_bars == 55
