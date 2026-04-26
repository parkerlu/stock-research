"""Step 2: Build per-stock per-day ML score cache via 5-seed ensemble.

Output: backend/cache/ml_scores.parquet  (ts_code, trade_date, ml_score)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES,
    precompute_dongli,
    precompute_maimai,
    precompute_gp_factors,
)


CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
CSF_PATH = ROOT / "models" / "cross_sectional_ranks.parquet"
SCORES_OUT = CACHE_DIR / "ml_scores.parquet"
ENSEMBLE = [ROOT / "app" / "services" / f"ml_filter_ens_{i}.json" for i in range(5)]


def vectorize_generic_features(df: pd.DataFrame) -> dict:
    """Return dict of 15 generic features (each is a length-N array)."""
    n = len(df)
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["vol"].values.astype(float)

    s_close = pd.Series(close)
    s_high = pd.Series(high)
    s_low = pd.Series(low)
    s_vol = pd.Series(vol)

    ma20 = s_close.rolling(20, min_periods=1).mean().shift(1).values
    ma60 = s_close.rolling(60, min_periods=1).mean().shift(1).values
    ma120 = s_close.rolling(120, min_periods=1).mean().shift(1).values

    high20 = s_high.rolling(20, min_periods=1).max().shift(1).values
    low20 = s_low.rolling(20, min_periods=1).min().shift(1).values
    high60 = s_high.rolling(60, min_periods=1).max().shift(1).values
    low60 = s_low.rolling(60, min_periods=1).min().shift(1).values

    rng20 = high20 - low20
    rng60 = high60 - low60
    pos20 = np.where(rng20 > 0, (close - low20) / np.where(rng20 > 0, rng20, 1) * 100, 50.0)
    pos60 = np.where(rng60 > 0, (close - low60) / np.where(rng60 > 0, rng60, 1) * 100, 50.0)

    def ret_n(n_lag):
        prev = s_close.shift(n_lag).values
        return np.where(prev > 0, close / np.where(prev > 0, prev, 1) - 1, 0.0) * 100

    ret5 = ret_n(5)
    ret20 = ret_n(20)
    ret60 = ret_n(60)

    above_ma20 = np.where(close > ma20, 1.0, 0.0)
    above_ma60 = np.where(close > ma60, 1.0, 0.0)
    above_ma120 = np.where(close > ma120, 1.0, 0.0)

    atr14 = (s_high - s_low).rolling(14, min_periods=1).mean().shift(1).values
    atr14_pct = np.where(close > 0, atr14 / close * 100, 0.0)

    vma20 = s_vol.rolling(20, min_periods=1).mean().shift(1).values
    vol_today_ratio = np.where(vma20 > 0, vol / np.where(vma20 > 0, vma20, 1), 1.0)
    vol_yest_ratio = np.where(vma20 > 0,
                              np.concatenate(([0.0], vol[:-1])) / np.where(vma20 > 0, vma20, 1),
                              1.0)

    prev_close = s_close.shift(1).values
    prev_prev_close = s_close.shift(2).values
    yest_drop_pct = np.where(prev_prev_close > 0,
                             prev_close / np.where(prev_prev_close > 0, prev_prev_close, 1) - 1,
                             0.0) * 100
    today_gain_pct = np.where(prev_close > 0,
                              close / np.where(prev_close > 0, prev_close, 1) - 1,
                              0.0) * 100

    # red_days_10: count of bars in last 10 where close < prev close
    is_red = (s_close < s_close.shift(1)).astype(float)
    red_days_10 = is_red.rolling(10, min_periods=1).sum().shift(1).values

    ma60_distance_pct = np.where(ma60 > 0, close / np.where(ma60 > 0, ma60, 1) - 1, 0.0) * 100

    return {
        "pos20": pos20,
        "pos60": pos60,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "above_ma20": above_ma20,
        "above_ma60": above_ma60,
        "above_ma120": above_ma120,
        "atr14_pct": atr14_pct,
        "vol_today_ratio": vol_today_ratio,
        "vol_yest_ratio": vol_yest_ratio,
        "yest_drop_pct": yest_drop_pct,
        "today_gain_pct": today_gain_pct,
        "red_days_10": red_days_10,
        "ma60_distance_pct": ma60_distance_pct,
    }


def build_features_for_stock(df_stock: pd.DataFrame, csf_lookup: dict) -> pd.DataFrame:
    """Build a DataFrame with all 39 features + (ts_code, trade_date)."""
    n = len(df_stock)
    if n < 130:
        return pd.DataFrame()

    feats = {}
    feats.update(vectorize_generic_features(df_stock))
    feats.update(precompute_maimai(df_stock))
    feats.update(precompute_dongli(df_stock))
    # GP features are vestigial (legacy 3 cols, model doesn't use these names)

    out = pd.DataFrame(feats)
    out["ts_code"] = df_stock["ts_code"].iloc[0]
    out["trade_date"] = df_stock["trade_date"].values

    # CSF — left-join from precomputed parquet
    ts = df_stock["ts_code"].iloc[0]
    csf_df = csf_lookup.get(ts)
    if csf_df is None or csf_df.empty:
        for k in ["csf_ret5_rank", "csf_ret20_rank", "csf_vol_ratio_5_20_rank",
                  "csf_pos20_rank", "csf_atr_pct_rank", "csf_money_flow_5_rank",
                  "csf_qmom_rank", "csf_close_to_ma20_rank",
                  "csf_max_drawdown_20_rank", "csf_high_low_corr_20_rank"]:
            out[k] = 0.5
    else:
        out = out.merge(csf_df, on="trade_date", how="left")
        for k in ["csf_ret5_rank", "csf_ret20_rank", "csf_vol_ratio_5_20_rank",
                  "csf_pos20_rank", "csf_atr_pct_rank", "csf_money_flow_5_rank",
                  "csf_qmom_rank", "csf_close_to_ma20_rank",
                  "csf_max_drawdown_20_rank", "csf_high_low_corr_20_rank"]:
            if k not in out.columns:
                out[k] = 0.5
            else:
                out[k] = out[k].fillna(0.5)

    return out


def main() -> None:
    t0 = time.time()
    print("Loading OHLCV cache...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    print(f"  {len(ohlcv):,} bars across {ohlcv['ts_code'].nunique()} stocks", flush=True)

    print("Loading CSF cache...", flush=True)
    csf = pd.read_parquet(CSF_PATH)
    csf["trade_date"] = pd.to_datetime(csf["trade_date"])
    csf_cols = ["csf_ret5_rank", "csf_ret20_rank", "csf_vol_ratio_5_20_rank",
                "csf_pos20_rank", "csf_atr_pct_rank", "csf_money_flow_5_rank",
                "csf_qmom_rank", "csf_close_to_ma20_rank",
                "csf_max_drawdown_20_rank", "csf_high_low_corr_20_rank"]
    csf_lookup = {ts: g[["trade_date"] + csf_cols].copy()
                  for ts, g in csf.groupby("ts_code")}
    print(f"  {len(csf_lookup)} stocks in CSF lookup", flush=True)

    print(f"Loading {len(ENSEMBLE)} ensemble models...", flush=True)
    boosters = []
    for p in ENSEMBLE:
        b = xgb.Booster()
        b.load_model(str(p))
        boosters.append(b)

    print("Building features + scoring all stocks...", flush=True)
    out_rows = []
    codes = sorted(ohlcv["ts_code"].unique())
    for k, ts in enumerate(codes):
        df_st = ohlcv[ohlcv["ts_code"] == ts].sort_values("trade_date").reset_index(drop=True)
        feats = build_features_for_stock(df_st, csf_lookup)
        if feats.empty:
            continue

        valid = feats.dropna(subset=FEATURE_NAMES).reset_index(drop=True)
        if valid.empty:
            continue

        X = valid[FEATURE_NAMES].values.astype(np.float32)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        preds = np.mean([b.predict(dmat) for b in boosters], axis=0)
        valid["ml_score"] = preds
        out_rows.append(valid[["ts_code", "trade_date", "ml_score"]])

        if (k + 1) % 100 == 0:
            print(f"  scored {k+1}/{len(codes)}", flush=True)

    print("Concatenating...", flush=True)
    scores = pd.concat(out_rows, ignore_index=True)
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_OUT, index=False)
    print(f"Saved {len(scores):,} score rows to {SCORES_OUT}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
