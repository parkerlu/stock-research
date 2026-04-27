"""Build LambdaRank score cache for all stocks.

Mirror of mine_426_step2_cache_scores.py but uses single LambdaRank model
(not ensemble). Output: cache/lr_scores.parquet (ts_code, trade_date, ml_score).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES,
    precompute_dongli, precompute_maimai, precompute_gp_factors,
)
from scripts.mine_426_step2_cache_scores import (  # noqa: E402
    vectorize_generic_features, build_features_for_stock,
)


CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
CSF_PATH = ROOT / "models" / "cross_sectional_ranks.parquet"
SCORES_OUT = CACHE_DIR / "lr_scores.parquet"
MODEL_PATH = ROOT / "models" / "lambdarank_v3.json"


def main() -> None:
    t0 = time.time()
    print(f"Loading model {MODEL_PATH}...", flush=True)
    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))

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
        preds_raw = booster.predict(dmat)
        preds = 1.0 / (1.0 + np.exp(-preds_raw))   # sigmoid → [0,1]
        valid["ml_score"] = preds
        out_rows.append(valid[["ts_code", "trade_date", "ml_score"]])

        if (k + 1) % 200 == 0:
            print(f"  scored {k+1}/{len(codes)}", flush=True)

    print("Concatenating...", flush=True)
    scores = pd.concat(out_rows, ignore_index=True)
    SCORES_OUT.parent.mkdir(parents=True, exist_ok=True)
    scores.to_parquet(SCORES_OUT, index=False)
    print(f"Saved {len(scores):,} rows → {SCORES_OUT}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
