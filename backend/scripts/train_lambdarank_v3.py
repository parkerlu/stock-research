"""Train LambdaRank on the cached compare dataset.

Same 39 features as XGB ensemble, same time split (2024-09-01), but objective
is rank:pairwise — model learns to rank candidates within each day rather
than predict absolute probability. Often better aligned with picking the
"top-K best candidates today" use case.

Reads:  models/compare_dataset.npz  (already built by train_patchtst_v3_compare.py)
Writes: models/lambdarank_v3.json

Run as a separate Python process (no OpenMP collision).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.ml_step1_train import FEATURE_NAMES  # noqa: E402

DATA_PATH = ROOT / "models" / "compare_dataset.npz"
OUT_PATH = ROOT / "models" / "lambdarank_v3.json"


def main() -> None:
    print(f"Loading {DATA_PATH}...", flush=True)
    d = np.load(DATA_PATH)
    Xtr = d["Xtr_xgb"]; ytr = d["ytr"]
    Xte = d["Xte_xgb"]; yte = d["yte"]
    gtr_groups = d["gtr"]   # symbol per row — used for grouping
    print(f"  train: {len(ytr):,}  test: {len(yte):,}", flush=True)

    # LambdaRank needs group structure: rows are partitioned into groups,
    # ranking is performed WITHIN each group. Two natural choices:
    #   1. group = trade_date  → rank stocks against each other on the same day
    #   2. group = symbol      → rank dates within a stock
    # (1) is what we want for "pick best stocks today". But our cached dataset
    # doesn't store trade_date per row — only symbol (gtr). Use symbol grouping
    # as a proxy: model learns to rank good vs bad bars within each stock.
    # This is suboptimal vs date-grouping but works with current cache.
    train_idx = np.argsort(gtr_groups, kind="stable")
    Xtr_s = Xtr[train_idx]
    ytr_s = ytr[train_idx]
    sorted_g = gtr_groups[train_idx]
    _, group_sizes = np.unique(sorted_g, return_counts=True)
    print(f"  train groups (by symbol): {len(group_sizes)}, "
          f"mean size: {group_sizes.mean():.0f}, min/max: {group_sizes.min()}/{group_sizes.max()}",
          flush=True)

    dtr = xgb.DMatrix(Xtr_s, label=ytr_s, feature_names=FEATURE_NAMES)
    dtr.set_group(group_sizes)
    dte = xgb.DMatrix(Xte, feature_names=FEATURE_NAMES)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": ["auc"],
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "verbosity": 0,
        "nthread": 8,
        "seed": 42,
    }

    print("Training LambdaRank...", flush=True)
    t0 = time.time()
    model = xgb.train(params, dtr, num_boost_round=300, verbose_eval=50)
    print(f"  trained in {time.time()-t0:.0f}s", flush=True)

    preds = model.predict(dte)
    # Re-scale to [0,1] for compatibility with score-based mining (use sigmoid)
    preds_norm = 1.0 / (1.0 + np.exp(-preds))

    auc_raw = roc_auc_score(yte, preds)
    auc_norm = roc_auc_score(yte, preds_norm)
    print(f"\nTest AUC (raw scores):       {auc_raw:.4f}", flush=True)
    print(f"Test AUC (sigmoid normalized): {auc_norm:.4f}", flush=True)
    print(f"\nthr      prec     n", flush=True)
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        bin_pred = (preds_norm >= thr).astype(int)
        n = int(bin_pred.sum())
        if n > 0:
            p = precision_score(yte, bin_pred, zero_division=0)
            print(f"{thr:.2f}    {p:.3f}    {n:>6,}", flush=True)

    model.save_model(str(OUT_PATH))
    print(f"\nSaved → {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
