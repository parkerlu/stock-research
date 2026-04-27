"""Train XGBoost ensemble on the cached compare dataset.

Reads:  models/compare_dataset.npz
Writes: models/compare_xgb_preds.npy

Run as a separate Python process (subprocess) — keeps OpenMP runtime isolated
from PyTorch's libomp (loading both in one process causes SIGSEGV on macOS).
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import sys
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from scripts.ml_step1_train import FEATURE_NAMES  # noqa: E402

OUT_DIR = ROOT / "models"
DATA_PATH = OUT_DIR / "compare_dataset.npz"
PRED_PATH = OUT_DIR / "compare_xgb_preds.npy"
Y_PATH = OUT_DIR / "compare_y_te.npy"

XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "eval_metric": "auc",
    "verbosity": 0,
    "nthread": 8,
}
XGB_ROUNDS = 200
XGB_SEEDS = [42, 123, 7, 2024, 88]


def main():
    print("Loading cached dataset...", flush=True)
    d = np.load(DATA_PATH)
    Xtr = d["Xtr_xgb"]; ytr = d["ytr"]
    Xte = d["Xte_xgb"]; yte = d["yte"]
    print(f"  train={len(ytr)}  test={len(yte)}", flush=True)

    print(f"\n--- XGBoost Ensemble ({len(XGB_SEEDS)} seeds) ---", flush=True)
    t0 = time.time()
    all_preds = []
    for seed in XGB_SEEDS:
        params = {**XGB_PARAMS, "seed": seed}
        dtr = xgb.DMatrix(Xtr, label=ytr, feature_names=FEATURE_NAMES)
        dte = xgb.DMatrix(Xte, feature_names=FEATURE_NAMES)
        m = xgb.train(params, dtr, num_boost_round=XGB_ROUNDS,
                      evals=[(dtr, "tr")], verbose_eval=False)
        preds = m.predict(dte)
        all_preds.append(preds)
        auc = roc_auc_score(yte, preds)
        print(f"  seed={seed}: AUC={auc:.4f}  ({time.time()-t0:.0f}s)", flush=True)
        # Free between seeds
        del dtr, dte, m

    ens = np.mean(all_preds, axis=0)
    ens_auc = roc_auc_score(yte, ens)
    print(f"\nEnsemble AUC: {ens_auc:.4f}", flush=True)

    np.save(PRED_PATH, ens)
    np.save(Y_PATH, yte)
    print(f"Saved {PRED_PATH}", flush=True)


if __name__ == "__main__":
    main()
