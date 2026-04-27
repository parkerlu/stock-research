"""
Evaluate combinations of XGBoost ensemble + PatchTST predictions.
"""
from __future__ import annotations

import asyncio
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from datetime import date

from scripts.ml_step1_train import (
    FEATURE_NAMES, compute_features, compute_label,
    passes_loose_trigger, load_candles,
)
from scripts.train_patchtst import (
    PatchTSTBinary, make_sequence, DEVICE, SEQ_LEN, N_FEATURES,
)


CUTOFF = date(2024, 1, 1)


async def collect_test() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collect post-CUTOFF samples that have BOTH XGB features and sequence."""
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await engine.dispose()

    X_xgb, X_seq, y = [], [], []
    for sym in codes:
        df = await load_candles(sym)
        if len(df) < SEQ_LEN + 130:
            continue
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            d = df["trade_date"].iloc[i]
            if d < CUTOFF:
                continue
            f = compute_features(df, i)
            if f is None:
                continue
            seq = make_sequence(df, i)
            if seq is None:
                continue
            label = compute_label(df, i)
            if label is None:
                continue
            X_xgb.append([f[k] for k in FEATURE_NAMES])
            X_seq.append(seq)
            y.append(label)
    return np.array(X_xgb), np.stack(X_seq), np.array(y, dtype=np.float32)


async def main() -> None:
    print("Collecting test samples...")
    X_xgb, X_seq, y = await collect_test()
    print(f"Test set: {len(y)} samples")

    # XGBoost ensemble predictions
    models = []
    for i in range(5):
        m = xgb.Booster()
        m.load_model(f"models/ml_filter_ens_{i}.json")
        models.append(m)
    dmat = xgb.DMatrix(X_xgb, feature_names=FEATURE_NAMES)
    xgb_pred = np.array([m.predict(dmat) for m in models]).mean(axis=0)
    print(f"XGB ensemble AUC: {roc_auc_score(y, xgb_pred):.4f}")

    # PatchTST predictions
    norm = np.load("models/patchtst_norm.npy")
    mean, std = norm[0], norm[1]
    X_seq_n = (X_seq - mean) / std
    model = PatchTSTBinary().to(DEVICE)
    model.load_state_dict(torch.load("models/patchtst_best.pt",
                                      map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        Xt = torch.from_numpy(X_seq_n.astype(np.float32)).to(DEVICE)
        # batch to avoid OOM
        preds = []
        for i in range(0, len(Xt), 1024):
            preds.append(torch.sigmoid(model(Xt[i:i+1024])).cpu().numpy())
    pt_pred = np.concatenate(preds)
    print(f"PatchTST AUC:     {roc_auc_score(y, pt_pred):.4f}")

    # Combined: average
    avg = (xgb_pred + pt_pred) / 2
    print(f"\nCombined (avg)    AUC: {roc_auc_score(y, avg):.4f}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        n = int((avg >= thr).sum())
        if n > 0:
            prec = precision_score(y, (avg >= thr).astype(int), zero_division=0)
            print(f"  thr={thr:.2f}: prec={prec:.3f}  n={n}")

    # Combined: weighted (60% XGB, 40% PT)
    w = 0.6 * xgb_pred + 0.4 * pt_pred
    print(f"\nCombined (60/40)  AUC: {roc_auc_score(y, w):.4f}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        n = int((w >= thr).sum())
        if n > 0:
            prec = precision_score(y, (w >= thr).astype(int), zero_division=0)
            print(f"  thr={thr:.2f}: prec={prec:.3f}  n={n}")

    # Combined: only act when BOTH agree (intersection)
    print(f"\nIntersection (both >= thr):")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        mask = (xgb_pred >= thr) & (pt_pred >= thr)
        n = int(mask.sum())
        if n > 0:
            prec = y[mask].mean()
            print(f"  thr={thr:.2f}: prec={prec:.3f}  n={n}")


if __name__ == "__main__":
    asyncio.run(main())
