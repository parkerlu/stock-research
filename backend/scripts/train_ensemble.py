"""
Train an ensemble of 5 XGBoost models with different random seeds.
Each member sees the same training data but different bagging samples,
producing slightly different decision boundaries. Predictions are averaged.

Output: models/ml_filter_ens_{0..4}.json
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from datetime import date

from scripts.ml_step1_train import (
    FEATURE_NAMES, compute_features, compute_label, passes_loose_trigger,
    load_candles,
)


CUTOFF = date(2024, 1, 1)
N_MODELS = 5
SEEDS = [42, 123, 7, 2024, 88]


async def build_dataset() -> pd.DataFrame:
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await engine.dispose()

    rows: list[dict] = []
    for sym in codes:
        df = await load_candles(sym)
        if len(df) < 130:
            continue
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            f = compute_features(df, i)
            if f is None:
                continue
            label = compute_label(df, i)
            if label is None:
                continue
            rows.append({**f, "label": label, "symbol": sym,
                         "trade_date": df["trade_date"].iloc[i]})
    return pd.DataFrame(rows)


def train_one(X, y, seed: int) -> xgb.Booster:
    dtrain = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    params = {
        "objective": "binary:logistic", "max_depth": 4, "eta": 0.05,
        "subsample": 0.85, "colsample_bytree": 0.85,
        "min_child_weight": 5, "verbosity": 0, "seed": seed,
    }
    return xgb.train(params, dtrain, num_boost_round=150)


def predict_ensemble(models: list[xgb.Booster], X) -> np.ndarray:
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    preds = np.array([m.predict(dmat) for m in models])
    return preds.mean(axis=0)


async def main() -> None:
    print("Building dataset...")
    data = await build_dataset()
    print(f"Total {len(data)} samples ({data['label'].mean()*100:.1f}% positive)")

    train = data[data["trade_date"] < CUTOFF]
    test = data[data["trade_date"] >= CUTOFF]
    print(f"Train: {len(train)}  |  Test: {len(test)}")

    X_train = train[FEATURE_NAMES].values
    y_train = train["label"].values
    X_test = test[FEATURE_NAMES].values
    y_test = test["label"].values

    os.makedirs("models", exist_ok=True)

    print(f"\nTraining ensemble of {N_MODELS} models...")
    models = []
    individual_aucs = []
    for i, seed in enumerate(SEEDS[:N_MODELS]):
        m = train_one(X_train, y_train, seed)
        models.append(m)
        # Individual AUC on test
        pred = m.predict(xgb.DMatrix(X_test, feature_names=FEATURE_NAMES))
        auc = roc_auc_score(y_test, pred)
        individual_aucs.append(auc)
        path = f"models/ml_filter_ens_{i}.json"
        m.save_model(path)
        print(f"  Model {i} (seed={seed}): test AUC = {auc:.4f}  → {path}")

    # Ensemble prediction
    ens_pred = predict_ensemble(models, X_test)
    ens_auc = roc_auc_score(y_test, ens_pred)
    print(f"\nIndividual AUCs:  mean = {np.mean(individual_aucs):.4f}, "
          f"min = {min(individual_aucs):.4f}, max = {max(individual_aucs):.4f}")
    print(f"Ensemble AUC:     {ens_auc:.4f}")
    print(f"Ensemble lift:    {ens_auc - np.mean(individual_aucs):+.4f}")

    print(f"\nEnsemble precision @ thresholds:")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        n = int((ens_pred >= thr).sum())
        if n > 0:
            prec = precision_score(y_test, (ens_pred >= thr).astype(int), zero_division=0)
            print(f"  thr={thr:.2f}: prec={prec:.3f}  n={n}")


if __name__ == "__main__":
    asyncio.run(main())
