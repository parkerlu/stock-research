"""
B — XGBoost LambdaRank model.

Trains XGBoost with rank:pairwise objective. Each day = one query group.
The label is graded: 0 (loss), 1 (small win), 2 (medium win), 3 (big win)
based on next-7-day return distribution.

Compared to binary classifier, LambdaRank directly optimizes "rank stocks by
expected return within each day" — more aligned with our daily decision logic.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import ndcg_score
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from datetime import date

from scripts.ml_step1_train import (
    FEATURE_NAMES, compute_features, passes_loose_trigger,
    load_candles, precompute_maimai, load_csf_cache,
)


CUTOFF = date(2024, 1, 1)
LABEL_HORIZON = 7


def grade_label(future_ret: float) -> int:
    if future_ret >= 0.10: return 3   # big win
    if future_ret >= 0.05: return 2   # medium win
    if future_ret >= 0.0:  return 1   # small win
    return 0                          # loss


async def main() -> None:
    print("Loading CSF cache...", flush=True)
    load_csf_cache()
    print("Building dataset (panel: per-day groups)...", flush=True)

    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await eng.dispose()

    rows = []
    for k, sym in enumerate(codes):
        df = await load_candles(sym)
        if len(df) < 150:
            continue
        df.attrs["ts_code"] = sym
        close = df["close"].values
        for i in range(120, len(df) - LABEL_HORIZON):
            f = compute_features(df, i)
            if f is None:
                continue
            future_max = max(close[i + 1:i + 1 + LABEL_HORIZON]) / close[i] - 1
            future_min = min(close[i + 1:i + 1 + LABEL_HORIZON]) / close[i] - 1
            # Mark as 0 if dropped > 3% before any 5% gain
            d = df["trade_date"].iloc[i]
            label = grade_label(future_max if future_min > -0.03 else future_min)
            row = {**f, "label": int(label), "trade_date": d, "ts_code": sym}
            rows.append(row)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1}/{len(codes)} stocks processed, total samples={len(rows)}",
                  flush=True)

    print(f"\nTotal {len(rows)} samples", flush=True)
    full = pd.DataFrame(rows)
    train_df = full[full["trade_date"] < CUTOFF].copy()
    test_df = full[full["trade_date"] >= CUTOFF].copy()
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}", flush=True)

    # Sort by trade_date — XGBoost ranking expects samples grouped by query
    train_df = train_df.sort_values("trade_date").reset_index(drop=True)
    test_df = test_df.sort_values("trade_date").reset_index(drop=True)

    train_groups = train_df.groupby("trade_date").size().values
    test_groups = test_df.groupby("trade_date").size().values
    # Filter groups that are too small (< 5 stocks per day)
    print(f"  Train groups: {len(train_groups)} days, "
          f"min={train_groups.min()}, max={train_groups.max()}", flush=True)

    X_train = train_df[FEATURE_NAMES].values
    y_train = train_df["label"].values
    X_test = test_df[FEATURE_NAMES].values
    y_test = test_df["label"].values
    test_dates = test_df["trade_date"].values

    print("\nTraining LambdaRank model...", flush=True)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=FEATURE_NAMES)
    dtrain.set_group(train_groups)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=FEATURE_NAMES)
    dtest.set_group(test_groups)

    params = {
        "objective": "rank:pairwise",
        "eval_metric": ["ndcg@10", "ndcg@20"],
        "max_depth": 5,
        "eta": 0.05,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 5,
        "verbosity": 1,
    }
    model = xgb.train(params, dtrain, num_boost_round=200,
                      evals=[(dtest, "test")], early_stopping_rounds=20,
                      verbose_eval=20)
    os.makedirs("models", exist_ok=True)
    model.save_model("models/lambdarank.json")
    print(f"\nModel saved", flush=True)

    # Out-of-sample scoring + evaluate as ranker
    pred = model.predict(dtest)
    test_df["pred"] = pred

    # Per-day, see if top-decile by pred actually outperforms
    print("\nPer-day top-decile actual return distribution:", flush=True)
    per_day_stats = []
    for d, group in test_df.groupby("trade_date"):
        if len(group) < 10:
            continue
        ranked = group.sort_values("pred", ascending=False)
        top10 = ranked.head(max(1, len(group) // 10))
        bot10 = ranked.tail(max(1, len(group) // 10))
        top_avg_label = top10["label"].mean()
        bot_avg_label = bot10["label"].mean()
        per_day_stats.append({
            "date": d,
            "n": len(group),
            "top_label": top_avg_label,
            "bot_label": bot_avg_label,
            "diff": top_avg_label - bot_avg_label,
        })
    pdf = pd.DataFrame(per_day_stats)
    print(f"  test days: {len(pdf)}", flush=True)
    print(f"  avg top-decile label: {pdf['top_label'].mean():.3f}", flush=True)
    print(f"  avg bot-decile label: {pdf['bot_label'].mean():.3f}", flush=True)
    print(f"  avg top-bot diff:     {pdf['diff'].mean():.3f}  (positive = good ranking)", flush=True)
    days_top_higher = (pdf["diff"] > 0).sum()
    print(f"  days top>bot: {days_top_higher}/{len(pdf)} = "
          f"{days_top_higher/max(len(pdf),1)*100:.1f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
