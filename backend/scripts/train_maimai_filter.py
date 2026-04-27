"""Train a per-stock filter on 买卖很准 'transition to all-clear' signals.

Signal definition:
  At bar i, jibuy=duanbuy=zhunbei=0 AND at bar i-1 at least one was 1.
  → "the maimai setup just ended" — user's actionable buy moment.

For each signal event, look forward LABEL_HORIZON bars.
  Label = SUCCESS if forward return reaches +TARGET (default +5%) before
                hitting -STOP (default -10%).
  Else  = FAIL.

Train a classifier on the feature vector at signal time.
Reports: signal count, success rate, model accuracy/precision at thresholds.

Usage:
  python -m scripts.train_maimai_filter --ts_code 603319.SH
  python -m scripts.train_maimai_filter --ts_code 603319.SH --multi-stock
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import asyncio
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES, compute_features, load_candles, precompute_maimai,
    precompute_dongli, load_csf_cache,
)


LABEL_HORIZON = 10
LABEL_TARGET = 0.05      # +5% wins
LABEL_STOP = -0.10       # -10% loss tolerated
MIN_HISTORY = 130        # bars of history required before signal


def find_maimai_transitions(df: pd.DataFrame) -> list[int]:
    """Return list of bar indices i where:
       - mm_jibuy + mm_duanbuy + mm_zhunbei == 0 at bar i
       - sum > 0 at bar i-1 (at least one was active and just turned off)
    """
    if len(df) < 30:
        return []
    mm = precompute_maimai(df)
    df.attrs["_mm_cache"] = mm  # cache for downstream feature compute

    j = mm["mm_jibuy_active"]
    d = mm["mm_duanbuy_active"]
    z = mm["mm_zhunbei_active"]
    sum_now = j + d + z
    sum_prev = np.concatenate(([0.0], sum_now[:-1]))

    sig = (sum_now == 0) & (sum_prev > 0)
    return [i for i in range(1, len(df)) if sig[i]]


def label_event(df: pd.DataFrame, i: int) -> int | None:
    """+1 if forward window hits TARGET before STOP, 0 otherwise.
    Returns None if not enough forward bars."""
    n = len(df)
    if i + LABEL_HORIZON >= n:
        return None
    close = df["close"].values
    entry = close[i]
    for j in range(1, LABEL_HORIZON + 1):
        ret = close[i + j] / entry - 1
        if ret >= LABEL_TARGET:
            return 1
        if ret <= LABEL_STOP:
            return 0
    return 0


def collect_for_stock(symbol: str) -> pd.DataFrame:
    """Async-friendly wrapper. Returns DataFrame of (features..., label, signal_idx, date)."""
    df = asyncio.run(load_candles(symbol))
    if len(df) < MIN_HISTORY:
        return pd.DataFrame()
    df.attrs["ts_code"] = symbol
    # Pre-cache for fast per-bar feature computation
    df.attrs["_dl_cache"] = precompute_dongli(df)

    sigs = find_maimai_transitions(df)
    rows = []
    for i in sigs:
        if i < MIN_HISTORY or i >= len(df) - 1:
            continue
        label = label_event(df, i)
        if label is None:
            continue
        feat = compute_features(df, i)
        if feat is None:
            continue
        rows.append({**feat, "label": label, "signal_idx": i,
                     "trade_date": df["trade_date"].iloc[i],
                     "symbol": symbol})
    return pd.DataFrame(rows)


def main():
    global LABEL_HORIZON, LABEL_TARGET, LABEL_STOP
    p = argparse.ArgumentParser()
    p.add_argument("--ts_code", default="603319.SH")
    p.add_argument("--multi-stock", action="store_true",
                   help="Train on all stocks; evaluate per-stock")
    p.add_argument("--horizon", type=int, default=LABEL_HORIZON)
    p.add_argument("--target", type=float, default=LABEL_TARGET)
    p.add_argument("--stop", type=float, default=LABEL_STOP)
    args = p.parse_args()
    LABEL_HORIZON = args.horizon
    LABEL_TARGET = args.target
    LABEL_STOP = args.stop

    load_csf_cache()
    t0 = time.time()

    if args.multi_stock:
        from sqlalchemy import distinct, select
        from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
        from app.config import settings
        from app.models.schema import DailyCandle, StockBasic

        async def list_codes():
            engine = create_async_engine(settings.database_url, echo=False)
            S = async_sessionmaker(engine, expire_on_commit=False)
            async with S() as db:
                rows = (await db.execute(
                    select(distinct(DailyCandle.ts_code))
                    .join(StockBasic, StockBasic.ts_code == DailyCandle.ts_code)
                    .where(StockBasic.is_active.is_(True))
                )).scalars().all()
            await engine.dispose()
            return list(rows)

        codes = asyncio.run(list_codes())
        print(f"Multi-stock training on {len(codes)} active stocks", flush=True)
        all_df = []
        for k, sym in enumerate(codes):
            try:
                d = collect_for_stock(sym)
                if not d.empty:
                    all_df.append(d)
            except Exception as e:
                print(f"  {sym}: error {e}", flush=True)
            if (k + 1) % 100 == 0:
                rows_so_far = sum(len(x) for x in all_df)
                print(f"  {k+1}/{len(codes)} stocks  signals so far: {rows_so_far}",
                      flush=True)
        data = pd.concat(all_df, ignore_index=True)
    else:
        print(f"Single-stock training on {args.ts_code}", flush=True)
        data = collect_for_stock(args.ts_code)

    if data.empty:
        print("No signals found.")
        return

    n_total = len(data)
    n_pos = int(data["label"].sum())
    print(f"\nTotal signals: {n_total}, success: {n_pos} ({n_pos/n_total*100:.1f}%)",
          flush=True)
    print(f"({time.time()-t0:.0f}s to collect)\n", flush=True)

    if n_total < 30:
        print("⚠️  Too few signals to train (<30). Try --multi-stock.")
        return

    # Train/test split, stratified by label
    X = data[FEATURE_NAMES].values
    y = data["label"].values

    if args.multi_stock:
        # Time-based split for multi-stock
        cutoff = pd.Timestamp("2024-09-01")
        data["trade_date"] = pd.to_datetime(data["trade_date"])
        is_train = data["trade_date"] < cutoff
        X_tr, y_tr = X[is_train.values], y[is_train.values]
        X_te, y_te = X[~is_train.values], y[~is_train.values]
        print(f"Train (<{cutoff.date()}): {len(y_tr)}  Test: {len(y_te)}",
              flush=True)
    else:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.30, stratify=y, random_state=42
        )

    # Train ensemble
    print("\n--- Training XGBoost (5 seeds) ---", flush=True)
    preds_list = []
    for seed in [42, 123, 7, 2024, 88]:
        params = {
            "objective": "binary:logistic",
            "max_depth": 4, "eta": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 5, "eval_metric": "auc",
            "verbosity": 0, "nthread": 8, "seed": seed,
        }
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
        dte = xgb.DMatrix(X_te, feature_names=FEATURE_NAMES)
        m = xgb.train(params, dtr, num_boost_round=200, verbose_eval=False)
        preds_list.append(m.predict(dte))
    preds = np.mean(preds_list, axis=0)

    auc = roc_auc_score(y_te, preds) if len(set(y_te)) > 1 else float("nan")
    print(f"\nTest AUC: {auc:.4f}", flush=True)
    print(f"\n{'thr':>6}  {'precision':>9}  {'kept':>6}/{'total':>6}  "
          f"{'kept_success':>13}", flush=True)
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        bin_pred = (preds >= thr).astype(int)
        n_kept = int(bin_pred.sum())
        if n_kept == 0:
            continue
        prec = precision_score(y_te, bin_pred, zero_division=0)
        n_succ = int(((bin_pred == 1) & (y_te == 1)).sum())
        print(f"  {thr:.2f}     {prec:.3f}    {n_kept:>4}/{len(y_te):>5}     "
              f"{n_succ}/{n_kept}", flush=True)

    # Save
    out = ROOT / "models" / f"maimai_filter_{args.ts_code.replace('.', '_')}.json"
    if args.multi_stock:
        out = ROOT / "models" / "maimai_filter_multi.json"
    # Save the FIRST seed's model for use as the canonical one
    dall = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    final = xgb.train(
        {"objective": "binary:logistic", "max_depth": 4, "eta": 0.05,
         "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5,
         "verbosity": 0, "nthread": 8, "seed": 42},
        dall, num_boost_round=200,
    )
    final.save_model(str(out))
    print(f"\nSaved final model → {out}", flush=True)

    # Top features
    imp = final.get_score(importance_type="gain")
    print(f"\nTop 10 features by gain:", flush=True)
    for k, v in sorted(imp.items(), key=lambda x: -x[1])[:10]:
        print(f"  {k:<28} {v:.1f}", flush=True)


if __name__ == "__main__":
    main()
