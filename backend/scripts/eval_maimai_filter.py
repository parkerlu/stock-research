"""Evaluate the trained maimai-filter on a single stock.

Shows every signal with:
  - signal_date
  - filter_score (model probability of success)
  - actual outcome (forward 10-bar TARGET hit before STOP?)
  - what % of signals would be kept at each threshold + their win rate

Usage:
  python -m scripts.eval_maimai_filter --ts_code 603319.SH
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import asyncio
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES, compute_features, load_candles,
    precompute_dongli, precompute_maimai, load_csf_cache,
)
from scripts.train_maimai_filter import (  # noqa: E402
    find_maimai_transitions, label_event,
    LABEL_HORIZON, LABEL_TARGET, LABEL_STOP, MIN_HISTORY,
)


MODEL_PATH = ROOT / "models" / "maimai_filter_multi.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ts_code", default="603319.SH")
    args = p.parse_args()

    load_csf_cache()

    booster = xgb.Booster()
    booster.load_model(str(MODEL_PATH))
    print(f"Loaded {MODEL_PATH}")

    df = asyncio.run(load_candles(args.ts_code))
    if len(df) < MIN_HISTORY:
        print(f"{args.ts_code}: not enough history")
        return
    df.attrs["ts_code"] = args.ts_code
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
        rows.append({
            **feat, "label": label, "i": i,
            "trade_date": df["trade_date"].iloc[i],
            "close": df["close"].iloc[i],
        })
    if not rows:
        print("no signals")
        return

    data = pd.DataFrame(rows)
    X = data[FEATURE_NAMES].values
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    scores = booster.predict(dmat)
    data["score"] = scores

    print(f"\n=== {args.ts_code} ===")
    print(f"Total maimai 'all-clear' signals: {len(data)}")
    print(f"Raw success rate: {data['label'].mean()*100:.1f}% "
          f"({int(data['label'].sum())}/{len(data)})")
    print(f"\nFilter score distribution: min {scores.min():.3f}, "
          f"max {scores.max():.3f}, mean {scores.mean():.3f}")

    print(f"\nThreshold sweep (kept = signals model rated ≥ thr):")
    print(f"{'thr':>6}  {'kept':>4}  {'win_rate':>8}  {'wins/total':>10}")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        kept = data[data["score"] >= thr]
        if len(kept) == 0:
            continue
        wr = kept["label"].mean() * 100
        print(f"  {thr:.2f}    {len(kept):>4}   {wr:>5.1f}%    "
              f"{int(kept['label'].sum())}/{len(kept)}")

    # Show all signals as a table, sorted by date
    print(f"\nDetailed signals (newest first, top 30):")
    print(f"{'date':>11}  {'close':>8}  {'score':>6}  {'outcome':>8}")
    show = data.sort_values("trade_date", ascending=False).head(30)
    for _, r in show.iterrows():
        outcome = "✓ +5%" if r["label"] == 1 else "✗ fail"
        print(f"  {str(r['trade_date']):>10}  {r['close']:>8.2f}  "
              f"{r['score']:>6.3f}    {outcome}")


if __name__ == "__main__":
    main()
