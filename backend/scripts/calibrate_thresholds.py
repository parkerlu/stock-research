"""
Per-stock adaptive thresholds. For each stock, find buy/exit thresholds that
maximize cumulative return on the TRAINING period (pre-2024). Save as a JSON
lookup that the strategy loads at inference time.

Stocks with too few signals fall back to global defaults.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from app.services.backtest_engine import run_backtest
from scripts.ml_step1_train import FEATURE_NAMES, compute_features, load_candles


CUTOFF = date(2024, 1, 1)
ENSEMBLE_PATHS = [f"models/ml_filter_ens_{i}.json" for i in range(5)]
DEFAULT_BUY = 0.55
DEFAULT_EXIT = 0.40
GRID_BUY = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
GRID_EXIT = [0.30, 0.35, 0.40, 0.45]


def load_ensemble() -> list[xgb.Booster]:
    out = []
    for p in ENSEMBLE_PATHS:
        m = xgb.Booster()
        m.load_model(p)
        out.append(m)
    return out


def predict_ensemble(models, X) -> np.ndarray:
    dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
    return np.array([m.predict(dmat) for m in models]).mean(axis=0)


def score_all_bars(models, df: pd.DataFrame) -> dict[int, float]:
    rows, idx = [], []
    for i in range(120, len(df)):
        f = compute_features(df, i)
        if f is None:
            continue
        rows.append([f[k] for k in FEATURE_NAMES])
        idx.append(i)
    if not rows:
        return {}
    scores = predict_ensemble(models, np.array(rows))
    return {i: float(s) for i, s in zip(idx, scores)}


def simulate(df: pd.DataFrame, scores: dict[int, float],
             buy_thr: float, exit_thr: float,
             stop_loss: float = -0.07, take_profit: float = 0.15,
             time_stop: int = 30) -> dict:
    """Simulate trades on df with given thresholds. Returns metrics."""
    close = df["close"].astype(float).values
    dates = df["trade_date"]
    candles = df.to_dict("records")
    signals = []
    in_pos = False
    entry_price = 0.0
    hold = 0

    for i in range(120, len(df)):
        s = scores.get(i)
        if s is None:
            continue
        c = close[i]
        d = dates.iloc[i]
        if not in_pos:
            if s >= buy_thr:
                signals.append({"date": d, "action": "buy"})
                entry_price = c
                hold = 0
                in_pos = True
        else:
            hold += 1
            pnl = (c - entry_price) / entry_price
            if pnl <= stop_loss or s <= exit_thr or pnl >= take_profit or hold >= time_stop:
                signals.append({"date": d, "action": "sell"})
                in_pos = False

    r = run_backtest(candles, signals, 10000)
    return {
        "ret": r["net_profit_pct"],
        "trades": len(r["trades"]),
        "win_rate": r["win_rate"],
        "max_dd": r["max_drawdown"],
    }


async def main() -> None:
    print("Loading ensemble...")
    models = load_ensemble()

    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await engine.dispose()

    thresholds: dict[str, dict] = {}
    skipped = 0
    used_default = 0

    for i, sym in enumerate(codes):
        df = await load_candles(sym)
        if len(df) < 130:
            skipped += 1
            continue

        # Restrict to TRAINING period only for calibration
        df_train = df[df["trade_date"] < CUTOFF].reset_index(drop=True)
        if len(df_train) < 130:
            skipped += 1
            continue

        scores = score_all_bars(models, df_train)
        if not scores:
            skipped += 1
            continue

        # Grid search on training period
        best = None
        best_ret = -float("inf")
        for buy in GRID_BUY:
            for exit_t in GRID_EXIT:
                if exit_t >= buy:
                    continue
                m = simulate(df_train, scores, buy, exit_t)
                # Need minimum trades for meaningful tuning
                if m["trades"] < 2:
                    continue
                if m["ret"] > best_ret:
                    best_ret = m["ret"]
                    best = {"buy": buy, "exit": exit_t, **m}

        if best is None:
            thresholds[sym] = {"buy": DEFAULT_BUY, "exit": DEFAULT_EXIT,
                               "source": "default"}
            used_default += 1
        else:
            thresholds[sym] = {
                "buy": best["buy"], "exit": best["exit"],
                "train_ret": round(best["ret"], 2),
                "train_trades": best["trades"],
                "train_win_rate": round(best["win_rate"], 3),
                "source": "tuned",
            }

        if (i + 1) % 50 == 0:
            print(f"  ...calibrated {i+1}/{len(codes)}")

    print(f"\nDone. Total stocks: {len(codes)}, skipped: {skipped}, "
          f"defaulted: {used_default}, tuned: {len(thresholds) - used_default}")

    out = "models/per_stock_thresholds.json"
    with open(out, "w") as f:
        json.dump(thresholds, f, indent=2, default=str)
    print(f"Saved to {out}")

    # Summary stats
    tuned = [v for v in thresholds.values() if v.get("source") == "tuned"]
    if tuned:
        buys = [v["buy"] for v in tuned]
        exits = [v["exit"] for v in tuned]
        print(f"\nTuned threshold distribution:")
        print(f"  buy:  mean={np.mean(buys):.2f}, "
              f"min={min(buys):.2f}, max={max(buys):.2f}")
        print(f"  exit: mean={np.mean(exits):.2f}, "
              f"min={min(exits):.2f}, max={max(exits):.2f}")
        from collections import Counter
        print(f"  buy distribution: {Counter(buys).most_common()}")
        print(f"  exit distribution: {Counter(exits).most_common()}")


if __name__ == "__main__":
    asyncio.run(main())
