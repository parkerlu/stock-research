"""
D — Daily ranking strategy using ML ensemble scores.

Each day, score all stocks with the trained XGBoost ensemble. Rank stocks
within each day by score. Buy top decile, hold until rank drops or stops hit.

This is fundamentally different from MLDirectDecision (per-bar binary decision)
— here we make a relative cross-sectional decision every day.
"""
from __future__ import annotations

import asyncio
import os
from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from scripts.ml_step1_train import (
    FEATURE_NAMES, compute_features, load_candles, load_csf_cache,
)


CUTOFF = date(2024, 1, 1)


def load_models():
    models = []
    for i in range(5):
        m = xgb.Booster()
        m.load_model(f"models/ml_filter_ens_{i}.json")
        models.append(m)
    return models


async def get_pool_codes(pool_id: int = 3) -> list[str]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == pool_id)
        )).scalars().all()
    await eng.dispose()
    return list(rows)


async def main() -> None:
    load_csf_cache()
    print(f"Loading 100-stock pool...", flush=True)
    codes = await get_pool_codes(3)

    # Load each stock, compute scores per bar
    print(f"Computing per-bar scores for {len(codes)} stocks...", flush=True)
    models = load_models()

    # Build daily score panel: dict[trade_date][ts_code] = score
    score_by_day: dict = {}
    candle_by_stock = {}
    for k, sym in enumerate(codes):
        df = await load_candles(sym)
        if len(df) < 130:
            continue
        df.attrs["ts_code"] = sym
        candle_by_stock[sym] = df
        # Score every bar
        rows, idx_list = [], []
        for i in range(120, len(df)):
            f = compute_features(df, i)
            if f is None:
                continue
            rows.append([f[k_] for k_ in FEATURE_NAMES])
            idx_list.append(i)
        if not rows:
            continue
        X = np.array(rows)
        dmat = xgb.DMatrix(X, feature_names=FEATURE_NAMES)
        preds = np.array([m.predict(dmat) for m in models]).mean(axis=0)
        for i, score in zip(idx_list, preds):
            d = df["trade_date"].iloc[i]
            score_by_day.setdefault(d, {})[sym] = float(score)
        if (k + 1) % 25 == 0:
            print(f"  scored {k + 1}/{len(codes)}", flush=True)

    print(f"\nDaily score panel: {len(score_by_day)} days", flush=True)

    # Run daily ranking strategy
    print(f"\nDaily ranking backtests (post-{CUTOFF})", flush=True)
    print(f"{'config':<22}  {'trades':>6}  {'win%':>6}  {'≥5%':>5}  "
          f"{'≥10%':>5}  {'avg':>6}  {'median':>6}", flush=True)
    print("-" * 70, flush=True)

    for top_pct, drop_pct, time_stop, name in [
        (0.10, 0.30, 60, "top10_drop30"),
        (0.10, 0.40, 60, "top10_drop40"),
        (0.20, 0.50, 60, "top20_drop50"),
        (0.05, 0.20, 60, "top5_drop20"),
        (0.20, 0.40, 30, "top20_drop40_ts30"),
    ]:
        positions: dict = {}  # sym -> {entry_price, entry_date, peak, hold_bars, trail_active}
        completed = []
        sorted_dates = sorted(d for d in score_by_day.keys() if d >= CUTOFF)
        for d in sorted_dates:
            day_scores = score_by_day.get(d, {})
            if len(day_scores) < 10:
                continue

            # Compute ranks
            scores_list = sorted(day_scores.items(), key=lambda x: -x[1])
            n = len(scores_list)
            top_k = max(int(n * top_pct), 1)

            # Drop threshold rank
            stock_ranks = {sym: rk for rk, (sym, _) in enumerate(scores_list)}
            drop_threshold_rank = int(n * (1 - drop_pct))

            # 1. Exit logic
            to_exit = []
            for sym, pos in positions.items():
                df = candle_by_stock.get(sym)
                if df is None:
                    continue
                row_match = df[df["trade_date"] == d]
                if len(row_match) == 0:
                    continue
                close = float(row_match["close"].iloc[0])
                pos["peak"] = max(pos["peak"], close)
                pos["hold_bars"] += 1
                pnl = close / pos["entry_price"] - 1
                if pnl >= 0.05:
                    pos["trail_active"] = True
                hard_stop = pos["entry_price"] * 0.93
                trail_stop = max(pos["entry_price"], pos["peak"] * 0.97) if pos["trail_active"] else hard_stop
                rk = stock_ranks.get(sym, n)

                reason = None
                if close <= hard_stop:
                    reason = "stop"
                elif pos["trail_active"] and close <= trail_stop:
                    reason = "trail"
                elif pos["hold_bars"] >= time_stop:
                    reason = "time"
                elif rk > drop_threshold_rank:
                    reason = "rank_drop"
                if reason:
                    completed.append({
                        "ts_code": sym,
                        "entry_date": pos["entry_date"],
                        "exit_date": d,
                        "entry_price": pos["entry_price"],
                        "exit_price": close,
                        "pnl": (close / pos["entry_price"] - 1) * 100,
                        "reason": reason,
                    })
                    to_exit.append(sym)
            for sym in to_exit:
                del positions[sym]

            # 2. Entry: pick top_k by score, skip already held + 涨停
            slots = max(top_k - len(positions), 0)
            for sym, score in scores_list[:top_k * 2]:  # consider 2x buffer
                if slots <= 0:
                    break
                if sym in positions:
                    continue
                df = candle_by_stock.get(sym)
                if df is None:
                    continue
                row_match = df[df["trade_date"] == d]
                if len(row_match) == 0:
                    continue
                close = float(row_match["close"].iloc[0])
                # 涨停 check
                idx_today = df.index[df["trade_date"] == d][0]
                if idx_today >= 1:
                    prev_close = float(df["close"].iloc[idx_today - 1])
                    if prev_close > 0:
                        gain = close / prev_close - 1
                        code6 = sym.split(".")[0]
                        limit = 0.20 if code6.startswith("688") or code6.startswith("30") else 0.10
                        if gain >= limit - 0.005:
                            continue
                positions[sym] = {
                    "entry_price": close,
                    "entry_date": d,
                    "peak": close,
                    "hold_bars": 0,
                    "trail_active": False,
                }
                slots -= 1

        # Force-close
        last_d = sorted_dates[-1] if sorted_dates else None
        for sym, pos in list(positions.items()):
            df = candle_by_stock.get(sym)
            if df is None:
                continue
            row_match = df[df["trade_date"] == last_d]
            if len(row_match) == 0:
                continue
            close = float(row_match["close"].iloc[0])
            completed.append({
                "ts_code": sym,
                "entry_date": pos["entry_date"],
                "exit_date": last_d,
                "entry_price": pos["entry_price"],
                "exit_price": close,
                "pnl": (close / pos["entry_price"] - 1) * 100,
                "reason": "end",
            })

        if not completed:
            print(f"{name:<22}  no trades", flush=True)
            continue
        pcts = [t["pnl"] for t in completed]
        wins = sum(1 for p in pcts if p > 0)
        print(f"{name:<22}  {len(completed):>6}  "
              f"{wins/len(completed)*100:>5.1f}%  "
              f"{sum(1 for p in pcts if p>=5)/len(pcts)*100:>4.1f}%  "
              f"{sum(1 for p in pcts if p>=10)/len(pcts)*100:>4.1f}%  "
              f"{np.mean(pcts):>+5.2f}%  {np.median(pcts):>+5.2f}%", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
