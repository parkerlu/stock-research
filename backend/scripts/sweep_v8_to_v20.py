"""
Sweep 13 strategy variants (V8-V20) on the 100-stock pool, post-2024.
Pick top 3 by composite metric and emit recommended preset configs.

All variants use the same trained ML ensemble (V7 model: 7d / +5% / -3% label).
Variants differ only in strategy decision logic (thresholds, exits, filters).

Composite metric: rank stocks by:
  win_rate (50%) + ≥5%_pct (30%) + (1 if avg_return > 0 else 0)*20
Then average across stocks. Higher = better.
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import date

import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle, StockPoolItem
from app.services.backtest_engine import run_backtest
from app.services.strategy_templates import MLDirectDecision


CUTOFF = date(2024, 1, 1)
HISTORY_START = date(2022, 1, 1)


VARIANTS = [
    # name, params
    ("V8_default",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 30, "use_adaptive": True}),

    ("V9_high_buy_thr",
     {"buy_threshold": 0.60, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 30, "use_adaptive": False}),

    ("V10_low_exit_thr",
     {"buy_threshold": 0.50, "exit_threshold": 0.25,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 30, "use_adaptive": True}),

    ("V11_tight_trail",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.03, "trail_pct": 0.025,
      "time_stop": 30, "use_adaptive": True}),

    ("V12_wide_trail",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.07, "trail_pct": 0.06,
      "time_stop": 30, "use_adaptive": True}),

    ("V13_fixed_5pct_target",
     # Disable trailing (huge activation) — only triggers on +5% target via take_profit
     # Strategy doesn't have explicit take_profit currently; use trail_activation=0.05
     # with trail_pct=0.001 to effectively lock at +5% on first touch
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.001,
      "time_stop": 30, "use_adaptive": True}),

    ("V14_short_time_stop",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 15, "use_adaptive": True}),

    ("V15_long_time_stop",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 60, "use_adaptive": True}),

    ("V16_tight_stop",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.05, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 30, "use_adaptive": True}),

    ("V17_loose_stop",
     {"buy_threshold": 0.50, "exit_threshold": 0.35,
      "stop_loss": -0.10, "trail_activation": 0.05, "trail_pct": 0.04,
      "time_stop": 30, "use_adaptive": True}),

    ("V18_balanced",
     {"buy_threshold": 0.55, "exit_threshold": 0.35,
      "stop_loss": -0.05, "trail_activation": 0.04, "trail_pct": 0.03,
      "time_stop": 30, "use_adaptive": True}),

    ("V19_quick_scalp",
     # Fast in/out — short time stop + tight trail
     {"buy_threshold": 0.50, "exit_threshold": 0.30,
      "stop_loss": -0.05, "trail_activation": 0.03, "trail_pct": 0.02,
      "time_stop": 15, "use_adaptive": True}),

    ("V20_swing",
     # Patient swing — wide trail + long time stop + higher target via trail
     {"buy_threshold": 0.55, "exit_threshold": 0.30,
      "stop_loss": -0.07, "trail_activation": 0.08, "trail_pct": 0.05,
      "time_stop": 60, "use_adaptive": True}),
]


_engine = None
_session_factory = None


def get_session_factory():
    global _engine, _session_factory
    if _engine is None:
        _engine = create_async_engine(settings.database_url, echo=False, pool_size=4)
        _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    return _session_factory


async def get_pool_codes(pool_id: int = 3) -> list[str]:
    Session = get_session_factory()
    async with Session() as db:
        rows = (await db.execute(
            select(StockPoolItem.ts_code).where(StockPoolItem.pool_id == pool_id)
        )).scalars().all()
    return list(rows)


async def load_candles(symbol: str) -> pd.DataFrame:
    Session = get_session_factory()
    async with Session() as db:
        rows = (await db.execute(
            select(DailyCandle).where(DailyCandle.ts_code == symbol)
            .order_by(DailyCandle.trade_date)
        )).scalars().all()
    return pd.DataFrame([{
        "trade_date": r.trade_date, "open": float(r.open), "high": float(r.high),
        "low": float(r.low), "close": float(r.close),
        "vol": float(r.vol or 0), "amount": float(r.amount) if r.amount else 0,
    } for r in rows])


def precompute_scores(candles_by_sym: dict) -> dict:
    """Score every (stock, bar) once. Uses vectorized maimai for ~100x speedup."""
    import numpy as np
    import xgboost as xgb
    from app.services.strategy_templates.ml_direct import (
        _FEATURE_NAMES, _load_ensemble, _csf_for,
    )
    from scripts.ml_step1_train import precompute_maimai
    models = _load_ensemble()
    if not models:
        return {}
    cache = {}
    print(f"  scoring {len(candles_by_sym)} stocks...", flush=True)
    for stock_idx, (sym, df) in enumerate(candles_by_sym.items()):
        df_h = df[df["trade_date"] >= HISTORY_START].reset_index(drop=True)
        if len(df_h) < 130:
            continue
        close = df_h["close"].astype(float).values
        high = df_h["high"].astype(float).values
        low = df_h["low"].astype(float).values
        vol = df_h["vol"].astype(float).values

        # Vectorized maimai for whole series at once
        mm = precompute_maimai(df_h)

        # Build feature rows
        rows, idx, feat_cache = [], [], {}
        for i in range(120, len(df_h)):
            c = close[i]
            ma20 = close[i - 20:i].mean()
            ma60 = close[i - 60:i].mean()
            ma120 = close[i - 120:i].mean()
            high20, low20 = high[i - 20:i].max(), low[i - 20:i].min()
            high60, low60 = high[i - 60:i].max(), low[i - 60:i].min()
            atr14 = (high[i - 14:i] - low[i - 14:i]).mean()
            vma20 = vol[i - 20:i].mean()
            f = {
                "pos20": (c - low20) / (high20 - low20) * 100 if high20 > low20 else 50,
                "pos60": (c - low60) / (high60 - low60) * 100 if high60 > low60 else 50,
                "ret5": (c / close[i - 5] - 1) * 100,
                "ret20": (c / close[i - 20] - 1) * 100,
                "ret60": (c / close[i - 60] - 1) * 100,
                "above_ma20": 1.0 if c > ma20 else 0.0,
                "above_ma60": 1.0 if c > ma60 else 0.0,
                "above_ma120": 1.0 if c > ma120 else 0.0,
                "atr14_pct": atr14 / c * 100,
                "vol_today_ratio": vol[i] / vma20 if vma20 > 0 else 1,
                "vol_yest_ratio": vol[i - 1] / vma20 if vma20 > 0 else 1,
                "yest_drop_pct": (close[i - 1] / close[i - 2] - 1) * 100,
                "today_gain_pct": (c / close[i - 1] - 1) * 100,
                "red_days_10": float((close[i - 10:i] < close[i - 11:i - 1]).sum()),
                "ma60_distance_pct": (c / ma60 - 1) * 100,
                "mm_below_floor": float(mm["mm_below_floor"][i]),
                "mm_below_ceiling": float(mm["mm_below_ceiling"][i]),
                "mm_jibuy_active": float(mm["mm_jibuy_active"][i]),
                "mm_duanbuy_active": float(mm["mm_duanbuy_active"][i]),
                "mm_jimai_state": float(mm["mm_jimai_state"][i]),
                "mm_zhunbei_active": float(mm["mm_zhunbei_active"][i]),
                "mm_shentou": float(mm["mm_shentou"][i]),
                "mm_dongxiang": float(mm["mm_dongxiang"][i]),
            }
            f.update(_csf_for(sym, df_h["trade_date"].iloc[i]))
            feat_cache[i] = f
            rows.append([f[k] for k in _FEATURE_NAMES])
            idx.append(i)
        if not rows:
            continue
        X = np.array(rows)
        dmat = xgb.DMatrix(X, feature_names=_FEATURE_NAMES)
        preds = np.array([m.predict(dmat) for m in models]).mean(axis=0)
        score_map = {i: float(s) for i, s in zip(idx, preds)}
        cache[sym] = (df_h, close, feat_cache, score_map)
        if (stock_idx + 1) % 25 == 0:
            print(f"    scored {stock_idx + 1}/{len(candles_by_sym)}", flush=True)
    return cache


def evaluate_one(params: dict, score_cache: dict) -> dict:
    """Run one variant against pre-computed scores. Fast."""
    from app.services.strategy_templates.ml_direct import _is_candidate, _load_thresholds

    thr_lookup = _load_thresholds() if params.get("use_adaptive") else {}
    default_buy = params["buy_threshold"]
    default_exit = params["exit_threshold"]
    sl = params["stop_loss"]
    trail_act = params["trail_activation"]
    trail_pct = params["trail_pct"]
    time_stop = params["time_stop"]

    total_trades = total_wins = stocks_traded = stocks_profit = 0
    sum_ret = 0.0
    all_pcts: list[float] = []

    for sym, (df_h, close, feat_cache, score_map) in score_cache.items():
        # Resolve thresholds for this stock
        if params.get("use_adaptive") and sym in thr_lookup:
            buy_thr = float(thr_lookup[sym].get("buy", default_buy))
            exit_thr = float(thr_lookup[sym].get("exit", default_exit))
        else:
            buy_thr = default_buy
            exit_thr = default_exit

        # Walk bars and simulate
        in_pos = False
        entry_price = 0.0
        peak = 0.0
        trail_active = False
        hold = 0
        signals: list[dict] = []
        for i in range(120, len(df_h)):
            score = score_map.get(i)
            if score is None:
                continue
            d = df_h["trade_date"].iloc[i]
            c = float(close[i])
            if not in_pos:
                if not _is_candidate(close, feat_cache, i, require_trend=False):
                    continue
                if score >= buy_thr:
                    signals.append({"date": d, "action": "buy"})
                    entry_price = c; peak = c
                    trail_active = False; hold = 0; in_pos = True
            else:
                hold += 1
                if c > peak: peak = c
                pnl = (c - entry_price) / entry_price if entry_price else 0
                hard_stop = entry_price * (1 + sl)
                if not trail_active and pnl >= trail_act:
                    trail_active = True
                ts_p = (max(entry_price, peak * (1 - trail_pct))
                        if trail_active else hard_stop)
                if c <= hard_stop:
                    signals.append({"date": d, "action": "sell"}); in_pos = False
                elif score <= exit_thr:
                    signals.append({"date": d, "action": "sell"}); in_pos = False
                elif trail_active and c <= ts_p:
                    signals.append({"date": d, "action": "sell"}); in_pos = False
                elif hold >= time_stop:
                    signals.append({"date": d, "action": "sell"}); in_pos = False

        # Filter to post-cutoff and run backtest
        in_p = False
        out: list[dict] = []
        for sig in signals:
            if sig["action"] == "buy" and sig["date"] >= CUTOFF:
                out.append(sig); in_p = True
            elif sig["action"] == "sell" and in_p:
                if sig["date"] >= CUTOFF:
                    out.append(sig); in_p = False
        post = [c for c in df_h.to_dict("records") if c["trade_date"] >= CUTOFF]
        if not post or not out:
            continue
        r = run_backtest(post, out, 10000)
        if not r["trades"]:
            continue
        pcts = [(t["exit_price"] - t["entry_price"]) / t["entry_price"] * 100
                for t in r["trades"]]
        wins = sum(1 for p in pcts if p > 0)
        total_trades += len(r["trades"]); total_wins += wins
        sum_ret += r["net_profit_pct"]; stocks_traded += 1
        if r["net_profit_pct"] > 0:
            stocks_profit += 1
        all_pcts.extend(pcts)

    if total_trades == 0 or stocks_traded == 0:
        return {"total_trades": 0, "wins": 0, "win_rate": 0,
                "stocks_traded": 0, "stocks_profit": 0, "stocks_profit_rate": 0,
                "avg_per_stock_ret": 0, "acc5_pct": 0, "acc10_pct": 0,
                "median_pnl": 0, "composite": 0}

    win_rate = total_wins / total_trades * 100
    acc5 = sum(1 for p in all_pcts if p >= 5) / len(all_pcts) * 100
    acc10 = sum(1 for p in all_pcts if p >= 10) / len(all_pcts) * 100
    avg_per_stock = sum_ret / stocks_traded
    median = sorted(all_pcts)[len(all_pcts) // 2]
    profit_stocks_rate = stocks_profit / stocks_traded * 100

    # Composite: balance win rate, ≥5% rate, profitability, and avg return
    composite = (
        win_rate * 0.35
        + acc5 * 0.25
        + profit_stocks_rate * 0.20
        + min(max(avg_per_stock, -50), 50) * 0.20
    )

    return {
        "total_trades": total_trades,
        "wins": total_wins,
        "win_rate": round(win_rate, 1),
        "stocks_traded": stocks_traded,
        "stocks_profit": stocks_profit,
        "stocks_profit_rate": round(profit_stocks_rate, 1),
        "avg_per_stock_ret": round(avg_per_stock, 2),
        "acc5_pct": round(acc5, 1),
        "acc10_pct": round(acc10, 1),
        "median_pnl": round(median, 2),
        "composite": round(composite, 2),
    }


async def main() -> None:
    codes = await get_pool_codes(3)
    print(f"Pool: {len(codes)} stocks\n")

    print("Pre-loading all candles...", flush=True)
    candles_by_sym = {}
    for i, c in enumerate(codes):
        df = await load_candles(c)
        if len(df) >= 130:
            candles_by_sym[c] = df
        if (i + 1) % 25 == 0:
            print(f"  loaded {i+1}/{len(codes)}", flush=True)
    print(f"Loaded {len(candles_by_sym)} stocks with sufficient data\n", flush=True)

    print("Pre-computing ML scores for all bars (one-time)...", flush=True)
    score_cache = precompute_scores(candles_by_sym)
    print(f"Cached scores for {len(score_cache)} stocks\n", flush=True)

    results = []
    for name, params in VARIANTS:
        print(f"=== {name} ===", flush=True)
        r = evaluate_one(params, score_cache)
        r["name"] = name
        r["params"] = params
        results.append(r)
        print(f"  trades={r['total_trades']:>4}  win%={r['win_rate']:>5.1f}  "
              f"≥5%={r['acc5_pct']:>5.1f}  ≥10%={r['acc10_pct']:>5.1f}  "
              f"avg/stock={r['avg_per_stock_ret']:+6.2f}%  "
              f"profit_stocks={r['stocks_profit_rate']:>5.1f}%  "
              f"composite={r['composite']:>6.2f}")

    # Rank by composite
    ranked = sorted(results, key=lambda x: -x["composite"])
    print(f"\n{'='*78}\n  RANKING (by composite)\n{'='*78}")
    for i, r in enumerate(ranked, 1):
        print(f"{i:>2}. {r['name']:<22} composite={r['composite']:>6.2f}  "
              f"win%={r['win_rate']:>5.1f}  avg={r['avg_per_stock_ret']:+6.2f}%  "
              f"trades={r['total_trades']}")

    print(f"\n=== TOP 3 ===")
    top3 = ranked[:3]
    for i, r in enumerate(top3, 1):
        print(f"\n#{i}: {r['name']}")
        print(f"  params: {r['params']}")
        print(f"  metrics: trades={r['total_trades']}  win={r['win_rate']:.1f}%  "
              f"≥5%={r['acc5_pct']:.1f}%  avg/stock={r['avg_per_stock_ret']:+.2f}%  "
              f"profit_stocks={r['stocks_profit_rate']:.1f}%")

    # Save full results for inspection
    os.makedirs("models", exist_ok=True)
    with open("models/sweep_v8_v20_results.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)
    print(f"\nFull results saved to models/sweep_v8_v20_results.json")


if __name__ == "__main__":
    asyncio.run(main())
