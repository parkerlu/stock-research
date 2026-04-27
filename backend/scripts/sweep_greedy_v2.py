"""
Greedy algorithm sweep — find the best trailing-stop logic to capture swing highs.

Tests 4 exit modes with ~100 configurations on 1005 stocks (post-2024 OOS):

  1. CLASSIC: simple % trail after activation
       — trail_stop = max(entry, peak × (1 - trail_pct)) once pnl ≥ activation

  2. ATR_TRAIL: ATR-multiplier trail (volatility-adaptive)
       — trail_stop = peak − atr_mult × ATR14
       — Stocks with high volatility get wider stops automatically

  3. STEP_TRAIL: discrete profit-milestone trail (lock-in ladder)
       — At +5% gain: stop = entry × 1.01 (lock 1% profit)
       — At +10%: stop = peak × (1 - 0.05)
       — At +20%: stop = peak × (1 - 0.04)
       — At +30%: stop = peak × (1 - 0.03)

  4. CHANDELIER: peak − K × ATR(N), classic swing trader exit

Picks top 3 by composite score (win rate, ≥5%, ≥10%, avg per stock).
"""
from __future__ import annotations

import asyncio
import itertools
import json
import os
import random
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from app.services.backtest_engine import run_backtest
from app.services.strategy_templates.ml_direct import (
    _FEATURE_NAMES, _load_ensemble, _csf_for, _is_candidate,
)
from scripts.ml_step1_train import precompute_maimai


CUTOFF = pd.Timestamp("2024-01-01").date()
HISTORY_START = pd.Timestamp("2022-01-01").date()


def precompute_features_and_atr(candles_by_sym: dict) -> dict:
    """Score every (stock, bar) once + cache ATR series for ATR-based exits."""
    models = _load_ensemble()
    if not models:
        return {}
    cache = {}
    for stock_idx, (sym, df) in enumerate(candles_by_sym.items()):
        df_h = df[df["trade_date"] >= HISTORY_START].reset_index(drop=True)
        if len(df_h) < 130:
            continue
        close = df_h["close"].astype(float).values
        high = df_h["high"].astype(float).values
        low = df_h["low"].astype(float).values
        vol = df_h["vol"].astype(float).values

        # ATR14
        prev_c = np.roll(close, 1); prev_c[0] = close[0]
        tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
        atr14 = pd.Series(tr).rolling(14, min_periods=5).mean().values

        mm = precompute_maimai(df_h)

        rows, idx, feat_cache = [], [], {}
        for i in range(120, len(df_h)):
            c = close[i]
            ma20 = close[i - 20:i].mean()
            ma60 = close[i - 60:i].mean()
            ma120 = close[i - 120:i].mean()
            high20, low20 = high[i - 20:i].max(), low[i - 20:i].min()
            high60, low60 = high[i - 60:i].max(), low[i - 60:i].min()
            atr14p = (high[i - 14:i] - low[i - 14:i]).mean()
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
                "atr14_pct": atr14p / c * 100,
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
        cache[sym] = (df_h, close, feat_cache, score_map, atr14)
        if (stock_idx + 1) % 100 == 0:
            print(f"    scored {stock_idx + 1}/{len(candles_by_sym)}", flush=True)
    return cache


def compute_exit(mode: str, params: dict, *,
                 entry_price: float, peak_price: float, current_close: float,
                 hold_bars: int, atr_today: float | None) -> tuple[bool, str]:
    """Returns (should_sell, reason). Hard stop is ALWAYS checked first."""
    sl = params["stop_loss"]
    time_stop = params["time_stop"]
    pnl = current_close / entry_price - 1

    # Hard stop
    if pnl <= sl:
        return True, "hard_stop"

    if mode == "CLASSIC":
        act = params["activation"]
        trail_pct = params["trail_pct"]
        if pnl >= act:
            stop = max(entry_price, peak_price * (1 - trail_pct))
            if current_close <= stop:
                return True, "classic_trail"

    elif mode == "ATR_TRAIL":
        atr_mult = params["atr_mult"]
        act = params["activation"]
        if pnl >= act and atr_today and atr_today > 0:
            stop = max(entry_price, peak_price - atr_mult * atr_today)
            if current_close <= stop:
                return True, "atr_trail"

    elif mode == "STEP_TRAIL":
        # Discrete ladder: locks in fixed profit at milestones
        if pnl >= 0.05 and current_close <= entry_price * 1.01:
            return True, "step_5_lock1"
        if pnl >= 0.10 and current_close <= peak_price * (1 - 0.05):
            return True, "step_10_trail5"
        if pnl >= 0.20 and current_close <= peak_price * (1 - 0.04):
            return True, "step_20_trail4"
        if pnl >= 0.30 and current_close <= peak_price * (1 - 0.03):
            return True, "step_30_trail3"

    elif mode == "CHANDELIER":
        # Always-on ATR trail from highest, no activation gate
        atr_mult = params["atr_mult"]
        if atr_today and atr_today > 0 and hold_bars >= 1:
            stop = peak_price - atr_mult * atr_today
            if current_close <= stop and current_close > entry_price * (1 + sl):
                return True, "chandelier"

    if hold_bars >= time_stop:
        return True, "time_stop"
    return False, ""


def evaluate_one(mode: str, params: dict, score_cache: dict) -> dict:
    total_trades = total_wins = stocks_traded = stocks_profit = 0
    sum_ret = 0.0
    all_pcts: list[float] = []
    durations: list[int] = []

    for sym, (df_h, close, feat_cache, score_map, atr14) in score_cache.items():
        buy_thr = params["buy_threshold"]
        in_pos = False
        entry_price = peak = 0.0
        hold = 0
        signals: list[dict] = []
        for i in range(120, len(df_h)):
            score = score_map.get(i)
            if score is None:
                continue
            d = df_h["trade_date"].iloc[i]
            c = float(close[i])
            if not in_pos:
                if not _is_candidate(close, feat_cache, i, require_trend=False, ts_code=sym):
                    continue
                if score >= buy_thr:
                    signals.append({"date": d, "action": "buy"})
                    entry_price = peak = c
                    hold = 0
                    in_pos = True
            else:
                hold += 1
                if c > peak:
                    peak = c
                atr_today = float(atr14[i]) if i < len(atr14) and not np.isnan(atr14[i]) else None
                should_sell, reason = compute_exit(
                    mode, params, entry_price=entry_price, peak_price=peak,
                    current_close=c, hold_bars=hold, atr_today=atr_today,
                )
                if should_sell:
                    signals.append({"date": d, "action": "sell"})
                    in_pos = False

        # Filter to post-cutoff and run backtest
        out: list[dict] = []
        in_p = False
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
        # Estimate avg duration via trade entry/exit dates
        for t in r["trades"]:
            d_in = pd.to_datetime(t["entry_date"])
            d_out = pd.to_datetime(t["exit_date"])
            durations.append((d_out - d_in).days)

    if total_trades == 0:
        return {"total_trades": 0, "win_rate": 0, "acc5_pct": 0, "acc10_pct": 0,
                "avg_per_stock_ret": 0, "stocks_profit_rate": 0,
                "median_pnl": 0, "avg_duration": 0}
    win_rate = total_wins / total_trades * 100
    acc5 = sum(1 for p in all_pcts if p >= 5) / len(all_pcts) * 100
    acc10 = sum(1 for p in all_pcts if p >= 10) / len(all_pcts) * 100
    return {
        "total_trades": total_trades,
        "win_rate": round(win_rate, 1),
        "acc5_pct": round(acc5, 1),
        "acc10_pct": round(acc10, 1),
        "avg_per_stock_ret": round(sum_ret / stocks_traded, 2),
        "stocks_profit_rate": round(stocks_profit / stocks_traded * 100, 1),
        "median_pnl": round(float(np.median(all_pcts)), 2),
        "avg_duration": round(float(np.mean(durations)), 1) if durations else 0,
    }


def gen_variants():
    """Build ~100 diverse variants across 4 exit modes."""
    vs = []

    # CLASSIC: 6×6 grid = 36 with -10% stop, 60-bar time stop
    for act in [0.05, 0.07, 0.09, 0.11, 0.13]:
        for tp in [0.025, 0.03, 0.04, 0.05, 0.06, 0.08]:
            vs.append(("CLASSIC", {
                "buy_threshold": 0.50, "stop_loss": -0.10,
                "activation": act, "trail_pct": tp, "time_stop": 60,
            }))

    # ATR_TRAIL: vary atr_mult and activation (24)
    for act in [0.03, 0.05, 0.08]:
        for atr_mult in [1.5, 2.0, 2.5, 3.0]:
            for ts in [45, 75]:
                vs.append(("ATR_TRAIL", {
                    "buy_threshold": 0.50, "stop_loss": -0.10,
                    "activation": act, "atr_mult": atr_mult, "time_stop": ts,
                }))

    # STEP_TRAIL: only one variant (logic is fixed); vary buy and time_stop (4)
    for buy in [0.45, 0.50]:
        for ts in [60, 90]:
            vs.append(("STEP_TRAIL", {
                "buy_threshold": buy, "stop_loss": -0.10, "time_stop": ts,
            }))

    # CHANDELIER: vary atr_mult and time_stop (16)
    for atr_mult in [1.5, 2.0, 2.5, 3.0]:
        for ts in [30, 45, 60, 90]:
            vs.append(("CHANDELIER", {
                "buy_threshold": 0.50, "stop_loss": -0.10,
                "atr_mult": atr_mult, "time_stop": ts,
            }))

    # Mix: ATR trail with lower buy threshold (12)
    for buy in [0.45, 0.50]:
        for atr_mult in [2.0, 2.5, 3.0]:
            for ts in [30, 60]:
                vs.append(("ATR_TRAIL", {
                    "buy_threshold": buy, "stop_loss": -0.10,
                    "activation": 0.05, "atr_mult": atr_mult, "time_stop": ts,
                }))

    return vs


async def get_codes() -> list[str]:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await eng.dispose()
    return list(rows)


async def load_candles(symbol: str) -> pd.DataFrame:
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(DailyCandle).where(DailyCandle.ts_code == symbol)
            .order_by(DailyCandle.trade_date)
        )).scalars().all()
    await eng.dispose()
    if not rows:
        return pd.DataFrame()
    la = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
    return pd.DataFrame([{
        "trade_date": r.trade_date,
        "open": round(float(r.open) * (float(r.adj_factor or 1.0) / la), 4),
        "high": round(float(r.high) * (float(r.adj_factor or 1.0) / la), 4),
        "low":  round(float(r.low)  * (float(r.adj_factor or 1.0) / la), 4),
        "close":round(float(r.close)* (float(r.adj_factor or 1.0) / la), 4),
        "vol": float(r.vol or 0), "amount": float(r.amount) if r.amount else 0,
    } for r in rows])


async def main() -> None:
    codes = await get_codes()
    # Use top 300 stocks by ts_code for speed (representative subset)
    codes = sorted(codes)[:300]
    print(f"Loading {len(codes)} stocks...", flush=True)
    candles = {}
    for k, c in enumerate(codes):
        df = await load_candles(c)
        if len(df) >= 130:
            candles[c] = df
        if (k + 1) % 50 == 0:
            print(f"  loaded {k + 1}", flush=True)
    print(f"Loaded {len(candles)}\nPre-computing scores + ATR...", flush=True)
    cache = precompute_features_and_atr(candles)
    print(f"Cached {len(cache)}\n", flush=True)

    variants = gen_variants()
    print(f"Testing {len(variants)} variants on {len(cache)} stocks...\n", flush=True)
    results = []
    t0 = time.time()
    for k, (mode, params) in enumerate(variants, 1):
        m = evaluate_one(mode, params, cache)
        m["mode"] = mode
        m["params"] = params
        results.append(m)
        if k % 20 == 0:
            print(f"  evaluated {k}/{len(variants)} ({time.time() - t0:.0f}s)", flush=True)

    # Composite score: balance win rate, ≥5% rate, avg/stock, frequency
    for r in results:
        r["composite"] = (
            r["win_rate"] * 0.30
            + r["acc5_pct"] * 0.25
            + r["stocks_profit_rate"] * 0.20
            + min(r["avg_per_stock_ret"], 50) * 0.15
            + min(r["acc10_pct"], 50) * 0.10
        )

    ranked = sorted(results, key=lambda x: -x["composite"])
    print(f"\n{'='*90}", flush=True)
    print(f"  TOP 15 CONFIGS by composite", flush=True)
    print(f"{'='*90}", flush=True)
    print(f"{'#':>2}  {'mode':<11}  {'trades':>5}  {'win':>5}  {'≥5%':>5}  {'≥10%':>5}  "
          f"{'avg':>6}  {'days':>4}  composite  params", flush=True)
    print("-" * 90, flush=True)
    for i, r in enumerate(ranked[:15], 1):
        p_str = ", ".join(f"{k}={v}" for k, v in r["params"].items() if k != "buy_threshold")
        print(f"{i:>2}  {r['mode']:<11}  {r['total_trades']:>5}  "
              f"{r['win_rate']:>4.1f}%  {r['acc5_pct']:>4.1f}%  "
              f"{r['acc10_pct']:>4.1f}%  {r['avg_per_stock_ret']:>+5.1f}%  "
              f"{r['avg_duration']:>4.0f}  {r['composite']:>7.2f}  | "
              f"buy={r['params']['buy_threshold']} {p_str[:50]}", flush=True)

    os.makedirs("models", exist_ok=True)
    with open("models/sweep_greedy_v2_results.json", "w") as f:
        json.dump(ranked, f, indent=2, default=str)
    print(f"\nFull results saved to models/sweep_greedy_v2_results.json", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
