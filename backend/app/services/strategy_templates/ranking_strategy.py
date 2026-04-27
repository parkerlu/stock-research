"""
RankingStrategy — daily top-decile ranking strategy.

Different from MLDirectDecision (which scores each bar independently with a
binary classifier). This strategy:

  1. Each day, scores ALL stocks in the universe with composite GP factor
  2. Buys top decile (top 10% by composite score) when entering position
  3. Sells when stock drops out of top 30% OR hits stop / target

Naturally high-frequency (signals every day) AND can be high-accuracy
(only top-ranked stocks are bought).

This is a MULTI-STOCK strategy — it operates on a panel, not per-stock.
The current backtest engine is single-stock so we provide a separate
panel-aware backtest method.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .base import StrategyTemplate


def _vwap_proxy(panel: dict) -> np.ndarray:
    return (panel["high"] + panel["low"] + panel["close"]) / 3


def _gp_quiet_momentum(panel: dict) -> np.ndarray:
    """Top GP V2 factor."""
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    vol = panel["vol"]
    vwap = _vwap_proxy(panel)
    safe_vwap = np.where(vwap > 0, vwap, 1)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / np.where(close[:-1] > 0, close[:-1], 1) - 1
    ts_max_ret = pd.DataFrame(ret).rolling(10, min_periods=1).max().values
    return ts_max_ret * (close / safe_vwap) * (low - high) * vol


def _ret_momentum(panel: dict, n: int) -> np.ndarray:
    return pd.DataFrame(panel["close"]).pct_change(n).fillna(0).values


def composite_score(panel: dict) -> np.ndarray:
    """Combine multiple factors via cross-sectional rank averaging.

    Each factor → daily rank percentile (0..1) → average.
    Resulting score is itself in [0, 1] roughly.
    """
    factors = {
        "qmom": _gp_quiet_momentum(panel),
        "mom20": _ret_momentum(panel, 20),
        "neg_max_dd": -_max_drawdown(panel, 20),  # negate so higher = better
    }
    T, N = panel["close"].shape
    composite = np.zeros((T, N), dtype=np.float64)
    for arr in factors.values():
        ranks = np.full_like(arr, 0.5, dtype=np.float64)
        for t in range(T):
            row = arr[t]
            valid = np.isfinite(row)
            if valid.sum() < 5:
                continue
            ranks[t, valid] = pd.Series(row[valid]).rank(pct=True).values
        composite += ranks
    composite /= len(factors)
    return composite


def _max_drawdown(panel: dict, n: int) -> np.ndarray:
    high_n = pd.DataFrame(panel["high"]).rolling(n, min_periods=1).max().values
    return panel["close"] / np.where(high_n > 0, high_n, 1) - 1


def panel_backtest(
    panel: dict,
    dates: list,
    codes: list[str],
    start_date,
    composite: np.ndarray,
    *,
    top_pct: float = 0.10,
    drop_pct: float = 0.30,
    stop_loss: float = -0.07,
    take_profit_trail: float = 0.05,
    trail_act: float = 0.05,
    time_stop: int = 60,
) -> dict:
    """Daily-rebalance ranking backtest.

    Each day:
      For each stock currently held, check exits (stop / trail / time / out-of-rank)
      Then for any "open slot", scan top decile of composite that's NOT a 涨停 day
      Buy at close

    For simplicity: equal-weight across held positions, max N positions.
    """
    T, N = composite.shape
    max_positions = max(int(N * top_pct), 5)

    positions: dict[int, dict] = {}  # stock_idx -> {entry_price, peak, hold_bars, trail_active}
    completed_trades: list[dict] = []

    start_idx = next((i for i, d in enumerate(dates) if d >= start_date), 0)

    for t in range(start_idx, T):
        day = dates[t]
        # 1. Exit logic for existing positions
        to_exit = []
        for j, pos in positions.items():
            close = panel["close"][t, j]
            if not np.isfinite(close) or close <= 0:
                continue
            entry = pos["entry_price"]
            peak = max(pos["peak"], close)
            pos["peak"] = peak
            pos["hold_bars"] += 1
            pnl = close / entry - 1
            # Check rank drop
            cs = composite[t, j]
            stock_rank = pd.Series(composite[t]).rank(pct=True).iloc[j] \
                if not np.isnan(composite[t, j]) else 0
            # Update trail-active
            if pnl >= trail_act:
                pos["trail_active"] = True
            hard_stop = entry * (1 + stop_loss)
            trail_stop = max(entry, peak * (1 - take_profit_trail)) if pos["trail_active"] else hard_stop

            if close <= hard_stop:
                reason = "stop"
            elif pos["trail_active"] and close <= trail_stop:
                reason = "trail"
            elif pos["hold_bars"] >= time_stop:
                reason = "time"
            elif stock_rank < (1 - drop_pct):  # dropped out of top 30%
                reason = "rank_drop"
            else:
                continue
            completed_trades.append({
                "ts_code": codes[j],
                "entry_date": pos["entry_date"],
                "exit_date": day,
                "entry_price": round(entry, 4),
                "exit_price": round(close, 4),
                "pnl_pct": round((close - entry) / entry * 100, 2),
                "reason": reason,
            })
            to_exit.append(j)
        for j in to_exit:
            del positions[j]

        # 2. Entry: pick top-pct stocks not currently held + not at limit-up
        n_open_slots = max_positions - len(positions)
        if n_open_slots <= 0:
            continue
        cs_today = composite[t]
        valid = np.isfinite(cs_today)
        if valid.sum() < 5:
            continue
        # Compute today's rank percentile per stock
        ranks = pd.Series(cs_today[valid]).rank(pct=True)
        rank_idx = np.where(valid)[0]
        rank_pairs = [(ri, ranks.iloc[i]) for i, ri in enumerate(rank_idx)]
        rank_pairs.sort(key=lambda x: -x[1])
        # Filter: skip already held, skip limit-up bars
        new_entries = []
        for j, r in rank_pairs:
            if r < (1 - top_pct):
                break
            if j in positions:
                continue
            # Limit-up filter
            close = panel["close"][t, j]
            prev_close = panel["close"][t - 1, j] if t > 0 else close
            if prev_close > 0:
                gain = close / prev_close - 1
                code6 = codes[j].split(".")[0]
                limit = 0.20 if code6.startswith("688") or code6.startswith("30") else 0.10
                if gain >= limit - 0.005:
                    continue
            new_entries.append(j)
            if len(new_entries) >= n_open_slots:
                break

        for j in new_entries:
            close = panel["close"][t, j]
            if not np.isfinite(close) or close <= 0:
                continue
            positions[j] = {
                "entry_price": close,
                "entry_date": day,
                "peak": close,
                "hold_bars": 0,
                "trail_active": False,
            }

    # Force-close any open positions at end
    last_t = T - 1
    for j, pos in positions.items():
        close = panel["close"][last_t, j]
        if not np.isfinite(close):
            continue
        completed_trades.append({
            "ts_code": codes[j],
            "entry_date": pos["entry_date"],
            "exit_date": dates[last_t],
            "entry_price": round(pos["entry_price"], 4),
            "exit_price": round(close, 4),
            "pnl_pct": round((close / pos["entry_price"] - 1) * 100, 2),
            "reason": "end_of_period",
        })

    pcts = [t["pnl_pct"] for t in completed_trades]
    if not pcts:
        return {"trades": [], "n_trades": 0, "win_rate": 0, "avg_pnl": 0,
                "total_return": 0, "acc5": 0, "acc10": 0}
    wins = sum(1 for p in pcts if p > 0)
    return {
        "trades": completed_trades,
        "n_trades": len(completed_trades),
        "win_rate": wins / len(completed_trades) * 100,
        "avg_pnl": float(np.mean(pcts)),
        "median_pnl": float(np.median(pcts)),
        "total_return": float(sum(pcts)),
        "acc5": sum(1 for p in pcts if p >= 5) / len(pcts) * 100,
        "acc10": sum(1 for p in pcts if p >= 10) / len(pcts) * 100,
    }
