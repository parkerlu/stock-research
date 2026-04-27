"""
Build cross-sectional rank features for the full universe.

For each trading day d, take all stocks alive on d, compute factor values,
and convert to within-day percentile rank (0..1). This neutralizes
absolute-magnitude effects so XGBoost can use the rank info directly.

Output: a parquet file mapping (ts_code, trade_date) → {feature_rank dict}
that ml_step1_train.py can load and merge.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle


# Feature definitions: (name, lambda taking panel-of-arrays returning (T,N))
# Each feature must be computable from open/high/low/close/vol/amount
FEATURE_FNS = {
    "csf_ret5": lambda p: pd.DataFrame(p["close"]).pct_change(5).fillna(0).values,
    "csf_ret20": lambda p: pd.DataFrame(p["close"]).pct_change(20).fillna(0).values,
    "csf_vol_ratio_5_20": lambda p: (
        pd.DataFrame(p["vol"]).rolling(5, min_periods=1).mean().values /
        np.where(pd.DataFrame(p["vol"]).rolling(20, min_periods=1).mean().values > 0,
                 pd.DataFrame(p["vol"]).rolling(20, min_periods=1).mean().values, 1)
    ),
    "csf_pos20": lambda p: _pos_in_range(p, 20),
    "csf_atr_pct": lambda p: (
        pd.DataFrame(p["high"] - p["low"]).rolling(14, min_periods=1).mean().values /
        np.where(p["close"] > 0, p["close"], 1) * 100
    ),
    "csf_money_flow_5": lambda p: (
        pd.DataFrame((p["close"] - (p["high"] + p["low"]) / 2) / np.where(p["high"] != p["low"], p["high"] - p["low"], 1) * p["vol"])
        .rolling(5, min_periods=1).sum().values
    ),
    "csf_qmom": lambda p: _gp_quiet_momentum(p),
    "csf_close_to_ma20": lambda p: (
        p["close"] / np.where(pd.DataFrame(p["close"]).rolling(20, min_periods=1).mean().values > 0,
                              pd.DataFrame(p["close"]).rolling(20, min_periods=1).mean().values, 1) - 1
    ),
    "csf_max_drawdown_20": lambda p: _max_drawdown(p, 20),
    "csf_high_low_corr_20": lambda p: _ts_corr_panel(p["high"], p["low"], 20),
}


def _pos_in_range(panel: dict, n: int) -> np.ndarray:
    high_n = pd.DataFrame(panel["high"]).rolling(n, min_periods=1).max().values
    low_n = pd.DataFrame(panel["low"]).rolling(n, min_periods=1).min().values
    rng = high_n - low_n
    return np.where(rng > 0, (panel["close"] - low_n) / np.where(rng > 0, rng, 1), 0.5)


def _gp_quiet_momentum(panel: dict) -> np.ndarray:
    """Top GP V2 factor: ts_max(ret,10) × close/vwap × (low-high) × vol."""
    close = panel["close"]
    high = panel["high"]
    low = panel["low"]
    vol = panel["vol"]
    vwap = (high + low + close) / 3
    safe_vwap = np.where(vwap > 0, vwap, 1)
    ret = np.zeros_like(close)
    ret[1:] = close[1:] / np.where(close[:-1] > 0, close[:-1], 1) - 1
    ts_max_ret = pd.DataFrame(ret).rolling(10, min_periods=1).max().values
    return ts_max_ret * (close / safe_vwap) * (low - high) * vol


def _max_drawdown(panel: dict, n: int) -> np.ndarray:
    high_n = pd.DataFrame(panel["high"]).rolling(n, min_periods=1).max().values
    return (panel["close"] / np.where(high_n > 0, high_n, 1)) - 1


def _ts_corr_panel(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(a)
    for j in range(a.shape[1]):
        sa = pd.Series(a[:, j])
        sb = pd.Series(b[:, j])
        out[:, j] = sa.rolling(n, min_periods=2).corr(sb).fillna(0).values
    return out


async def main() -> None:
    print("Loading all stocks...", flush=True)
    eng = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(eng, expire_on_commit=False)
    async with Session() as db:
        from sqlalchemy import distinct
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    print(f"  {len(codes)} stocks in DB", flush=True)

    per_stock: dict[str, pd.DataFrame] = {}
    for k, code in enumerate(codes):
        async with Session() as db:
            rows = (await db.execute(
                select(DailyCandle).where(DailyCandle.ts_code == code)
                .order_by(DailyCandle.trade_date)
            )).scalars().all()
        if len(rows) < 130:
            continue
        latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
        df = pd.DataFrame([{
            "trade_date": r.trade_date,
            "open": float(r.open) * (float(r.adj_factor or 1.0) / latest_adj),
            "high": float(r.high) * (float(r.adj_factor or 1.0) / latest_adj),
            "low":  float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj),
            "close":float(r.close)* (float(r.adj_factor or 1.0) / latest_adj),
            "vol": float(r.vol or 0),
            "amount": float(r.amount or 0),
        } for r in rows]).set_index("trade_date")
        per_stock[code] = df
        if (k + 1) % 50 == 0:
            print(f"  loaded {k + 1}/{len(codes)}", flush=True)
    await eng.dispose()
    print(f"  total {len(per_stock)} stocks with sufficient data", flush=True)

    # Determine common date range (use union: each stock contributes its valid dates)
    all_dates = sorted(set().union(*[set(df.index) for df in per_stock.values()]))
    print(f"  total {len(all_dates)} unique dates", flush=True)

    codes_used = list(per_stock.keys())
    T = len(all_dates)
    N = len(codes_used)
    print(f"  building panel T={T} × N={N}...", flush=True)

    # Build panel arrays — use dict lookup (O(1)) instead of list.index (O(T))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    cols = ["open", "high", "low", "close", "vol", "amount"]
    panel = {}
    for col in cols:
        arr = np.full((T, N), np.nan, dtype=np.float32)
        for j, c in enumerate(codes_used):
            df = per_stock[c]
            indices = np.array([date_to_idx[d] for d in df.index], dtype=np.int64)
            arr[indices, j] = df[col].values.astype(np.float32)
        panel[col] = arr
        print(f"    panel[{col}] built", flush=True)

    print(f"  computing {len(FEATURE_FNS)} features...", flush=True)
    feature_panels: dict[str, np.ndarray] = {}
    for name, fn in FEATURE_FNS.items():
        try:
            feature_panels[name] = fn(panel).astype(np.float32)
            print(f"    {name}: shape={feature_panels[name].shape}", flush=True)
        except Exception as e:
            print(f"    {name}: FAILED — {e}", flush=True)

    # Cross-sectional rank: for each row, rank stocks (NaN-aware)
    print(f"\n  cross-sectional ranking each day...", flush=True)
    rank_panels: dict[str, np.ndarray] = {}
    for name, arr in feature_panels.items():
        ranks = np.full_like(arr, 0.5, dtype=np.float32)
        for t in range(T):
            row = arr[t]
            valid = ~np.isnan(row)
            if valid.sum() < 5:
                continue
            valid_vals = row[valid]
            # Rank to [0, 1]
            order = pd.Series(valid_vals).rank(pct=True).values.astype(np.float32)
            ranks[t, valid] = order
        rank_panels[name] = ranks
        print(f"    ranked {name}", flush=True)

    # Save as long-format parquet: (ts_code, trade_date, csf_*_rank)
    print(f"\n  serializing to parquet...", flush=True)
    rows_out = []
    for j, code in enumerate(codes_used):
        df = per_stock[code]
        for d in df.index:
            t_idx = date_to_idx[d]
            row = {"ts_code": code, "trade_date": d}
            for name in feature_panels:
                row[f"{name}_rank"] = float(rank_panels[name][t_idx, j])
            rows_out.append(row)
    out_df = pd.DataFrame(rows_out)
    out_path = "models/cross_sectional_ranks.parquet"
    os.makedirs("models", exist_ok=True)
    out_df.to_parquet(out_path, index=False)
    print(f"  saved {len(out_df)} rows × {len(out_df.columns)} cols to {out_path}",
          flush=True)
    print(f"  feature columns: {[c for c in out_df.columns if c.startswith('csf_')]}",
          flush=True)


if __name__ == "__main__":
    asyncio.run(main())
