"""V2 mining with LambdaRank scores. Output: docs/strategy_mine/lr/"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Hot-patch the score path before importing the v2 mining module
from scripts import mine_426_v2_mine as v2  # noqa: E402

v2.SCORES_PATH = ROOT / "cache" / "lr_scores.parquet"
v2.OUT_DIR = ROOT.parent / "docs" / "strategy_mine" / "lr"


def patched_load():
    """Drop-in replacement using lr_scores instead of ml_scores."""
    import pandas as pd
    print("Loading caches (LR scores)...", flush=True)
    ohlcv = pd.read_parquet(ROOT / "cache" / "ohlcv.parquet")
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    scores = pd.read_parquet(v2.SCORES_PATH)
    scores["trade_date"] = pd.to_datetime(scores["trade_date"])
    indic = pd.read_parquet(ROOT / "cache" / "indicators.parquet")
    indic["trade_date"] = pd.to_datetime(indic["trade_date"])
    print(f"  OHLCV: {len(ohlcv):,}, scores: {len(scores):,}, indic: {len(indic):,}", flush=True)
    merged = ohlcv.merge(scores, on=["ts_code", "trade_date"], how="left")
    merged = merged.merge(indic, on=["ts_code", "trade_date"], how="left")
    merged["ml_score"] = merged["ml_score"].fillna(0.0)

    indic_cols = [c for c in indic.columns if c not in ("ts_code", "trade_date")]
    per_stock = {}; dates_per_stock = {}
    for ts, g in merged.groupby("ts_code"):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if len(g) < 60:
            continue
        ind = {c: g[c].values.astype(float) for c in indic_cols}
        per_stock[ts] = {
            "close": g["close"].values.astype(float),
            "high": g["high"].values.astype(float),
            "low": g["low"].values.astype(float),
            "score": g["ml_score"].values.astype(float),
            "ind": ind,
        }
        dates_per_stock[ts] = [d.strftime("%Y-%m-%d") for d in g["trade_date"]]
    print(f"  per-stock data: {len(per_stock)} stocks", flush=True)
    return per_stock, dates_per_stock


v2.load_per_stock_data = patched_load
v2.main()
