"""Build (60-bar input, 5-bar future log-returns target) dataset for LSTM.

For each (stock, bar_i) where i >= 60 and i + 5 < N:
  X[i] = OHLCV[i-60:i] normalized to first-bar close (so inputs are scale-free)
  y[i] = log(close[i+1..i+5] / close[i])    (5 future returns)

Time split: 2024-09-01 (train < cutoff < test).
Stride: take every 3rd bar to reduce dataset size while keeping coverage.

Output: backend/models/lstm_dataset.npz with Xtr/ytr/Xte/yte.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
OUT_PATH = ROOT / "models" / "lstm_dataset.npz"
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

INPUT_LEN = 60
HORIZON = 5
N_CHANNELS = 5  # OHLCV
STRIDE = 3
CUTOFF = pd.Timestamp("2024-09-01")


def main():
    t0 = time.time()
    print(f"Loading {OHLCV_PATH}...", flush=True)
    df = pd.read_parquet(OHLCV_PATH)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    print(f"  {len(df):,} bars × {df['ts_code'].nunique()} stocks", flush=True)

    # Filter to active stocks only via stock_basic
    from sqlalchemy import create_engine, text
    eng = create_engine("postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db")
    active = pd.read_sql(
        text("SELECT ts_code FROM stock_basic WHERE is_active = true"), eng
    )["ts_code"].tolist()
    print(f"  active stocks: {len(active)}", flush=True)
    df = df[df["ts_code"].isin(set(active))]
    print(f"  after filter: {len(df):,} bars", flush=True)

    Xtr_list, ytr_list, Xte_list, yte_list = [], [], [], []
    Xtr_dates, Xte_dates = [], []
    codes = sorted(df["ts_code"].unique())
    for k, ts in enumerate(codes):
        g = df[df["ts_code"] == ts].sort_values("trade_date").reset_index(drop=True)
        n = len(g)
        if n < INPUT_LEN + HORIZON + 1:
            continue
        ohlcv = g[["open", "high", "low", "close", "vol"]].values.astype(np.float32)
        closes = g["close"].values.astype(np.float32)
        dates = g["trade_date"].values
        for i in range(INPUT_LEN, n - HORIZON, STRIDE):
            window = ohlcv[i - INPUT_LEN:i]
            anchor_close = window[-1, 3]
            if anchor_close <= 0 or window[0, 3] <= 0:
                continue
            # Normalize OHLC to anchor close, vol to first-bar vol
            norm = window.copy()
            norm[:, :4] = norm[:, :4] / anchor_close
            v0 = max(window[0, 4], 1.0)
            norm[:, 4] = norm[:, 4] / v0

            future = closes[i:i + HORIZON]
            if (future <= 0).any():
                continue
            tgt = np.log(future / anchor_close).astype(np.float32)

            d_anchor = dates[i - 1]
            if d_anchor < np.datetime64(CUTOFF):
                Xtr_list.append(norm); ytr_list.append(tgt); Xtr_dates.append(d_anchor)
            else:
                Xte_list.append(norm); yte_list.append(tgt); Xte_dates.append(d_anchor)

        if (k + 1) % 200 == 0:
            print(f"  processed {k+1}/{len(codes)} stocks  "
                  f"train={len(Xtr_list)} test={len(Xte_list)}", flush=True)

    Xtr = np.stack(Xtr_list); ytr = np.stack(ytr_list)
    Xte = np.stack(Xte_list); yte = np.stack(yte_list)
    print(f"\nFinal: train {Xtr.shape} test {Xte.shape}", flush=True)
    print(f"  ytr stats: mean={ytr.mean():.4f}, std={ytr.std():.4f}, "
          f"day-1 ret mean={ytr[:,0].mean()*100:.2f}%", flush=True)
    np.savez(OUT_PATH, Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte)
    print(f"Saved → {OUT_PATH}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
