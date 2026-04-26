"""Step 1: Cache OHLCV for 1004 stocks (2019-01 → 2026-04, backward-adjusted).

Output: backend/cache/ohlcv.parquet
Schema: ts_code, trade_date, open, high, low, close, vol, amount
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text


DB_URL = "postgresql+psycopg2://stock:__REDACTED_DB_PASSWORD__@localhost:5432/stock_db"
START_DATE = "2019-01-01"
END_DATE = "2026-12-31"
OUT_PATH = Path(__file__).parent.parent / "cache" / "ohlcv.parquet"


def main() -> None:
    eng = create_engine(DB_URL)
    t0 = time.time()
    print("Loading raw candles from DB...", flush=True)

    sql = text("""
        SELECT ts_code, trade_date, open, high, low, close, vol, amount, adj_factor
        FROM daily_candle
        WHERE trade_date >= :start AND trade_date <= :end
        ORDER BY ts_code, trade_date
    """)
    df = pd.read_sql(sql, eng, params={"start": START_DATE, "end": END_DATE})
    print(f"  {len(df):,} rows loaded in {time.time()-t0:.1f}s", flush=True)

    df["adj_factor"] = df["adj_factor"].fillna(1.0).astype(float)
    for c in ("open", "high", "low", "close", "vol", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(float)

    print("Backward-adjusting prices per stock...", flush=True)
    latest = df.groupby("ts_code")["adj_factor"].transform("last")
    factor = df["adj_factor"] / latest
    for c in ("open", "high", "low", "close"):
        df[c] = (df[c] * factor).round(4)
    df = df.drop(columns=["adj_factor"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(df):,} rows × {df['ts_code'].nunique()} stocks to {OUT_PATH}", flush=True)
    print(f"Date range: {df['trade_date'].min()} → {df['trade_date'].max()}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
