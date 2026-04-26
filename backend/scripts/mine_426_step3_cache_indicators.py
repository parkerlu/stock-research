"""Step 3: Cache all TDX/momentum indicators per (ts_code, trade_date).

Output: backend/cache/indicators.parquet
Columns: ts_code, trade_date, + many indicator columns:
  动力线: dl_value, dl_stage_bottom, dl_stage_watch, dl_liquidate, dl_short_sell, dl_velocity
  买卖很准: mm_jibuy, mm_duanbuy, mm_zhunbei, mm_shentou, mm_dongxiang, mm_jimai
  KDJ: k, d, j  (daily) + weekly k_w, d_w
  DMI: pdi, mdi, adx
  RSI: rsi14
  MACD: macd_diff, macd_dea, macd_hist
  Bollinger: boll_upper, boll_lower, boll_width
  Channel: high20, low20, high50, low50, donch_pos20
  Volatility: atr14, atr14_pct, vol_60
  Trend: sma10, sma20, sma30, sma60
  ATR: atr14
  Volume: vma20, vol_spike (vol/vma20)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import precompute_dongli, precompute_maimai  # noqa: E402


CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
OUT_PATH = CACHE_DIR / "indicators.parquet"


def precompute_kdj(df: pd.DataFrame, n: int = 9) -> dict:
    """Standard KDJ (n=9, 3, 3)."""
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    s_high = pd.Series(high).rolling(n, min_periods=1).max().values
    s_low = pd.Series(low).rolling(n, min_periods=1).min().values
    rng = s_high - s_low
    rsv = np.where(rng > 0, (close - s_low) / np.where(rng > 0, rng, 1) * 100, 50.0)
    # K = EMA(RSV, 3) using α=1/3 → K_t = (2/3)*K_{t-1} + (1/3)*RSV_t
    k = np.zeros(len(close))
    d = np.zeros(len(close))
    k[0] = 50.0
    d[0] = 50.0
    for i in range(1, len(close)):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    j = 3 * k - 2 * d
    return {"kdj_k": k, "kdj_d": d, "kdj_j": j}


def precompute_dmi(df: pd.DataFrame, n: int = 14) -> dict:
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)

    prev_c = np.concatenate([[close[0]], close[:-1]])
    prev_h = np.concatenate([[high[0]], high[:-1]])
    prev_l = np.concatenate([[low[0]], low[:-1]])

    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    hd = high - prev_h
    ld = prev_l - low
    pos_dm = np.where((hd > 0) & (hd > ld), hd, 0.0)
    neg_dm = np.where((ld > 0) & (ld > hd), ld, 0.0)

    tr_n = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    pos_dm_n = pd.Series(pos_dm).ewm(alpha=1 / n, adjust=False).mean().values
    neg_dm_n = pd.Series(neg_dm).ewm(alpha=1 / n, adjust=False).mean().values
    pdi = np.where(tr_n > 0, 100 * pos_dm_n / np.where(tr_n > 0, tr_n, 1), 0)
    mdi = np.where(tr_n > 0, 100 * neg_dm_n / np.where(tr_n > 0, tr_n, 1), 0)
    denom = pdi + mdi
    dx = np.where(denom > 0, 100 * np.abs(pdi - mdi) / np.where(denom > 0, denom, 1), 0)
    adx = pd.Series(dx).ewm(alpha=1 / n, adjust=False).mean().values
    return {"dmi_pdi": pdi, "dmi_mdi": mdi, "dmi_adx": adx}


def precompute_rsi(df: pd.DataFrame, n: int = 14) -> np.ndarray:
    close = df["close"].values.astype(float)
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1 / n, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1 / n, adjust=False).mean().values
    rs = np.where(avg_loss > 0, avg_gain / np.where(avg_loss > 0, avg_loss, 1), 100.0)
    rsi = 100 - 100 / (1 + rs)
    return rsi


def precompute_macd(df: pd.DataFrame) -> dict:
    close = pd.Series(df["close"].astype(float))
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    diff = (ema12 - ema26).values
    dea = pd.Series(diff).ewm(span=9, adjust=False).mean().values
    hist = diff - dea
    return {"macd_diff": diff, "macd_dea": dea, "macd_hist": hist}


def precompute_bollinger(df: pd.DataFrame, n: int = 20, k: float = 2.0) -> dict:
    close = pd.Series(df["close"].astype(float))
    mid = close.rolling(n, min_periods=1).mean().values
    std = close.rolling(n, min_periods=1).std().fillna(0).values
    upper = mid + k * std
    lower = mid - k * std
    width = np.where(mid > 0, (upper - lower) / np.where(mid > 0, mid, 1), 0.0)
    return {"boll_mid": mid, "boll_upper": upper, "boll_lower": lower, "boll_width": width}


def precompute_donch(df: pd.DataFrame) -> dict:
    high = pd.Series(df["high"].astype(float))
    low = pd.Series(df["low"].astype(float))
    close = df["close"].values.astype(float)

    h20 = high.rolling(20, min_periods=1).max().shift(1).fillna(method="bfill").values
    l20 = low.rolling(20, min_periods=1).min().shift(1).fillna(method="bfill").values
    h50 = high.rolling(50, min_periods=1).max().shift(1).fillna(method="bfill").values
    l50 = low.rolling(50, min_periods=1).min().shift(1).fillna(method="bfill").values

    return {"high20": h20, "low20": l20, "high50": h50, "low50": l50}


def precompute_atr(df: pd.DataFrame, n: int = 14) -> dict:
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    close = df["close"].values.astype(float)
    prev_c = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr = pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().values
    atr_pct = np.where(close > 0, atr / np.where(close > 0, close, 1) * 100, 0.0)
    return {"atr14": atr, "atr14_pct": atr_pct}


def precompute_smas(df: pd.DataFrame) -> dict:
    close = pd.Series(df["close"].astype(float))
    return {
        "sma10": close.rolling(10, min_periods=1).mean().values,
        "sma20": close.rolling(20, min_periods=1).mean().values,
        "sma30": close.rolling(30, min_periods=1).mean().values,
        "sma60": close.rolling(60, min_periods=1).mean().values,
    }


def precompute_volume(df: pd.DataFrame) -> dict:
    vol = pd.Series(df["vol"].astype(float))
    vma20 = vol.rolling(20, min_periods=1).mean().values
    spike = np.where(vma20 > 0, vol.values / np.where(vma20 > 0, vma20, 1), 1.0)
    return {"vma20": vma20, "vol_spike": spike}


def build_for_stock(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["ts_code"] = df["ts_code"].values
    out["trade_date"] = df["trade_date"].values

    feats: dict = {}
    feats.update(precompute_dongli(df))
    feats.update(precompute_maimai(df))
    feats.update(precompute_kdj(df))
    feats.update(precompute_dmi(df))
    feats["rsi14"] = precompute_rsi(df)
    feats.update(precompute_macd(df))
    feats.update(precompute_bollinger(df))
    feats.update(precompute_donch(df))
    feats.update(precompute_atr(df))
    feats.update(precompute_smas(df))
    feats.update(precompute_volume(df))

    for k, v in feats.items():
        out[k] = v

    return out


def main() -> None:
    t0 = time.time()
    print("Loading OHLCV...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    print(f"  {len(ohlcv):,} bars × {ohlcv['ts_code'].nunique()} stocks", flush=True)

    print("Computing indicators per stock...", flush=True)
    parts = []
    codes = sorted(ohlcv["ts_code"].unique())
    for k, ts in enumerate(codes):
        df_st = ohlcv[ohlcv["ts_code"] == ts].sort_values("trade_date").reset_index(drop=True)
        if len(df_st) < 60:
            continue
        parts.append(build_for_stock(df_st))
        if (k + 1) % 200 == 0:
            print(f"  done {k+1}/{len(codes)}", flush=True)

    print("Concatenating...", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"Saved {len(out):,} rows × {len(out.columns)} cols to {OUT_PATH}", flush=True)
    print(f"Total time: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
