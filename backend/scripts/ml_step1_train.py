"""
Step 1: Train XGBoost classifier to predict whether a VolCapit-style entry
signal will succeed (i.e., reach +5% within 10 bars before hitting -5% stop).

Strategy: instead of training only on actual VolCapit triggers (~50 samples),
relax the trigger to gather more training data:
  - Yesterday's drop > 2% (not 3%, looser)
  - Yesterday's volume > MA20 (any multiple, looser)
  - Today's close > yesterday's close (simple bounce)

This generates ~5x more candidate signals across all stocks. Each candidate is
labeled with whether it would have succeeded under VolCapit-Greedy exit logic
(target +5%, stop -5%, time stop 10).

Features: 15 engineered features capturing trend / position / volatility / volume.
Output: trained model saved to models/ml_filter_v1.json
"""
from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GroupKFold
from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle


FEATURE_NAMES = [
    # Generic price/trend/vol/volatility (15)
    "pos20", "pos60", "ret5", "ret20", "ret60",
    "above_ma20", "above_ma60", "above_ma120",
    "atr14_pct", "vol_today_ratio", "vol_yest_ratio",
    "yest_drop_pct", "today_gain_pct", "red_days_10", "ma60_distance_pct",
    # 买卖很准 (Maimai Henzhun) signals (8)
    "mm_below_floor", "mm_below_ceiling",
    "mm_jibuy_active", "mm_duanbuy_active", "mm_jimai_state",
    "mm_zhunbei_active", "mm_shentou", "mm_dongxiang",
    # Cross-sectional rank features (10) — daily percentile rank across all 504 stocks.
    # These give XGBoost the info "this stock is in the top 20% by metric X today",
    # which a single per-stock value cannot convey.
    "csf_ret5_rank", "csf_ret20_rank",
    "csf_vol_ratio_5_20_rank", "csf_pos20_rank",
    "csf_atr_pct_rank", "csf_money_flow_5_rank",
    "csf_qmom_rank",                    # GP-mined "quiet momentum" rank
    "csf_close_to_ma20_rank",
    "csf_max_drawdown_20_rank",
    "csf_high_low_corr_20_rank",
    # 动力线 (Dongli Xian, 6) — TDX stage indicator
    "dl_value",
    "dl_stage_bottom_recent",   # CROSS up 0.2 in last 5 bars (buy signal)
    "dl_stage_watch_recent",    # CROSS up 0.5 in last 5 bars
    "dl_liquidate_recent",      # CROSS down 3.5 in last 5 bars
    "dl_short_sell_recent",     # CROSS down 3.2 in last 5 bars
    "dl_velocity",              # 5-day delta of 动力线
]

LABEL_HORIZON = 7         # tighter window — favor fast clean bounces
LABEL_TARGET = 0.05       # +5% wins
LABEL_STOP = -0.03        # -3% strict stop (was -5%) — penalize drawdown


async def load_candles(symbol: str) -> pd.DataFrame:
    """Load OHLC with backward adjustment so prices match the chart display.
    Attaches ts_code to df.attrs so compute_features can look up CSF features."""
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        stmt = (select(DailyCandle).where(DailyCandle.ts_code == symbol)
                .order_by(DailyCandle.trade_date))
        rows = (await db.execute(stmt)).scalars().all()
    await engine.dispose()
    if not rows:
        return pd.DataFrame()
    latest_adj = float(rows[-1].adj_factor) if rows[-1].adj_factor else 1.0
    df = pd.DataFrame([{
        "trade_date": r.trade_date,
        "open": round(float(r.open) * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "high": round(float(r.high) * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "low":  round(float(r.low)  * (float(r.adj_factor or 1.0) / latest_adj), 4),
        "close":round(float(r.close)* (float(r.adj_factor or 1.0) / latest_adj), 4),
        "vol": float(r.vol or 0),
    } for r in rows])
    df.attrs["ts_code"] = symbol
    return df


_CSF_CACHE: pd.DataFrame | None = None


def load_csf_cache(path: str = "models/cross_sectional_ranks.parquet") -> pd.DataFrame:
    global _CSF_CACHE
    if _CSF_CACHE is not None:
        return _CSF_CACHE
    import os
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df["trade_date"] = pd.to_datetime(df["trade_date"]).dt.date
    df = df.set_index(["ts_code", "trade_date"])
    _CSF_CACHE = df
    return _CSF_CACHE


def compute_features(df: pd.DataFrame, i: int) -> dict | None:
    """Feature vector for entry at bar i. Returns None if not enough history."""
    if i < 120 or i >= len(df) - 1:
        return None

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["vol"].values

    c = close[i]

    # Trend / MAs
    ma20 = close[i - 20:i].mean()
    ma60 = close[i - 60:i].mean()
    ma120 = close[i - 120:i].mean()

    # Position in 20/60-day range
    high20, low20 = high[i - 20:i].max(), low[i - 20:i].min()
    high60, low60 = high[i - 60:i].max(), low[i - 60:i].min()
    pos20 = (c - low20) / (high20 - low20) * 100 if high20 > low20 else 50
    pos60 = (c - low60) / (high60 - low60) * 100 if high60 > low60 else 50

    # Returns
    ret5 = (c / close[i - 5] - 1) * 100
    ret20 = (c / close[i - 20] - 1) * 100
    ret60 = (c / close[i - 60] - 1) * 100

    # Volatility
    atr14 = (high[i - 14:i] - low[i - 14:i]).mean()
    atr14_pct = atr14 / c * 100

    # Volume ratios
    vma20 = vol[i - 20:i].mean()
    vol_today = vol[i] / vma20 if vma20 > 0 else 1
    vol_yest = vol[i - 1] / vma20 if vma20 > 0 else 1

    # Yesterday drop / today gain
    yest_drop = (close[i - 1] / close[i - 2] - 1) * 100
    today_gain = (c / close[i - 1] - 1) * 100

    # Recent weakness
    red_days = int((close[i - 10:i] < close[i - 11:i - 1]).sum())

    # --- 买卖很准 (Maimai Henzhun) features (read from cache when available) ---
    mm_cache = df.attrs.get("_mm_cache")
    if mm_cache is None:
        mm_cache = precompute_maimai(df)
        df.attrs["_mm_cache"] = mm_cache
    mm = {k: float(mm_cache[k][i]) for k in mm_cache}

    # --- GP-mined quiet-momentum factors (per-stock raw — kept for legacy) ---
    gp_cache = df.attrs.get("_gp_cache")
    if gp_cache is None:
        gp_cache = precompute_gp_factors(df)
        df.attrs["_gp_cache"] = gp_cache
    gp = {k: float(gp_cache[k][i]) for k in gp_cache}

    # --- 动力线 (Dongli Xian) features ---
    dl_cache = df.attrs.get("_dl_cache")
    if dl_cache is None:
        dl_cache = precompute_dongli(df)
        df.attrs["_dl_cache"] = dl_cache
    dl = {k: float(dl_cache[k][i]) for k in dl_cache}

    # --- Cross-sectional rank features (looked up from precomputed parquet) ---
    csf = {}
    csf_cache = load_csf_cache()
    if not csf_cache.empty:
        ts_code = df.attrs.get("ts_code")
        if ts_code:
            try:
                row = csf_cache.loc[(ts_code, df["trade_date"].iloc[i])]
                for k in row.index:
                    csf[k] = float(row[k])
            except KeyError:
                pass
    # Fill missing csf with neutral 0.5
    for k in ["csf_ret5_rank", "csf_ret20_rank", "csf_vol_ratio_5_20_rank",
              "csf_pos20_rank", "csf_atr_pct_rank", "csf_money_flow_5_rank",
              "csf_qmom_rank", "csf_close_to_ma20_rank",
              "csf_max_drawdown_20_rank", "csf_high_low_corr_20_rank"]:
        if k not in csf:
            csf[k] = 0.5

    return {
        "pos20": pos20,
        "pos60": pos60,
        "ret5": ret5,
        "ret20": ret20,
        "ret60": ret60,
        "above_ma20": 1.0 if c > ma20 else 0.0,
        "above_ma60": 1.0 if c > ma60 else 0.0,
        "above_ma120": 1.0 if c > ma120 else 0.0,
        "atr14_pct": atr14_pct,
        "vol_today_ratio": vol_today,
        "vol_yest_ratio": vol_yest,
        "yest_drop_pct": yest_drop,
        "today_gain_pct": today_gain,
        "red_days_10": float(red_days),
        "ma60_distance_pct": (c / ma60 - 1) * 100,
        **mm,
        **csf,
        **dl,
    }


def precompute_dongli(df: pd.DataFrame) -> dict:
    """动力线 (Dongli Xian) features — TDX stage indicator from 超级极品底 family.

    Core formula:
        动力线 = EMA((close - LLV(low, 10)) / (HHV(high, 25) - LLV(low, 10)) * 4, 4)
    Range typically 0-4. Signals fire when 动力线 crosses key thresholds:
        阶段底部: cross above 0.2  → reversal-from-bottom buy signal
        阶段关注: cross above 0.5  → momentum building
        清仓:    cross below 3.5  → top forming
        短线卖出: cross below 3.2  → near-term sell
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)
    if n < 30:
        z = np.zeros(n)
        return {f"dl_{k}": z.copy() for k in
                ["value", "stage_bottom_recent", "stage_watch_recent",
                 "liquidate_recent", "short_sell_recent", "velocity"]}

    s_low = pd.Series(low)
    s_high = pd.Series(high)
    s_close = pd.Series(close)

    var2 = s_low.rolling(10, min_periods=1).min().values   # LLV(low, 10)
    var33 = s_high.rolling(25, min_periods=1).max().values  # HHV(high, 25)
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values  # EMA(_, 4)

    # CROSS detections (vectorized)
    prev_d = np.concatenate(([dongli[0]], dongli[:-1]))
    cross_up_02 = ((prev_d <= 0.2) & (dongli > 0.2)).astype(float)   # 阶段底部
    cross_up_05 = ((prev_d <= 0.5) & (dongli > 0.5)).astype(float)   # 阶段关注
    cross_dn_35 = ((prev_d >= 3.5) & (dongli < 3.5)).astype(float)   # 清仓
    cross_dn_32 = ((prev_d >= 3.2) & (dongli < 3.2)).astype(float)   # 短线卖出

    # "Recent N bars" rolling — was the cross in the last 5 bars?
    bottom_recent = pd.Series(cross_up_02).rolling(5, min_periods=1).max().values
    watch_recent = pd.Series(cross_up_05).rolling(5, min_periods=1).max().values
    liquidate_recent = pd.Series(cross_dn_35).rolling(5, min_periods=1).max().values
    short_sell_recent = pd.Series(cross_dn_32).rolling(5, min_periods=1).max().values

    # 5-day velocity (rate of change)
    velocity = np.zeros(n)
    velocity[5:] = dongli[5:] - dongli[:-5]

    return {
        "dl_value": dongli,
        "dl_stage_bottom_recent": bottom_recent,
        "dl_stage_watch_recent": watch_recent,
        "dl_liquidate_recent": liquidate_recent,
        "dl_short_sell_recent": short_sell_recent,
        "dl_velocity": velocity,
    }


def precompute_gp_factors(df: pd.DataFrame) -> dict:
    """GP-mined "quiet momentum" factor cluster.

    Discovered by genetic programming with cross-sectional IC fitness on a
    60-stock panel. Cluster IC ≈ 0.106, IR ≈ 0.535.

    Identifies stocks that had a single-day spike in the last 10 days, are
    currently above their daily VWAP, with small intraday range and modest
    volume — interpreted as institutional accumulation followed by quiet
    consolidation (筹码锁定).

    Returns rolling-z-scored versions for stable cross-stock comparability.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["vol"].values.astype(float)
    n = len(close)
    z = np.zeros(n)
    if n < 30:
        return {f"gp_qmom_{k}": z.copy() for k in ["close", "high", "low"]}

    # VWAP proxy: typical price (H+L+C)/3 — close enough for daily bars
    vwap_proxy = (high + low + close) / 3.0
    safe_vwap = np.where(vwap_proxy > 0, vwap_proxy, 1.0)

    # 1-day return
    ret = np.zeros(n)
    ret[1:] = close[1:] / np.where(close[:-1] > 0, close[:-1], 1) - 1

    # ts_max(ret, 10): max single-day return in the last 10 bars
    ts_max_ret_10 = pd.Series(ret).rolling(10, min_periods=1).max().values

    # range_neg = low - high, always ≤ 0; combined with vol gives a "noise" term
    range_neg = low - high

    cv = close / safe_vwap
    hv = high / safe_vwap
    lv = low / safe_vwap

    f_close = ts_max_ret_10 * cv * range_neg * vol
    f_high = ts_max_ret_10 * hv * range_neg * vol
    f_low = ts_max_ret_10 * lv * range_neg * vol

    def rolling_z(arr):
        s = pd.Series(arr)
        m = s.rolling(60, min_periods=20).mean()
        sd = s.rolling(60, min_periods=20).std().replace(0, np.nan)
        z_arr = ((s - m) / sd).fillna(0).values
        # Clip extreme z-scores to avoid outlier dominance in tree splits
        return np.clip(z_arr, -5.0, 5.0)

    return {
        "gp_qmom_close": rolling_z(f_close),
        "gp_qmom_high": rolling_z(f_high),
        "gp_qmom_low": rolling_z(f_low),
    }


def precompute_maimai(df: pd.DataFrame) -> dict:
    """Vectorized maimai feature computation for the whole stock series.
    Returns dict of arrays keyed by mm_* feature name.
    Much faster than calling _compute_maimai per bar.
    """
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    n = len(close)
    if n < 30:
        zeros = np.zeros(n)
        return {f"mm_{k}": zeros.copy() for k in
                ["below_floor", "below_ceiling", "jibuy_active", "duanbuy_active",
                 "jimai_state", "zhunbei_active", "shentou", "dongxiang"]}

    typ = (close + high + low) / 3.0
    s = pd.Series(typ)
    ban = s.rolling(5, min_periods=5).mean().values         # MA5(typical)
    ban_s = pd.Series(ban)
    hao = ban_s.rolling(10, min_periods=10).max().values    # ceiling
    floor = ban_s.rolling(10, min_periods=10).min().values  # floor

    below_floor = np.where(np.isnan(floor), 0.0,
                           (close < floor).astype(float))
    below_ceiling = np.where(np.isnan(hao), 0.0,
                             (close < hao).astype(float))

    # 急买/短买奇准: any close<floor in last 5/10 bars
    bf = pd.Series(below_floor)
    jibuy = (bf.rolling(5, min_periods=1).max() > 0).astype(float).values
    duanbuy = (bf.rolling(10, min_periods=1).max() > 0).astype(float).values
    bc = pd.Series(below_ceiling)
    jimai = np.where(bc.rolling(5, min_periods=1).max() > 0, 100.0, 50.0)

    # DMI on 5-bar window
    prev_c = np.concatenate([[close[0]], close[:-1]])
    prev_h = np.concatenate([[high[0]], high[:-1]])
    prev_l = np.concatenate([[low[0]], low[:-1]])
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    hd = high - prev_h
    ld = prev_l - low
    pos_hd = np.where((hd > 0) & (hd > ld), hd, 0.0)
    pos_ld = np.where((ld > 0) & (ld > hd), ld, 0.0)
    td = pd.Series(tr).rolling(5, min_periods=5).sum().values
    dmp = pd.Series(pos_hd).rolling(5, min_periods=5).sum().values
    dmm = pd.Series(pos_ld).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td > 0, dmp * 100.0 / td, 0.0)
        fuzhu = np.where(td > 0, dmm * 100.0 / td, 0.0)
        denom = fuzhu + shentou
        dx_raw = np.where(denom > 0, np.abs(fuzhu - shentou) / denom * 100.0, 0.0)
    dongxiang = pd.Series(dx_raw).rolling(3, min_periods=1).mean().values

    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    # NaN → 0 in all output arrays
    def clean(a):
        return np.where(np.isnan(a), 0.0, a)

    return {
        "mm_below_floor": below_floor,
        "mm_below_ceiling": below_ceiling,
        "mm_jibuy_active": jibuy,
        "mm_duanbuy_active": duanbuy,
        "mm_jimai_state": clean(jimai),
        "mm_zhunbei_active": zhunbei,
        "mm_shentou": clean(shentou),
        "mm_dongxiang": clean(dongxiang),
    }


def _compute_maimai(close, high, low, i: int) -> dict:
    """Compute 买卖很准 features at bar i. Needs i >= 30 for stable DMI."""
    # MA5 of typical price, then 10-bar HHV/LLV → ceiling/floor channel
    n = i + 1
    typ = (close + high + low) / 3
    ban_window = 5
    floor_window = 10
    # ban[k] = mean(typ[k-4..k])
    ban = np.full(n, np.nan)
    for k in range(ban_window - 1, n):
        ban[k] = typ[k - ban_window + 1:k + 1].mean()
    if i < floor_window + ban_window:
        return {f"mm_{k}": 0.0 for k in
                ["below_floor", "below_ceiling", "jibuy_active", "duanbuy_active",
                 "jimai_state", "zhunbei_active", "shentou", "dongxiang"]}

    # hao = HHV(ban, 10), maimai = LLV(ban, 10) at bar i and recent bars
    def hhv_at(k):
        s = ban[k - floor_window + 1:k + 1]
        s = s[~np.isnan(s)]
        return s.max() if len(s) else np.nan

    def llv_at(k):
        s = ban[k - floor_window + 1:k + 1]
        s = s[~np.isnan(s)]
        return s.min() if len(s) else np.nan

    hao_i = hhv_at(i)
    floor_i = llv_at(i)
    below_floor = 1.0 if not np.isnan(floor_i) and close[i] < floor_i else 0.0
    below_ceiling = 1.0 if not np.isnan(hao_i) and close[i] < hao_i else 0.0

    # 急买奇准 / 短买奇准: any close<floor in last 5 / 10 bars
    jibuy = 0.0
    duanbuy = 0.0
    for k in range(max(0, i - 4), i + 1):
        f = llv_at(k)
        if not np.isnan(f) and close[k] < f:
            jibuy = 1.0
            break
    for k in range(max(0, i - 9), i + 1):
        f = llv_at(k)
        if not np.isnan(f) and close[k] < f:
            duanbuy = 1.0
            break

    # 急卖奇准: 100 if any close<ceiling in last 5
    jimai = 50.0
    for k in range(max(0, i - 4), i + 1):
        h = hhv_at(k)
        if not np.isnan(h) and close[k] < h:
            jimai = 100.0
            break

    # DMI / 准备现金
    if i < 6:
        return {
            "mm_below_floor": below_floor, "mm_below_ceiling": below_ceiling,
            "mm_jibuy_active": jibuy, "mm_duanbuy_active": duanbuy,
            "mm_jimai_state": jimai, "mm_zhunbei_active": 0.0,
            "mm_shentou": 0.0, "mm_dongxiang": 0.0,
        }
    tr = np.zeros(n)
    hd = np.zeros(n)
    ld = np.zeros(n)
    for k in range(1, n):
        prev_c = close[k - 1]
        tr[k] = max(high[k] - low[k], abs(high[k] - prev_c), abs(low[k] - prev_c))
        hd[k] = high[k] - high[k - 1]
        ld[k] = low[k - 1] - low[k]
    win = 5
    td = tr[i - win + 1:i + 1].sum()
    dmp = sum(hd[k] for k in range(i - win + 1, i + 1) if hd[k] > 0 and hd[k] > ld[k])
    dmm = sum(ld[k] for k in range(i - win + 1, i + 1) if ld[k] > 0 and ld[k] > hd[k])
    if td <= 0:
        return {
            "mm_below_floor": below_floor, "mm_below_ceiling": below_ceiling,
            "mm_jibuy_active": jibuy, "mm_duanbuy_active": duanbuy,
            "mm_jimai_state": jimai, "mm_zhunbei_active": 0.0,
            "mm_shentou": 0.0, "mm_dongxiang": 0.0,
        }
    shentou = dmp * 100.0 / td
    fuzhu = dmm * 100.0 / td
    denom = fuzhu + shentou
    dongxiang_raw = abs(fuzhu - shentou) / denom * 100 if denom > 0 else 0
    # 3-bar MA of dongxiang_raw
    raws = []
    for k in range(max(i - 2, 0), i + 1):
        td_k = tr[max(0, k - win + 1):k + 1].sum()
        if td_k <= 0:
            raws.append(0.0); continue
        dmp_k = sum(hd[m] for m in range(max(0, k - win + 1), k + 1)
                    if hd[m] > 0 and hd[m] > ld[m])
        dmm_k = sum(ld[m] for m in range(max(0, k - win + 1), k + 1)
                    if ld[m] > 0 and ld[m] > hd[m])
        st_k = dmp_k * 100 / td_k
        fz_k = dmm_k * 100 / td_k
        d = fz_k + st_k
        raws.append(abs(fz_k - st_k) / d * 100 if d > 0 else 0)
    dongxiang = float(np.mean(raws))
    zhunbei = 1.0 if dongxiang > 88 and shentou < 5.8 else 0.0

    return {
        "mm_below_floor": below_floor,
        "mm_below_ceiling": below_ceiling,
        "mm_jibuy_active": jibuy,
        "mm_duanbuy_active": duanbuy,
        "mm_jimai_state": jimai,
        "mm_zhunbei_active": zhunbei,
        "mm_shentou": float(shentou),
        "mm_dongxiang": dongxiang,
    }


def passes_loose_trigger(df: pd.DataFrame, i: int) -> bool:
    """Candidate filter — OR of three patterns:
      1. Yesterday dropped >2% with decent volume + today up (short-term reversal setup)
      2. 买卖很准 准备现金 = 1 (DMI capitulation)
      3. mm_below_floor: yesterday=1, today=0 (recovered from broken floor)
    """
    if i < 22:
        return False
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    vol = df["vol"].values

    # Pattern 1: short-term oversold reversal
    yest_drop = close[i - 1] / close[i - 2] - 1
    vma20 = vol[i - 21:i - 1].mean()
    cond1 = yest_drop < -0.02
    cond2 = vol[i - 1] > vma20 * 0.9 if vma20 > 0 else False
    cond3 = close[i] > close[i - 1]
    if cond1 and cond2 and cond3:
        return True

    # Pattern 2 + 3: 买卖很准 signals
    mm = df.attrs.get("_mm_cache")
    if mm is None:
        mm = precompute_maimai(df)
        df.attrs["_mm_cache"] = mm
    if mm["mm_zhunbei_active"][i] == 1.0:
        return True
    if i >= 1:
        if (mm["mm_below_floor"][i - 1] == 1.0
                and mm["mm_below_floor"][i] == 0.0
                and close[i] > close[i - 1]):
            return True

    # Pattern 4: 动力线 阶段底部 (CROSS up 0.2 in last 5 bars)
    dl = df.attrs.get("_dl_cache")
    if dl is None:
        dl = precompute_dongli(df)
        df.attrs["_dl_cache"] = dl
    if dl["dl_stage_bottom_recent"][i] == 1.0 and close[i] >= close[i - 1]:
        return True

    return False


def compute_label(df: pd.DataFrame, i: int) -> int | None:
    """Walk forward LABEL_HORIZON bars; 1 if reached LABEL_TARGET first, 0 if hit LABEL_STOP."""
    if i + LABEL_HORIZON >= len(df):
        return None

    close = df["close"].values
    entry = close[i]
    for j in range(1, LABEL_HORIZON + 1):
        ret = close[i + j] / entry - 1
        if ret >= LABEL_TARGET:
            return 1
        if ret <= LABEL_STOP:
            return 0
    # Neither target nor stop hit within horizon → label as 0
    # (we only want clean fast bounces, not "drift up slightly")
    return 0


async def build_dataset() -> pd.DataFrame:
    """Build training dataset across all stocks in DB."""
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        from sqlalchemy import distinct
        codes = (await db.execute(
            select(distinct(DailyCandle.ts_code))
        )).scalars().all()
    await engine.dispose()

    rows: list[dict] = []
    for symbol in codes:
        df = await load_candles(symbol)
        if len(df) < 130:
            continue
        n_signals = 0
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            feat = compute_features(df, i)
            if feat is None:
                continue
            label = compute_label(df, i)
            if label is None:
                continue
            row = {**feat, "label": label, "symbol": symbol,
                   "trade_date": df["trade_date"].iloc[i]}
            rows.append(row)
            n_signals += 1
        print(f"  {symbol}: {len(df)} bars → {n_signals} candidates")

    return pd.DataFrame(rows)


def train_and_evaluate(data: pd.DataFrame) -> xgb.Booster:
    print(f"\nDataset: {len(data)} samples ({data['label'].sum()} positive, "
          f"baseline {data['label'].mean()*100:.1f}%)")

    X = data[FEATURE_NAMES].values
    y = data["label"].values
    groups = data["symbol"].values

    # GroupKFold by symbol — train on some stocks, validate on others
    n_groups = len(set(groups))
    n_splits = min(n_groups, 4)
    gkf = GroupKFold(n_splits=n_splits)

    print(f"\n--- {n_splits}-fold cross validation (split by stock) ---")
    fold_metrics = []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
        train_groups = sorted(set(groups[tr_idx]))
        val_groups = sorted(set(groups[va_idx]))
        dtrain = xgb.DMatrix(X[tr_idx], label=y[tr_idx])
        dval = xgb.DMatrix(X[va_idx], label=y[va_idx])
        params = {
            "objective": "binary:logistic",
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "min_child_weight": 5,
            "eval_metric": "auc",
            "verbosity": 0,
        }
        model = xgb.train(params, dtrain, num_boost_round=200,
                          evals=[(dval, "val")], early_stopping_rounds=20,
                          verbose_eval=False)
        pred = model.predict(dval)
        pred_bin = (pred >= 0.5).astype(int)
        m = {
            "fold": fold,
            "val_groups": val_groups,
            "n_val": len(va_idx),
            "auc": roc_auc_score(y[va_idx], pred),
            "acc": accuracy_score(y[va_idx], pred_bin),
            "prec": precision_score(y[va_idx], pred_bin, zero_division=0),
            "recall": recall_score(y[va_idx], pred_bin, zero_division=0),
            "high_conf_prec": (
                precision_score(y[va_idx], (pred >= 0.65).astype(int), zero_division=0)
                if (pred >= 0.65).sum() >= 5 else None
            ),
            "high_conf_n": int((pred >= 0.65).sum()),
        }
        fold_metrics.append(m)
        hc_str = "—" if m["high_conf_prec"] is None else f"{m['high_conf_prec']:.3f}"
        print(f"Fold {fold}: val={val_groups} n={m['n_val']:4d} "
              f"AUC={m['auc']:.3f} acc={m['acc']:.3f} "
              f"prec@0.5={m['prec']:.3f} recall={m['recall']:.3f} "
              f"prec@0.65 (n={m['high_conf_n']})={hc_str}")

    avg_auc = np.mean([m["auc"] for m in fold_metrics])
    print(f"\nMean AUC: {avg_auc:.3f}")

    # Final: train on all data
    print("\n--- Training final model on all data ---")
    dall = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    final_model = xgb.train(
        {**{
            "objective": "binary:logistic", "max_depth": 4, "eta": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 5, "verbosity": 0,
        }},
        dall, num_boost_round=150,
    )

    # Feature importance
    imp = final_model.get_score(importance_type="gain")
    print("\nFeature importance (gain):")
    for k, v in sorted(imp.items(), key=lambda x: -x[1]):
        print(f"  {k:<22}{v:.1f}")

    return final_model, fold_metrics


async def main() -> None:
    print("Building dataset...")
    data = await build_dataset()
    if len(data) == 0:
        print("No data!"); return

    model, metrics = train_and_evaluate(data)

    # Save artifacts
    out_dir = "models"
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "ml_filter_v1.json")
    model.save_model(model_path)
    print(f"\nModel saved to {model_path}")

    # Save data for inspection
    data.to_csv(os.path.join(out_dir, "ml_step1_data.csv"), index=False)
    with open(os.path.join(out_dir, "ml_step1_metrics.json"), "w") as f:
        json.dump([{k: v for k, v in m.items() if k != "val_groups"} | {"val_groups": m["val_groups"]}
                   for m in metrics], f, indent=2, default=str)
    print(f"Data + metrics saved to {out_dir}/")


if __name__ == "__main__":
    asyncio.run(main())
