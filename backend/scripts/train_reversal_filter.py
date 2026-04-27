"""Train ML filter on a UNION of reversal-buy candidate events.

Candidate event types (any one fires → bar i is a candidate):
  1. 买卖很准 broad: any of 急买/短买/准备/急卖/短卖 just turned off (1→0)
  2. 动力线 stage_bottom: cross above 0.2
  3. KDJ J cross above 0 (oversold rebound)
  4. RSI(14) cross above 30 (oversold rebound)

For each candidate, label = 1 if forward 10 bars hit +5% before -10%, else 0.
Train 5-seed XGBoost ensemble on 39 features.
Goal: legitimate 10+ candidates/year per stock; ML filter retains ~50% of them.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import asyncio
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES, compute_features, load_candles,
    precompute_dongli, precompute_maimai, load_csf_cache,
)


LABEL_HORIZON = 10
LABEL_TARGET = 0.05
LABEL_STOP = -0.10
MIN_HISTORY = 130


def _shift1(a):
    return np.concatenate(([a[0]], a[:-1]))


def _compute_signals(close, high, low):
    """All 5 maimai lines + 动力线 + KDJ J + RSI (vectorized)."""
    # ---- maimai (TDX-correct LLV) ----
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    maimai_thr = ban_s.rolling(10, min_periods=10).min().values
    hao_thr = ban_s.rolling(10, min_periods=10).max().values
    bb_buy = pd.Series(np.where(np.isnan(maimai_thr), 0.0,
                                (close < maimai_thr).astype(float)))
    jibuy = (bb_buy.rolling(5, min_periods=5).min() > 0).astype(float).values
    duanbuy = (bb_buy.rolling(10, min_periods=10).min() > 0).astype(float).values
    bs_sell = pd.Series(np.where(np.isnan(hao_thr), 0.0,
                                  (close < hao_thr).astype(float)))
    jisell = (bs_sell.rolling(5, min_periods=1).max() > 0).astype(float).values
    duansell = (bs_sell.rolling(10, min_periods=1).max() > 0).astype(float).values

    # 准备 (DMI-5)
    prev_h = _shift1(high); prev_l = _shift1(low); prev_c = _shift1(close)
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    hd = high - prev_h; ld = prev_l - low
    pos_hd = np.where((hd > 0) & (hd > ld), hd, 0.0)
    pos_ld = np.where((ld > 0) & (ld > hd), ld, 0.0)
    td = pd.Series(tr).rolling(5, min_periods=5).sum().values
    dmp = pd.Series(pos_hd).rolling(5, min_periods=5).sum().values
    dmm = pd.Series(pos_ld).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td > 0, dmp * 100 / td, 0)
        fuzhu = np.where(td > 0, dmm * 100 / td, 0)
        denom = fuzhu + shentou
        dx = np.where(denom > 0, np.abs(fuzhu - shentou) / denom * 100, 0)
    dongxiang = pd.Series(dx).rolling(3, min_periods=1).mean().values
    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    # ---- 动力线 ----
    var2 = pd.Series(low).rolling(10, min_periods=1).min().values
    var33 = pd.Series(high).rolling(25, min_periods=1).max().values
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values
    pdl = _shift1(dongli)
    dl_cross_02 = ((pdl <= 0.2) & (dongli > 0.2))

    # ---- KDJ ----
    n = len(close)
    h9 = pd.Series(high).rolling(9, min_periods=1).max().values
    l9 = pd.Series(low).rolling(9, min_periods=1).min().values
    rng9 = h9 - l9
    rsv = np.where(rng9 > 0, (close - l9) / np.where(rng9 > 0, rng9, 1) * 100, 50.0)
    k = np.zeros(n); d = np.zeros(n); k[0] = 50; d[0] = 50
    for i in range(1, n):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    j = 3 * k - 2 * d
    pj = _shift1(j)
    kdj_cross_0 = ((pj <= 0) & (j > 0))

    # ---- RSI ----
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    ag = pd.Series(gain).ewm(alpha=1 / 14, adjust=False).mean().values
    al = pd.Series(loss).ewm(alpha=1 / 14, adjust=False).mean().values
    rs = np.where(al > 0, ag / np.where(al > 0, al, 1), 100.0)
    rsi = 100 - 100 / (1 + rs)
    prsi = _shift1(rsi)
    rsi_cross_30 = ((prsi <= 30) & (rsi > 30))

    return {
        "jibuy": jibuy, "duanbuy": duanbuy, "zhunbei": zhunbei,
        "jisell": jisell, "duansell": duansell,
        "dl_cross_02": dl_cross_02,
        "kdj_cross_0": kdj_cross_0,
        "rsi_cross_30": rsi_cross_30,
    }


def candidate_mask(close, high, low) -> tuple[np.ndarray, dict]:
    """Union of all candidate events. Returns (mask, source_flags)."""
    s = _compute_signals(close, high, low)
    j_off = (_shift1(s["jibuy"]) > 0) & (s["jibuy"] == 0)
    d_off = (_shift1(s["duanbuy"]) > 0) & (s["duanbuy"] == 0)
    z_off = (_shift1(s["zhunbei"]) > 0) & (s["zhunbei"] == 0)
    js_off = (_shift1(s["jisell"]) > 0) & (s["jisell"] == 0)
    ds_off = (_shift1(s["duansell"]) > 0) & (s["duansell"] == 0)
    mm_any = j_off | d_off | z_off | js_off | ds_off
    union = mm_any | s["dl_cross_02"] | s["kdj_cross_0"] | s["rsi_cross_30"]
    sources = {
        "mm_buy_off": j_off | d_off | z_off,
        "mm_sell_off": js_off | ds_off,
        "dl_cross_02": s["dl_cross_02"],
        "kdj_cross_0": s["kdj_cross_0"],
        "rsi_cross_30": s["rsi_cross_30"],
    }
    return union, sources


def label_event(df, i):
    n = len(df)
    if i + LABEL_HORIZON >= n:
        return None
    close = df["close"].values
    entry = close[i]
    for j in range(1, LABEL_HORIZON + 1):
        ret = close[i + j] / entry - 1
        if ret >= LABEL_TARGET:
            return 1
        if ret <= LABEL_STOP:
            return 0
    return 0


def collect_for_stock(symbol: str) -> pd.DataFrame:
    df = asyncio.run(load_candles(symbol))
    if len(df) < MIN_HISTORY:
        return pd.DataFrame()
    df.attrs["ts_code"] = symbol
    df.attrs["_dl_cache"] = precompute_dongli(df)
    df.attrs["_mm_cache"] = precompute_maimai(df)

    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    union, srcs = candidate_mask(close, high, low)
    cand_idx = np.where(union)[0]

    rows = []
    for i in cand_idx:
        if i < MIN_HISTORY or i >= len(df) - 1:
            continue
        label = label_event(df, i)
        if label is None:
            continue
        feat = compute_features(df, i)
        if feat is None:
            continue
        rows.append({
            **feat, "label": label,
            "trade_date": df["trade_date"].iloc[i], "symbol": symbol,
            "src_mm_buy": int(srcs["mm_buy_off"][i]),
            "src_mm_sell": int(srcs["mm_sell_off"][i]),
            "src_dl": int(srcs["dl_cross_02"][i]),
            "src_kdj": int(srcs["kdj_cross_0"][i]),
            "src_rsi": int(srcs["rsi_cross_30"][i]),
        })
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ts_code", default=None,
                   help="single-stock mode (just diagnostic, no train)")
    p.add_argument("--multi-stock", action="store_true")
    args = p.parse_args()

    load_csf_cache()
    t0 = time.time()

    if args.ts_code and not args.multi_stock:
        d = collect_for_stock(args.ts_code)
        if d.empty:
            print("no signals"); return
        years = (pd.to_datetime(d["trade_date"].max()) -
                 pd.to_datetime(d["trade_date"].min())).days / 365.25
        print(f"\n{args.ts_code}: {len(d)} candidates over {years:.1f}y "
              f"= {len(d)/years:.1f}/yr")
        print(f"  raw success rate: {d['label'].mean()*100:.1f}%")
        for src in ["src_mm_buy", "src_mm_sell", "src_dl", "src_kdj", "src_rsi"]:
            cnt = d[src].sum()
            wr = d[d[src] == 1]["label"].mean() * 100 if cnt > 0 else 0
            print(f"  {src}: {cnt} events, {wr:.1f}% raw success")
        return

    from sqlalchemy import distinct, select
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from app.config import settings
    from app.models.schema import DailyCandle, StockBasic

    async def list_codes():
        e = create_async_engine(settings.database_url, echo=False)
        S = async_sessionmaker(e, expire_on_commit=False)
        async with S() as db:
            rows = (await db.execute(
                select(distinct(DailyCandle.ts_code))
                .join(StockBasic, StockBasic.ts_code == DailyCandle.ts_code)
                .where(StockBasic.is_active.is_(True))
            )).scalars().all()
        await e.dispose()
        return list(rows)

    codes = asyncio.run(list_codes())
    print(f"Multi-stock training on {len(codes)} active stocks", flush=True)
    all_df = []
    for k, sym in enumerate(codes):
        try:
            d = collect_for_stock(sym)
            if not d.empty:
                all_df.append(d)
        except Exception as e:
            pass
        if (k + 1) % 100 == 0:
            tot = sum(len(x) for x in all_df)
            print(f"  {k+1}/{len(codes)}  candidates={tot}", flush=True)

    data = pd.concat(all_df, ignore_index=True)
    n = len(data); n_pos = int(data["label"].sum())
    print(f"\nTotal candidates: {n}  success: {n_pos} ({n_pos/n*100:.1f}%)")
    for src in ["src_mm_buy", "src_mm_sell", "src_dl", "src_kdj", "src_rsi"]:
        cnt = int(data[src].sum())
        if cnt > 0:
            wr = data[data[src] == 1]["label"].mean() * 100
            print(f"  {src}: {cnt} ({wr:.1f}% success)")

    # Time split
    cutoff = pd.Timestamp("2024-09-01")
    data["trade_date"] = pd.to_datetime(data["trade_date"])
    is_train = data["trade_date"] < cutoff
    X = data[FEATURE_NAMES].values
    y = data["label"].values
    X_tr, y_tr = X[is_train.values], y[is_train.values]
    X_te, y_te = X[~is_train.values], y[~is_train.values]
    print(f"\nTrain: {len(y_tr)}  Test: {len(y_te)}")

    print("\n--- Training XGBoost (5 seeds) ---")
    preds_list = []
    for seed in [42, 123, 7, 2024, 88]:
        params = {
            "objective": "binary:logistic",
            "max_depth": 4, "eta": 0.05,
            "subsample": 0.85, "colsample_bytree": 0.85,
            "min_child_weight": 5, "eval_metric": "auc",
            "verbosity": 0, "nthread": 8, "seed": seed,
        }
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
        dte = xgb.DMatrix(X_te, feature_names=FEATURE_NAMES)
        m = xgb.train(params, dtr, num_boost_round=200, verbose_eval=False)
        preds_list.append(m.predict(dte))
    preds = np.mean(preds_list, axis=0)

    auc = roc_auc_score(y_te, preds)
    print(f"\nTest AUC: {auc:.4f}")
    print(f"\n{'thr':>6}  {'kept':>6}  {'precision':>9}")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        bp = (preds >= thr).astype(int)
        n_k = int(bp.sum())
        if n_k == 0:
            continue
        prec = precision_score(y_te, bp, zero_division=0)
        print(f"  {thr:.2f}    {n_k:>5}    {prec:.3f}")

    out = ROOT / "models" / "reversal_filter_multi.json"
    dall = xgb.DMatrix(X, label=y, feature_names=FEATURE_NAMES)
    final = xgb.train(
        {"objective": "binary:logistic", "max_depth": 4, "eta": 0.05,
         "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 5,
         "verbosity": 0, "nthread": 8, "seed": 42},
        dall, num_boost_round=200,
    )
    final.save_model(str(out))
    print(f"\nSaved → {out}  ({time.time()-t0:.0f}s total)")


if __name__ == "__main__":
    main()
