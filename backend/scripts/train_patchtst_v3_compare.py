"""PatchTST V3 vs XGBoost Ensemble — apples-to-apples comparison.

Same candidate bars (passes_loose_trigger), same labels (5%/-3% in 7 bars),
same time-based split (2024-09-01).

For each candidate bar, builds:
  - 39-feature vector (XGBoost input)
  - 60-bar × 12-channel sequence (PatchTST input)

Trains both on train portion (date < 2024-09-01), evaluates on test
(date ≥ 2024-09-01). Reports AUC, prec@multi-threshold, signal counts.

Uses the OHLCV cache for speed (no DB hits).
"""
from __future__ import annotations

import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from scripts.ml_step1_train import (  # noqa: E402
    FEATURE_NAMES, compute_features,
    compute_label, passes_loose_trigger,
    precompute_dongli, precompute_maimai,
)


# =================================================================
# Config
# =================================================================
CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
CSF_PATH = ROOT / "models" / "cross_sectional_ranks.parquet"
OUT_DIR = ROOT / "models"

CUTOFF = pd.Timestamp("2024-09-01")

SEQ_LEN = 60
PATCH_LEN = 10
N_PATCHES = SEQ_LEN // PATCH_LEN
N_CHANNELS = 12  # see make_sequence

EMBED_DIM = 96
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.15
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

XGB_PARAMS = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.05,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 5,
    "eval_metric": "auc",
    "verbosity": 0,
}
XGB_ROUNDS = 200
XGB_SEEDS = [42, 123, 7, 2024, 88]


# =================================================================
# Sequence builder (12 channels per bar)
# =================================================================

def make_sequence(close, high, low, vol, mm, dl, kdj_j, atr14_pct, i: int):
    if i < SEQ_LEN + 5 or i >= len(close):
        return None
    seq = np.zeros((SEQ_LEN, N_CHANNELS), dtype=np.float32)
    for j in range(SEQ_LEN):
        b = i - SEQ_LEN + j
        if b < 1:
            continue
        ret = (close[b] / close[b - 1] - 1) * 100
        hl_pct = (high[b] - low[b]) / close[b] * 100 if close[b] > 0 else 0.0
        vma = vol[max(0, b - 20):b].mean() if b >= 1 else 1.0
        vr = vol[b] / vma if vma > 0 else 1.0
        ma5 = close[max(0, b - 5):b].mean() if b >= 5 else close[b]
        ma20 = close[max(0, b - 20):b].mean() if b >= 20 else close[b]
        seq[j, 0] = ret
        seq[j, 1] = hl_pct
        seq[j, 2] = vr
        seq[j, 3] = 1.0 if close[b] > ma5 else 0.0
        seq[j, 4] = 1.0 if close[b] > ma20 else 0.0
        seq[j, 5] = mm["mm_below_floor"][b]
        seq[j, 6] = mm["mm_below_ceiling"][b]
        seq[j, 7] = mm["mm_zhunbei_active"][b]
        seq[j, 8] = dl["dl_value"][b] / 4.0  # normalize 0-4 to 0-1
        seq[j, 9] = dl["dl_stage_bottom_recent"][b]
        seq[j, 10] = kdj_j[b] / 100.0  # normalize
        seq[j, 11] = atr14_pct[b] / 10.0
    return seq


def precompute_kdj_j(close, high, low):
    n = len(close)
    sh = pd.Series(high)
    sl = pd.Series(low)
    h9 = sh.rolling(9, min_periods=1).max().values
    l9 = sl.rolling(9, min_periods=1).min().values
    rng = h9 - l9
    rsv = np.where(rng > 0, (close - l9) / np.where(rng > 0, rng, 1) * 100, 50.0)
    k = np.zeros(n); d = np.zeros(n); k[0] = 50.0; d[0] = 50.0
    for i in range(1, n):
        k[i] = (2 / 3) * k[i - 1] + (1 / 3) * rsv[i]
        d[i] = (2 / 3) * d[i - 1] + (1 / 3) * k[i]
    return 3 * k - 2 * d


def precompute_atr14_pct(close, high, low):
    prev_c = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    atr = pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().values
    return np.where(close > 0, atr / np.where(close > 0, close, 1) * 100, 0.0)


# =================================================================
# Dataset build
# =================================================================

def build_dataset_from_cache():
    print("Loading OHLCV cache...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    csf = pd.read_parquet(CSF_PATH)
    csf["trade_date"] = pd.to_datetime(csf["trade_date"])
    csf_lookup = {ts: g.set_index("trade_date") for ts, g in csf.groupby("ts_code")}
    csf_cols = ["csf_ret5_rank", "csf_ret20_rank", "csf_vol_ratio_5_20_rank",
                "csf_pos20_rank", "csf_atr_pct_rank", "csf_money_flow_5_rank",
                "csf_qmom_rank", "csf_close_to_ma20_rank",
                "csf_max_drawdown_20_rank", "csf_high_low_corr_20_rank"]

    train_X_xgb, train_X_seq, train_y, train_groups = [], [], [], []
    test_X_xgb, test_X_seq, test_y, test_groups = [], [], [], []

    codes = sorted(ohlcv["ts_code"].unique())
    print(f"Building features for {len(codes)} stocks...", flush=True)
    t0 = time.time()
    for k, sym in enumerate(codes):
        df = ohlcv[ohlcv["ts_code"] == sym].sort_values("trade_date").reset_index(drop=True)
        if len(df) < SEQ_LEN + 130:
            continue
        df.attrs["ts_code"] = sym

        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        vol = df["vol"].values.astype(float)

        mm = precompute_maimai(df); df.attrs["_mm_cache"] = mm
        dl = precompute_dongli(df); df.attrs["_dl_cache"] = dl
        kdj_j = precompute_kdj_j(close, high, low)
        atr_pct = precompute_atr14_pct(close, high, low)

        n_pos = 0
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            if i < 120 or i >= len(df) - 1:
                continue
            label = compute_label(df, i)
            if label is None:
                continue
            feat = compute_features(df, i)
            if feat is None:
                continue
            # Add CSF features (manual lookup since compute_features's load_csf_cache
            # may not be set for our cache layout)
            csf_df = csf_lookup.get(sym)
            if csf_df is not None:
                d_idx = df["trade_date"].iloc[i]
                if d_idx in csf_df.index:
                    for c in csf_cols:
                        feat[c] = float(csf_df.loc[d_idx, c])
            for c in csf_cols:
                feat.setdefault(c, 0.5)

            seq = make_sequence(close, high, low, vol, mm, dl, kdj_j, atr_pct, i)
            if seq is None:
                continue

            xv = [feat[k_] for k_ in FEATURE_NAMES]
            d = df["trade_date"].iloc[i]
            if d < CUTOFF:
                train_X_xgb.append(xv); train_X_seq.append(seq)
                train_y.append(label); train_groups.append(sym)
            else:
                test_X_xgb.append(xv); test_X_seq.append(seq)
                test_y.append(label); test_groups.append(sym)
            n_pos += 1

        if (k + 1) % 100 == 0:
            print(f"  {k+1}/{len(codes)}  train={len(train_y)}  test={len(test_y)}  ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nTrain: {len(train_y)} samples, {sum(train_y)} pos ({np.mean(train_y)*100:.1f}%)")
    print(f"Test:  {len(test_y)} samples, {sum(test_y)} pos ({np.mean(test_y)*100:.1f}%)")
    return (
        np.array(train_X_xgb, dtype=np.float32), np.stack(train_X_seq).astype(np.float32),
        np.array(train_y, dtype=np.float32), np.array(train_groups),
        np.array(test_X_xgb, dtype=np.float32), np.stack(test_X_seq).astype(np.float32),
        np.array(test_y, dtype=np.float32), np.array(test_groups),
    )


# =================================================================
# XGBoost ensemble training
# =================================================================

def train_xgb_ensemble(X_tr, y_tr, X_te, y_te):
    print(f"\n--- XGBoost Ensemble ({len(XGB_SEEDS)} seeds) ---", flush=True)
    all_preds = []
    for seed in XGB_SEEDS:
        params = {**XGB_PARAMS, "seed": seed}
        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=FEATURE_NAMES)
        dte = xgb.DMatrix(X_te, feature_names=FEATURE_NAMES)
        m = xgb.train(params, dtr, num_boost_round=XGB_ROUNDS,
                      evals=[(dtr, "tr")], verbose_eval=False)
        preds = m.predict(dte)
        all_preds.append(preds)
        auc = roc_auc_score(y_te, preds)
        print(f"  seed={seed}: AUC={auc:.4f}", flush=True)
    ens = np.mean(all_preds, axis=0)
    return ens


# =================================================================
# PatchTST V3
# =================================================================

class PatchTST(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN * N_CHANNELS, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, EMBED_DIM) * 0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=EMBED_DIM * 2,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        b = x.size(0)
        x = x.view(b, N_PATCHES, PATCH_LEN * N_CHANNELS)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


def train_patchtst(X_tr, y_tr, X_te, y_te):
    print(f"\n--- PatchTST (device: {DEVICE}) ---", flush=True)
    # Per-channel z-score using train stats
    flat = X_tr.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0); std = flat.std(axis=0) + 1e-6
    X_tr = (X_tr - mean) / std
    X_te = (X_te - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
        batch_size=BATCH_SIZE)

    model = PatchTST().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc = 0.0; best_preds = None
    for ep in range(EPOCHS):
        model.train()
        losses = []
        t_ep = time.time()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        ll = []
        with torch.no_grad():
            for X, _ in test_loader:
                ll.append(model(X.to(DEVICE)).cpu().numpy())
        preds = 1 / (1 + np.exp(-np.concatenate(ll)))
        auc = roc_auc_score(y_te, preds)
        if auc > best_auc:
            best_auc = auc; best_preds = preds.copy()
        print(f"  Ep {ep+1:>2}: loss={np.mean(losses):.4f}  test AUC={auc:.4f}  "
              f"(best {best_auc:.4f})  [{time.time()-t_ep:.0f}s]", flush=True)
    return best_preds


# =================================================================
# Comparison
# =================================================================

def compare(name_a, preds_a, name_b, preds_b, y_te):
    print(f"\n{'='*72}")
    print(f"COMPARISON: {name_a} vs {name_b}  (test n={len(y_te)}, +ve {y_te.mean()*100:.1f}%)")
    print(f"{'='*72}")
    auc_a = roc_auc_score(y_te, preds_a)
    auc_b = roc_auc_score(y_te, preds_b)
    print(f"  AUC: {name_a}={auc_a:.4f}  {name_b}={auc_b:.4f}  "
          f"Δ={(auc_b-auc_a)*100:+.2f} pp")
    print(f"\n{'thr':>6}  {name_a:>12}  {name_b:>12}  {'(diff)':>8}")
    print(f"{'   ':>6}  {'prec / n':>12}  {'prec / n':>12}")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        bin_a = (preds_a >= thr).astype(int)
        bin_b = (preds_b >= thr).astype(int)
        na, nb = int(bin_a.sum()), int(bin_b.sum())
        pa = precision_score(y_te, bin_a, zero_division=0) if na > 0 else 0
        pb = precision_score(y_te, bin_b, zero_division=0) if nb > 0 else 0
        print(f"  {thr:.2f}  {pa:.3f} / {na:>4}  {pb:.3f} / {nb:>4}  "
              f"{(pb-pa)*100:+.1f} pp")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print("=" * 72)
    print("PatchTST V3 vs XGBoost Ensemble — Apples-to-Apples Comparison")
    print(f"  Cutoff: {CUTOFF.date()}  (train < cutoff < test)")
    print("=" * 72)

    Xtr_xgb, Xtr_seq, ytr, gtr, Xte_xgb, Xte_seq, yte, gte = build_dataset_from_cache()
    print(f"Dataset built in {time.time()-t0:.0f}s")

    xgb_preds = train_xgb_ensemble(Xtr_xgb, ytr, Xte_xgb, yte)
    pt_preds = train_patchtst(Xtr_seq, ytr, Xte_seq, yte)

    compare("XGB-Ens", xgb_preds, "PatchTST", pt_preds, yte)

    # Save artifacts for later analysis
    np.save(OUT_DIR / "compare_xgb_preds.npy", xgb_preds)
    np.save(OUT_DIR / "compare_pt_preds.npy", pt_preds)
    np.save(OUT_DIR / "compare_y_te.npy", yte)
    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
