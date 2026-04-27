"""Build PatchTST score cache for all stocks in cache/ohlcv.parquet.

Mirrors mine_426_step2_cache_scores.py but uses PatchTST instead of XGBoost.
Output schema matches: ts_code, trade_date, ml_score (so downstream mining
can swap caches transparently).

Run separately from XGB: this script imports torch only (no xgboost), so
OpenMP runtimes don't collide.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


# ----- Architecture (must match compare_pt_only.py) -----
SEQ_LEN = 60
PATCH_LEN = 10
N_PATCHES = SEQ_LEN // PATCH_LEN
N_CHANNELS = 12
EMBED_DIM = 96
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.15
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 1024

CACHE_DIR = ROOT / "cache"
OHLCV_PATH = CACHE_DIR / "ohlcv.parquet"
OUT_PATH = CACHE_DIR / "pt_scores.parquet"
MODELS_DIR = ROOT / "models"
MODEL_PATH = MODELS_DIR / "patchtst_v3.pt"
NORM_PATH = MODELS_DIR / "patchtst_v3_norm.npy"


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


# ----- Same precompute helpers used during training -----

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


def build_sequences_for_stock(df: pd.DataFrame):
    """Return (sequences, valid_indices) — vectorized 60×12 windows for every
    bar where i >= SEQ_LEN + 5. Each sequence is float32."""
    n = len(df)
    if n < SEQ_LEN + 5:
        return None, None

    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["vol"].astype(float).values

    s_close = pd.Series(close)
    s_vol = pd.Series(vol)

    # Channel pre-computes (per-bar values, length n)
    ret = np.zeros(n)
    ret[1:] = (close[1:] / np.where(close[:-1] > 0, close[:-1], 1) - 1) * 100
    hl_pct = np.where(close > 0, (high - low) / np.where(close > 0, close, 1) * 100, 0.0)
    vma = s_vol.rolling(20, min_periods=1).mean().shift(1).fillna(method="bfill").values
    vr = np.where(vma > 0, vol / np.where(vma > 0, vma, 1), 1.0)
    ma5 = s_close.rolling(5, min_periods=1).mean().values
    ma20 = s_close.rolling(20, min_periods=1).mean().values
    above_ma5 = (close > ma5).astype(float)
    above_ma20 = (close > ma20).astype(float)

    # 买卖很准
    typ = (close + high + low) / 3.0
    ban = pd.Series(typ).rolling(5, min_periods=5).mean().values
    ban_s = pd.Series(ban)
    hao = ban_s.rolling(10, min_periods=10).max().values
    floor10 = ban_s.rolling(10, min_periods=10).min().values
    below_floor = np.where(np.isnan(floor10), 0.0, (close < floor10).astype(float))
    below_ceiling = np.where(np.isnan(hao), 0.0, (close < hao).astype(float))
    # DMI for 准备
    prev_h = np.concatenate(([high[0]], high[:-1]))
    prev_l = np.concatenate(([low[0]], low[:-1]))
    prev_c = np.concatenate(([close[0]], close[:-1]))
    tr = np.maximum.reduce([high - low, np.abs(high - prev_c), np.abs(low - prev_c)])
    hd = high - prev_h; ld = prev_l - low
    pos_hd = np.where((hd > 0) & (hd > ld), hd, 0.0)
    pos_ld = np.where((ld > 0) & (ld > hd), ld, 0.0)
    td = pd.Series(tr).rolling(5, min_periods=5).sum().values
    dmp = pd.Series(pos_hd).rolling(5, min_periods=5).sum().values
    dmm = pd.Series(pos_ld).rolling(5, min_periods=5).sum().values
    with np.errstate(divide="ignore", invalid="ignore"):
        shentou = np.where(td > 0, dmp * 100 / td, 0.0)
        fuzhu = np.where(td > 0, dmm * 100 / td, 0.0)
        denom = fuzhu + shentou
        dx_raw = np.where(denom > 0, np.abs(fuzhu - shentou) / denom * 100, 0.0)
    dongxiang = pd.Series(dx_raw).rolling(3, min_periods=1).mean().values
    zhunbei = ((dongxiang > 88) & (shentou < 5.8)).astype(float)

    # 动力线
    var2 = pd.Series(low).rolling(10, min_periods=1).min().values
    var33 = pd.Series(high).rolling(25, min_periods=1).max().values
    rng = var33 - var2
    raw = np.where(rng > 0, (close - var2) / np.where(rng > 0, rng, 1) * 4, 0.0)
    dongli = pd.Series(raw).ewm(span=4, adjust=False).mean().values
    dl_norm = dongli / 4.0
    prev_d = np.concatenate(([dongli[0]], dongli[:-1]))
    cross_up_02 = ((prev_d <= 0.2) & (dongli > 0.2)).astype(float)
    dl_bottom_recent = pd.Series(cross_up_02).rolling(5, min_periods=1).max().values

    kdj_j = precompute_kdj_j(close, high, low) / 100.0
    atr_pct = precompute_atr14_pct(close, high, low) / 10.0

    # Stack channels (n, 12)
    ch = np.stack([
        ret, hl_pct, vr, above_ma5, above_ma20,
        below_floor, below_ceiling, zhunbei,
        dl_norm, dl_bottom_recent, kdj_j, atr_pct,
    ], axis=1).astype(np.float32)

    valid = []
    seqs = []
    for i in range(SEQ_LEN + 5, n):
        seqs.append(ch[i - SEQ_LEN:i])
        valid.append(i)
    return np.stack(seqs), np.array(valid)


def main() -> None:
    t0 = time.time()
    print(f"Loading model {MODEL_PATH}...", flush=True)
    model = PatchTST().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    norm = np.load(NORM_PATH)
    mean = norm[0]; std = norm[1]

    print("Loading OHLCV cache...", flush=True)
    ohlcv = pd.read_parquet(OHLCV_PATH)
    ohlcv["trade_date"] = pd.to_datetime(ohlcv["trade_date"])
    codes = sorted(ohlcv["ts_code"].unique())
    print(f"  {len(ohlcv):,} bars × {len(codes)} stocks", flush=True)

    out_rows = []
    for k, ts in enumerate(codes):
        df = ohlcv[ohlcv["ts_code"] == ts].sort_values("trade_date").reset_index(drop=True)
        seqs, valid_idx = build_sequences_for_stock(df)
        if seqs is None or len(seqs) == 0:
            continue
        # Normalize using training stats
        seqs = (seqs - mean) / std

        preds = []
        with torch.no_grad():
            for s in range(0, len(seqs), BATCH_SIZE):
                batch = torch.from_numpy(seqs[s:s + BATCH_SIZE]).to(DEVICE)
                logits = model(batch).cpu().numpy()
                preds.append(1.0 / (1.0 + np.exp(-logits)))
        preds = np.concatenate(preds)

        rows = pd.DataFrame({
            "ts_code": ts,
            "trade_date": df["trade_date"].iloc[valid_idx].values,
            "ml_score": preds.astype(float),
        })
        out_rows.append(rows)

        if (k + 1) % 100 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (k + 1) * (len(codes) - k - 1)
            n_so_far = sum(len(r) for r in out_rows)
            print(f"  {k+1}/{len(codes)} stocks  scored={n_so_far:,}  "
                  f"elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

    print("Concatenating...", flush=True)
    scores = pd.concat(out_rows, ignore_index=True)
    scores.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(scores):,} rows → {OUT_PATH}", flush=True)
    print(f"Total: {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
