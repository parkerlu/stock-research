"""
PatchTST V2 — longer sequence (60 bars), richer features (8 channels including
买卖很准 binary signals as time-series channels).

Channels per bar:
  0: return %
  1: high-low % (volatility)
  2: vol / MA20(vol) ratio
  3: close > MA5 (binary)
  4: close > MA20 (binary)
  5: mm_below_floor (binary)
  6: mm_below_ceiling (binary)
  7: mm_zhunbei_active (binary)

Trained on the same candidate-bar selection as XGBoost (passes_loose_trigger),
with the same 5%/10-bar label.
"""
from __future__ import annotations

import asyncio
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, roc_auc_score
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from torch.utils.data import DataLoader, TensorDataset

from app.config import settings
from app.models.schema import DailyCandle
from datetime import date

from scripts.ml_step1_train import (
    compute_label, passes_loose_trigger, load_candles, precompute_maimai,
)


SEQ_LEN = 60
PATCH_LEN = 10
N_PATCHES = SEQ_LEN // PATCH_LEN
N_FEATURES = 8

EMBED_DIM = 96
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.15
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 25
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CUTOFF = date(2024, 1, 1)


def make_sequence(df: pd.DataFrame, mm: dict, i: int) -> np.ndarray | None:
    if i < SEQ_LEN + 5:
        return None
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    vol = df["vol"].values.astype(float)

    # Pre-compute MAs once (outside loop) is faster but for now keep simple
    seq = np.zeros((SEQ_LEN, N_FEATURES), dtype=np.float32)
    for j in range(SEQ_LEN):
        b = i - SEQ_LEN + j
        ret = (close[b] / close[b - 1] - 1) * 100 if b > 0 else 0.0
        hl_pct = (high[b] - low[b]) / close[b] * 100 if close[b] > 0 else 0.0
        vma = vol[max(0, b - 20):b].mean() if b >= 1 else 1.0
        vr = vol[b] / vma if vma > 0 else 1.0
        ma5 = close[max(0, b - 5):b].mean() if b >= 1 else close[b]
        ma20 = close[max(0, b - 20):b].mean() if b >= 1 else close[b]
        seq[j, 0] = ret
        seq[j, 1] = hl_pct
        seq[j, 2] = vr
        seq[j, 3] = 1.0 if close[b] > ma5 else 0.0
        seq[j, 4] = 1.0 if close[b] > ma20 else 0.0
        seq[j, 5] = mm["mm_below_floor"][b]
        seq[j, 6] = mm["mm_below_ceiling"][b]
        seq[j, 7] = mm["mm_zhunbei_active"][b]
    return seq


async def build_sequences():
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        codes = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await engine.dispose()

    train_X, train_y, test_X, test_y = [], [], [], []
    for sym in codes:
        df = await load_candles(sym)
        if len(df) < SEQ_LEN + 130:
            continue
        mm = precompute_maimai(df)
        df.attrs["_mm_cache"] = mm
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            seq = make_sequence(df, mm, i)
            if seq is None:
                continue
            label = compute_label(df, i)
            if label is None:
                continue
            d = df["trade_date"].iloc[i]
            if d < CUTOFF:
                train_X.append(seq); train_y.append(label)
            else:
                test_X.append(seq); test_y.append(label)
    return (np.stack(train_X), np.array(train_y, dtype=np.float32),
            np.stack(test_X), np.array(test_y, dtype=np.float32))


class PatchTSTV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN * N_FEATURES, EMBED_DIM)
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
        x = x.view(b, N_PATCHES, PATCH_LEN * N_FEATURES)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


async def main() -> None:
    print(f"Device: {DEVICE}")
    print("Building sequences (~5 minutes for 504 stocks)...")
    Xtr, ytr, Xte, yte = await build_sequences()
    print(f"Train: {Xtr.shape}, +ve {ytr.mean()*100:.1f}%")
    print(f"Test:  {Xte.shape}, +ve {yte.mean()*100:.1f}%")

    # Per-channel z-score using train stats
    mean = Xtr.reshape(-1, N_FEATURES).mean(axis=0)
    std = Xtr.reshape(-1, N_FEATURES).std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    np.save("models/patchtst_v2_norm.npy", np.stack([mean, std]))

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=BATCH_SIZE)

    os.makedirs("models", exist_ok=True)
    model = PatchTSTV2().to(DEVICE)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_path = "models/patchtst_v2_best.pt"
    for ep in range(EPOCHS):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            losses.append(loss.item())
        model.eval()
        ll, yy = [], []
        with torch.no_grad():
            for X, y in test_loader:
                X = X.to(DEVICE)
                ll.append(model(X).cpu().numpy()); yy.append(y.numpy())
        preds = 1 / (1 + np.exp(-np.concatenate(ll)))
        ys = np.concatenate(yy)
        auc = roc_auc_score(ys, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), best_path)
        print(f"Ep {ep+1:>2}: loss={np.mean(losses):.4f}  val AUC={auc:.4f}  "
              f"(best {best_auc:.4f})")

    # Reload best, eval at thresholds
    model.load_state_dict(torch.load(best_path, map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        preds = []
        for X, _ in test_loader:
            preds.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    preds = np.concatenate(preds)
    print(f"\nFinal AUC: {roc_auc_score(yte, preds):.4f}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75]:
        n = int((preds >= thr).sum())
        if n > 0:
            print(f"  thr={thr:.2f}: prec={precision_score(yte, (preds>=thr).astype(int), zero_division=0):.3f}  n={n}")


if __name__ == "__main__":
    asyncio.run(main())
