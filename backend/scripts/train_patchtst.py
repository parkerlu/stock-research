"""
Train a small PatchTST classifier: input = 60-bar sequence of [return,
high-low%, vol_ratio]; output = same binary label as XGBoost (≥+5% in next
10 bars before -5% stop).

Architecture:
  - 60 bars → 6 patches of 10 bars each (3 features per bar = 30-d patch)
  - Linear projection to 64-d
  - 2 transformer encoder layers (4 heads, MLP 128)
  - Mean pool → linear → sigmoid

Trained on MPS. Goal: see if sequence dynamics add value over aggregate features.
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

from scripts.ml_step1_train import compute_label, passes_loose_trigger, load_candles


SEQ_LEN = 60
PATCH_LEN = 10
N_PATCHES = SEQ_LEN // PATCH_LEN
N_FEATURES = 3
EMBED_DIM = 64
N_HEADS = 4
N_LAYERS = 2
DROPOUT = 0.1
BATCH_SIZE = 256
LR = 1e-3
EPOCHS = 30
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
CUTOFF = date(2024, 1, 1)


def make_sequence(df: pd.DataFrame, i: int) -> np.ndarray | None:
    """Returns (SEQ_LEN, 3) array: [return_pct, hl_range_pct, vol_ratio]."""
    if i < SEQ_LEN + 5:
        return None
    close = df["close"].astype(float).values
    high = df["high"].astype(float).values
    low = df["low"].astype(float).values
    vol = df["vol"].astype(float).values

    seq = []
    for j in range(i - SEQ_LEN, i):
        if j == 0:
            ret = 0.0
        else:
            ret = (close[j] / close[j - 1] - 1) * 100
        hl_pct = (high[j] - low[j]) / close[j] * 100 if close[j] > 0 else 0.0
        vma = vol[max(0, j - 20):j].mean() if j >= 1 else 1
        vr = vol[j] / vma if vma > 0 else 1
        seq.append([ret, hl_pct, vr])
    return np.array(seq, dtype=np.float32)


async def build_sequences() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
        for i in range(len(df)):
            if not passes_loose_trigger(df, i):
                continue
            seq = make_sequence(df, i)
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


class PatchTSTBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(PATCH_LEN * N_FEATURES, EMBED_DIM)
        self.pos_embed = nn.Parameter(torch.randn(1, N_PATCHES, EMBED_DIM) * 0.02)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=N_HEADS, dim_feedforward=128,
            dropout=DROPOUT, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=N_LAYERS)
        self.head = nn.Sequential(
            nn.LayerNorm(EMBED_DIM),
            nn.Linear(EMBED_DIM, 32),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        # x: (B, SEQ_LEN, N_FEATURES)
        b = x.size(0)
        x = x.view(b, N_PATCHES, PATCH_LEN * N_FEATURES)
        x = self.patch_embed(x) + self.pos_embed
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.head(x).squeeze(-1)


def train(model, train_loader, val_loader):
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        losses = []
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = crit(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())

        model.eval()
        all_logits = []
        all_y = []
        with torch.no_grad():
            for X, y in val_loader:
                X = X.to(DEVICE)
                all_logits.append(model(X).cpu().numpy())
                all_y.append(y.numpy())
        preds = 1 / (1 + np.exp(-np.concatenate(all_logits)))
        ys = np.concatenate(all_y)
        auc = roc_auc_score(ys, preds)
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), "models/patchtst_best.pt")
        print(f"Epoch {epoch+1:>2}: loss={np.mean(losses):.4f}  val AUC={auc:.4f}  "
              f"(best {best_auc:.4f})")


async def main() -> None:
    print(f"Device: {DEVICE}")
    print("Building sequence dataset (this may take a minute)...")
    Xtr, ytr, Xte, yte = await build_sequences()
    print(f"Train: {Xtr.shape}, positive {ytr.mean()*100:.1f}%")
    print(f"Test:  {Xte.shape}, positive {yte.mean()*100:.1f}%")

    # Normalize features per-channel using train stats
    mean = Xtr.reshape(-1, N_FEATURES).mean(axis=0)
    std = Xtr.reshape(-1, N_FEATURES).std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    np.save("models/patchtst_norm.npy", np.stack([mean, std]))

    train_ds = TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr))
    test_ds = TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    os.makedirs("models", exist_ok=True)
    model = PatchTSTBinary().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    train(model, train_loader, test_loader)

    # Load best, evaluate at thresholds
    model.load_state_dict(torch.load("models/patchtst_best.pt", map_location=DEVICE))
    model.eval()
    with torch.no_grad():
        preds = []
        for X, _ in test_loader:
            preds.append(torch.sigmoid(model(X.to(DEVICE))).cpu().numpy())
    preds = np.concatenate(preds)

    print(f"\nFinal test AUC: {roc_auc_score(yte, preds):.4f}")
    for thr in [0.50, 0.55, 0.60, 0.65, 0.70]:
        n = int((preds >= thr).sum())
        if n > 0:
            prec = precision_score(yte, (preds >= thr).astype(int), zero_division=0)
            print(f"  thr={thr:.2f}: prec={prec:.3f}  n={n}")


if __name__ == "__main__":
    asyncio.run(main())
