"""Train PatchTST on the cached compare dataset.

Reads:  models/compare_dataset.npz
Writes: models/compare_pt_preds.npy

Run as a separate Python process — isolates torch's OpenMP from XGBoost's.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Architecture must match data layout (60 bars × 12 channels)
SEQ_LEN = 60
PATCH_LEN = 10
N_PATCHES = SEQ_LEN // PATCH_LEN
N_CHANNELS = 12

EMBED_DIM = 96
N_HEADS = 4
N_LAYERS = 3
DROPOUT = 0.15
BATCH_SIZE = 512
LR = 1e-3
EPOCHS = 20
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

OUT_DIR = ROOT / "models"
DATA_PATH = OUT_DIR / "compare_dataset.npz"
PRED_PATH = OUT_DIR / "compare_pt_preds.npy"
MODEL_PATH = OUT_DIR / "patchtst_v3.pt"
NORM_PATH = OUT_DIR / "patchtst_v3_norm.npy"


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


def main():
    print(f"Loading cached dataset... (device={DEVICE})", flush=True)
    d = np.load(DATA_PATH)
    Xtr = d["Xtr_seq"].astype(np.float32)
    Xte = d["Xte_seq"].astype(np.float32)
    ytr = d["ytr"].astype(np.float32)
    yte = d["yte"].astype(np.float32)
    print(f"  train={len(ytr)}  test={len(yte)}", flush=True)

    # Per-channel z-score using train stats
    flat = Xtr.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0); std = flat.std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=BATCH_SIZE)

    model = PatchTST().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    best_auc = 0.0
    best_preds = None
    best_state = None

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
        auc = roc_auc_score(yte, preds)
        if auc > best_auc:
            best_auc = auc
            best_preds = preds.copy()
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(f"Ep {ep+1:>2}: loss={np.mean(losses):.4f}  test AUC={auc:.4f}  "
              f"(best {best_auc:.4f})  [{time.time()-t_ep:.0f}s]", flush=True)

    np.save(PRED_PATH, best_preds)
    np.save(NORM_PATH, np.stack([mean, std]))
    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
    print(f"\nSaved preds → {PRED_PATH}", flush=True)
    print(f"Saved model → {MODEL_PATH}  (best AUC {best_auc:.4f})", flush=True)
    print(f"Saved norm  → {NORM_PATH}", flush=True)


if __name__ == "__main__":
    main()
