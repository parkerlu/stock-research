"""Train LSTM forecaster (60 OHLCV bars → 5 future log-returns)."""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "4")

import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

DATA_PATH = ROOT / "models" / "lstm_dataset.npz"
MODEL_PATH = ROOT / "models" / "lstm_forecast.pt"
NORM_PATH = ROOT / "models" / "lstm_forecast_norm.npy"

INPUT_LEN = 60
HORIZON = 5
N_CHANNELS = 5
HIDDEN_DIM = 96
N_LAYERS = 2
DROPOUT = 0.20
BATCH = 1024
LR = 1e-3
EPOCHS = 25
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class LSTMForecaster(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=N_CHANNELS, hidden_size=HIDDEN_DIM,
            num_layers=N_LAYERS, dropout=DROPOUT, batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(HIDDEN_DIM),
            nn.Linear(HIDDEN_DIM, 64),
            nn.GELU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, HORIZON),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)


def main():
    print(f"Device: {DEVICE}", flush=True)
    print(f"Loading {DATA_PATH}...", flush=True)
    d = np.load(DATA_PATH)
    Xtr = d["Xtr"]; ytr = d["ytr"]
    Xte = d["Xte"]; yte = d["yte"]
    print(f"  train {Xtr.shape}  test {Xte.shape}", flush=True)

    # Per-channel z-score using train stats (after the per-window normalization
    # already applied during dataset build — this just centers/scales further)
    flat = Xtr.reshape(-1, N_CHANNELS)
    mean = flat.mean(axis=0); std = flat.std(axis=0) + 1e-6
    Xtr = (Xtr - mean) / std
    Xte = (Xte - mean) / std
    np.save(NORM_PATH, np.stack([mean, std]))
    print(f"  norm: mean={mean}, std={std}", flush=True)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=BATCH, shuffle=True, drop_last=True,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xte), torch.from_numpy(yte)),
        batch_size=BATCH,
    )

    model = LSTMForecaster().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Params: {n_params:,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    crit = nn.MSELoss()

    best_loss = float("inf")
    best_state = None
    for ep in range(EPOCHS):
        model.train()
        losses = []
        t_ep = time.time()
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = crit(pred, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
        model.eval()
        te_losses = []
        all_preds = []; all_y = []
        with torch.no_grad():
            for X, y in test_loader:
                pred = model(X.to(DEVICE)).cpu()
                te_losses.append(crit(pred, y).item())
                all_preds.append(pred.numpy()); all_y.append(y.numpy())
        te_loss = np.mean(te_losses)
        # Direction accuracy on day-1 forecast
        preds_arr = np.concatenate(all_preds, axis=0)
        ys_arr = np.concatenate(all_y, axis=0)
        dir_acc = float(((preds_arr[:, 0] > 0) == (ys_arr[:, 0] > 0)).mean())
        # MAE per horizon
        mae = np.abs(preds_arr - ys_arr).mean(axis=0)
        print(f"Ep {ep+1:>2}: tr={np.mean(losses):.5f}  te={te_loss:.5f}  "
              f"dir1={dir_acc*100:.1f}%  mae={[f'{m*100:.2f}%' for m in mae]}  "
              f"[{time.time()-t_ep:.0f}s]", flush=True)
        if te_loss < best_loss:
            best_loss = te_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        torch.save(best_state, MODEL_PATH)
    print(f"\nSaved best model → {MODEL_PATH} (te_loss={best_loss:.5f})", flush=True)


if __name__ == "__main__":
    main()
