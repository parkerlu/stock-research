"""LSTM 5-day forecast — singleton model loader + per-stock predict."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

INPUT_LEN = 60
HORIZON = 5
N_CHANNELS = 5
HIDDEN_DIM = 96
N_LAYERS = 2
DROPOUT = 0.20

_MODEL: "nn.Module | None" = None
_NORM: tuple[np.ndarray, np.ndarray] | None = None
_DEVICE = torch.device("cpu")  # CPU is fine for single inference; avoids MPS contention with FastAPI
_LOCK = threading.Lock()


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


def _candidate_paths(name: str) -> list[str]:
    base = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))  # /app
    return [
        os.path.join(base, "models", name),                 # /app/models/...
        os.path.join(base, "app", "services", name),        # /app/app/services/...
        os.path.abspath(os.path.join("models", name)),
    ]


def _load() -> tuple["nn.Module | None", tuple[np.ndarray, np.ndarray] | None]:
    global _MODEL, _NORM
    if _MODEL is not None and _NORM is not None:
        return _MODEL, _NORM
    with _LOCK:
        if _MODEL is not None and _NORM is not None:
            return _MODEL, _NORM
        model_path = next((p for p in _candidate_paths("lstm_forecast.pt")
                           if os.path.exists(p)), None)
        norm_path = next((p for p in _candidate_paths("lstm_forecast_norm.npy")
                          if os.path.exists(p)), None)
        if not model_path or not norm_path:
            return None, None
        m = LSTMForecaster().to(_DEVICE)
        m.load_state_dict(torch.load(model_path, map_location=_DEVICE))
        m.eval()
        n = np.load(norm_path)
        _MODEL = m
        _NORM = (n[0], n[1])
        return _MODEL, _NORM


def forecast(df: pd.DataFrame) -> dict | None:
    """Run 5-day forecast on the last 60 bars of df.

    Returns: {
        anchor_close, anchor_date,
        forecast: [{day:1..5, close, low, high}],   # bands = ±1 stddev of test residuals
    } or None if model unavailable.
    """
    model, norm = _load()
    if model is None or norm is None:
        return None
    if len(df) < INPUT_LEN:
        return None
    mean, std = norm

    window = df.iloc[-INPUT_LEN:][["open", "high", "low", "close", "vol"]].values.astype(np.float32)
    anchor_close = float(window[-1, 3])
    if anchor_close <= 0:
        return None
    norm_window = window.copy()
    norm_window[:, :4] /= anchor_close
    v0 = max(window[0, 4], 1.0)
    norm_window[:, 4] /= v0
    norm_window = (norm_window - mean) / (std + 1e-6)

    x = torch.from_numpy(norm_window[None, :, :]).float().to(_DEVICE)
    with torch.no_grad():
        pred_log_returns = model(x).cpu().numpy()[0]   # shape (5,)

    # Convert to predicted closes
    closes = anchor_close * np.exp(pred_log_returns)
    # Confidence band as ±1σ multiplier in log space (test residuals).
    band_std = np.array([0.019, 0.028, 0.034, 0.040, 0.044])  # measured MAE/horizon
    band = np.exp(band_std)   # multiplier, e.g. 1.019 ≈ ±1.9% on day-1

    anchor_date = df["trade_date"].iloc[-1]
    if hasattr(anchor_date, "isoformat"):
        anchor_date_str = anchor_date.isoformat()
    else:
        anchor_date_str = str(anchor_date)

    return {
        "anchor_close": round(anchor_close, 4),
        "anchor_date": anchor_date_str,
        "forecast": [
            {
                "day": d + 1,
                "close": round(float(closes[d]), 4),
                "low":   round(float(closes[d] / band[d]), 4),
                "high":  round(float(closes[d] * band[d]), 4),
                "log_return": round(float(pred_log_returns[d]), 6),
            }
            for d in range(HORIZON)
        ],
    }
