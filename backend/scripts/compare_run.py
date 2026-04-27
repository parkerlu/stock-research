"""Orchestrator for the PatchTST vs XGBoost comparison.

Runs each training in a separate Python subprocess to keep their OpenMP
runtimes from clashing. Then loads the saved predictions and prints the
side-by-side report.

Prerequisite: models/compare_dataset.npz must already exist (built by
train_patchtst_v3_compare.py — that script's dataset-build path is unchanged
and still works fine; it's only the in-process XGB+torch training that
crashes).
"""
from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score, roc_auc_score

ROOT = Path(__file__).parent.parent
PY = sys.executable
OUT_DIR = ROOT / "models"


def run_step(name: str, script: str) -> None:
    print(f"\n{'='*72}\n[{name}] launching subprocess: {script}\n{'='*72}", flush=True)
    t0 = time.time()
    rc = subprocess.run([PY, script], cwd=str(ROOT)).returncode
    dur = time.time() - t0
    if rc != 0:
        print(f"[{name}] FAILED rc={rc}  ({dur:.0f}s)", flush=True)
        sys.exit(rc)
    print(f"[{name}] OK  ({dur:.0f}s)", flush=True)


def report():
    yte = np.load(OUT_DIR / "compare_y_te.npy")
    xgb_p = np.load(OUT_DIR / "compare_xgb_preds.npy")
    pt_p = np.load(OUT_DIR / "compare_pt_preds.npy")

    auc_xgb = roc_auc_score(yte, xgb_p)
    auc_pt = roc_auc_score(yte, pt_p)
    print(f"\n{'='*72}")
    print(f"FINAL COMPARISON  (test n={len(yte)}, +ve {yte.mean()*100:.1f}%)")
    print(f"{'='*72}")
    print(f"AUC: XGB-Ens={auc_xgb:.4f}   PatchTST={auc_pt:.4f}   "
          f"Δ={(auc_pt-auc_xgb)*100:+.2f} pp")
    print(f"\n{'thr':>6}  {'XGB-Ens (prec / n)':>22}  {'PatchTST (prec / n)':>22}  {'Δprec':>8}")
    for thr in [0.30, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70]:
        bx = (xgb_p >= thr).astype(int); bp = (pt_p >= thr).astype(int)
        nx, np_ = int(bx.sum()), int(bp.sum())
        px = precision_score(yte, bx, zero_division=0) if nx > 0 else 0
        pp = precision_score(yte, bp, zero_division=0) if np_ > 0 else 0
        print(f"  {thr:.2f}  {px:.3f} / {nx:>6,}        {pp:.3f} / {np_:>6,}      "
              f"{(pp-px)*100:+.1f} pp")


def main():
    if not (OUT_DIR / "compare_dataset.npz").exists():
        print("ERROR: models/compare_dataset.npz missing.")
        print("Run scripts/train_patchtst_v3_compare.py first to build it.")
        sys.exit(1)

    run_step("XGB", "scripts/compare_xgb_only.py")
    run_step("PatchTST", "scripts/compare_pt_only.py")
    report()


if __name__ == "__main__":
    main()
