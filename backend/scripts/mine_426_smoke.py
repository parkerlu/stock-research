"""Smoke test: 3 families × 50 stocks to verify engine works before full mine."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.mine_426_step4_mine import load_per_stock_data, evaluate_candidate
from scripts.mine_426_strategies import FAMILIES


def main():
    t0 = time.time()
    per_stock, dates = load_per_stock_data()
    # Take first 50 stocks
    codes = list(per_stock.keys())[:50]
    sub_data = {c: per_stock[c] for c in codes}
    sub_dates = {c: dates[c] for c in codes}
    print(f"Subset: {len(sub_data)} stocks", flush=True)

    test_families = ["A01_ml_classic", "C01_dl_bottom_classic", "F04_dl_ji"]
    fam_map = {f.family_id: f for f in FAMILIES}

    for fid in test_families:
        fam = fam_map[fid]
        t = time.time()
        r = evaluate_candidate(fam, fam.default_params, sub_data, sub_dates)
        if r["full"]:
            f = r["full"]
            print(f"  {fid:<28}  trades={f['n_trades']:>4}  "
                  f"win={f['win_rate']*100:>5.1f}%  avgRet={f['avg_ret']*100:>+5.2f}%  "
                  f"totRet={f['total_ret']*100:>+6.1f}%  stocks={r['stocks_traded']:>3}  "
                  f"({time.time()-t:.1f}s)", flush=True)
        else:
            print(f"  {fid:<28}  no trades", flush=True)

    print(f"\nTotal: {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
