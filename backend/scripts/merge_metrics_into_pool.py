"""Merge eval metrics from data/strategy_metrics.json into the running
strategy pool (data/strategy_pool.json).

Strategies marked with `source: preset` keep their preset description but
get their metrics overwritten with the freshly computed values."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
METRICS = ROOT / "data" / "strategy_metrics.json"
POOL = ROOT / "data" / "strategy_pool.json"

if not METRICS.exists():
    print(f"missing {METRICS}; run eval_all_strategies.py first")
    sys.exit(1)
if not POOL.exists():
    print(f"missing {POOL}; will be auto-seeded on next API call")
    sys.exit(1)

metrics = json.loads(METRICS.read_text())
pool = json.loads(POOL.read_text())
entries = pool.get("entries", [])

updated = 0
for e in entries:
    tid = e["template_id"]
    m = metrics.get(tid)
    if not m or "trades" not in m:
        continue
    e["metrics"] = {
        "trades":  m.get("trades", 0),
        "stocks":  m.get("stocks", 0),
        "win_rate": m.get("win_rate", 0),
        "avg_ret":  m.get("avg_ret", 0),
        "avg_mdd":  m.get("avg_mdd", 0),
        "max_loss": m.get("max_loss", 0),
    }
    updated += 1

POOL.write_text(json.dumps({"entries": entries}, ensure_ascii=False, indent=2))
print(f"Updated {updated}/{len(entries)} entries → {POOL}")
