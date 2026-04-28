"""Batch-evaluate every registered strategy on a 20-stock liquid pool.

For each (template, stock) pair, run the API's /api/backtests/try (in-process)
and record win_rate / avg_return / total_return / max_drawdown / n_trades.
Aggregate across the pool, then write back into strategy_pool.json.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

# Smaller pool for faster eval (8 stocks across sectors)
EVAL_POOL = [
    "600519.SH",  # 茅台 (大白马)
    "600036.SH",  # 招商银行 (金融)
    "002594.SZ",  # 比亚迪 (新能源)
    "300059.SZ",  # 东方财富 (券商互联网)
    "002230.SZ",  # 科大讯飞 (科技成长)
    "603319.SH",  # 陕鼓动力 (周期)
    "688111.SH",  # 金山办公 (科创板软件)
    "600476.SH",  # 湘邮科技 (中小盘)
]


async def evaluate_template(template_id: str, codes: list[str]) -> dict:
    """Run all (stock, template_id) pairs and aggregate."""
    from app.db import async_session
    from app.services.factory_service import get_candle_dicts
    from app.services.backtest_engine import run_backtest
    from app.services.strategy_templates import TEMPLATE_REGISTRY

    cls = TEMPLATE_REGISTRY.get(template_id)
    if cls is None:
        return {"error": "not registered"}

    end = date.today()
    start = date(end.year - 8, 1, 1)

    all_trades = []
    n_stocks_with_trades = 0
    n_total = 0

    async with async_session() as db:
        for ts in codes:
            candles = await get_candle_dicts(db, ts, start, end)
            if not candles or len(candles) < 130:
                continue
            df = pd.DataFrame(candles)
            tpl = cls()
            if hasattr(tpl, "ts_code"):
                tpl.ts_code = ts
            try:
                signals = tpl.generate_signals(df)
            except Exception:
                continue
            if not signals:
                continue
            result = run_backtest(candles, signals, initial_capital=10000)
            trades = result.get("trades", [])
            if not trades:
                continue
            n_stocks_with_trades += 1
            for t in trades:
                t["ts_code"] = ts
                t["mdd"] = result.get("max_drawdown", 0.0)
                all_trades.append(t)

    if not all_trades:
        return {"trades": 0, "stocks": 0}

    df_t = pd.DataFrame(all_trades)
    pnl_pcts = df_t.apply(lambda r: (r["exit_price"] / r["entry_price"] - 1) * 100, axis=1).values
    win_rate = float((pnl_pcts > 0).mean())
    avg_ret = float(pnl_pcts.mean())
    max_loss = float(pnl_pcts.min())
    # Avg per-stock MDD (each stock has its own MDD)
    avg_mdd = float(df_t.groupby("ts_code")["mdd"].first().mean())

    return {
        "trades": int(len(df_t)),
        "stocks": int(n_stocks_with_trades),
        "win_rate": round(win_rate * 100, 1),
        "avg_ret": round(avg_ret, 2),
        "max_loss": round(max_loss, 2),
        "avg_mdd": round(avg_mdd * 100, 1),
    }


async def main():
    from app.services.strategy_templates import TEMPLATE_REGISTRY

    out_path = ROOT / "data" / "strategy_metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Resume if file exists
    out: dict[str, dict] = {}
    if out_path.exists():
        try:
            out = json.loads(out_path.read_text())
            print(f"Resuming from {out_path} ({len(out)} done)", flush=True)
        except Exception:
            pass

    t0 = time.time()
    todo = [t for t in TEMPLATE_REGISTRY.keys() if t not in out]
    print(f"Evaluating {len(todo)} templates × {len(EVAL_POOL)} stocks...",
          flush=True)
    for k, tpl in enumerate(todo):
        t1 = time.time()
        try:
            metrics = await evaluate_template(tpl, EVAL_POOL)
        except Exception as e:
            metrics = {"error": str(e)[:80]}
        out[tpl] = metrics
        elapsed = time.time() - t1
        print(f"  [{k+1}/{len(todo)}] {tpl:<22} "
              f"trd={metrics.get('trades', 0):>4} "
              f"win={metrics.get('win_rate', 0):>5.1f}% "
              f"avg={metrics.get('avg_ret', 0):>+5.2f}% "
              f"mdd={metrics.get('avg_mdd', 0):>5.1f}% "
              f"({elapsed:.0f}s)", flush=True)
        # Persist incrementally so partial results survive a kill
        out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))

    print(f"\nSaved → {out_path}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    asyncio.run(main())
