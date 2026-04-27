"""
Test MLDirectDecision strategy on all 504 stocks in DB.

For each stock: backtest the ML-driven strategy on full available history.
Aggregate: total trades, win rate, total PnL, distribution.
"""
from __future__ import annotations

import asyncio
from datetime import date

import pandas as pd
from sqlalchemy import select, distinct
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from app.config import settings
from app.models.schema import DailyCandle
from app.services.backtest_engine import run_backtest
from app.services.strategy_templates import MLDirectDecision


async def get_all_codes() -> list[str]:
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(select(distinct(DailyCandle.ts_code)))).scalars().all()
    await engine.dispose()
    return sorted(rows)


async def load_candles(symbol: str) -> pd.DataFrame:
    engine = create_async_engine(settings.database_url, echo=False)
    Session = async_sessionmaker(engine, expire_on_commit=False)
    async with Session() as db:
        rows = (await db.execute(
            select(DailyCandle).where(DailyCandle.ts_code == symbol)
            .order_by(DailyCandle.trade_date)
        )).scalars().all()
    await engine.dispose()
    return pd.DataFrame([{
        "trade_date": r.trade_date, "open": float(r.open), "high": float(r.high),
        "low": float(r.low), "close": float(r.close),
        "vol": float(r.vol or 0), "amount": float(r.amount) if r.amount else 0,
    } for r in rows])


def evaluate(strategy, df: pd.DataFrame, post_cutoff: date) -> dict:
    candles = df.to_dict("records")
    sigs = strategy.generate_signals(df)
    # Keep only post-cutoff signals (full out-of-sample period)
    in_pos = False
    out_sigs = []
    for s in sigs:
        if s["action"] == "buy":
            if s["date"] >= post_cutoff:
                out_sigs.append(s)
                in_pos = True
        elif s["action"] == "sell" and in_pos:
            if s["date"] >= post_cutoff:
                out_sigs.append(s)
                in_pos = False
    post_candles = [c for c in candles if c["trade_date"] >= post_cutoff]
    if not post_candles:
        return None
    result = run_backtest(post_candles, out_sigs, 10000)
    trades = result["trades"]
    if not trades:
        return {"trades": 0, "wins": 0, "ret": 0,
                "dd": result["max_drawdown"]*100}
    pcts = [(t["exit_price"]-t["entry_price"])/t["entry_price"]*100 for t in trades]
    wins = sum(1 for p in pcts if p > 0)
    return {
        "trades": len(trades),
        "wins": wins,
        "ret": result["net_profit_pct"],
        "dd": result["max_drawdown"]*100,
        "pcts": pcts,
    }


async def main() -> None:
    codes = await get_all_codes()
    print(f"Testing on {len(codes)} stocks (post-2024-01-01, fully out-of-sample)\n")

    cutoff = date(2024, 1, 1)

    # Compare configs
    configs = [
        ("V2: Adaptive thr + fixed +15% target", {
            "use_adaptive": True, "buy_threshold": 0.55,
            "stop_loss": -0.07, "trail_activation": 999.0, "trail_pct": 999.0,
            "time_stop": 30,
        }),
        ("V3: Adaptive thr + GREEDY trailing (more buys)", {
            "use_adaptive": True, "buy_threshold": 0.55,
            "stop_loss": -0.07, "trail_activation": 0.05, "trail_pct": 0.04,
            "time_stop": 30,
        }),
    ]

    for label, cfg in configs:
        strategy_factory = lambda c=cfg: MLDirectDecision(**c)

        all_results = []
        total_trades = 0
        total_wins = 0
        total_ret = 0.0
        stocks_traded = 0
        stocks_profit = 0
        all_pcts: list[float] = []

        for i, sym in enumerate(codes):
            df = await load_candles(sym)
            if len(df) < 130:
                continue
            strategy = strategy_factory()
            strategy.ts_code = sym  # let adaptive config look up per-stock thr
            r = evaluate(strategy, df, cutoff)
            if r is None or r["trades"] == 0:
                continue
            all_results.append((sym, r))
            total_trades += r["trades"]
            total_wins += r["wins"]
            total_ret += r["ret"]
            all_pcts.extend(r["pcts"])
            stocks_traded += 1
            if r["ret"] > 0:
                stocks_profit += 1

        print(f"\n{'='*78}")
        print(f"  {label}")
        print(f"{'='*78}")
        print(f"产生交易的股票数:     {stocks_traded} / {len(codes)}")
        print(f"盈利股票数:           {stocks_profit} "
              f"({stocks_profit/max(stocks_traded,1)*100:.1f}%)")
        print(f"总交易笔数:           {total_trades}")
        print(f"胜率 (pnl>0):         "
              f"{total_wins/max(total_trades,1)*100:.1f}% "
              f"({total_wins}/{total_trades})")
        if all_pcts:
            avg = sum(all_pcts) / len(all_pcts)
            print(f"平均单笔收益:         {avg:+.2f}%")
            sp = sorted(all_pcts)
            print(f"中位数:               {sp[len(sp)//2]:+.2f}%")
            print(f"最好/最差:            {max(all_pcts):+.2f}% / "
                  f"{min(all_pcts):+.2f}%")
            for thr in [3, 5, 10, 15]:
                n = sum(1 for p in all_pcts if p >= thr)
                print(f"达到 ≥{thr:>2}% 的笔数:    {n} ({n/len(all_pcts)*100:.1f}%)")
        print(f"各股平均收益(等权):   {total_ret/max(stocks_traded,1):+.2f}%")


if __name__ == "__main__":
    asyncio.run(main())
