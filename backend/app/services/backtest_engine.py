"""Backtest engine — full position only (no fractional).

Each buy signal opens one full position; each sell closes it. No averaging in,
no scaling out, no tiered sizing.
"""
from __future__ import annotations

from datetime import date


def run_backtest(
    candles: list[dict],
    signals: list[dict],
    initial_capital: float = 10000.0,
    **_legacy_kwargs,
) -> dict:
    """Run a single-position backtest.

    Args:
        candles: list of dicts with keys: trade_date, open, high, low, close, vol, amount.
        signals: list of dicts with keys: date, action ("buy"/"sell").
        initial_capital: starting capital.
        **_legacy_kwargs: ignored (e.g. obsolete position_ratios).
    """
    signal_map: dict[date, str] = {sig["date"]: sig["action"] for sig in signals}

    cash = initial_capital
    in_position = False
    holding: dict | None = None  # {shares, entry_price, entry_date}

    completed_trades: list[dict] = []
    actions: list[dict] = []
    equity_curve: list[dict] = []

    for candle in candles:
        td = candle["trade_date"]
        close = float(candle["close"])
        action = signal_map.get(td)

        if action == "buy" and not in_position and close > 0 and cash > 0:
            shares = cash / close
            holding = {"shares": shares, "entry_price": close, "entry_date": td}
            actions.append({
                "date": td.isoformat() if isinstance(td, date) else str(td),
                "type": "buy",
                "price": round(close, 4),
                "shares": round(shares, 4),
                "amount": round(cash, 4),
                "position_level": 1,
            })
            cash = 0.0
            in_position = True

        elif action == "sell" and in_position and holding is not None:
            shares = holding["shares"]
            entry_price = holding["entry_price"]
            proceeds = shares * close
            cost = shares * entry_price
            pnl = proceeds - cost
            pnl_pct = (pnl / cost * 100) if cost > 0 else 0.0

            completed_trades.append({
                "entry_date": holding["entry_date"].isoformat()
                              if isinstance(holding["entry_date"], date)
                              else str(holding["entry_date"]),
                "exit_date": td.isoformat() if isinstance(td, date) else str(td),
                "entry_price": round(entry_price, 4),
                "exit_price": round(close, 4),
                "shares": round(shares, 4),
                "pnl": round(pnl, 4),
            })
            actions.append({
                "date": td.isoformat() if isinstance(td, date) else str(td),
                "type": "sell",
                "price": round(close, 4),
                "shares": round(shares, 4),
                "amount": round(proceeds, 4),
                "position_level": 0,
                "pnl": round(pnl, 4),
                "pnl_pct": round(pnl_pct, 2),
            })
            cash = proceeds
            holding = None
            in_position = False

        equity = cash + (holding["shares"] * close if holding else 0.0)
        equity_curve.append({
            "date": td.isoformat() if isinstance(td, date) else str(td),
            "equity": round(equity, 4),
        })

    if candles:
        last_close = float(candles[-1]["close"])
        final_capital = cash + (holding["shares"] * last_close if holding else 0.0)
    else:
        final_capital = initial_capital

    net_profit = final_capital - initial_capital
    net_profit_pct = (net_profit / initial_capital * 100) if initial_capital else 0.0

    trading_days = len(candles)
    annualized_return = (
        (final_capital / initial_capital) ** (252 / trading_days) - 1
        if trading_days > 1 and final_capital > 0 and initial_capital > 0 else 0.0
    )

    max_drawdown = 0.0
    peak = 0.0
    for point in equity_curve:
        eq = point["equity"]
        if eq > peak:
            peak = eq
        if peak > 0:
            dd = (peak - eq) / peak
            if dd > max_drawdown:
                max_drawdown = dd

    total_trades = len(completed_trades)
    winning = sum(1 for t in completed_trades if t["pnl"] > 0)
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    gross_profit = sum(t["pnl"] for t in completed_trades if t["pnl"] > 0)
    gross_loss = abs(sum(t["pnl"] for t in completed_trades if t["pnl"] < 0))
    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = 9999.0
    else:
        profit_factor = 0.0

    return {
        "net_profit": round(net_profit, 4),
        "net_profit_pct": round(net_profit_pct, 4),
        "annualized_return": round(annualized_return, 6),
        "max_drawdown": round(max_drawdown, 6),
        "win_rate": round(win_rate, 4),
        "total_trades": total_trades,
        "profit_factor": round(profit_factor, 4),
        "final_capital": round(final_capital, 4),
        "trades": completed_trades,
        "actions": actions,
        "equity_curve": equity_curve,
    }
