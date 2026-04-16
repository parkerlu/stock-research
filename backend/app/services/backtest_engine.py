"""Backtest engine with three-tier position sizing.

Pure Python, no database dependency. Can be called from factory or standalone.
"""
from __future__ import annotations

from datetime import date


def run_backtest(
    candles: list[dict],
    signals: list[dict],
    initial_capital: float = 10000.0,
    position_ratios: list[int] | None = None,
) -> dict:
    """Run a backtest with three-tier position sizing.

    Args:
        candles: List of dicts with keys: trade_date, open, high, low, close, vol, amount.
                 Must be sorted by trade_date ascending.
        signals: List of dicts with keys: date (date object), action ("buy"/"sell").
        initial_capital: Starting capital.
        position_ratios: Three-element list like [40, 30, 30] summing to 100.

    Returns:
        Dict with metrics: net_profit, net_profit_pct, annualized_return, max_drawdown,
        win_rate, total_trades, profit_factor, final_capital, trades, equity_curve.
    """
    if position_ratios is None:
        position_ratios = [40, 30, 30]

    # Build signal lookup: date -> action
    signal_map: dict[date, str] = {}
    for sig in signals:
        signal_map[sig["date"]] = sig["action"]

    # State
    cash = initial_capital
    position_level = 0  # 0=空仓, 1=1/3仓, 2=2/3仓, 3=满仓
    holdings: list[dict] = []  # [{shares, entry_price, entry_date}]

    completed_trades: list[dict] = []
    equity_curve: list[dict] = []

    for candle in candles:
        td = candle["trade_date"]
        close = float(candle["close"])
        action = signal_map.get(td)

        if action == "buy" and position_level < 3:
            amount = initial_capital * position_ratios[position_level] / 100.0
            if amount > 0 and close > 0:
                shares = amount / close
                cash -= amount
                holdings.append({
                    "shares": shares,
                    "entry_price": close,
                    "entry_date": td,
                })
                position_level += 1

        elif action == "sell" and position_level > 0:
            total_shares = sum(h["shares"] for h in holdings)
            total_cost = sum(h["shares"] * h["entry_price"] for h in holdings)
            proceeds = total_shares * close
            pnl = proceeds - total_cost
            first_entry = min(h["entry_date"] for h in holdings)

            completed_trades.append({
                "entry_date": first_entry.isoformat() if isinstance(first_entry, date) else str(first_entry),
                "exit_date": td.isoformat() if isinstance(td, date) else str(td),
                "entry_price": round(total_cost / total_shares, 4) if total_shares else 0,
                "exit_price": close,
                "shares": round(total_shares, 4),
                "pnl": round(pnl, 4),
            })

            cash += proceeds
            holdings.clear()
            position_level = 0

        # Record daily equity
        equity = cash + sum(h["shares"] * close for h in holdings)
        equity_curve.append({
            "date": td.isoformat() if isinstance(td, date) else str(td),
            "equity": round(equity, 4),
        })

    # Final capital (mark-to-market)
    if candles:
        last_close = float(candles[-1]["close"])
        final_capital = cash + sum(h["shares"] * last_close for h in holdings)
    else:
        final_capital = initial_capital

    # Compute metrics
    net_profit = final_capital - initial_capital
    net_profit_pct = (net_profit / initial_capital * 100) if initial_capital else 0.0

    # Annualized return
    trading_days = len(candles)
    if trading_days > 1 and final_capital > 0 and initial_capital > 0:
        annualized_return = (final_capital / initial_capital) ** (252 / trading_days) - 1
    else:
        annualized_return = 0.0

    # Max drawdown
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

    # Win rate
    total_trades = len(completed_trades)
    winning = sum(1 for t in completed_trades if t["pnl"] > 0)
    win_rate = winning / total_trades if total_trades > 0 else 0.0

    # Profit factor
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
        "equity_curve": equity_curve,
    }
