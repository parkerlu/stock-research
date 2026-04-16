"""Tests for the backtest engine with three-tier position sizing."""
from datetime import date
import pytest
from app.services.backtest_engine import run_backtest


def _candle(d: str, close: float, open_: float = 0, high: float = 0, low: float = 0, vol: int = 1000):
    """Helper to build a candle dict."""
    return {
        "trade_date": date.fromisoformat(d),
        "open": open_ or close,
        "high": high or close,
        "low": low or close,
        "close": close,
        "vol": vol,
        "amount": close * vol,
    }


class TestBasicBacktest:
    def test_single_buy_sell_profit(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),
            _candle("2020-01-03", 11.0),
            _candle("2020-01-04", 12.0),
            _candle("2020-01-05", 13.0),
        ]
        signals = [
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(800.0)
        assert result["final_capital"] == pytest.approx(10800.0)
        assert result["win_rate"] == pytest.approx(1.0)

    def test_single_buy_sell_loss(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),
            _candle("2020-01-03", 9.0),
            _candle("2020-01-04", 8.0),
        ]
        signals = [
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(-800.0)
        assert result["win_rate"] == pytest.approx(0.0)

    def test_no_signals_no_trades(self):
        candles = [_candle("2020-01-01", 10.0), _candle("2020-01-02", 11.0)]
        result = run_backtest(candles, [], initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 0
        assert result["net_profit"] == pytest.approx(0.0)
        assert result["final_capital"] == pytest.approx(10000.0)


class TestThreeTierPosition:
    def test_three_buys_then_sell(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),
            _candle("2020-01-03", 10.0),
            _candle("2020-01-04", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(2000.0)
        assert result["final_capital"] == pytest.approx(12000.0)

    def test_buy_at_full_position_ignored(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),
            _candle("2020-01-03", 10.0),
            _candle("2020-01-04", 10.0),
            _candle("2020-01-05", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "buy"},
            {"date": date(2020, 1, 5), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["net_profit"] == pytest.approx(2000.0)

    def test_sell_at_empty_position_ignored(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 11.0),
            _candle("2020-01-03", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "sell"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(4000 / 11 * 12 - 4000, rel=1e-4)

    def test_partial_position_sell_clears_all(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 10.0),
            _candle("2020-01-03", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "buy"},
            {"date": date(2020, 1, 3), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 1
        assert result["net_profit"] == pytest.approx(1400.0)


class TestMetrics:
    def test_equity_curve_length(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 11.0),
            _candle("2020-01-03", 12.0),
        ]
        result = run_backtest(candles, [], initial_capital=10000, position_ratios=[40, 30, 30])
        assert len(result["equity_curve"]) == 3

    def test_max_drawdown(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 12.0),
            _candle("2020-01-03", 8.0),
            _candle("2020-01-04", 11.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[100, 0, 0])
        assert result["max_drawdown"] == pytest.approx(1 / 3, rel=1e-3)

    def test_profit_factor(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 12.0),
            _candle("2020-01-03", 12.0),
            _candle("2020-01-04", 11.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 2
        assert result["win_rate"] == pytest.approx(0.5)
        gross_profit = 800.0
        gross_loss = abs(4000 / 12 * 11 - 4000)
        assert result["profit_factor"] == pytest.approx(gross_profit / gross_loss, rel=1e-3)

    def test_trades_detail(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert len(result["trades"]) == 1
        trade = result["trades"][0]
        assert trade["entry_date"] == "2020-01-01"
        assert trade["exit_date"] == "2020-01-02"
        assert trade["pnl"] == pytest.approx(800.0)

    def test_annualized_return(self):
        candles = [_candle(f"2020-01-{str(i+1).zfill(2)}", 10.0 + i * 0.1) for i in range(20)]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 20), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert "annualized_return" in result
        assert isinstance(result["annualized_return"], float)


class TestMultipleCycles:
    def test_two_full_cycles(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 12.0),
            _candle("2020-01-03", 12.0),
            _candle("2020-01-04", 15.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
            {"date": date(2020, 1, 2), "action": "sell"},
            {"date": date(2020, 1, 3), "action": "buy"},
            {"date": date(2020, 1, 4), "action": "sell"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 2
        assert result["net_profit"] > 0

    def test_open_position_at_end(self):
        candles = [
            _candle("2020-01-01", 10.0),
            _candle("2020-01-02", 12.0),
        ]
        signals = [
            {"date": date(2020, 1, 1), "action": "buy"},
        ]
        result = run_backtest(candles, signals, initial_capital=10000, position_ratios=[40, 30, 30])
        assert result["total_trades"] == 0
        assert result["final_capital"] == pytest.approx(6000 + 400 * 12)
