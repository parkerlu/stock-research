"""Tests for the factory service (unit tests)."""
from datetime import date, timedelta
import numpy as np
import pytest

from app.services.factory_service import evaluate_single_candidate, filter_and_rank


def _make_candle_dicts(n: int = 500) -> list[dict]:
    np.random.seed(42)
    base = 100.0
    candles = []
    for i in range(n):
        d = date(2015, 1, 1) + timedelta(days=i)
        change = np.random.randn() * 2
        base = max(base + change, 10)
        candles.append({
            "trade_date": d,
            "open": base * 0.99,
            "high": base * 1.02,
            "low": base * 0.98,
            "close": base,
            "vol": 1000000,
            "amount": base * 1000000,
        })
    return candles


class TestEvaluateSingleCandidate:
    def test_returns_result_dict(self):
        candles = _make_candle_dicts(200)
        candidate = {
            "template_id": "ma_crossover",
            "name": "MA_5_20",
            "params": {"fast_period": 5, "slow_period": 20},
        }
        result = evaluate_single_candidate(candles, candidate, [40, 30, 30])
        assert "name" in result
        assert "metrics" in result
        assert "net_profit" in result["metrics"]

    def test_short_data_returns_zero_trades(self):
        candles = _make_candle_dicts(5)
        candidate = {
            "template_id": "ma_crossover",
            "name": "MA_5_20",
            "params": {"fast_period": 5, "slow_period": 20},
        }
        result = evaluate_single_candidate(candles, candidate, [40, 30, 30])
        assert result["metrics"]["total_trades"] == 0


class TestFilterAndRank:
    def test_filters_negative_profit(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.1, "annualized_return": 0.2}},
            {"name": "b", "metrics": {"net_profit": -50, "max_drawdown": 0.1, "annualized_return": -0.1}},
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "a"

    def test_filters_high_drawdown(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.5, "annualized_return": 0.2}},
            {"name": "b", "metrics": {"net_profit": 100, "max_drawdown": 0.3, "annualized_return": 0.15}},
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 1
        assert filtered[0]["name"] == "b"

    def test_top_50_limit(self):
        results = [
            {"name": f"s{i}", "metrics": {"net_profit": 100 + i, "max_drawdown": 0.1, "annualized_return": 0.01 * i}}
            for i in range(100)
        ]
        filtered = filter_and_rank(results)
        assert len(filtered) == 50
        returns = [r["metrics"]["annualized_return"] for r in filtered]
        assert returns == sorted(returns, reverse=True)

    def test_sorted_by_annualized_return(self):
        results = [
            {"name": "a", "metrics": {"net_profit": 100, "max_drawdown": 0.1, "annualized_return": 0.1}},
            {"name": "b", "metrics": {"net_profit": 200, "max_drawdown": 0.2, "annualized_return": 0.3}},
            {"name": "c", "metrics": {"net_profit": 150, "max_drawdown": 0.15, "annualized_return": 0.2}},
        ]
        filtered = filter_and_rank(results)
        assert [r["name"] for r in filtered] == ["b", "c", "a"]
