from datetime import date

import pytest

from app.services.quote_service import aggregate_weekly, aggregate_monthly


class TestAggregation:
    def test_aggregate_weekly(self):
        """Mon 2025-01-06 to Fri 2025-01-10 is one week."""
        rows = [
            {"trade_date": date(2025, 1, 6), "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 7), "open": 11.0, "high": 13.0, "low": 10.0, "close": 12.0, "vol": 150, "amount": 1500.0},
            {"trade_date": date(2025, 1, 8), "open": 12.0, "high": 14.0, "low": 11.0, "close": 13.0, "vol": 200, "amount": 2000.0},
            {"trade_date": date(2025, 1, 9), "open": 13.0, "high": 15.0, "low": 10.5, "close": 14.0, "vol": 120, "amount": 1200.0},
            {"trade_date": date(2025, 1, 10), "open": 14.0, "high": 16.0, "low": 12.0, "close": 15.0, "vol": 180, "amount": 1800.0},
        ]
        result = aggregate_weekly(rows)

        assert len(result) == 1
        week = result[0]
        assert week["open"] == 10.0
        assert week["close"] == 15.0
        assert week["high"] == 16.0
        assert week["low"] == 9.0
        assert week["volume"] == 750
        assert week["amount"] == 7500.0

    def test_aggregate_weekly_partial_week(self):
        """A week with only 3 trading days (e.g. holiday week)."""
        rows = [
            {"trade_date": date(2025, 1, 27), "open": 10.0, "high": 12.0, "low": 9.5, "close": 11.5, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 28), "open": 11.5, "high": 13.0, "low": 11.0, "close": 12.5, "vol": 110, "amount": 1100.0},
            {"trade_date": date(2025, 1, 29), "open": 12.5, "high": 14.0, "low": 12.0, "close": 13.5, "vol": 120, "amount": 1200.0},
        ]
        result = aggregate_weekly(rows)

        assert len(result) == 1
        assert result[0]["open"] == 10.0
        assert result[0]["close"] == 13.5

    def test_aggregate_monthly(self):
        """Two months of data should produce two bars."""
        rows = [
            {"trade_date": date(2025, 1, 2), "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "vol": 100, "amount": 1000.0},
            {"trade_date": date(2025, 1, 31), "open": 11.0, "high": 15.0, "low": 10.0, "close": 14.0, "vol": 200, "amount": 2000.0},
            {"trade_date": date(2025, 2, 3), "open": 14.0, "high": 16.0, "low": 13.0, "close": 15.0, "vol": 150, "amount": 1500.0},
            {"trade_date": date(2025, 2, 28), "open": 15.0, "high": 18.0, "low": 14.0, "close": 17.0, "vol": 180, "amount": 1800.0},
        ]
        result = aggregate_monthly(rows)

        assert len(result) == 2
        jan = result[0]
        assert jan["open"] == 10.0
        assert jan["close"] == 14.0
        assert jan["high"] == 15.0
        assert jan["low"] == 9.0
        assert jan["volume"] == 300
