"""Tests for strategy templates."""
from datetime import date, timedelta
import numpy as np
import pandas as pd
import pytest

from app.services.strategy_templates.base import StrategyTemplate
from app.services.strategy_templates.ma_crossover import MACrossover
from app.services.strategy_templates.rsi import RSIOverboughtOversold
from app.services.strategy_templates.macd import MACDCrossover
from app.services.strategy_templates.bollinger import BollingerBreakout
from app.services.strategy_templates.kdj import KDJCrossover
from app.services.strategy_templates.combined import MARSICombined, MACDVolumeCombined, BollingerRSICombined


def _make_df(n: int = 100, start_price: float = 10.0) -> pd.DataFrame:
    np.random.seed(42)
    dates = [date(2020, 1, 1) + timedelta(days=i) for i in range(n)]
    closes = start_price + np.cumsum(np.random.randn(n) * 0.5)
    closes = np.maximum(closes, 1.0)
    return pd.DataFrame({
        "trade_date": dates,
        "open": closes * 0.99,
        "high": closes * 1.02,
        "low": closes * 0.98,
        "close": closes,
        "vol": [1000000] * n,
        "amount": closes * 1000000,
    })


class TestMACrossover:
    def test_generate_signals_returns_list(self):
        df = _make_df(100)
        t = MACrossover(fast_period=5, slow_period=20)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)
        for sig in signals:
            assert sig["action"] in ("buy", "sell")

    def test_signals_have_valid_dates(self):
        df = _make_df(100)
        t = MACrossover(fast_period=5, slow_period=20)
        signals = t.generate_signals(df)
        df_dates = set(df["trade_date"])
        for sig in signals:
            assert sig["date"] in df_dates

    def test_name_format(self):
        t = MACrossover(fast_period=5, slow_period=20)
        assert t.name == "MA_5_20"
        assert t.template_id == "ma_crossover"

    def test_parameter_candidates(self):
        candidates = MACrossover.parameter_candidates()
        assert len(candidates) > 0
        for params in candidates:
            assert params["fast_period"] < params["slow_period"]

    def test_no_signals_for_short_data(self):
        df = _make_df(5)
        t = MACrossover(fast_period=5, slow_period=20)
        signals = t.generate_signals(df)
        assert signals == []


class TestRSI:
    def test_generate_signals(self):
        df = _make_df(200)
        t = RSIOverboughtOversold(period=14, overbought=70, oversold=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = RSIOverboughtOversold(period=14, overbought=70, oversold=30)
        assert t.name == "RSI_14_70_30"
        assert t.template_id == "rsi"

    def test_candidates(self):
        c = RSIOverboughtOversold.parameter_candidates()
        assert len(c) >= 20
        for p in c:
            assert p["oversold"] < p["overbought"] - 15


class TestMACD:
    def test_generate_signals(self):
        df = _make_df(200)
        t = MACDCrossover(fast=12, slow=26, signal=9)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = MACDCrossover(fast=12, slow=26, signal=9)
        assert t.name == "MACD_12_26_9"

    def test_candidates(self):
        c = MACDCrossover.parameter_candidates()
        assert len(c) >= 30
        for p in c:
            assert p["fast"] < p["slow"]


class TestBollinger:
    def test_generate_signals(self):
        df = _make_df(200)
        t = BollingerBreakout(period=20, std_dev=2.0)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = BollingerBreakout(period=20, std_dev=2.0)
        assert t.name == "BB_20_2.0"

    def test_candidates(self):
        c = BollingerBreakout.parameter_candidates()
        assert len(c) >= 20


class TestKDJ:
    def test_generate_signals(self):
        df = _make_df(200)
        t = KDJCrossover(k_period=9, d_period=3)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_name(self):
        t = KDJCrossover(k_period=9, d_period=3)
        assert t.name == "KDJ_9_3"

    def test_candidates(self):
        c = KDJCrossover.parameter_candidates()
        assert len(c) >= 20


class TestMARSI:
    def test_signals(self):
        df = _make_df(200)
        t = MARSICombined(ma_period=20, rsi_period=14, overbought=70, oversold=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = MARSICombined.parameter_candidates()
        assert 20 <= len(c) <= 200


class TestMACDVolume:
    def test_signals(self):
        df = _make_df(200)
        t = MACDVolumeCombined(fast=12, slow=26, signal=9, vol_ma=10)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = MACDVolumeCombined.parameter_candidates()
        assert 30 <= len(c) <= 150


class TestBollingerRSI:
    def test_signals(self):
        df = _make_df(200)
        t = BollingerRSICombined(bb_period=20, bb_std=2.0, rsi_period=14, rsi_ob=70, rsi_os=30)
        signals = t.generate_signals(df)
        assert isinstance(signals, list)

    def test_candidates_count(self):
        c = BollingerRSICombined.parameter_candidates()
        assert 30 <= len(c) <= 150


class TestTemplateRegistry:
    def test_registry_has_all_templates(self):
        from app.services.strategy_templates import TEMPLATE_REGISTRY
        assert "ma_crossover" in TEMPLATE_REGISTRY
        assert "rsi" in TEMPLATE_REGISTRY
        assert "macd" in TEMPLATE_REGISTRY
        assert "bollinger" in TEMPLATE_REGISTRY
        assert "kdj" in TEMPLATE_REGISTRY
        assert "ma_rsi" in TEMPLATE_REGISTRY
        assert "macd_vol" in TEMPLATE_REGISTRY
        assert "bb_rsi" in TEMPLATE_REGISTRY

    def test_total_candidates_in_range(self):
        from app.services.strategy_templates import generate_all_candidates
        candidates = generate_all_candidates()
        assert 300 <= len(candidates) <= 800, f"Got {len(candidates)} candidates, expected 300-800"
