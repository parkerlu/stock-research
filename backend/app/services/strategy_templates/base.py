"""Base class for strategy templates."""
from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd


class StrategyTemplate(ABC):
    """Abstract base for all strategy templates."""

    template_id: str = ""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable strategy name, e.g. 'MA_5_20'."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> list[dict]:
        """Generate buy/sell signals from OHLCV DataFrame.

        Args:
            df: DataFrame with columns: trade_date, open, high, low, close, vol, amount.

        Returns:
            List of dicts: [{"date": date, "action": "buy"|"sell"}, ...]
        """

    @staticmethod
    @abstractmethod
    def parameter_candidates() -> list[dict]:
        """Return all valid parameter combinations for this template."""
