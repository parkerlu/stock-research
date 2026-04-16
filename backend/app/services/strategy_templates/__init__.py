"""Strategy template registry."""
from __future__ import annotations

from .ma_crossover import MACrossover
from .rsi import RSIOverboughtOversold
from .macd import MACDCrossover
from .bollinger import BollingerBreakout
from .kdj import KDJCrossover
from .combined import MARSICombined, MACDVolumeCombined, BollingerRSICombined

TEMPLATE_REGISTRY: dict[str, type] = {
    "ma_crossover": MACrossover,
    "rsi": RSIOverboughtOversold,
    "macd": MACDCrossover,
    "bollinger": BollingerBreakout,
    "kdj": KDJCrossover,
    "ma_rsi": MARSICombined,
    "macd_vol": MACDVolumeCombined,
    "bb_rsi": BollingerRSICombined,
}


def generate_all_candidates() -> list[dict]:
    """Generate all strategy candidates across all registered templates."""
    candidates = []
    for template_id, cls in TEMPLATE_REGISTRY.items():
        for params in cls.parameter_candidates():
            instance = cls(**params)
            candidates.append({
                "template_id": template_id,
                "name": instance.name,
                "params": params,
            })
    return candidates
