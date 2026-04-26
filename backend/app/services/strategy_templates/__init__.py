"""Strategy template registry — ML-driven strategies with cross-sectional features."""
from __future__ import annotations

from .ml_direct import (
    MLDirectATRSwing,
    MLDirectClassicGreedy,
    MLDirectDecision,
    MLDirectFastTurnover,
    MLDirectHighFreq,
    MLDirectHighFreqV2,
    MLDirectMaxFreq,
    MLDirectTop1TightLock,
    MLDirectTop2MidGreedy,
    MLDirectTop3HighGreedy,
    Mined426_1,
    Mined426_2,
    Mined426_3,
    Mined426_4,
    Mined426_5,
)

TEMPLATE_REGISTRY: dict[str, type] = {
    # 🏆 Mined-426 — output of 50-family × 20-param mining (2026-04-26).
    # All pass: win ≥ 75%, avg ≥ 6%, max_loss ≤ 10%, total ≥ 10%, IS+OOS.
    "426-1": Mined426_1,  # 准备+ATR (75% win, +9.6% avg, OOS 87% win)
    "426-2": Mined426_2,  # 准备+ATR tight (79% win, +6.3% avg, OOS 88% win)
    "426-3": Mined426_3,  # 急买+KDJ (87% win, +7.4% avg, OOS 91% win)
    "426-4": Mined426_4,  # KDJ+ATR (76% win, +12.3% avg, OOS 82% win)
    "426-5": Mined426_5,  # KDJ+ATR wide (76% win, +13.2% avg, OOS 81% win)
    # 🎯 优化版 trail (吃波段高点)
    "ml_direct_atr_swing": MLDirectATRSwing,            # ATR 自适应，单股 +36%，46% ≥10%
    "ml_direct_classic_greedy": MLDirectClassicGreedy,  # 经典 trail，78.6% 胜率
    # 🥇 High-quality (low frequency, very high accuracy)
    "ml_direct_top1_tight_lock": MLDirectTop1TightLock,
    "ml_direct_top2_mid_greedy": MLDirectTop2MidGreedy,
    "ml_direct_top3_high_greedy": MLDirectTop3HighGreedy,
    # ⚡ High-frequency
    "ml_direct_fast_turnover": MLDirectFastTurnover,
    "ml_direct_high_freq_v2": MLDirectHighFreqV2,
    "ml_direct_max_freq": MLDirectMaxFreq,
    # Legacy
    "ml_direct_high_freq": MLDirectHighFreq,
    "ml_direct": MLDirectDecision,
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
