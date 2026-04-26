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
)
from .mined_426_v2 import (
    Mined426V2_1, Mined426V2_2, Mined426V2_3, Mined426V2_4, Mined426V2_5,
)

TEMPLATE_REGISTRY: dict[str, type] = {
    # 🏆 Mined-426 V2 — diverse top 5 (50 concept-orthogonal families × 20 LHS).
    # 5 conceptually different signal sources; pairwise Jaccard ≤ 17.7%.
    "426-1": Mined426V2_1,  # KDJ K/D 金叉  | IS 75.9% win, +12.0% avg | OOS 83.1%
    "426-2": Mined426V2_2,  # BB 下轨反弹    | IS 75.4% win, +6.4% avg  | OOS 83.6%
    "426-3": Mined426V2_3,  # 动力线 上穿 0.5 | IS 83.4% win, +6.7% avg  | OOS 86.3%
    "426-4": Mined426V2_4,  # RSI 上穿 30    | IS 91.0% win, +6.0% avg  | OOS 89.9%
    "426-5": Mined426V2_5,  # 买卖很准 准备首发| IS 77.9% win, +6.6% avg  | OOS 88.4%
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
