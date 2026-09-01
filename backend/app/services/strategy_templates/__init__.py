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
    Mined426V2_6, Mined426V2_7, Mined426V2_8, Mined426V2_9, Mined426V2_10,
)
from .maimai_filter import (
    MaimaiFilter30, MaimaiFilter40, MaimaiFilter50, MaimaiFilter55,
    MaimaiFilterBroad40, MaimaiFilterBroad50,
    MaimaiPure40, MaimaiPure50,
)
from .maimai_zhun import MaimaiZhun
from .reversal_filter import Reversal30, Reversal40, Reversal50, Reversal55, Reversal60
from .tdx_classics import TDXTripleGold, TDXMACDGoldPit, TDXExpmaTrio
from .diao_di import DiaoDiPure, DiaoDi20, DiaoDi30, DiaoDiDiv

TEMPLATE_REGISTRY: dict[str, type] = {
    # 🏆 Mined-426 V2 — top 10 from re-mining on 2031 stocks (2026-04-27).
    # 6 distinct concepts (KDJ x2, 动力线 x2, SMA x2, Breakout x2, RSI, BB).
    "426-1":  Mined426V2_1,   # 动力线 上穿 0.5     | IS 84.6%/+6.7% OOS 87.8%
    "426-2":  Mined426V2_2,   # KDJ K/D 金叉       | IS 76.9%/+7.6% OOS 80.7%
    "426-3":  Mined426V2_3,   # RSI 上穿 30        | IS 75.1%/+7.3% OOS 79.3%
    "426-4":  Mined426V2_4,   # KDJ J 上穿 20      | IS 80.8%/+9.6% OOS 85.7%/+10.7%
    "426-5":  Mined426V2_5,   # 动力线 上穿 1.0     | IS 78.6%/+9.0% OOS 83.2%
    "426-6":  Mined426V2_6,   # SMA 5/20 金叉      | IS 94.9%/+7.3% OOS 91.7%/+16.5%
    "426-7":  Mined426V2_7,   # 20日高点突破        | IS 77.1%/+7.2% OOS 77.3%/+11.7%
    "426-8":  Mined426V2_8,   # SMA 5/20 慢版      | IS 90.6%/+8.7% OOS 93.8%/+23.6%
    "426-9":  Mined426V2_9,   # 50日高点突破        | IS 80.4%/+7.9% OOS 77.8%/+11.6%
    "426-10": Mined426V2_10,  # BB 上轨突破        | IS 77.6%/+6.8% OOS 82.1%/+13.3%
    # 🎯 买卖很准 合并版 — 纯指标边沿信号 (与副图指标同一实现, 2026-08-16)
    "mmhz":        MaimaiZhun,             # 买: 买线非0→0; 卖: 卖线<100→100
    # 🎯 买卖很准 ML-filter — 87K 信号训练，贪婪 ATR 退出
    "mm-30":       MaimaiFilter30,         # 高频版 thr 0.30
    "mm-40":       MaimaiFilter40,         # 中频 thr 0.40
    "mm-50":       MaimaiFilter50,         # sweet spot thr 0.50
    "mm-55":       MaimaiFilter55,         # 严格 thr 0.55
    "mm-broad-40": MaimaiFilterBroad40,    # 5 信号 OR 触发 + thr 0.40
    "mm-broad-50": MaimaiFilterBroad50,    # 5 信号 OR 触发 + thr 0.50
    # 🔄 纯买卖很准双向 — 买入用 mm 买信号关闭，退出用 mm 卖信号触发
    "mm-pure-40":  MaimaiPure40,           # 买/卖都来自 买卖很准 + ATR backup
    "mm-pure-50":  MaimaiPure50,           # 严格版
    # 🔄 多源反转融合 — 买卖很准 + 动力线 + KDJ + RSI 全部，345K 信号训练
    "rev-30": Reversal30,    # thr 0.30 高频版 (~10/yr)
    "rev-40": Reversal40,    # thr 0.40 中频版
    "rev-50": Reversal50,    # thr 0.50 sweet spot — 目标 10/年
    "rev-55": Reversal55,    # thr 0.55 严格 + 宽 trail
    "rev-60": Reversal60,    # thr 0.60 极严
    # 📚 通达信社区经典策略（无 ML，纯指标信号 + 贪婪 ATR 退出）
    "tdx-triple-gold":  TDXTripleGold,    # 三金叉共振 (MA+MACD+KDJ)
    "tdx-macd-pit":     TDXMACDGoldPit,   # MACD 黄金坑抄底
    "tdx-expma-trio":   TDXExpmaTrio,     # EXPMA 三步擒牛
    # 🎯 抄底逃顶 — Pine Momentum Line 移植
    "dt-30":   DiaoDi30,      # 🏆 深 V + ml_score ≥ 0.30 (rank 12/42, 80% win)
    "dt-20":   DiaoDi20,      # 深 V + ml_score ≥ 0.20 (中频版)
    "dt-pure": DiaoDiPure,    # 深 V 反转无 ML (高频低准)
    "dt-div":  DiaoDiDiv,     # 底背驰 (实验性，样本少)
    # 🌀 缠论 (Chan Theory) — 含包关系 + 分型 + 笔 + MACD 背驰
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
