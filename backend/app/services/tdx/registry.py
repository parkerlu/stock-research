"""
TDX indicator registry. Each registered indicator is a module that exposes
`name`, `label`, `pane`, `min_bars`, and `compute(df) -> IndicatorResult`.
"""

from __future__ import annotations

from types import ModuleType

from app.services.tdx.indicators import (
    chande_momentum,
    chao_di_tao_ding,
    didian_zuhe,
    dongli_xian,
    dual_kdj,
    ma,
    maimai_henzhun,
    multi_kdj,
    pretty_good_oscillator,
    relative_volatility_index,
    zhuli_lasheng,
    zhuli_lasheng_tiqian,
)

_INDICATORS: dict[str, ModuleType] = {
    ma.name: ma,
    multi_kdj.name: multi_kdj,
    dongli_xian.name: dongli_xian,
    # 虚拟盘当前策略, 与 strategy_templates 同源
    dual_kdj.name: dual_kdj,
    maimai_henzhun.name: maimai_henzhun,
    didian_zuhe.name: didian_zuhe,
    chao_di_tao_ding.name: chao_di_tao_ding,
    zhuli_lasheng.name: zhuli_lasheng,
    zhuli_lasheng_tiqian.name: zhuli_lasheng_tiqian,
    # Pine 移植 — 24 个候选里净边际前 3 (已扣随机入场基线, 见各模块 docstring)
    relative_volatility_index.name: relative_volatility_index,
    pretty_good_oscillator.name: pretty_good_oscillator,
    chande_momentum.name: chande_momentum,
}


def all_indicators() -> list[ModuleType]:
    return list(_INDICATORS.values())


def get(name: str) -> ModuleType | None:
    return _INDICATORS.get(name)
