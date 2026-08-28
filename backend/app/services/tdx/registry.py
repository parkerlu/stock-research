"""
TDX indicator registry. Each registered indicator is a module that exposes
`name`, `label`, `pane`, `min_bars`, and `compute(df) -> IndicatorResult`.
"""

from __future__ import annotations

from types import ModuleType

from app.services.tdx.indicators import (
    chao_di_tao_ding,
    didian_zuhe,
    dongli_xian,
    ma,
    maimai_henzhun,
    multi_kdj,
    zhuli_lasheng,
    zhuli_lasheng_tiqian,
)

_INDICATORS: dict[str, ModuleType] = {
    ma.name: ma,
    multi_kdj.name: multi_kdj,
    dongli_xian.name: dongli_xian,
    maimai_henzhun.name: maimai_henzhun,
    didian_zuhe.name: didian_zuhe,
    chao_di_tao_ding.name: chao_di_tao_ding,
    zhuli_lasheng.name: zhuli_lasheng,
    zhuli_lasheng_tiqian.name: zhuli_lasheng_tiqian,
}


def all_indicators() -> list[ModuleType]:
    return list(_INDICATORS.values())


def get(name: str) -> ModuleType | None:
    return _INDICATORS.get(name)
