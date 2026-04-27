"""
TDX indicator registry. Each registered indicator is a module that exposes
`name`, `label`, `pane`, `min_bars`, and `compute(df) -> IndicatorResult`.
"""

from __future__ import annotations

from types import ModuleType

from app.services.tdx.indicators import (
    didian_zuhe,
    dongli_xian,
    ma,
    maimai_henzhun,
    multi_kdj,
)

_INDICATORS: dict[str, ModuleType] = {
    ma.name: ma,
    multi_kdj.name: multi_kdj,
    dongli_xian.name: dongli_xian,
    maimai_henzhun.name: maimai_henzhun,
    didian_zuhe.name: didian_zuhe,
}


def all_indicators() -> list[ModuleType]:
    return list(_INDICATORS.values())


def get(name: str) -> ModuleType | None:
    return _INDICATORS.get(name)
