"""
TDX indicator registry. Each registered indicator is a module that exposes
`name`, `label`, `pane`, `min_bars`, and `compute(df) -> IndicatorResult`.
"""

from __future__ import annotations

from types import ModuleType

from app.services.tdx.indicators import multi_kdj

_INDICATORS: dict[str, ModuleType] = {
    multi_kdj.name: multi_kdj,
}


def all_indicators() -> list[ModuleType]:
    return list(_INDICATORS.values())


def get(name: str) -> ModuleType | None:
    return _INDICATORS.get(name)
