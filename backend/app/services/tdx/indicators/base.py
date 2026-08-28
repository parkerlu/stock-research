"""
Common data structures for TDX indicator output.

These dataclasses define the shape of any TDX indicator's computed result,
designed to cover the four visual element types used across TDX formulas:
plain lines, horizontal reference lines, filled bands (STICKLINE), and
markers (DRAWICON).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol

import pandas as pd


PaneType = Literal["main", "sub"]
IconType = Literal["dot", "triangle_up", "triangle_down"]
RenderType = Literal["line", "bar"]


@dataclass
class IndicatorLine:
    name: str
    values: list[float | None]
    color: str
    thickness: int = 1
    # "bar" 渲染成从 0 起的实心柱（对应 TDX 的柱状副图）。同一序列在图例里
    # 照常显示数值，只是画法不同。多条 bar 按声明顺序叠画，后画的盖在前面，
    # 所以由高到低声明可以得到分层色带。
    render: RenderType = "line"


@dataclass
class IndicatorHLine:
    name: str
    value: float
    color: str
    dashed: bool = False


@dataclass
class IndicatorBand:
    # A band is a set of rectangles at the given timestamps, each bar-wide,
    # spanning y1..y2. Used to render TDX STICKLINE output.
    timestamps: list[int]
    y1: float
    y2: float
    color: str
    opacity: float = 0.3


@dataclass
class IndicatorMarker:
    timestamp: int
    value: float
    color: str
    icon: IconType = "dot"


@dataclass
class IndicatorResult:
    name: str
    label: str
    pane: PaneType = "sub"
    y_axis_range: tuple[float, float] | None = None
    timestamps: list[int] = field(default_factory=list)
    lines: list[IndicatorLine] = field(default_factory=list)
    hlines: list[IndicatorHLine] = field(default_factory=list)
    bands: list[IndicatorBand] = field(default_factory=list)
    markers: list[IndicatorMarker] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class IndicatorSpec(Protocol):
    """Interface every registered TDX indicator must implement."""

    name: str
    label: str
    pane: PaneType
    min_bars: int

    def compute(self, df: pd.DataFrame) -> IndicatorResult:
        ...


def series_to_json(s: pd.Series) -> list[float | None]:
    """Convert a numeric series to a list where NaN becomes None (JSON-safe)."""
    return [None if pd.isna(v) else float(v) for v in s.tolist()]
