"""heatops.problems — 物理问题定义：参数 dataclass、解析解与无量纲化换算。"""

from .annulus import (
    AnnulusParams,
    analytical_steady_state,
    analytical_steady_state_dimensionless,
)
from .cartesian import (
    CartesianHeatParams,
    HeatLineParams,
    heat_line_cooling,
    heat_line_heating,
)

__all__ = [
    "AnnulusParams",
    "analytical_steady_state",
    "analytical_steady_state_dimensionless",
    "CartesianHeatParams",
    "HeatLineParams",
    "heat_line_cooling",
    "heat_line_heating",
]
