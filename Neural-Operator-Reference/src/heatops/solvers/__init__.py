"""heatops.solvers — 纯 numpy/scipy 的数值求解器（FDM / C-N，函数式接口）。"""

from .adr_1d import solve_adr_robin
from .annulus_1d import (
    check_energy_balance,
    check_physical_plausibility,
    solve_annulus_steady,
    solve_annulus_transient_cn,
)
from .heat_1d import solve_heat_1d_explicit, solve_heat_1d_source_explicit
from .heat_2d import solve_heat_2d_robin

__all__ = [
    "solve_adr_robin",
    "solve_annulus_steady",
    "solve_annulus_transient_cn",
    "check_energy_balance",
    "check_physical_plausibility",
    "solve_heat_1d_explicit",
    "solve_heat_1d_source_explicit",
    "solve_heat_2d_robin",
]
