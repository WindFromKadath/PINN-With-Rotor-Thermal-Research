"""一维热传导（带插值源项）Crank-Nicolson 求解器，Robin 边界虚点消元。

方程：``u_t = alpha * u_xx + s(x)``，其中 ``s(x) = source_scale·v(x) + constant_source``，
``v(x)`` 由传感点 ``(source_points, source_values)`` 经 ``interp1d`` 插值得到。

边界为 Newton 冷却约定（``beta = h/k ≥ 0``，单位 1/m，n 为外法向，
``du/dn = -beta·(u - u0)``）：

- 左端：``du/dx = beta_left·(u - u0_left)``；
- 右端：``du/dx = -beta_right·(u - u0_right)``。

虚点代入中心差分后与 C-N 时间离散联立，得到三对角系统，
隐式侧用 ``scipy.linalg.solve_banded`` 求解，显式侧用三对角乘积求值。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.interpolate import interp1d

from ._tridiag import solve_tridiag, tridiag_matvec


def solve_adr_robin(
    xmin: float,
    xmax: float,
    tmin: float,
    tmax: float,
    alpha: float,
    source_values: np.ndarray,
    source_points: np.ndarray,
    beta_left: float,
    u0_left: float,
    beta_right: float,
    u0_right: float,
    u_initial: float | Callable[[np.ndarray], np.ndarray],
    source_scale: float = 30.0,
    constant_source: float = 100.0,
    nx: int = 101,
    nt: int = 101,
    interp_kind: str = "linear",
) -> dict:
    """求解一维热传导方程（C-N + Robin 虚点消元）。

    Args:
        xmin, xmax: 空间区间端点，单位 m。
        tmin, tmax: 时间区间端点，单位 s。
        alpha: 热扩散系数，单位 m²/s。
        source_values: 源函数在传感点处的取值 v（无量纲或约定单位）。
        source_points: 传感点坐标，单位 m。
        beta_left, beta_right: 左右端 Robin 系数，单位 1/m。
        u0_left, u0_right: 左右端环境温度，单位 °C。
        u_initial: 初始温度场（°C），标量或 ``callable(x) -> ndarray``。
        source_scale: 源项传感值放大系数，单位 °C/s。
        constant_source: 源项常值部分，单位 °C/s。
        nx, nt: 空间/时间网格点数。
        interp_kind: ``interp1d`` 插值方式。

    Returns:
        dict，含 ``x``（(nx,)，m）、``t``（(nt,)，s）、``u``（(nx, nt)，°C）。
    """
    x = np.linspace(xmin, xmax, nx)
    t = np.linspace(tmin, tmax, nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    r = alpha * dt / (2.0 * dx**2)
    g_left = dx * beta_left
    g_right = dx * beta_right

    source_points = np.asarray(source_points, dtype=np.float64).ravel()
    source_values = np.asarray(source_values, dtype=np.float64).ravel()
    v_interp = interp1d(
        source_points, source_values, kind=interp_kind,
        bounds_error=False, fill_value="extrapolate",
    )
    s = source_scale * v_interp(x) + constant_source  # °C/s

    u = u_initial(x) if callable(u_initial) else np.full(nx, float(u_initial))
    u = np.asarray(u, dtype=np.float64)
    u_all = np.zeros((nx, nt))
    u_all[:, 0] = u

    # 隐式侧 A（三对角）：内部 (1+2r) 对角、-r 旁对角；
    # 边界行代入虚点 u_{-1} = u_1 - 2dx·beta·(u_0-u0)（左端，右端类似）
    imp_lower = np.full(nx, -r)
    imp_diag = np.full(nx, 1.0 + 2.0 * r)
    imp_upper = np.full(nx, -r)
    imp_diag[0] = 1.0 + 2.0 * r * (1.0 + g_left)
    imp_upper[0] = -2.0 * r
    imp_diag[-1] = 1.0 + 2.0 * r * (1.0 + g_right)
    imp_lower[-1] = -2.0 * r

    # 显式侧 B（三对角）
    exp_lower = np.full(nx, r)
    exp_diag = np.full(nx, 1.0 - 2.0 * r)
    exp_upper = np.full(nx, r)
    exp_diag[0] = 1.0 - 2.0 * r * (1.0 + g_left)
    exp_upper[0] = 2.0 * r
    exp_diag[-1] = 1.0 - 2.0 * r * (1.0 + g_right)
    exp_lower[-1] = 2.0 * r

    # Robin 环境温度引入的常值源项（两时间层各贡献 2·r·g·u0）
    rhs_left = 4.0 * r * g_left * u0_left
    rhs_right = 4.0 * r * g_right * u0_right

    for n in range(nt - 1):
        rhs = tridiag_matvec(exp_lower, exp_diag, exp_upper, u) + dt * s
        rhs[0] += rhs_left
        rhs[-1] += rhs_right
        u = solve_tridiag(imp_lower, imp_diag, imp_upper, rhs)
        u_all[:, n + 1] = u

    return {"x": x, "t": t, "u": u_all}
