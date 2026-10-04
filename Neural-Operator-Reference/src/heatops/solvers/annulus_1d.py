"""轴对称圆环热传导求解器（径向一维，纯 numpy/scipy）。

离散约定：

- 稳态 ``solve_annulus_steady``：内部为控制面通量守恒形式的中心差分
  （``(1/r)·d/dr(r·dT/dr) = 0``，二阶精度）；内边界 Neumann 一阶前向差分
  ``-k·(T1-T0)/dr = q_in``；外边界 Robin 一阶后向差分
  ``-k·(T_N-T_{N-1})/dr = h·(T_N - T_inf)``。
- 瞬态 ``solve_annulus_transient_cn``：Crank-Nicolson（时间二阶、无条件稳定），
  内部为控制体积守恒形式；边界按半控制体积（厚度 dr/2）推导，
  内边界源项 ``b[0] += q_in·2/dr``、外边界源项 ``b[-1] += h·T_inf·2/dr``。
  三对角系统一律用 ``scipy.linalg.solve_banded`` 求解，不做稠密求逆。

与原研究档案的差异（bug 修正）：档案瞬态脚本中边界行的扩散系数多了
``r_face/r_node`` 因子，与半控制体体积 ``r_node·dr/2`` 不一致，使其稳态极限的
壳层温降约为正确值的 2 倍（外壁温度偏差约 2%）。本实现改为半控制体一致系数
``2·k·r_face/(r_node·dr²)``，稳态时严格满足每弧度的能量守恒
``q_in·R_in = h·(T(R_out)-T_inf)·R_out``。
"""

from __future__ import annotations

import numpy as np

from ..problems.annulus import AnnulusParams
from ._tridiag import solve_tridiag, tridiag_matvec


def solve_annulus_steady(params: AnnulusParams, nr: int = 200) -> dict:
    """求解稳态轴对称径向温度场。

    Args:
        params: 圆环物理参数。
        nr: 径向网格点数。

    Returns:
        dict，含 ``r``（(nr,)，单位 m）、``T``（(nr,)，单位 °C）、``dr``（网格间距 m）。
    """
    r = np.linspace(params.R_in, params.R_out, nr)
    dr = r[1] - r[0]
    k, h, t_inf, q_in = params.k, params.h, params.T_inf, params.q_in

    lower = np.zeros(nr)
    diag = np.zeros(nr)
    upper = np.zeros(nr)
    b = np.zeros(nr)

    # 内边界（一阶前向）：(k/dr)·T0 - (k/dr)·T1 = q_in
    diag[0] = k / dr
    upper[0] = -k / dr
    b[0] = q_in

    # 内部点：控制面半径 r∓dr/2 的守恒中心差分
    r_w = r[1:-1] - 0.5 * dr
    r_e = r[1:-1] + 0.5 * dr
    lower[1:-1] = r_w / dr**2
    diag[1:-1] = -(r_w + r_e) / dr**2
    upper[1:-1] = r_e / dr**2

    # 外边界（一阶后向）：(k/dr)·T_{N-1} - (k/dr + h)·T_N = -h·T_inf
    lower[-1] = k / dr
    diag[-1] = -(k / dr + h)
    b[-1] = -h * t_inf

    T = solve_tridiag(lower, diag, upper, b)
    return {"r": r, "T": T, "dr": dr}


def solve_annulus_transient_cn(
    params: AnnulusParams,
    nr: int,
    dt: float,
    nt: int,
    q_in: float | None = None,
    T0: float | np.ndarray | None = None,
    save_every: int = 1,
) -> dict:
    """Crank-Nicolson 求解瞬态轴对称温度场。

    Args:
        params: 圆环物理参数。
        nr: 径向网格点数。
        dt: 时间步长，单位 s（C-N 无条件稳定；稳态极限与 dt 无关）。
        nt: 时间点数（含 t=0 初始帧）。
        q_in: 内边界热流密度覆盖值（W/m²），None 时使用 ``params.q_in``。
        T0: 初始温度场（°C），标量或 (nr,) 数组；None 表示均匀 ``T_inf``。
        save_every: 每隔多少步保存一帧（首末帧总是保存）。

    Returns:
        dict，含 ``r``（(nr,)，m）、``t``（(n_frames,)，s）、
        ``T_history``（(n_frames, nr)，°C）。
    """
    q = params.q_in if q_in is None else q_in
    r = np.linspace(params.R_in, params.R_out, nr)
    dr = r[1] - r[0]
    k, h, t_inf = params.k, params.h, params.T_inf
    g = params.rho * params.cp / dt  # 时间项系数，单位 J/(s·m³·K)

    # 内部点控制面系数：kw/ke 单位 J/(s·m³·K)
    r_w = r - 0.5 * dr
    r_e = r + 0.5 * dr
    kw = k * r_w / (r * dr**2)
    ke = k * r_e / (r * dr**2)

    imp_lower = np.zeros(nr)
    imp_diag = np.zeros(nr)
    imp_upper = np.zeros(nr)
    exp_lower = np.zeros(nr)
    exp_diag = np.zeros(nr)
    exp_upper = np.zeros(nr)

    # 内部行 i = 1..nr-2
    imp_lower[1:-1] = -0.5 * kw[1:-1]
    imp_diag[1:-1] = g + 0.5 * (kw[1:-1] + ke[1:-1])
    imp_upper[1:-1] = -0.5 * ke[1:-1]
    exp_lower[1:-1] = 0.5 * kw[1:-1]
    exp_diag[1:-1] = g - 0.5 * (kw[1:-1] + ke[1:-1])
    exp_upper[1:-1] = 0.5 * ke[1:-1]

    # 内边界行（半控制体 [r0, r0+dr/2]，体积/弧度 ≈ r0·dr/2）：
    # rho·cp·dT0/dt = 2·k·r_e/(r0·dr²)·(T1-T0) + 2·q_in/dr
    d0 = 2.0 * k * r_e[0] / (r[0] * dr**2)
    imp_diag[0] = g + 0.5 * d0
    imp_upper[0] = -0.5 * d0
    exp_diag[0] = g - 0.5 * d0
    exp_upper[0] = 0.5 * d0

    # 外边界行（半控制体 [rN-dr/2, rN]）：
    # rho·cp·dT_N/dt = 2·k·r_w/(rN·dr²)·(T_{N-1}-T_N) + 2·h·(T_inf-T_N)/dr
    d_n = 2.0 * k * r_w[-1] / (r[-1] * dr**2)
    h_row = 2.0 * h / dr
    imp_diag[-1] = g + 0.5 * (d_n + h_row)
    imp_lower[-1] = -0.5 * d_n
    exp_diag[-1] = g - 0.5 * (d_n + h_row)
    exp_lower[-1] = 0.5 * d_n

    # 边界源项（常值，C-N 两时间层平均后仍为原值），单位 J/(s·m³)
    src_inner = q * 2.0 / dr
    src_outer = h * t_inf * 2.0 / dr

    if T0 is None:
        T = np.full(nr, t_inf)
    else:
        T = np.broadcast_to(np.asarray(T0, dtype=np.float64), (nr,)).copy()

    frames = [T.copy()]
    steps = [0]
    for n in range(1, nt):
        b = tridiag_matvec(exp_lower, exp_diag, exp_upper, T)
        b[0] += src_inner
        b[-1] += src_outer
        T = solve_tridiag(imp_lower, imp_diag, imp_upper, b)
        if n % save_every == 0 or n == nt - 1:
            frames.append(T.copy())
            steps.append(n)

    return {"r": r, "t": np.asarray(steps) * dt, "T_history": np.asarray(frames)}


def check_energy_balance(T: np.ndarray, params: AnnulusParams) -> dict:
    """稳态能量平衡校核（单位轴向长度，单位 W/m）。

    输入热流 ``Q_in = q_in·2π·R_in``；输出热流 ``Q_out = h·(T(R_out)-T_inf)·2π·R_out``。

    Returns:
        dict，含 ``q_in_total``、``q_out_total``（W/m）与
        ``relative_error``（|Q_in-Q_out|/|Q_in|，无量纲）。
    """
    T = np.asarray(T, dtype=np.float64)
    q_in_total = params.q_in * 2.0 * np.pi * params.R_in
    q_out_total = params.h * (T[-1] - params.T_inf) * 2.0 * np.pi * params.R_out
    return {
        "q_in_total": float(q_in_total),
        "q_out_total": float(q_out_total),
        "relative_error": float(abs(q_in_total - q_out_total) / abs(q_in_total)),
    }


def check_physical_plausibility(T: np.ndarray, r: np.ndarray, params: AnnulusParams) -> dict:
    """物理合理性三件套：单调性、边界热流/ Robin 残差、温度范围。

    物理预期（内热源流入、外边界对流冷却）：温度沿 r 单调不增，
    内壁最热，全场不低于环境温度。

    Returns:
        dict：
        - ``is_monotonic_decreasing``：温度是否单调不增（允许 1e-9 °C 数值抖动）；
        - ``inner_flux_relative_error``：内边界前向差分热流与 q_in 的相对偏差；
        - ``outer_robin_relative_residual``：外边界 Robin 残差的相对量级；
        - ``above_ambient``：全场是否不低于 T_inf；
        - ``max_temp`` / ``min_temp``：温度范围（°C）。
    """
    T = np.asarray(T, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    dr = r[1] - r[0]

    q_inner_calc = -params.k * (T[1] - T[0]) / dr
    inner_err = abs(q_inner_calc - params.q_in) / params.q_in

    lhs = -params.k * (T[-1] - T[-2]) / dr
    rhs = params.h * (T[-1] - params.T_inf)
    robin_res = abs(lhs - rhs) / max(abs(lhs), abs(rhs))

    return {
        "is_monotonic_decreasing": bool(np.all(np.diff(T) <= 1e-9)),
        "inner_flux_relative_error": float(inner_err),
        "outer_robin_relative_residual": float(robin_res),
        "above_ambient": bool(np.all(T >= params.T_inf - 1e-9)),
        "max_temp": float(np.max(T)),
        "min_temp": float(np.min(T)),
    }
