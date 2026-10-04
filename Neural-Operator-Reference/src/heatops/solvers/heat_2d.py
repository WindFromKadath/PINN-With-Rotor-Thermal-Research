"""2D 单位正方形热传导显式求解器，Robin 边界虚点法（二阶精度）。

方程：``u_t = alpha * Δu + f(x, y)``，域 ``[0, 1]²``，网格约定 ``indexing='ij'``
（数组第一维对应 x 轴）。四边 Robin/Newton 冷却 ``du/dn = -beta·(u - u0)``，
虚点消元后边界点的法向二阶差分为
``(2·u_inner - 2·u_b + 2·dx·beta·(u0 - u_b)) / dx²``，角点两个方向同时修正。

稳定性：显式 FTCS 理论上限 ``dt ≤ dx²/(4·alpha)``，考虑虚点边界收紧取
``dt_max = 0.22·dx²/alpha``；期望帧间隔超限时自动加密时间步（0.9 倍安全余量）
并重新调整 dt 精确对齐 T_max，保存帧按总步数均匀抽取。
"""

from __future__ import annotations

import numpy as np


def solve_heat_2d_robin(
    alpha: float,
    f_grid: np.ndarray,
    u0: float,
    n: int,
    T_max: float,
    nt_save: int,
    beta: float,
) -> dict:
    """显式 FTCS + 虚点法 Robin 边界求解 2D 热传导。

    Args:
        alpha: 热扩散系数，单位 m²/s。
        f_grid: 热源场，形状 (n, n)（第一维为 x），单位 °C/s。
        u0: 初始温度与环境温度（°C），初始场均匀。
        n: 每边网格点数。
        T_max: 物理时长，单位 s。
        nt_save: 保存帧数（含 t=0）。
        beta: Robin 系数 h/k，单位 1/m。

    Returns:
        dict，含 ``u``（(nt_save, n, n)，°C）、``t``（(nt_save,)，s）、
        ``x``/``y``（(n,)，m）、``dt``（实际步长，s）。

    Raises:
        FloatingPointError: 迭代中出现 NaN/Inf（通常为 alpha 或 f_grid 过大导致发散）。
    """
    f = np.asarray(f_grid, dtype=np.float64)
    if f.shape != (n, n):
        raise ValueError(f"f_grid 形状应为 ({n}, {n})，实际为 {f.shape}")

    dx = 1.0 / (n - 1)
    dt_max_allowed = 0.22 * dx**2 / alpha
    suggested_dt = T_max / (nt_save - 1)
    if suggested_dt > dt_max_allowed:
        dt = 0.9 * dt_max_allowed
        total_steps = int(np.ceil(T_max / dt))
        dt = T_max / total_steps  # 重新调整，精确到达 T_max
    else:
        dt = suggested_dt
        total_steps = nt_save - 1

    # 均匀帧索引（间距 ≥ 1，必不重复，含首末帧）
    save_idx = np.rint(np.linspace(0, total_steps, nt_save)).astype(np.int64)
    save_set = set(save_idx.tolist())

    u = np.full((n, n), float(u0))
    frames = [u.copy()]

    c = 2.0 * dx * beta  # 虚点修正公共因子
    adt = alpha * dt

    for step in range(1, total_steps + 1):
        u_next = u.copy()

        # 内部点
        u_next[1:-1, 1:-1] = u[1:-1, 1:-1] + dt * f[1:-1, 1:-1] + adt * (
            (u[2:, 1:-1] - 2.0 * u[1:-1, 1:-1] + u[:-2, 1:-1]) / dx**2
            + (u[1:-1, 2:] - 2.0 * u[1:-1, 1:-1] + u[1:-1, :-2]) / dx**2
        )

        # 左 (i=0) / 右 (i=n-1) 边界：x 方向虚点修正，y 方向常规中心差分
        u_next[0, 1:-1] = u[0, 1:-1] + dt * f[0, 1:-1] + adt * (
            (2.0 * u[1, 1:-1] - 2.0 * u[0, 1:-1] + c * (u0 - u[0, 1:-1])) / dx**2
            + (u[0, 2:] - 2.0 * u[0, 1:-1] + u[0, :-2]) / dx**2
        )
        u_next[-1, 1:-1] = u[-1, 1:-1] + dt * f[-1, 1:-1] + adt * (
            (2.0 * u[-2, 1:-1] - 2.0 * u[-1, 1:-1] + c * (u0 - u[-1, 1:-1])) / dx**2
            + (u[-1, 2:] - 2.0 * u[-1, 1:-1] + u[-1, :-2]) / dx**2
        )

        # 下 (j=0) / 上 (j=n-1) 边界：y 方向虚点修正，x 方向常规中心差分
        u_next[1:-1, 0] = u[1:-1, 0] + dt * f[1:-1, 0] + adt * (
            (u[2:, 0] - 2.0 * u[1:-1, 0] + u[:-2, 0]) / dx**2
            + (2.0 * u[1:-1, 1] - 2.0 * u[1:-1, 0] + c * (u0 - u[1:-1, 0])) / dx**2
        )
        u_next[1:-1, -1] = u[1:-1, -1] + dt * f[1:-1, -1] + adt * (
            (u[2:, -1] - 2.0 * u[1:-1, -1] + u[:-2, -1]) / dx**2
            + (2.0 * u[1:-1, -2] - 2.0 * u[1:-1, -1] + c * (u0 - u[1:-1, -1])) / dx**2
        )

        # 四角：两个方向均虚点修正（ni/nj 为内向邻居索引）
        for i, j in ((0, 0), (0, -1), (-1, 0), (-1, -1)):
            ni = 1 if i == 0 else -2
            nj = 1 if j == 0 else -2
            lap_x = (2.0 * u[ni, j] - 2.0 * u[i, j] + c * (u0 - u[i, j])) / dx**2
            lap_y = (2.0 * u[i, nj] - 2.0 * u[i, j] + c * (u0 - u[i, j])) / dx**2
            u_next[i, j] = u[i, j] + dt * f[i, j] + adt * (lap_x + lap_y)

        u = u_next

        if not np.all(np.isfinite(u)):
            raise FloatingPointError(
                f"第 {step} 步出现 NaN/Inf，请检查 alpha、f_grid 或网格参数"
            )
        if step in save_set:
            frames.append(u.copy())

    x = np.linspace(0.0, 1.0, n)
    return {
        "u": np.asarray(frames),
        "t": save_idx * dt,
        "x": x,
        "y": x.copy(),
        "dt": dt,
    }
