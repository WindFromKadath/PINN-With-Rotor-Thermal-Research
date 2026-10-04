"""一维热传导显式差分（FTCS）数据生成求解器，Dirichlet 零边界。

两条原研究档案线的重写：

- 散热线 ``solve_heat_1d_explicit``：``u_t = a·u_xx``，随机初始条件，
  CFL 自动步长（理论上限 dx²/(2a) 的 0.8 倍），均匀帧采样并精确对齐 T_max；
- 带源加热线 ``solve_heat_1d_source_explicit``：``u_t = a·u_xx + f(x)``，
  帧间隔可能超过 CFL 上限，内部自动划分为等长子步（每帧内子步数
  ``ceil(dt_frame / (0.9·dx²/(2a)))``）。

初始条件由 ``ic_sampler`` 注入，协议为 ``callable(x, rng) -> ndarray``，
采样器见 ``heatops.fields.initial_conditions``；种子经 ``numpy Generator`` 显式控制。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

_ICSampler = Callable[[np.ndarray, np.random.Generator], np.ndarray]


def _check_bc(bc: str) -> None:
    if bc != "dirichlet0":
        raise ValueError(f"不支持的边界条件：{bc!r}（当前仅实现 'dirichlet0'）")


def solve_heat_1d_explicit(
    a: float,
    nx: int,
    nt: int,
    T_max: float,
    ic_sampler: _ICSampler,
    n_samples: int,
    seed: int | None = None,
    bc: str = "dirichlet0",
) -> dict:
    """散热线：显式 FTCS 求解 ``u_t = a·u_xx``，两端 Dirichlet 零边界。

    Args:
        a: 热扩散系数，单位 m²/s。
        nx: 空间网格点数（区间 [0, 1] m）。
        nt: 保存的时间帧数（含 t=0）。
        T_max: 物理时长，单位 s。
        ic_sampler: 初始条件采样器 ``(x, rng) -> (nx,)``，单位 °C。
        n_samples: 样本数。
        seed: 随机种子（None 为不可复现）。
        bc: 边界条件，当前仅支持 ``'dirichlet0'``。

    Returns:
        dict，含 ``u0``（(B, nx)，°C）、``u``（(B, nx, nt)，°C）、
        ``x``（(nx,)，m）、``t``（(nt,)，s）。
    """
    _check_bc(bc)
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]

    dt_safe = 0.8 * dx**2 / (2.0 * a)  # CFL 安全步长
    # 步数取到既能满足稳定性又能容纳 nt 个不重复帧索引
    n_steps = max(int(np.ceil(T_max / dt_safe)), nt - 1)
    dt = T_max / n_steps  # 重新调整，保证精确到达 T_max
    frame_idx = np.rint(np.linspace(0, n_steps, nt)).astype(np.int64)
    save_mask = np.zeros(n_steps + 1, dtype=bool)
    save_mask[frame_idx] = True

    rng = np.random.default_rng(seed)
    u0_all = np.empty((n_samples, nx))
    u_all = np.empty((n_samples, nx, nt))

    for m in range(n_samples):
        u = np.asarray(ic_sampler(x, rng), dtype=np.float64).copy()
        u[0] = 0.0
        u[-1] = 0.0  # Dirichlet 零边界与初始条件的相容性
        u0_all[m] = u

        frames = np.empty((nx, nt))
        col = 0
        if save_mask[0]:
            frames[:, col] = u
            col += 1
        for n in range(1, n_steps + 1):
            lap = np.zeros(nx)
            lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
            u = u + a * dt * lap
            u[0] = 0.0
            u[-1] = 0.0
            if save_mask[n]:
                frames[:, col] = u
                col += 1
        u_all[m] = frames

    return {"u0": u0_all, "u": u_all, "x": x, "t": frame_idx * dt}


def solve_heat_1d_source_explicit(
    a: float,
    nx: int,
    nt: int,
    dt_frame: float,
    source: np.ndarray,
    ic_sampler: _ICSampler,
    n_samples: int,
    seed: int | None = None,
    bc: str = "dirichlet0",
) -> dict:
    """带源加热线：显式 FTCS 求解 ``u_t = a·u_xx + f(x)``，两端 Dirichlet 零边界。

    Args:
        a: 热扩散系数，单位 m²/s。
        nx: 空间网格点数（区间 [0, 1] m）。
        nt: 保存的时间帧数（含 t=0）。
        dt_frame: 帧间隔，单位 s；超过 CFL 上限时自动子步划分。
        source: 热源强度分布 f(x)，形状 (nx,)，单位 °C/s。
        ic_sampler: 初始条件采样器 ``(x, rng) -> (nx,)``，单位 °C。
        n_samples: 样本数。
        seed: 随机种子（None 为不可复现）。
        bc: 边界条件，当前仅支持 ``'dirichlet0'``。

    Returns:
        dict，含 ``u0``（(B, nx)，°C）、``u``（(B, nx, nt)，°C）、
        ``x``（(nx,)，m）、``t``（(nt,)，s）、``steps_per_frame``（每帧子步数）。
    """
    _check_bc(bc)
    x = np.linspace(0.0, 1.0, nx)
    dx = x[1] - x[0]
    f = np.asarray(source, dtype=np.float64)
    if f.shape != (nx,):
        raise ValueError(f"source 形状应为 ({nx},)，实际为 {f.shape}")

    dt_limit = 0.9 * dx**2 / (2.0 * a)  # CFL 安全线（理论上限的 0.9 倍）
    steps_per_frame = max(1, int(np.ceil(dt_frame / dt_limit)))
    dt_sub = dt_frame / steps_per_frame

    rng = np.random.default_rng(seed)
    u0_all = np.empty((n_samples, nx))
    u_all = np.empty((n_samples, nx, nt))

    for m in range(n_samples):
        u = np.asarray(ic_sampler(x, rng), dtype=np.float64).copy()
        u[0] = 0.0
        u[-1] = 0.0
        u0_all[m] = u

        frames = np.empty((nx, nt))
        frames[:, 0] = u
        for j in range(1, nt):
            for _ in range(steps_per_frame):
                lap = np.zeros(nx)
                lap[1:-1] = (u[2:] - 2.0 * u[1:-1] + u[:-2]) / dx**2
                u = u + dt_sub * (a * lap + f)
                u[0] = 0.0
                u[-1] = 0.0
            frames[:, j] = u
        u_all[m] = frames

    return {
        "u0": u0_all,
        "u": u_all,
        "x": x,
        "t": np.arange(nt) * dt_frame,
        "steps_per_frame": steps_per_frame,
    }
