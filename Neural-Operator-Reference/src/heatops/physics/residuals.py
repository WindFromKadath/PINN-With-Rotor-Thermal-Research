"""热传导方程的 autograd PDE 残差（1D 直角坐标 + 2D 圆环极坐标）。

1D 残差定义：``r = u_t - a·u_xx``，对 (x, t) 坐标网格做一阶/二阶自动微分。
与原研究档案 DataPINO 线的逐点实现一致（对全域格点计算，含边界与 t=0）。

2D 极坐标残差定义：``r = u_t - alpha_norm·(u_rr + u_r/r + u_θθ/r²)``，
对无量纲 (t*, r*, θ) 坐标网格 autograd（无量纲化约定见
``heatops.problems.annulus`` 与函数 docstring）。
"""

from __future__ import annotations

import torch
from torch.autograd import grad


def heat_residual_1d(pred: torch.Tensor, grid: torch.Tensor, a: float) -> torch.Tensor:
    """1D 热方程残差 ``u_t - a·u_xx``（无热源散热线）。

    Args:
        pred: 模型输出，形状 (B, nx, nt, 1)，必须由 ``grid`` 参与计算图得到
            （即前向传播时 grid 已 ``requires_grad_(True)``）。
        grid: (x, t) 坐标网格，形状 (B, nx, nt, 2)，末维通道顺序 (x, t)。
        a: 热扩散系数，单位 m²/s（来自 config / ``problems``）。

    Returns:
        与 ``pred`` 同形状 (B, nx, nt, 1) 的残差张量，计算图保持连通
        （``create_graph=True``，供损失反向传播）。

    边界裁切约定：autograd 导数在全域逐点定义，本函数**不裁切**，返回含边界
    与 t=0 的全域残差（档案约定，损失对全部格点取均值；边界与初时刻另由
    bc/ic 损失项约束）。若只需内部残差，由调用方切片
    ``r[:, 1:-1, 1:, :]``。
    """
    if not grid.requires_grad:
        raise ValueError(
            "grid 必须 requires_grad=True 且 pred 由该 grid 前向得到，"
            "否则无法对坐标求导（见 heatops.physics.losses.compute_loss_terms_1d）"
        )
    u_g = grad(pred, grid, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    u_x = u_g[..., 0:1]
    u_t = u_g[..., 1:2]
    u_xx = grad(u_x, grid, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][..., 0:1]
    return u_t - a * u_xx


def heat_residual_polar(pred: torch.Tensor, grid: torch.Tensor, alpha_norm: float) -> torch.Tensor:
    """2D 圆环极坐标无量纲热方程残差 ``u_t - alpha_norm · ∇²u``（无内热源线）。

    极坐标 Laplacian：``∇²u = u_rr + u_r / r + u_θθ / r²``。方程无源项——
    内边界热流 q 仅经 Neumann 边界进入（档案极坐标 PINO/PIDON 线的共同设计，
    见 ``heatops.cli.run_pidon_annulus`` docstring）。

    Args:
        pred: 模型输出，形状 (B, nt, nr, ntheta)，必须由 ``grid`` 参与计算图得到
            （即前向传播时 grid 已 ``requires_grad_(True)``）。
        grid: (t, r, θ) 坐标网格，形状 (B, nt, nr, ntheta, 3)，末维通道顺序 (t, r, θ)；
            坐标为**无量纲物理坐标**（如 ``t* = t / t_max``、``r* = r / R_out ∈
            [R_in*, 1]``、``θ ∈ [0, 2π]``），r 通道必须处处 > 0（圆环保证）。
        alpha_norm: 无量纲热扩散系数；取 ``t* = t / t_max``、``r* = r / R_out``
            时为 ``alpha · t_max / R_out²``（换算见 ``problems.annulus.AnnulusParams``）。

    Returns:
        与 ``pred`` 同形状 (B, nt, nr, ntheta) 的残差张量，计算图保持连通
        （``create_graph=True``，供损失反向传播）。

    与档案实现的差异（D103 量纲统一）：档案 ``datapino_2d_polar_train.py`` 在
    [0,1]³ 归一化网格上 autograd，再在损失内用链式法则逐维除以 Δr/Δθ/Δt 还原
    物理梯度，且 BC 项混用归一化量与物理量纲；本库把坐标缩放集中在网格构造
    （CLI 按 ``problems`` 的无量纲化生成 grid），grid 直接携带无量纲物理坐标，
    autograd 一阶/二阶导数即无量纲梯度，残差与 BC 项同在无量纲域计算，
    无逐维链式还原与量纲混用。``r*`` 不归一化到 [0,1] 是有意为之：
    ``1/r*`` 与 ``1/r*²`` 项需要无量纲半径的真实值。

    边界裁切约定与 ``heat_residual_1d`` 一致：不裁切，返回全域残差；
    PDE 点采样由 ``heatops.physics.losses`` 的采样工具负责。
    """
    if not grid.requires_grad:
        raise ValueError(
            "grid 必须 requires_grad=True 且 pred 由该 grid 前向得到，"
            "否则无法对坐标求导（见 heatops.physics.losses.compute_loss_terms_polar）"
        )
    if grid.shape[-1] != 3:
        raise ValueError(f"grid 末维应为 3 个坐标通道 (t, r, θ)，实际形状 {tuple(grid.shape)}")
    g = grad(pred, grid, grad_outputs=torch.ones_like(pred), create_graph=True)[0]
    u_t = g[..., 0]
    u_r = g[..., 1]
    u_th = g[..., 2]
    u_rr = grad(u_r, grid, grad_outputs=torch.ones_like(u_r), create_graph=True)[0][..., 1]
    u_thth = grad(u_th, grid, grad_outputs=torch.ones_like(u_th), create_graph=True)[0][..., 2]
    r = grid[..., 1]
    laplacian = u_rr + u_r / r + u_thth / r**2
    return u_t - alpha_norm * laplacian
