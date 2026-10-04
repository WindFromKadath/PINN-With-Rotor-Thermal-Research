"""热传导算子学习的损失项与加权组合（1D 直角坐标线 + 2D 圆环极坐标线）。

1D 线四项损失与原研究档案 DataPINO 线 ``compute_all_losses`` 的语义一致：

- ``data``: 逐点 MSE ``mean((pred - u_true)²)``；
- ``pde``: 输出场有限差分残差 ``mean((u_t - a·u_xx)²)``，仅取时空内部点；
- ``ic``: 初始条件 MSE ``mean((pred[:, :, 0, 0] - u0)²)``（t=0 帧）；
- ``bc``: Dirichlet 零边界 MSE ``mean(pred[:, 0, :, :]²) + mean(pred[:, -1, :, :]²)``。

三种训练变体（纯数据 / 纯物理 / 混合）由 ``LossWeights`` 的不同取值产生，
共用同一模型与训练循环。

极坐标线（``*_polar`` 函数，张量约定 (B, nt, nr, ntheta)，坐标通道 (t, r, θ)）
重写自档案 ``datapino_2d_polar_train.py``，按 D103 统一在无量纲域计算：

- ``data``: 掩码加权 MSE（极坐标网格铺满圆环域时退化为全域 MSE）；
- ``pde``: 极坐标残差 MSE，支持随机点采样（采样率 config 化，档案为 24GB
  显存妥协只采样 20% 随机点）；
- ``ic``: t*=0 帧与零初始场（均匀 T_inf 的无量纲形式）的 MSE；
- ``bc_inner``: 无量纲 Neumann ``-dT*/dr* = q*`` ⇒ ``T_r* + q* = 0``；
- ``bc_outer``: 无量纲 Robin ``-dT*/dr* = Bi·T*`` ⇒ ``T_r* + Bi·T* = 0``；
- ``periodic``: θ 两端场值与 T_θ 对齐（档案约定：即使 FNO 谱卷积本身周期，
  显式约束可加速收敛并减少数值漂移）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch.autograd import grad

from heatops.physics.residuals import heat_residual_1d, heat_residual_polar


@dataclass(frozen=True)
class LossWeights:
    """四项损失的权重（纯数据训练跳过 PDE 差分，日志该项记录为零）。

    Attributes:
        data: 数据项权重（纯数据 FNO 变体为 1，其余 0）。
        pde: PDE 残差项权重（档案混合约定 10）。
        ic: 初始条件项权重（档案约定 200）。
        bc: 边界条件项权重（档案约定 100）。
    """

    data: float = 1.0
    pde: float = 0.0
    ic: float = 0.0
    bc: float = 0.0


def data_loss_1d(pred: torch.Tensor, u_true: torch.Tensor) -> torch.Tensor:
    """数据项：``mean((pred - u_true)²)``，标量。"""
    return torch.mean((pred - u_true) ** 2)


def ic_loss_1d(pred: torch.Tensor, u0: torch.Tensor) -> torch.Tensor:
    """初始条件项：t=0 帧与 u0 的 MSE，标量。"""
    return torch.mean((pred[:, :, 0, 0] - u0) ** 2)


def bc_loss_1d(pred: torch.Tensor) -> torch.Tensor:
    """Dirichlet 零边界项：x 两端（x=0 与 x=L）全场 MSE 之和，标量。"""
    return torch.mean(pred[:, 0, :, :] ** 2) + torch.mean(pred[:, -1, :, :] ** 2)


def pde_loss_1d(pred: torch.Tensor, grid: torch.Tensor, a: float) -> torch.Tensor:
    """点式光滑场的 autograd 残差；非局部 FNO 训练使用输出场差分。"""
    return torch.mean(heat_residual_1d(pred, grid, a) ** 2)


def compute_loss_terms_1d(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    u0: torch.Tensor,
    u_true: torch.Tensor,
    grid: torch.Tensor,
    a: float,
    derivative_method: str = "finite_difference",
    compute_pde: bool = True,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """计算四项损失：前向后沿输出场网格差分，梯度仍能回传至网络权重。

    Args:
        model: 算子模型 ``(u0, grid) -> pred``，pred 形状 (B, nx, nt, 1)。
        u0: 批初始条件 (B, nx)。
        u_true: 批真值场 (B, nx, nt, 1)。
        grid: 坐标网格 (B, nx, nt, 2)，只提供差分坐标，不计算坐标 VJP。
        a: 热扩散系数，单位 m²/s。
        derivative_method: 算子训练仅接受 finite_difference。
        compute_pde: False 时不计算 PDE 项，日志记录零。

    Returns:
        ``(terms, pred)``：terms 为 ``{"data", "pde", "ic", "bc"}`` 四个
        标量张量的字典；pred 为本次前向输出（供调用方复用，避免二次前向）。
    """
    from .finite_difference import residual_1d
    if derivative_method != "finite_difference":
        raise ValueError("算子网络必须使用 finite_difference 输出场导数；解析场可单独调用 autograd residual API")
    grid_g = grid.detach()
    pred = model(u0, grid_g)
    terms = {
        "data": data_loss_1d(pred, u_true),
        "pde": torch.mean(residual_1d(pred, grid_g, a)[:, 1:-1, 1:-1] ** 2) if compute_pde else pred.new_zeros(()),
        "ic": ic_loss_1d(pred, u0),
        "bc": bc_loss_1d(pred),
    }
    return terms, pred


def weighted_loss_1d(terms: dict[str, torch.Tensor], weights: LossWeights) -> torch.Tensor:
    """加权组合：``w_data·data + w_pde·pde + w_ic·ic + w_bc·bc``，标量。

    terms 必须包含四个键；跳过的 PDE 项用标量零表示。
    """
    missing = {"data", "pde", "ic", "bc"} - set(terms)
    if missing:
        raise KeyError(f"terms 缺少损失项：{sorted(missing)}")
    return (
        weights.data * terms["data"]
        + weights.pde * terms["pde"]
        + weights.ic * terms["ic"]
        + weights.bc * terms["bc"]
    )


# ----------------------------------------------------------------------
# 2D 圆环极坐标线（张量约定 (B, nt, nr, ntheta)，坐标通道 (t, r, θ)，无量纲域）
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class LossWeightsPolar:
    """极坐标 PINO 六项损失的权重（默认值沿用档案极坐标线）。

    Attributes:
        data: 数据项权重（档案 1.0）。
        pde: PDE 残差项权重（档案 0.05）。
        ic: 初始条件项权重（档案 2.0）。
        bc_inner: 内壁 Neumann 项权重（档案 5.0）。
        bc_outer: 外壁 Robin 项权重（档案 2.0）。
        periodic: θ 周期项权重（档案 2.0）。
    """

    data: float = 1.0
    pde: float = 0.05
    ic: float = 2.0
    bc_inner: float = 5.0
    bc_outer: float = 2.0
    periodic: float = 2.0


def masked_data_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """掩码加权数据项：掩码内（True）逐点 MSE 的均值，标量。

    ``mask=None`` 时退化为全域 MSE（极坐标网格恰好铺满圆环域的情形）。
    与档案 ``torch.mean(((pred - y) * mask)²)`` 的差异：档案除以**全部**格点
    数使损失随掩码占比缩放；本实现除以掩码内点数（真均值），不同几何间可比。
    """
    diff2 = (pred - target) ** 2
    if mask is None:
        return torch.mean(diff2)
    m = torch.broadcast_to(mask.to(diff2.dtype), diff2.shape)
    return torch.sum(diff2 * m) / torch.clamp(m.sum(), min=1.0)


def random_point_mask(
    shape: tuple[int, ...],
    rate: float,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """PDE 损失随机点采样掩码：每点以概率 ``rate`` 独立选中（Bernoulli）。

    随机源为 torch 全局 RNG（训练循环已由 ``set_seed`` 显式播种，同种子可复现）。
    ``rate >= 1`` 直接返回全 True；采样结果为空时回退为全 True（避免 0/0 NaN，
    并保证小批小网格下损失仍有定义）。
    """
    if not 0.0 <= rate:
        raise ValueError(f"rate 须非负，实际为 {rate}")
    if rate >= 1.0:
        return torch.ones(shape, dtype=torch.bool, device=device)
    mask = torch.rand(shape, device=device) < rate
    if not bool(mask.any()):
        return torch.ones(shape, dtype=torch.bool, device=device)
    return mask


def periodic_loss_theta(pred: torch.Tensor, dudtheta: torch.Tensor | None = None) -> torch.Tensor:
    """θ 周期对齐项：θ 首末两端场值 MSE，可选叠加 T_θ 两端 MSE，标量。

    约定 θ 网格含两端点（0 与 2π 为同一物理母线），末维为 θ。
    档案极坐标线两项都启用；仅场对齐时传 ``dudtheta=None``。
    """
    loss = torch.mean((pred[..., 0] - pred[..., -1]) ** 2)
    if dudtheta is not None:
        loss = loss + torch.mean((dudtheta[..., 0] - dudtheta[..., -1]) ** 2)
    return loss


def compute_loss_terms_polar(
    model: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    q_field: torch.Tensor,
    u_true: torch.Tensor,
    grid: torch.Tensor,
    alpha_norm: float,
    bi: float,
    ic_target: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pde_sample_rate: float = 1.0,
    derivative_method: str = "finite_difference",
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """极坐标六项损失：输出场差分，PDE 取时间/半径内部及唯一角度点。

    Args:
        model: 算子模型 ``(q_field, grid) -> pred``，pred 形状 (B, nt, nr, ntheta)。
        q_field: 批输入参数场 (B, nt, nr, ntheta)，由归一化热流 q* 广播构成；
            内壁 BC 的 q* 取自其内半径环 ``q_field[:, :, 0, :]``（档案约定：
            输入场为常数 q* 的广播，内环取值即该样本的 q*）。
        u_true: 批真值场 (B, nt, nr, ntheta)，与 pred 同域（无量纲）。
        grid: 坐标网格 (B, nt, nr, ntheta, 3)，只提供差分坐标。
        alpha_norm: 无量纲热扩散系数（见 ``heat_residual_polar``）。
        bi: 外边界 Biot 数（来自 ``problems`` / config）。
        ic_target: t*=0 帧目标场；默认 None 表示零初始场
            （均匀 T_inf 起算的无量纲形式 T*(0)=0）。
        mask: 可选数据域掩码（可广播到 pred），传给 ``masked_data_loss``。
        pde_sample_rate: PDE 残差随机点采样率（档案为 24GB 显存妥协取 0.2；
            1.0 表示全域）。

    Returns:
        ``(terms, pred)``：terms 含 ``data/pde/ic/bc_inner/bc_outer/periodic``
        六个标量；pred 为本次前向输出（供调用方复用）。

    PDE 的角度导数采用循环中心差分；周期 BC 在两端分别用单侧差分，
    以免循环差分复制端点导数而使该项恒为零。全部差分对输出场可导。
    """
    from .finite_difference import polar_derivatives, first_derivative
    if derivative_method != "finite_difference":
        raise ValueError("算子网络必须使用 finite_difference 输出场导数")
    grid_g = grid.detach()
    pred = model(q_field, grid_g)

    residual, t_r, _ = polar_derivatives(pred, grid_g, alpha_norm)
    t_th_boundary = first_derivative(pred, grid_g[0, 0, 0, :, 2], 3)
    residual = residual[:, 1:-1, 1:-1, :-1]
    if pde_sample_rate < 1.0:
        pm = random_point_mask(tuple(residual.shape), pde_sample_rate, device=residual.device)
        pde = torch.mean(residual[pm] ** 2)
    else:
        pde = torch.mean(residual ** 2)

    ic_tgt = torch.zeros_like(pred[:, 0]) if ic_target is None else ic_target
    terms = {
        "data": masked_data_loss(pred, u_true, mask),
        "pde": pde,
        "ic": torch.mean((pred[:, 0] - ic_tgt) ** 2),
        "bc_inner": torch.mean((t_r[:, :, 0, :] + q_field[:, :, 0, :]) ** 2),
        "bc_outer": torch.mean((t_r[:, :, -1, :] + bi * pred[:, :, -1, :]) ** 2),
        "periodic": periodic_loss_theta(pred, t_th_boundary),
    }
    return terms, pred


def weighted_loss_polar(terms: dict[str, torch.Tensor], weights: LossWeightsPolar) -> torch.Tensor:
    """加权组合六项极坐标损失，标量。

    terms 必须包含全部六个键（即使权重为零，也要求已计算，以保证日志完整）。
    """
    required = {"data", "pde", "ic", "bc_inner", "bc_outer", "periodic"}
    missing = required - set(terms)
    if missing:
        raise KeyError(f"terms 缺少损失项：{sorted(missing)}")
    return (
        weights.data * terms["data"]
        + weights.pde * terms["pde"]
        + weights.ic * terms["ic"]
        + weights.bc_inner * terms["bc_inner"]
        + weights.bc_outer * terms["bc_outer"]
        + weights.periodic * terms["periodic"]
    )
