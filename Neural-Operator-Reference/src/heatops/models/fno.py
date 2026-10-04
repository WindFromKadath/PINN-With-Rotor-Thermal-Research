"""FNO 热算子模型：输入函数广播 + 坐标通道拼接 → neuralop FNO →（可选）硬约束 ansatz。

重写自原研究档案 DataPINO 线的 ``HybridPINO``（archive-snapshot，未入库权重）：

- 输入通道：初始条件 u0 沿时间广播 + (x, t) 坐标，共 3 通道（``indexing='ij'``，
  空间维在前）；
- 骨干：``neuralop.models.FNO``（2D 时空谱卷积）；
- 可选硬约束 ansatz：``u = Softplus(u0·exp(-a·t) + t·FNO_out)``。
  注意 ``exp(-a·t)`` 是**经验衰减先验而非解析解**（1D 热方程的解析衰减率为
  模态相关的 ``exp(-a·k²·π²·t)``，档案注释「基础解部分」并不准确）；
  ansatz 的代数效果是 t=0 时输出恰为 ``Softplus(u0)``（网络项被 t 乘没），
  从而在 Softplus 逼近意义下近初始条件，Dirichlet 零边界仍由 bc 损失软约束；
- 输出经 Softplus 保证非负（散热数据非负）。

本模块无 import 副作用；物理参数 ``a`` 仅经构造参数显式传入（来自 config）。

另含 3D 时空变体 ``HeatOperatorFNO3D``（圆环极坐标线）：输入通道
[参数场 q*, t, r, θ]，维序 (t, r, θ) 与档案 ``datapino_2d_polar_train.py``
的 ``n_modes=(8, 12, 12)`` 对应（时间 8 模态，r/θ 各 12）。
"""

from __future__ import annotations

import torch
import torch.nn as nn
from neuralop.models import FNO


class HeatOperatorFNO(nn.Module):
    """u0(x) → u(x, t) 的 FNO 算子模型（1D 空间 + 时间）。

    Args:
        n_modes: FNO 各维谱模态数，如 (16, 16) 对应 (x, t) 两维。
        hidden_channels: FNO 隐层通道数。
        in_channels: 输入通道数（默认 3：u0 广播 + x + t）。
        out_channels: 输出通道数（默认 1：温度场）。
        use_ansatz: 是否启用硬约束 ansatz ``u0·exp(-a·t) + t·FNO``；
            关闭时输出为 ``Softplus(FNO)`` 直接回归。
        a: 热扩散系数（m²/s），仅用于 ansatz 的经验衰减因子；
            须由 config / ``problems`` 显式传入，不在本模块硬编码。

    输入与输出约定：
        ``forward(u0, grid)`` 中 ``u0`` 形状 (B, nx)，``grid`` 形状
        (B, nx, nt, 2)（末维通道顺序 (x, t)，见 ``heatops.data.make_xt_grid``）；
        输出形状 (B, nx, nt, 1)，非负。
    """

    def __init__(
        self,
        n_modes: tuple[int, int] = (16, 16),
        hidden_channels: int = 64,
        in_channels: int = 3,
        out_channels: int = 1,
        use_ansatz: bool = True,
        a: float = 0.4,
    ):
        super().__init__()
        self.fno = FNO(
            n_modes=tuple(n_modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )
        self.sp = nn.Softplus()
        self.use_ansatz = bool(use_ansatz)
        self.a = float(a)

    def forward(self, u0: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """前向传播；PDE 空间/时间导数由输出场差分计算。"""
        nt = grid.shape[2]
        u0_ext = u0.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nt, 1)
        x_in = torch.cat([u0_ext, grid], dim=-1).permute(0, 3, 1, 2).contiguous()
        out = self.fno(x_in).permute(0, 2, 3, 1).contiguous()  # (B, nx, nt, 1)
        if self.use_ansatz:
            t = grid[..., 1:2]
            out = u0_ext * torch.exp(-self.a * t) + t * out
        return self.sp(out)


class HeatOperatorFNO3D(nn.Module):
    """q*(场) → u(t*, r*, θ) 的 3D 时空 FNO 算子模型（圆环极坐标瞬态线）。

    重写自档案 ``datapino_2d_polar_train.py`` 的 ``PolarPINO``，
    接口与维度约定保持一致，物理有效性需另行验证：

    - 输入通道 4：``[q*, t, r, θ]``（参数场广播 + 三个坐标通道）；
    - 维序 (t, r, θ)，``n_modes=(8, 12, 12)`` 依次对应时间/r/θ 的谱模态数；
    - 输出为 FNO 原始回归（无 Softplus/ansatz——档案极坐标线即如此；
      无量纲升温场 T* ≥ 0 由数据与物理损失软约束，不硬编码）。

    Args:
        n_modes: FNO 各维谱模态数，如 (8, 12, 12) 对应 (t, r, θ) 三维。
        hidden_channels: FNO 隐层通道数。
        in_channels: 输入通道数（默认 4：q* + t + r + θ）。
        out_channels: 输出通道数（默认 1：温度场）。

    输入与输出约定：
        ``forward(q_field, grid)`` 中 ``q_field`` 形状 (B, nt, nr, ntheta)
        （归一化热流 q* 的广播场），``grid`` 形状 (B, nt, nr, ntheta, 3)
        （末维通道顺序 (t, r, θ)，无量纲物理坐标，见
        ``heatops.physics.residuals.heat_residual_polar``）；
        输出形状 (B, nt, nr, ntheta)。PDE/BC 导数由输出场差分计算，
        非局部模型的坐标 VJP 不等于逐点物理导数。
    """

    def __init__(
        self,
        n_modes: tuple[int, int, int] = (8, 12, 12),
        hidden_channels: int = 32,
        in_channels: int = 4,
        out_channels: int = 1,
    ):
        super().__init__()
        self.fno = FNO(
            n_modes=tuple(n_modes),
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
        )

    def forward(self, q_field: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
        """前向传播；PDE 空间/时间导数由输出场差分计算。"""
        x_in = torch.cat([q_field.unsqueeze(1), grid.permute(0, 4, 1, 2, 3)], dim=1)
        return self.fno(x_in).squeeze(1)
