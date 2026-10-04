"""离散场注入 PDE 残差的插值工具（``torch.nn.functional.grid_sample``）。

重写自原研究档案 PINNs 线的 ``get_interpolation_function``，用于把 FDM 网格上
的离散场（如 32×32 热源）作为可微注入项接入 PINN/PI-DeepONet 的 PDE 残差。

坐标映射约定（修正档案注释的歧义并推广到任意矩形域）：

- 离散场 ``f_grid`` 形状 ``(nx, ny)``，网格约定 ``indexing='ij'``（第一维对应
  x、第二维对应 y），均匀覆盖 ``x_range × y_range``；
- 场在内存中排布为 NCHW ``(1, 1, nx, ny)``，即 H 维 ↔ x、W 维 ↔ y；
- ``F.grid_sample`` 采样网格的最后一维约定为 ``[..., 0]`` 沿 W（即 y）、
  ``[..., 1]`` 沿 H（即 x），``align_corners=True`` 时归一化坐标 -1/1 恰好
  对齐场的首/末格点——因此物理查询点 ``(x, y)`` 必须按 ``[y_norm, x_norm]``
  的顺序拼装采样网格（档案注释「[...,0] 是横轴(y), [...,1] 是纵轴(x)」说的
  就是这件事，但表述容易读反）；
- 域外查询按 ``padding_mode="border"`` 取最近边界值（常数外延），避免默认
  ``zeros`` 填充在边界外产生虚假的 0。
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def make_grid_field_sampler(
    f_grid: np.ndarray | torch.Tensor,
    x_range: tuple[float, float] = (0.0, 1.0),
    y_range: tuple[float, float] = (0.0, 1.0),
    device: str | torch.device | None = None,
    dtype: torch.dtype = torch.float32,
):
    """由离散场构造双线性插值函数，供 PDE 残差按任意坐标查询。

    Args:
        f_grid: 离散场，形状 ``(nx, ny)``（第一维 x、第二维 y，均匀网格）。
        x_range, y_range: 场覆盖的物理区间 ``(min, max)``。
        device: 场张量驻留设备；None 表示 CPU。查询点会在采样前转换到该设备
            与 dtype。
        dtype: 场张量 dtype（默认 float32，与 deepxde 默认实数一致）。

    Returns:
        callable ``f(points)``：输入形状 ``(N, d)`` 的坐标张量（**前两列**为
        ``(x, y)``，其余列如 t 被忽略），返回形状 ``(N, 1)`` 的场值张量。
        对查询点可微性说明：grid_sample 对采样网格可微，但本工具用于注入
        **数据场**（残差中不对场值求导），梯度只应流向网络权重。
    """
    f = torch.as_tensor(np.asarray(f_grid), dtype=dtype)
    if f.dim() != 2:
        raise ValueError(f"f_grid 应为 (nx, ny) 二维数组，实际形状 {tuple(f.shape)}")
    x0, x1 = map(float, x_range)
    y0, y1 = map(float, y_range)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"x_range/y_range 须为递增区间，实际为 {x_range} × {y_range}")
    f_tensor = f.reshape(1, 1, *f.shape).to(device)

    def sampler(points: torch.Tensor) -> torch.Tensor:
        pts = points.to(device=f_tensor.device, dtype=f_tensor.dtype)
        # 物理坐标 → [-1, 1] 归一化；grid 末维顺序 [y_norm, x_norm]（见模块 docstring）
        gx = 2.0 * (pts[:, 0:1] - x0) / (x1 - x0) - 1.0
        gy = 2.0 * (pts[:, 1:2] - y0) / (y1 - y0) - 1.0
        grid = torch.cat([gy, gx], dim=1).view(1, 1, -1, 2)
        f_val = F.grid_sample(
            f_tensor, grid, mode="bilinear", padding_mode="border", align_corners=True
        )
        return f_val.view(-1, 1)

    return sampler
