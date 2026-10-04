"""Differentiable field derivatives on tensor-product grids (including FNO outputs).

Derivatives are taken along the sampled output field, rather than by differentiating
the sum of a nonlocal operator's outputs with respect to its input coordinates.
Nonuniform increasing axes are supported; PDE losses use interior time/radius points.
"""
from __future__ import annotations
import torch


def first_derivative(u: torch.Tensor, axis: torch.Tensor, dim: int) -> torch.Tensor:
    axis = axis.detach().to(device=u.device, dtype=u.dtype)
    if axis.numel() < 3 or not bool(torch.all(torch.diff(axis) > 0)):
        raise ValueError("差分坐标轴必须至少有三个严格递增的点")
    return torch.gradient(u, spacing=(axis,), dim=(dim,), edge_order=2)[0]


def second_derivative(u: torch.Tensor, axis: torch.Tensor, dim: int) -> torch.Tensor:
    axis = axis.detach().to(device=u.device, dtype=u.dtype)
    if axis.numel() < 3 or not bool(torch.all(torch.diff(axis) > 0)):
        raise ValueError("差分坐标轴必须至少有三个严格递增的点")
    v = u.movedim(dim, -1)
    h = torch.diff(axis)
    middle = 2 * ((v[..., 2:] - v[..., 1:-1]) / h[1:]
                  - (v[..., 1:-1] - v[..., :-2]) / h[:-1]) / (h[1:] + h[:-1])
    # Boundary values are one-sided three-point curvature estimates. They are not
    # included in PDE losses; boundary first derivatives have their own BC loss.
    return torch.cat((middle[..., :1], middle, middle[..., -1:]), -1).movedim(-1, dim)


def periodic_derivatives(u: torch.Tensor, axis: torch.Tensor, dim: int):
    """Central periodic stencils; axis contains both ends of one full period."""
    axis = axis.detach().to(device=u.device, dtype=u.dtype)
    if axis.numel() < 4 or not bool(torch.all(torch.diff(axis) > 0)):
        raise ValueError("周期差分坐标轴必须至少有四个严格递增的点（含重复端点）")
    h = torch.diff(axis)
    if not torch.allclose(h, h[:1].expand_as(h), rtol=1e-4, atol=1e-6):
        raise ValueError("周期角度差分要求均匀网格")
    v = u.movedim(dim, -1)[..., :-1]
    left, right = torch.roll(v, 1, -1), torch.roll(v, -1, -1)
    d1, d2 = (right - left) / (2 * h[0]), (right - 2 * v + left) / h[0]**2
    return tuple(torch.cat((d, d[..., :1]), -1).movedim(-1, dim) for d in (d1, d2))


def residual_1d(u: torch.Tensor, grid: torch.Tensor, a: float):
    x, t = grid[0, :, 0, 0], grid[0, 0, :, 1]
    return first_derivative(u, t, 2) - a * second_derivative(u, x, 1)


def polar_derivatives(u: torch.Tensor, grid: torch.Tensor, alpha_norm: float):
    t, r, theta = grid[0, :, 0, 0, 0], grid[0, 0, :, 0, 1], grid[0, 0, 0, :, 2]
    if not bool(torch.all(r > 0)):
        raise ValueError("极坐标半径必须为正")
    u_t = first_derivative(u, t, 1)
    u_r = first_derivative(u, r, 2)
    u_rr = second_derivative(u, r, 2)
    u_theta, u_thetatheta = periodic_derivatives(u, theta, 3)
    rad = grid[..., 1]
    return u_t - alpha_norm * (u_rr + u_r / rad + u_thetatheta / rad**2), u_r, u_theta
