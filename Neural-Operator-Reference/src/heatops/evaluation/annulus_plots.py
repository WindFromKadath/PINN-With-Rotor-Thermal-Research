"""2D 圆环（极坐标）绘图辅助：极坐标 pcolormesh 三联图、三角剖分掩码云图、径向剖面线。

与 ``evaluation.plots`` 同样约定：matplotlib Agg 后端、函数内延迟导入、
无 import 副作用。圆环空洞处理：极坐标图天然无空洞（域即 [r_in, r_out]×[0, 2π]）；
Cartesian 三角剖分云图按三角形质心半径掩码，显式挖去内孔与外域。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_polar_frame_triplet(
    r: np.ndarray,
    theta: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    path: str | Path,
    model_label: str = "Model",
    frame_label: str = "",
) -> Path:
    """单时刻帧极坐标三联图：真值 / 预测 / 绝对误差（场形状 (nr, ntheta)）。

    极坐标投影 ``pcolormesh(θ, r, u)``；误差子图标题附最大值。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    u_true = np.asarray(u_true, dtype=np.float64)
    u_pred = np.asarray(u_pred, dtype=np.float64)
    R, Theta = np.meshgrid(r, theta, indexing="ij")
    error = np.abs(u_true - u_pred)
    panels = [
        (u_true, "Reference", "magma", ""),
        (u_pred, f"{model_label} Prediction", "magma", ""),
        (error, "Absolute Error", "Reds", f" (Max: {error.max():.2e})"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), subplot_kw={"projection": "polar"})
    for ax, (data, title, cmap, suffix) in zip(axes, panels):
        im = ax.pcolormesh(Theta, R, data, cmap=cmap, shading="auto")
        ax.set_title(title + suffix)
        fig.colorbar(im, ax=ax)
    if frame_label:
        fig.suptitle(frame_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_annulus_triangulation(
    x: np.ndarray,
    y: np.ndarray,
    u: np.ndarray,
    path: str | Path,
    r_in: float | None = None,
    r_out: float | None = None,
    center: tuple[float, float] | None = None,
    title: str = "Annulus Field",
) -> Path:
    """Cartesian 点云三角剖分掩码云图（圆环空洞处理）。

    对散点 (x, y, u) 做 Delaunay 三角剖分，按三角形**质心半径**掩码：
    质心落在 ``[r_in, r_out]`` 之外的三角形不绘制，从而挖去内孔与外域
    （直接 tripcolor 会用三角形跨过空洞）。

    Args:
        x, y, u: 等长一维数组（散点坐标与场值）。
        r_in, r_out: 圆环内/外半径；均为 None 时不掩码（整域绘制）。
        center: 圆心；默认取 x/y 范围中点。
    """
    plt = _plt()
    import matplotlib.tri as mtri

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    u = np.asarray(u, dtype=np.float64).ravel()
    cx, cy = center if center is not None else (0.5 * (x.min() + x.max()), 0.5 * (y.min() + y.max()))

    tri = mtri.Triangulation(x, y)
    if r_in is not None and r_out is not None:
        tri_cx = x[tri.triangles].mean(axis=1)
        tri_cy = y[tri.triangles].mean(axis=1)
        tri_r = np.hypot(tri_cx - cx, tri_cy - cy)
        tri.set_mask((tri_r < r_in) | (tri_r > r_out))

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.tripcolor(tri, u, cmap="magma", shading="flat")
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_radial_profiles(
    r: np.ndarray,
    series: dict[str, np.ndarray],
    path: str | Path,
    title: str = "Radial Temperature Profiles",
    xlabel: str = "r (m)",
    ylabel: str = "T",
) -> Path:
    """径向剖面线对比：``{标签: 剖面数组 (nr,)}`` 逐条绘制。"""
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 6))
    for label, values in series.items():
        plt.plot(r, np.asarray(values, dtype=np.float64), label=label)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path
