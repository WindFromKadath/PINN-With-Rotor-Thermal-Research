"""圆环几何工具：Cartesian 网格上的圆环掩码、边界环带提取与径向法向导数。

用途：COMSOL Cartesian CSV 数据的几何掩码（NaN 提取的交叉校验）、边界环带
（内/外壁）提取，以及直角坐标梯度向径向法向的投影。圆环 PINO 极坐标训练线
本身不需要掩码（极坐标网格恰好铺满圆环域），本模块服务于数据管线与
直角坐标参考实现（直角 PINO 按 D104 不迁移，此处仅保留几何工具）。

实现参考原研究档案 ``datapino_2d_train.py`` 的 ``get_separated_indices_adaptive``
（binary_erosion 提取边界 + 半径中值分内外）与其法向导数径向投影写法，
按本库约定重写为纯 numpy 显式参数函数（物理参数由调用方从 ``problems`` 传入）。
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage


def annulus_mask(
    nx: int,
    ny: int | None = None,
    *,
    r_in: float,
    r_out: float,
    domain_size: float = 1.0,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """在 Cartesian 方形网格上生成圆环布尔掩码。

    网格覆盖 ``[0, domain_size]²``（``indexing='ij'``，第一维为 x），
    掩码为 ``r_in <= dist(point, center) <= r_out``（含边界像素）。

    Args:
        nx, ny: 网格点数；``ny=None`` 时取 nx（方网格）。
        r_in, r_out: 圆环内/外半径（与 domain_size 同单位）。
        domain_size: 方形域边长。
        center: 圆心 (cx, cy)；默认域中心 ``(domain_size/2, domain_size/2)``。

    Returns:
        形状 (nx, ny) 的 bool 数组，圆环内为 True。
    """
    ny = nx if ny is None else ny
    cx, cy = center if center is not None else (domain_size / 2, domain_size / 2)
    x = np.linspace(0.0, domain_size, nx)
    y = np.linspace(0.0, domain_size, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.hypot(X - cx, Y - cy)
    return (dist >= r_in) & (dist <= r_out)


def extract_boundary_bands(
    mask: np.ndarray,
    *,
    r_in: float,
    r_out: float,
    domain_size: float = 1.0,
    center: tuple[float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """由圆环掩码提取内/外边界环带（各 1 像素宽，按半径中值分内外）。

    边界提取：``boundary = mask XOR binary_erosion(mask)``（4-连通结构元），
    再按 ``dist < (r_in + r_out)/2`` 分内壁/外壁（档案自适应分辨率写法）。

    Args:
        mask: 形状 (nx, ny) 的 bool 圆环掩码。
        r_in, r_out, domain_size, center: 与 ``annulus_mask`` 同义；
            网格坐标从形状与 domain_size 重建，须与生成 mask 时一致。

    Returns:
        ``(inner_band, outer_band)`` 两个同形状 bool 数组，互不相交。
    """
    mask = np.asarray(mask, dtype=bool)
    nx, ny = mask.shape
    cx, cy = center if center is not None else (domain_size / 2, domain_size / 2)
    x = np.linspace(0.0, domain_size, nx)
    y = np.linspace(0.0, domain_size, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dist = np.hypot(X - cx, Y - cy)

    struct = ndimage.generate_binary_structure(2, 1)
    erosion = ndimage.binary_erosion(mask, structure=struct)
    boundary = mask ^ erosion

    r_mid = 0.5 * (r_in + r_out)
    inner_band = boundary & (dist < r_mid)
    outer_band = boundary & (dist >= r_mid)
    return inner_band, outer_band


def radial_normal_component(
    dudx: np.ndarray,
    dudy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    center: tuple[float, float] | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """直角坐标梯度向径向（圆环法向）投影：``du/dn = u_x·cosθ + u_y·sinθ``。

    圆环内/外壁的外法向分别为 -r̂ 与 +r̂；本函数返回**径向分量** ``du/dr``，
    符号换算（内壁法向取负）由调用方按边界侧处理（档案 ``datapino_2d_train.py``
    的 ``dT_dn`` 即此径向分量）。

    Args:
        dudx, dudy: 梯度分量，形状 (nx, ny)（``indexing='ij'``）。
        x, y: 坐标网格，形状 (nx, ny)（如 ``np.meshgrid(xv, yv, indexing='ij')``）。
        center: 圆心；默认取坐标范围中点。
        eps: 圆心处除零保护。

    Returns:
        与 ``dudx`` 同形状的径向导数数组。
    """
    dudx = np.asarray(dudx, dtype=np.float64)
    dudy = np.asarray(dudy, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if center is None:
        center = (0.5 * (float(x.min()) + float(x.max())), 0.5 * (float(y.min()) + float(y.max())))
    rel_x = x - center[0]
    rel_y = y - center[1]
    norm = np.sqrt(rel_x**2 + rel_y**2) + eps
    return dudx * (rel_x / norm) + dudy * (rel_y / norm)
