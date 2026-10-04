"""三对角系统的组装、乘积与求解工具（仅供本包内部使用）。

约定：``lower[i]`` 为第 i 行次对角系数（乘 ``x[i-1]``，``lower[0]`` 不使用），
``upper[i]`` 为第 i 行超对角系数（乘 ``x[i+1]``，``upper[-1]`` 不使用）。
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import solve_banded


def tridiag_matvec(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, x: np.ndarray
) -> np.ndarray:
    """三对角矩阵与向量的乘积，避免显式构造稠密矩阵。"""
    y = diag * x
    y[:-1] += upper[:-1] * x[1:]
    y[1:] += lower[1:] * x[:-1]
    return y


def solve_tridiag(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, b: np.ndarray
) -> np.ndarray:
    """按 (1,1) 带状格式组装并调用 scipy.linalg.solve_banded 求解三对角系统。"""
    n = diag.shape[0]
    ab = np.zeros((3, n))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, b)
