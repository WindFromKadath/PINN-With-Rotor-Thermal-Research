"""高斯随机场（GRF）采样器：RBF 核协方差 + Cholesky 分解。

不使用 deepxde 的 ``dde.data.GRF``（深度学习依赖留给上层），协方差矩阵由
sklearn 的 RBF 核构造：``K(x, y) = exp(-||x-y||² / (2·length_scale²))``，
加 1e-8 抖动的单位阵保证 Cholesky 分解稳定。随机性由构造时传入的
``numpy.random.Generator`` 种子显式控制：同种子两次采样逐点一致。
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import kernels as _gp_kernels


class GRF2DGenerator:
    """单位正方形 ``[0, 1]²`` 上的二维 GRF 采样器（网格约定 ``indexing='ij'``）。

    Args:
        length_scale: RBF 核长度尺度（相关长度），无量纲（域边长归一）。
        n_points: 每边网格点数 N，场形状为 (N, N)。
        seed: 随机种子；相同种子构造的采样器产生完全相同的样本序列。
    """

    def __init__(self, length_scale: float = 0.2, n_points: int = 32, seed: int | None = None):
        self.length_scale = float(length_scale)
        self.n_points = int(n_points)
        x = np.linspace(0.0, 1.0, self.n_points)
        X, Y = np.meshgrid(x, x, indexing="ij")
        points = np.column_stack([X.ravel(), Y.ravel()])
        K = _gp_kernels.RBF(length_scale=self.length_scale)(points)
        n_dof = self.n_points**2
        self._chol = np.linalg.cholesky(K + 1e-8 * np.eye(n_dof))
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        n_samples: int,
        mean_f: float = 120.0,
        std_f: float = 30.0,
        clip_negative: bool = True,
    ) -> np.ndarray:
        """采样随机场。

        Args:
            n_samples: 样本数。
            mean_f: 场均值（作为热源时单位 °C/s）。
            std_f: 场波动（标准差，与均值同单位）。
            clip_negative: 是否将负值截断为 0（热源非负物理约束，默认开启）。

        Returns:
            形状 (n_samples, N, N) 的随机场数组（float64）。
        """
        n_dof = self.n_points**2
        z = self._rng.standard_normal((n_dof, n_samples))
        f = mean_f + std_f * (self._chol @ z).T
        if clip_negative:
            f = np.maximum(f, 0.0)
        return f.reshape(n_samples, self.n_points, self.n_points)


class GRF1DGenerator:
    """一维区间上的 GRF 采样器，接口与 :class:`GRF2DGenerator` 一致。

    Args:
        length_scale: RBF 核长度尺度（相关长度），无量纲（域边长归一）。
        n_points: 网格点数。
        x_range: 区间端点 ``(x_min, x_max)``，单位 m。
        seed: 随机种子；相同种子构造的采样器产生完全相同的样本序列。
    """

    def __init__(
        self,
        length_scale: float = 0.2,
        n_points: int = 101,
        x_range: tuple[float, float] = (0.0, 1.0),
        seed: int | None = None,
    ):
        self.length_scale = float(length_scale)
        self.n_points = int(n_points)
        self.x = np.linspace(x_range[0], x_range[1], self.n_points)
        K = _gp_kernels.RBF(length_scale=self.length_scale)(self.x[:, None])
        self._chol = np.linalg.cholesky(K + 1e-8 * np.eye(self.n_points))
        self._rng = np.random.default_rng(seed)

    def sample(
        self,
        n_samples: int,
        mean_f: float = 0.0,
        std_f: float = 1.0,
        clip_negative: bool = False,
    ) -> np.ndarray:
        """采样随机场。

        Args:
            n_samples: 样本数。
            mean_f: 场均值。
            std_f: 场波动（标准差）。
            clip_negative: 是否将负值截断为 0；1D 场常用于可正可负的
                源函数/初始条件，默认不截断。

        Returns:
            形状 (n_samples, n_points) 的随机场数组（float64）。
        """
        z = self._rng.standard_normal((self.n_points, n_samples))
        f = mean_f + std_f * (self._chol @ z).T
        if clip_negative:
            f = np.maximum(f, 0.0)
        return f
