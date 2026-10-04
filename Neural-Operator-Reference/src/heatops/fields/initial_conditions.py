"""一维初始条件与热源场采样器。

所有随机采样器遵循统一协议：``callable(x, rng) -> ndarray``，形状与 ``x``
相同；随机性完全由调用方传入的 ``numpy.random.Generator`` 控制，
同一种子序列完全可复现。各参数的默认值对应原研究档案两条 1D 数据线的约定。
"""

from __future__ import annotations

import numpy as np


def sample_random_sines(
    x: np.ndarray,
    rng: np.random.Generator,
    n_terms_range: tuple[int, int] = (1, 4),
    freq_range: tuple[int, int] = (1, 5),
    amp_range: tuple[float, float] = (0.5, 1.5),
) -> np.ndarray:
    """随机正弦叠加初始条件：``Σ amp·sin(k·π·x)``，单位 °C。

    项数在 ``n_terms_range``、频率 k 在 ``freq_range`` 内整数均匀采样
    （区间为左闭右开，与档案的 ``np.random.randint`` 语义一致），
    幅值在 ``amp_range`` 内连续均匀采样。各阶正弦在 x=0、1 处均为零，
    自动相容 Dirichlet 零边界。
    """
    x = np.asarray(x, dtype=np.float64)
    u = np.zeros_like(x)
    n_terms = int(rng.integers(n_terms_range[0], n_terms_range[1]))
    for _ in range(n_terms):
        amp = rng.uniform(amp_range[0], amp_range[1])
        k = int(rng.integers(freq_range[0], freq_range[1]))
        u += amp * np.sin(k * np.pi * x)
    return u


def sample_square_wave(
    x: np.ndarray,
    rng: np.random.Generator,
    amp_range: tuple[float, float] = (0.8, 1.2),
) -> np.ndarray:
    """方波初始条件（°C）：中间 1/2 区间取均匀随机幅值，其余为零。

    非零段按网格索引 ``[nx//4, 3·nx//4)`` 取值，端点为零，相容 Dirichlet 零边界。
    """
    x = np.asarray(x, dtype=np.float64)
    nx = x.shape[0]
    u = np.zeros(nx)
    u[nx // 4 : (3 * nx) // 4] = rng.uniform(amp_range[0], amp_range[1])
    return u


def sample_sine_or_square(x: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """散热线档案约定的混合初始条件（°C）。

    先采随机正弦叠加，再以 50% 概率改为方波（覆盖前者）；随机数消耗顺序
    与档案数据生成器一致。
    """
    u = sample_random_sines(x, rng)
    if rng.random() > 0.5:
        u = sample_square_wave(x, rng)
    return u


def sample_single_sine(
    x: np.ndarray,
    rng: np.random.Generator,
    amp: float = 0.5,
    k_range: tuple[int, int] = (1, 4),
) -> np.ndarray:
    """单正弦参数化初始条件：``amp·sin(k·π·x)``（°C），k ∈ [1, 4) 整数均匀采样。

    对应带源加热线档案约定（amp=0.5，k ∈ {1, 2, 3}）。
    """
    x = np.asarray(x, dtype=np.float64)
    k = int(rng.integers(k_range[0], k_range[1]))
    return amp * np.sin(k * np.pi * x)


def gaussian_heat_source(
    x: np.ndarray,
    amplitude: float = 5.0,
    center: float = 0.5,
    sharpness: float = 100.0,
) -> np.ndarray:
    """高斯热源场 ``f(x) = amplitude·exp(-sharpness·(x-center)²)``，单位 °C/s。

    确定性函数（无随机性）；默认值对应带源加热线档案约定
    ``5·exp(-100·(x-0.5)²)``。
    """
    x = np.asarray(x, dtype=np.float64)
    return amplitude * np.exp(-sharpness * (x - center) ** 2)
