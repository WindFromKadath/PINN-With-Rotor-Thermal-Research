"""fields 层校核：GRF 种子复现性与统计量级、初始条件形状与边界相容性。"""

import numpy as np

from heatops.fields import (
    GRF1DGenerator,
    GRF2DGenerator,
    gaussian_heat_source,
    sample_random_sines,
    sample_single_sine,
    sample_sine_or_square,
    sample_square_wave,
)


def test_grf2d_seed_reproducibility():
    """同种子两次采样逐点一致；不同种子产生不同场。"""
    g1 = GRF2DGenerator(length_scale=0.2, n_points=16, seed=42)
    g2 = GRF2DGenerator(length_scale=0.2, n_points=16, seed=42)
    a1, a2 = g1.sample(4), g2.sample(4)
    assert np.array_equal(a1, a2)

    g3 = GRF2DGenerator(length_scale=0.2, n_points=16, seed=43)
    assert not np.allclose(g3.sample(4), a1)


def test_grf2d_magnitude_and_clip():
    """均值/波动量级检查（固定种子下为确定值，阈值宽松）与负值截断。"""
    g = GRF2DGenerator(length_scale=0.2, n_points=32, seed=0)
    f = g.sample(256, mean_f=120.0, std_f=30.0)
    assert f.shape == (256, 32, 32)
    assert abs(f.mean() - 120.0) < 5.0
    assert 20.0 < f.std() < 40.0
    assert np.all(f >= 0.0)

    # 均值为 0 时截断应产生大量零值
    f0 = g.sample(16, mean_f=0.0, std_f=30.0)
    assert np.all(f0 >= 0.0)
    assert np.any(f0 == 0.0)


def test_grf1d_seed_and_shape():
    g1 = GRF1DGenerator(length_scale=0.2, n_points=64, seed=11)
    g2 = GRF1DGenerator(length_scale=0.2, n_points=64, seed=11)
    assert np.array_equal(g1.sample(3), g2.sample(3))
    assert g1.sample(5).shape == (5, 64)

    g3 = GRF1DGenerator(length_scale=0.2, n_points=64, seed=12)
    assert not np.allclose(g3.sample(3), g1.sample(3))

    # 量级检查（固定种子确定值）
    g4 = GRF1DGenerator(length_scale=0.2, n_points=101, seed=1)
    f = g4.sample(512, mean_f=5.0, std_f=2.0)
    assert abs(f.mean() - 5.0) < 0.5
    assert 1.0 < f.std() < 3.0


def test_initial_conditions_shapes_and_dirichlet():
    """各初始条件采样器：形状一致、端点满足 Dirichlet 零边界、种子可复现。"""
    x = np.linspace(0.0, 1.0, 64)
    samplers = [sample_random_sines, sample_square_wave, sample_single_sine, sample_sine_or_square]
    for sampler in samplers:
        u = sampler(x, np.random.default_rng(123))
        assert u.shape == x.shape
        assert u[0] == 0.0
        assert abs(u[-1]) < 1e-12
        # 同种子复现
        u_rep = sampler(x, np.random.default_rng(123))
        assert np.array_equal(u, u_rep)


def test_square_wave_profile():
    """方波：非零段为中间 1/2 区间，幅值在约定范围内。"""
    x = np.linspace(0.0, 1.0, 64)
    u = sample_square_wave(x, np.random.default_rng(9))
    assert np.all(u[:16] == 0.0)
    assert np.all(u[48:] == 0.0)
    assert 0.8 <= u[16] <= 1.2
    assert np.all(u[16:48] == u[16])


def test_single_sine_frequency_range():
    """单正弦：amp·sin(k·π·x)，k ∈ {1, 2, 3}（档案约定 k∈[1,4)）。"""
    x = np.linspace(0.0, 1.0, 64)
    for seed in range(10):
        u = sample_single_sine(x, np.random.default_rng(seed))
        residual = min(
            np.linalg.norm(u - 0.5 * np.sin(k * np.pi * x)) for k in (1, 2, 3)
        )
        assert residual < 1e-12


def test_gaussian_heat_source():
    """高斯热源：峰值与位置、对称性、远端衰减。"""
    x = np.linspace(0.0, 1.0, 101)
    f = gaussian_heat_source(x)
    assert f.shape == x.shape
    assert np.isclose(f.max(), 5.0, rtol=1e-12)
    assert x[np.argmax(f)] == 0.5
    assert np.allclose(f, f[::-1])  # 关于中心对称
    assert f[0] < 1e-8  # 端点处近似为零
