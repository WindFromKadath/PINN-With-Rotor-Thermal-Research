"""problems 层校核：无量纲化往返恒等、解析解边界残差、特征数数值。"""

import numpy as np

from heatops.problems import (
    AnnulusParams,
    CartesianHeatParams,
    analytical_steady_state,
    analytical_steady_state_dimensionless,
    heat_line_cooling,
    heat_line_heating,
)


def test_annulus_dimensionless_roundtrip():
    """物理量 → 无量纲 → 物理应在浮点精度内恢复原值（温度/半径/时间）。"""
    p = AnnulusParams()
    rng = np.random.default_rng(0)

    T = rng.uniform(p.T_inf, 4000.0, 64)
    assert np.allclose(
        p.to_dimensional_temperature(p.to_dimensionless_temperature(T)), T,
        rtol=1e-12, atol=1e-10,
    )

    r = rng.uniform(p.R_in, p.R_out, 64)
    assert np.allclose(
        p.to_dimensional_radius(p.to_dimensionless_radius(r)), r,
        rtol=1e-12, atol=1e-14,
    )

    t = rng.uniform(0.0, 1e5, 64)
    assert np.allclose(
        p.to_dimensional_time(p.to_dimensionless_time(t)), t,
        rtol=1e-12, atol=1e-9,
    )


def test_annulus_dimensionless_analytic_matches_dimensional():
    """无量纲解析解经反无量纲化后应与物理形式解析解逐点一致。"""
    p = AnnulusParams()
    r_star = np.linspace(p.R_in_star, 1.0, 50)
    T_star = analytical_steady_state_dimensionless(r_star, p)
    T_direct = analytical_steady_state(p.to_dimensional_radius(r_star), p)
    assert np.allclose(p.to_dimensional_temperature(T_star), T_direct, rtol=1e-9, atol=1e-9)


def test_analytical_solution_boundary_residuals():
    """解析解满足内边界 Neumann 与外边界 Robin（细网格数值梯度验证）。

    np.gradient（edge_order=2）在 20001 点网格上的端点导数误差约 1e-5 °C/m，
    对应通量残差相对量级约 1e-8，阈值取 1e-6 仍留有百倍余量。
    """
    p = AnnulusParams()
    r = np.linspace(p.R_in, p.R_out, 20001)
    T = analytical_steady_state(r, p)
    dTdr = np.gradient(T, r, edge_order=2)

    # 内边界 Neumann：-k·dT/dr = q_in
    inner_flux = -p.k * dTdr[0]
    assert abs(inner_flux - p.q_in) / p.q_in < 1e-6

    # 外边界 Robin：-k·dT/dr = h·(T - T_inf)
    lhs = -p.k * dTdr[-1]
    rhs = p.h * (T[-1] - p.T_inf)
    assert abs(lhs - rhs) / abs(rhs) < 1e-6


def test_annulus_characteristic_numbers():
    """特征数与原研究档案报告值一致：Bi≈0.0294、t_scale≈1696 s、C1*≈-0.3401。"""
    p = AnnulusParams()
    assert np.isclose(p.alpha, 1.274e-5, rtol=1e-3)
    assert np.isclose(p.Bi, 0.0294, rtol=1e-9)
    assert np.isclose(p.t_scale, 1696.0, rtol=1e-3)
    # C1* = -R_in/R_out = -0.340136…，档案报告值 -0.3401 为四位舍入
    assert np.isclose(p.c1_star, -0.3401, atol=5e-5)
    assert p.c1_star == -p.R_in_star
    assert np.isclose(p.T_scale, 294.0, rtol=1e-6)


def test_cartesian_roundtrip():
    """2D 直角问题：温度与热源归一化/反归一化往返恒等。"""
    p = CartesianHeatParams()
    rng = np.random.default_rng(1)

    u = rng.uniform(p.u0, 500.0, (8, 8))
    assert np.allclose(p.denormalize_temperature(p.normalize_temperature(u)), u, rtol=1e-12, atol=1e-10)

    f = rng.uniform(0.0, p.f_max, (8, 8))
    assert np.allclose(p.denormalize_source(p.normalize_source(f)), f, rtol=1e-12, atol=1e-10)

    # 默认参数数值约定
    assert p.alpha == 0.01 and p.beta == 5.0 and p.u0 == 20.0
    assert p.f_mean == 100.0 and p.f_std == 30.0 and p.f_max == 150.0


def test_heat_line_variants_and_roundtrip():
    """1D 线参数变体：散热 a=0.4 / 加热 a=0.1；温度与时间归一化往返恒等。"""
    p_cool = heat_line_cooling()
    p_heat = heat_line_heating()
    assert p_cool.a == 0.4
    assert p_heat.a == 0.1
    assert np.isclose(p_cool.t_char, 1.0 / 0.4, rtol=1e-12)

    rng = np.random.default_rng(2)
    u = rng.uniform(-1.0, 1.0, 32)
    assert np.allclose(p_cool.denormalize_temperature(p_cool.normalize_temperature(u)), u, rtol=1e-12, atol=1e-12)

    t = rng.uniform(0.0, 10.0, 32)
    assert np.allclose(p_heat.denormalize_time(p_heat.normalize_time(t)), t, rtol=1e-12, atol=1e-12)
