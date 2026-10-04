"""1D/2D 直角坐标求解器校核：能量衰减、带源加热、Robin 均匀不变量、ADR 稳态极限。"""

import numpy as np

from heatops.evaluation import relative_l2
from heatops.fields import gaussian_heat_source, sample_single_sine, sample_square_wave
from heatops.solvers import (
    solve_adr_robin,
    solve_heat_1d_explicit,
    solve_heat_1d_source_explicit,
    solve_heat_2d_robin,
)


def test_heat_1d_dirichlet_energy_decay():
    """散热线：Dirichlet 零边界下非负初始场的总热量单调不增。

    r = a·dt/dx² ∈ (0, 0.5) 时显式更新为非负系数的凸组合，
    相邻边界节点的热量被边界移除，总和严格不增。
    """
    out = solve_heat_1d_explicit(
        a=0.4, nx=32, nt=25, T_max=0.1,
        ic_sampler=sample_square_wave, n_samples=4, seed=0,
    )
    assert out["u"].shape == (4, 32, 25)
    assert out["u0"].shape == (4, 32)
    assert np.isclose(out["t"][-1], 0.1, rtol=0, atol=1e-12)
    assert np.all(np.isfinite(out["u"]))
    assert np.all(out["u"] >= 0.0)  # 非负初值在稳定显式格式下保持非负

    dx = out["x"][1] - out["x"][0]
    heat = out["u"].sum(axis=1) * dx  # (B, nt) 总热量
    assert np.all(np.diff(heat, axis=1) <= 1e-12)


def test_heat_1d_seed_reproducible():
    """同一种子两次生成逐点一致。"""
    kw = dict(a=0.4, nx=16, nt=8, T_max=0.02, ic_sampler=sample_single_sine, n_samples=2)
    out1 = solve_heat_1d_explicit(**kw, seed=7)
    out2 = solve_heat_1d_explicit(**kw, seed=7)
    assert np.array_equal(out1["u"], out2["u"])
    assert np.array_equal(out1["u0"], out2["u0"])


def test_heat_1d_source_heating():
    """带源加热线：终态平均温度高于初态且数值有限；CFL 子步划分生效。"""
    nx, nt, dt_frame = 32, 20, 0.01
    x = np.linspace(0.0, 1.0, nx)
    out = solve_heat_1d_source_explicit(
        a=0.1, nx=nx, nt=nt, dt_frame=dt_frame,
        source=gaussian_heat_source(x),
        ic_sampler=sample_single_sine, n_samples=3, seed=3,
    )
    assert out["u"].shape == (3, nx, nt)
    assert np.all(np.isfinite(out["u"]))
    # dt_frame=0.01 超过 CFL 安全线（约 9·dx²/(2a)·0.9 ≈ 4.5e-3），必须子步划分
    assert out["steps_per_frame"] > 1
    assert np.allclose(out["t"], np.arange(nt) * dt_frame)

    mean_initial = out["u0"].mean(axis=1)
    mean_final = out["u"][:, :, -1].mean(axis=1)
    assert np.all(mean_final > mean_initial)


def test_heat_2d_robin_uniform_invariance():
    """2D Robin：均匀初值 + 零源 + 环境温度等于初值时，场应严格保持均匀。"""
    n, u0 = 16, 20.0
    out = solve_heat_2d_robin(
        alpha=0.01, f_grid=np.zeros((n, n)), u0=u0, n=n, T_max=0.05, nt_save=5, beta=5.0,
    )
    assert out["u"].shape == (5, n, n)
    assert np.allclose(out["u"], u0, rtol=0, atol=1e-12)
    assert np.isclose(out["t"][-1], 0.05, rtol=0, atol=1e-12)


def test_heat_2d_robin_auto_stability_refinement():
    """期望帧间隔超过稳定上限时自动加密步长，且末帧精确对齐 T_max、无发散。"""
    n = 16
    dx = 1.0 / (n - 1)
    dt_max = 0.22 * dx**2 / 0.01
    T_max, nt_save = 0.5, 5
    assert T_max / (nt_save - 1) > dt_max  # 确认触发自动加密分支

    rng = np.random.default_rng(5)
    f_grid = 50.0 * rng.random((n, n))  # 非负热源，°C/s
    out = solve_heat_2d_robin(
        alpha=0.01, f_grid=f_grid, u0=20.0, n=n, T_max=T_max, nt_save=nt_save, beta=5.0,
    )
    assert out["dt"] < dt_max
    assert np.isclose(out["t"][-1], T_max, rtol=0, atol=1e-12)
    assert out["u"].shape == (nt_save, n, n)
    assert np.all(np.isfinite(out["u"]))
    # 非负热源 + 环境温度 20：全场温度不低于 20
    assert np.all(out["u"] >= 20.0 - 1e-9)


def test_adr_robin_steady_state_quadratic():
    """ADR（C-N + Robin）长时极限收敛到常源抛物线稳态解析解。

    对称常源 s、两端相同 beta/u0 时稳态解为
    u(x) = u0e + s/(8·alpha) + s/(2·alpha·beta) - s·(x-0.5)²/(2·alpha)。
    虚点消元对二次剖面精确，误差仅含时间尾部：
    最慢模态衰减率 alpha·μ₁² ≈ 0.26（μ₁≈2.28 满足 μ·tan(μ/2)=5），
    t=50 s ≈ 13τ，残余约 450·e^-13 ≈ 1e-3 °C，阈值 RelL2 < 2e-3 留有余量。
    """
    alpha, s_const, beta, u0e = 0.05, 100.0, 5.0, 20.0
    nx, nt, tmax = 51, 201, 50.0
    out = solve_adr_robin(
        xmin=0.0, xmax=1.0, tmin=0.0, tmax=tmax,
        alpha=alpha,
        source_values=np.zeros(nx), source_points=np.linspace(0.0, 1.0, nx),
        beta_left=beta, u0_left=u0e, beta_right=beta, u0_right=u0e,
        u_initial=u0e,
        source_scale=0.0, constant_source=s_const,
        nx=nx, nt=nt,
    )
    x = out["x"]
    u_steady_analytic = u0e + s_const / (8.0 * alpha) + s_const / (2.0 * alpha * beta) \
        - s_const * (x - 0.5) ** 2 / (2.0 * alpha)
    assert relative_l2(u_steady_analytic, out["u"][:, -1]) < 2e-3
