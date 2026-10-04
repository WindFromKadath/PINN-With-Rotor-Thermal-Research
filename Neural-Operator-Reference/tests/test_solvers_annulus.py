"""圆环求解器校核：稳态 FDM 对解析解、能量平衡、瞬态 C-N 长时极限、单调性。"""

import numpy as np

from heatops.problems import AnnulusParams, analytical_steady_state
from heatops.solvers import (
    check_energy_balance,
    check_physical_plausibility,
    solve_annulus_steady,
    solve_annulus_transient_cn,
)


def test_steady_fdm_matches_analytical():
    """H101：Nr=200 稳态 FDM 与解析解最大相对误差 <2%。

    误差由边界一阶单侧差分主导（内部为二阶守恒格式），
    实际量级约 0.2%，阈值 2% 为假设表既定判据。
    """
    p = AnnulusParams()
    out = solve_annulus_steady(p, nr=200)
    T_an = analytical_steady_state(out["r"], p)
    rel = np.max(np.abs(out["T"] - T_an) / np.abs(T_an))
    assert rel < 0.02


def test_steady_energy_balance():
    """H101：稳态能量平衡 Q_in≈Q_out 相对误差 <1%。

    离散守恒误差来自边界半面半径取值（O(dr)），Nr=200 时约 0.7%。
    """
    p = AnnulusParams()
    out = solve_annulus_steady(p, nr=200)
    res = check_energy_balance(out["T"], p)
    assert res["q_in_total"] > 0.0
    assert res["relative_error"] < 0.01


def test_steady_monotonicity_and_boundary_ordering():
    """物理合理性：温度沿 r 单调不增，内壁最热，全场不低于环境温度。"""
    p = AnnulusParams()
    out = solve_annulus_steady(p, nr=200)
    T, r = out["T"], out["r"]

    assert np.all(np.diff(T) <= 0.0)
    assert T[0] > T[-1] > p.T_inf

    plaus = check_physical_plausibility(T, r, p)
    assert plaus["is_monotonic_decreasing"]
    assert plaus["above_ambient"]
    assert plaus["inner_flux_relative_error"] < 1e-9  # 内边界为方程行，残差仅含求解器舍入
    assert plaus["outer_robin_relative_residual"] < 1e-9  # 外边界同理


def test_transient_cn_converges_to_steady():
    """瞬态 C-N 长时极限收敛到稳态解析解。

    阈值依据：
    - 半控制体边界修正在稳态时严格满足每弧度能量守恒，离散稳态与解析解的
      偏差仅剩内部 O(dr²) 截断，nr=64 时约 1e-4 相对量级；
    - 时间尾部按集中参数估计 τ ≈ rho·cp·(R_out²-R_in²)/(2·h·R_out) ≈ 2.6e4 s，
      t_final = 2.5e5 s ≈ 10τ，残余约 3500·e^-10 ≈ 0.16 °C（相对约 5e-5）；
    - 取 0.5% 阈值，对上述两项之和留约一个数量级余量。

    C-N 无条件稳定且稳态极限与 dt 无关，取 dt=25 s 控制步数（共 1e4 步）。
    """
    p = AnnulusParams()
    nr, dt, t_final = 64, 25.0, 2.5e5
    nt = int(round(t_final / dt)) + 1
    out = solve_annulus_transient_cn(p, nr=nr, dt=dt, nt=nt, save_every=2500)

    assert out["T_history"].shape[1] == nr
    assert np.isclose(out["t"][-1], t_final, rtol=0, atol=dt)

    T_an = analytical_steady_state(out["r"], p)
    rel = np.max(np.abs(out["T_history"][-1] - T_an) / np.abs(T_an))
    assert rel < 0.005


def test_transient_cn_zero_flux_uniform():
    """零内热流 + h→0 时均匀初始场应保持均匀（绝热不变量）。"""
    p = AnnulusParams()
    out = solve_annulus_transient_cn(p, nr=16, dt=1.0, nt=11, q_in=0.0, T0=p.T_inf)
    # q_in=0 但 h>0 且 T=T_inf：Robin 通量为零，均匀场为稳态解
    assert np.allclose(out["T_history"], p.T_inf, atol=1e-10)
