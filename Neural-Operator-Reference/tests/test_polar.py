"""polar：极坐标残差与直角残差一致性（H102）、周期损失、几何工具与极坐标损失项。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heatops.models.fno import HeatOperatorFNO3D
from heatops.physics.geometry import (
    annulus_mask,
    extract_boundary_bands,
    radial_normal_component,
)
from heatops.physics.losses import (
    LossWeightsPolar,
    compute_loss_terms_polar,
    masked_data_loss,
    periodic_loss_theta,
    random_point_mask,
    weighted_loss_polar,
)
from heatops.physics.residuals import heat_residual_polar
from heatops.problems import AnnulusParams, analytical_steady_state_dimensionless


def _field_xy(x, y):
    """光滑测试场（对 x/θ 均周期相容：经 (r cosθ, r sinθ) 定义即天然 θ 周期）。"""
    return torch.sin(2 * x) * torch.cos(2 * y) + 0.25 * x * y**2 + 0.1 * x


def _polar_grid(nr=48, ntheta=64, nt=3, r_lo=0.4, r_hi=1.0):
    """无量纲 (t*, r*, θ) 网格 (1, nt, nr, ntheta, 3)，float64。"""
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    r = torch.linspace(r_lo, r_hi, nr, dtype=torch.float64)
    th = torch.linspace(0.0, 2 * np.pi, ntheta, dtype=torch.float64)
    T, R, Th = torch.meshgrid(t, r, th, indexing="ij")
    return torch.stack([T, R, Th], dim=-1).unsqueeze(0)


def test_polar_laplacian_matches_cartesian():
    """H102：同一光滑场上极坐标链式 Laplacian 与直角坐标 autograd Laplacian 一致。

    场经 u(x, y) 定义；极坐标侧 u(r cosθ, r sinθ) 对 (r, θ) autograd 组装
    ``u_rr + u_r/r + u_θθ/r²``，直角侧对同一批物理点的 (x, y) autograd 组装
    ``u_xx + u_yy``；两者均为同一解析函数的精确微分，内部点应一致到浮点量级
    （判据按 H102：相对偏差 <1%，r 边界带除外）。
    """
    grid_p = _polar_grid().requires_grad_(True)
    r, th = grid_p[..., 1], grid_p[..., 2]
    pred_p = _field_xy(r * torch.cos(th), r * torch.sin(th))  # 与 t 无关 ⇒ u_t ≡ 0

    residual = heat_residual_polar(pred_p, grid_p, alpha_norm=1.0)
    lap_polar = -residual  # 稳态场：residual = -alpha·laplacian

    x, y = r * torch.cos(th), r * torch.sin(th)
    grid_c = torch.stack([x.detach(), y.detach()], dim=-1).requires_grad_(True)
    pred_c = _field_xy(grid_c[..., 0], grid_c[..., 1])
    g1 = torch.autograd.grad(pred_c, grid_c, torch.ones_like(pred_c), create_graph=True)[0]
    u_xx = torch.autograd.grad(g1[..., 0], grid_c, torch.ones_like(g1[..., 0]), create_graph=True)[0][..., 0]
    u_yy = torch.autograd.grad(g1[..., 1], grid_c, torch.ones_like(g1[..., 1]), create_graph=True)[0][..., 1]
    lap_cart = u_xx + u_yy

    lp = lap_polar[0, 0].detach().numpy()  # (nr, ntheta)
    lc = lap_cart[0, 0].detach().numpy()
    interior = np.s_[1:-1, :]  # r 边界带除外（H102 scope）
    floor = 0.01 * np.abs(lc[interior]).max()
    rel = np.abs(lp[interior] - lc[interior]) / np.maximum(np.abs(lc[interior]), floor)
    assert float(rel.max()) < 0.01


def test_polar_residual_vanishes_on_dimensionless_steady_solution():
    """无量纲稳态解析解 u* = C1*·ln r* + C2*（轴对称、定常）的残差为 0。

    解析解满足 u_rr + u_r/r = 0 且 u_t = u_θθ = 0；该测试直接校验 1/r 项组装
    （对应 problems.annulus 的圆环稳态解析解）。
    """
    params = AnnulusParams()
    grid = _polar_grid(nr=32, ntheta=16, nt=3, r_lo=params.R_in_star, r_hi=1.0).requires_grad_(True)
    r = grid[..., 1]
    pred = params.c1_star * torch.log(r) + params.c2_star
    pred = pred + 0.0 * grid[..., 0]  # 保持对 t 通道的计算图连通

    r_res = heat_residual_polar(pred, grid, alpha_norm=1.0)
    assert r_res.shape == pred.shape
    assert float(r_res.detach().abs().max()) < 1e-9

    # 与 problems 的解析解函数逐点一致（同一实现的两侧）
    r_np = r.detach().numpy()
    assert np.allclose(pred.detach().numpy(), analytical_steady_state_dimensionless(r_np, params))


def test_polar_residual_requires_grad_guard():
    """grid 未启用 requires_grad 时给出明确报错而非 autograd 晦涩异常。"""
    grid = torch.zeros(1, 2, 4, 4, 3)
    pred = torch.zeros(1, 2, 4, 4)
    with pytest.raises(ValueError, match="requires_grad"):
        heat_residual_polar(pred, grid, alpha_norm=1.0)
    grid_bad = torch.zeros(1, 2, 4, 4, 2).requires_grad_(True)
    with pytest.raises(ValueError, match="坐标通道"):
        heat_residual_polar(torch.zeros(1, 2, 4, 4), grid_bad, alpha_norm=1.0)


def test_periodic_loss_zero_on_periodic_field():
    """θ 周期场（含两端点同母线）的周期损失为 0；非周期场 > 0。"""
    th = torch.linspace(0.0, 2 * np.pi, 33, dtype=torch.float64)
    base = torch.cos(th).expand(2, 3, 4, 33)  # (B, nt, nr, ntheta)，u[...,0]==u[...,-1]
    dudth = (-torch.sin(th)).expand(2, 3, 4, 33)
    assert float(periodic_loss_theta(base)) == pytest.approx(0.0)
    assert float(periodic_loss_theta(base, dudth)) == pytest.approx(0.0)

    non_periodic = th.expand(2, 3, 4, 33)  # u = θ，两端不对齐
    assert float(periodic_loss_theta(non_periodic)) > 1.0


def test_random_point_mask_rate_and_fallback():
    """采样率统计正确、rate>=1 全选、空采样回退全选、负 rate 报错。"""
    torch.manual_seed(0)
    m = random_point_mask((2000,), 0.2)
    frac = float(m.to(torch.float32).mean())
    assert 0.1 < frac < 0.3
    assert bool(random_point_mask((8,), 1.0).all())
    assert bool(random_point_mask((8,), 0.0).all())  # 空采样回退（docstring 约定）
    with pytest.raises(ValueError, match="rate"):
        random_point_mask((8,), -0.1)


def test_masked_data_loss_equals_masked_mean():
    """掩码加权 data loss 等于掩码内真均值（而非档案的掩码后全域均值）。"""
    pred = torch.arange(20, dtype=torch.float64).reshape(4, 5)
    target = torch.zeros(4, 5, dtype=torch.float64)
    mask = torch.zeros(4, 5, dtype=torch.bool)
    mask[1, 2] = mask[3, 4] = True
    loss = masked_data_loss(pred, target, mask)
    manual = ((pred[1, 2] ** 2 + pred[3, 4] ** 2) / 2).item()
    assert float(loss) == pytest.approx(manual)
    assert float(masked_data_loss(pred, target)) == pytest.approx(float((pred**2).mean()))


def test_annulus_mask_and_boundary_bands():
    """mask 形状/dtype/位置正确；边界环带按半径分内外、互不相交且贴边界。"""
    n, r_in, r_out, L = 41, 0.2, 0.4, 1.0
    mask = annulus_mask(n, r_in=r_in, r_out=r_out, domain_size=L)
    assert mask.shape == (n, n) and mask.dtype == bool
    assert not mask[n // 2, n // 2]  # 圆心在内孔
    assert mask[32, 20]  # (x, y) = (0.8, 0.5)，dist = 0.3 ∈ [0.2, 0.4]
    assert not mask[0, 0]  # 角点在外域

    inner, outer = extract_boundary_bands(mask, r_in=r_in, r_out=r_out, domain_size=L)
    assert inner.shape == mask.shape == outer.shape
    assert not (inner & outer).any()  # 互不相交
    assert (inner | outer).any() and not ((inner | outer) & ~mask).any()  # 边界 ⊆ 掩码

    cx = cy = L / 2
    xv = np.linspace(0, L, n)
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    dist = np.hypot(X - cx, Y - cy)
    pix = L / (n - 1)
    assert abs(dist[inner].mean() - r_in) < 2 * pix  # 内环带贴 r_in
    assert abs(dist[outer].mean() - r_out) < 2 * pix  # 外环带贴 r_out


def test_radial_normal_component_of_radial_field():
    """径向场 u = dist（梯度即径向单位向量）的径向法向分量恒为 1。"""
    n = 41
    xv = np.linspace(0.0, 1.0, n)
    X, Y = np.meshgrid(xv, xv, indexing="ij")
    center = (0.5, 0.5)
    d = np.hypot(X - center[0], Y - center[1])
    dudx = (X - center[0]) / np.maximum(d, 1e-12)
    dudy = (Y - center[1]) / np.maximum(d, 1e-12)
    comp = radial_normal_component(dudx, dudy, X, Y, center=center)
    off_center = d > 0.01
    assert np.allclose(comp[off_center], 1.0, atol=1e-9)


def test_fno3d_forward_shape_and_coordinate_grad():
    """HeatOperatorFNO3D 输出形状 (B, nt, nr, ntheta) 且对坐标网格可导（PINO 前提）。"""
    torch.manual_seed(0)
    model = HeatOperatorFNO3D(n_modes=(2, 3, 3), hidden_channels=8)
    grid = _polar_grid(nr=8, ntheta=8, nt=4).to(torch.float32).expand(2, -1, -1, -1, -1).clone()
    grid.requires_grad_(True)
    q_field = torch.rand(2, 4, 8, 8)
    pred = model(q_field, grid)
    assert pred.shape == (2, 4, 8, 8)
    g = torch.autograd.grad(pred.sum(), grid)[0]
    assert g.shape == grid.shape and bool(g.abs().sum() > 0)


def test_compute_loss_terms_polar_keys_and_weighted_sum():
    """compute_loss_terms_polar 返回六项标量；加权组合数值正确、缺键报错。"""
    nt, nr, ntheta = 3, 8, 8
    grid = _polar_grid(nr=nr, ntheta=ntheta, nt=nt).to(torch.float32).expand(2, -1, -1, -1, -1)

    def stub_model(q_field, grid_):
        return (
            torch.sin(np.pi * grid_[..., 1]) * torch.cos(grid_[..., 2])
            + 0.1 * q_field
            + 0.0 * grid_[..., 0]
        )

    q_field = torch.rand(2, nt, nr, ntheta)
    u_true = torch.rand(2, nt, nr, ntheta)
    terms, pred = compute_loss_terms_polar(
        stub_model, q_field, u_true, grid, alpha_norm=1.0, bi=0.03, pde_sample_rate=0.5
    )
    assert set(terms) == {"data", "pde", "ic", "bc_inner", "bc_outer", "periodic"}
    assert all(v.dim() == 0 for v in terms.values())
    assert pred.shape == u_true.shape

    weights = LossWeightsPolar(data=1.0, pde=0.05, ic=2.0, bc_inner=5.0, bc_outer=2.0, periodic=2.0)
    total = weighted_loss_polar(terms, weights)
    manual = sum(float(w) * float(terms[k].detach()) for k, w in vars(weights).items())
    assert float(total) == pytest.approx(manual)

    with pytest.raises(KeyError, match="bc_inner"):
        weighted_loss_polar({"data": torch.tensor(1.0)}, weights)
