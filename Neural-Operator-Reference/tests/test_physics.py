"""physics：解析场 PDE 残差归零、损失项形状与加权组合。"""

import pytest

torch = pytest.importorskip("torch")

from heatops.physics.losses import LossWeights, compute_loss_terms_1d, weighted_loss_1d
from heatops.physics.residuals import heat_residual_1d


def test_residual_vanishes_on_analytic_solution():
    """解析解 u = exp(-a·π²·t)·sin(π·x) 的残差 u_t - a·u_xx ≈ 0（float64 量级）。

    u_t = -a·π²·u 且 u_xx = -π²·u，解析残差恒为 0；autograd 对该复合解析函数
    的导数精确到浮点舍入，故残差量级为 1e-12 以下。
    """
    a = 0.4
    x = torch.linspace(0.0, 1.0, 33, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, 25, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    grid = torch.stack([X, T], dim=-1).unsqueeze(0).requires_grad_(True)  # (1,nx,nt,2)
    pred = (torch.exp(-a * torch.pi**2 * grid[..., 1:2]) * torch.sin(torch.pi * grid[..., 0:1]))

    r = heat_residual_1d(pred, grid, a)
    assert r.shape == pred.shape  # 全域同形状返回（含边界，见模块 docstring 裁切约定）
    assert float(r.detach().abs().max()) < 1e-9


def test_residual_requires_grad_guard():
    """grid 未启用 requires_grad 时给出明确报错而非 autograd 晦涩异常。"""
    grid = torch.zeros(1, 4, 4, 2)
    pred = torch.zeros(1, 4, 4, 1)
    with pytest.raises(ValueError, match="requires_grad"):
        heat_residual_1d(pred, grid, a=0.4)


def test_weighted_loss_combination():
    """加权组合数值正确、标量形状、零权重归零、缺键报错。"""
    terms = {
        "data": torch.tensor(2.0),
        "pde": torch.tensor(3.0),
        "ic": torch.tensor(5.0),
        "bc": torch.tensor(7.0),
    }
    total = weighted_loss_1d(terms, LossWeights(data=1.0, pde=10.0, ic=200.0, bc=100.0))
    assert total.dim() == 0
    assert float(total) == pytest.approx(2.0 + 30.0 + 1000.0 + 700.0)

    assert float(weighted_loss_1d(terms, LossWeights(data=1.0))) == pytest.approx(2.0)
    assert float(weighted_loss_1d(terms, LossWeights(data=0.0, pde=0.0, ic=0.0, bc=0.0))) == 0.0

    with pytest.raises(KeyError, match="bc"):
        weighted_loss_1d({"data": torch.tensor(1.0)}, LossWeights())


def test_compute_loss_terms_keys_and_scalars():
    """compute_loss_terms_1d 返回四项标量且 pred 形状保持（用小线性模型替身）。"""
    nx, nt = 8, 6
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    X, T = torch.meshgrid(x, t, indexing="ij")
    grid = torch.stack([X, T], dim=-1).to(torch.float32).expand(2, -1, -1, -1)
    u0 = torch.rand(2, nx)
    u_true = torch.rand(2, nx, nt, 1)

    # 替身模型：pred = u0 广播 + sin(πx)·exp(-t)（对 grid 可导）
    def stub_model(u0_, grid_):
        base = u0_.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, nt, 1)
        return base + torch.sin(torch.pi * grid_[..., 0:1]) * torch.exp(-grid_[..., 1:2])

    terms, pred = compute_loss_terms_1d(stub_model, u0, u_true, grid, a=0.4)
    assert set(terms) == {"data", "pde", "ic", "bc"}
    assert all(v.dim() == 0 for v in terms.values())
    assert pred.shape == u_true.shape
