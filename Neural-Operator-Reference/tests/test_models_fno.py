"""HeatOperatorFNO：输出形状/非负性、硬约束 ansatz 的 t=0 行为。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from heatops.data.datasets import make_xt_grid
from heatops.models.fno import HeatOperatorFNO


def _small_model(**kw) -> HeatOperatorFNO:
    torch.manual_seed(0)
    return HeatOperatorFNO(n_modes=(8, 8), hidden_channels=16, **kw)


def _grid(nx: int, nt: int, batch: int) -> torch.Tensor:
    x = torch.linspace(0.0, 1.0, nx, dtype=torch.float64)
    t = torch.linspace(0.0, 0.5, nt, dtype=torch.float64)
    return make_xt_grid(x, t).expand(batch, -1, -1, -1)


def test_output_shape_and_nonnegative():
    """输出 (B, nx, nt, 1) 且经 Softplus 非负；use_ansatz=False 路径同样可用。"""
    model = _small_model()
    u0 = torch.rand(3, 16)
    pred = model(u0, _grid(16, 12, 3))
    assert pred.shape == (3, 16, 12, 1)
    assert torch.all(pred >= 0.0)

    model_plain = _small_model(use_ansatz=False)
    pred_plain = model_plain(u0, _grid(16, 12, 3))
    assert pred_plain.shape == (3, 16, 12, 1)
    assert torch.all(pred_plain >= 0.0)


def test_ansatz_t0_equals_softplus_of_ic():
    """t=0 时 ansatz 输出恰为 Softplus(u0)（代数恒等，与网络权重无关）。

    ansatz ``u = Softplus(u0·exp(-a·t) + t·FNO)`` 在 t=0 处网络项被 t 乘没、
    衰减因子为 1，故 ``pred(t=0) == Softplus(u0)`` 精确成立。

    Softplus 与 IC 的逼近关系：``Softplus(s) - s = ln(1 + e^{-s})``，仅当
    u0 明显大于 0 时二者接近（如 u0 ≥ 3 时偏差 ≤ 0.049）；u0=0 处偏差为
    ln2 ≈ 0.69。因此 ansatz 只是「近似满足」初始条件，IC 仍由 ic 损失项约束，
    档案注释把 ``u0·exp(-a·t)`` 称作「基础解部分」并不准确（解析衰减率随
    模态变化，为 exp(-a·k²·π²·t)）。
    """
    model = _small_model()
    u0 = 3.0 + torch.rand(4, 16)  # u0 ∈ [3, 4)，Softplus 逼近误差 ≤ 0.049
    pred = model(u0, _grid(16, 12, 4))
    softplus_u0 = torch.nn.functional.softplus(u0)
    assert torch.allclose(pred[:, :, 0, 0], softplus_u0, atol=1e-6)
    # Softplus(u0) ≈ u0 的逼近质量（大正值区间）
    assert float((pred[:, :, 0, 0] - u0).detach().abs().max()) < 0.05


def test_ansatz_t0_independent_of_weights():
    """t=0 输出对网络权重不变：两个不同初始化模型给出相同的 Softplus(u0)。"""
    u0 = torch.rand(2, 16)
    pred_a = _small_model()(u0, _grid(16, 12, 2))
    torch.manual_seed(123)
    pred_b = HeatOperatorFNO(n_modes=(8, 8), hidden_channels=16)(u0, _grid(16, 12, 2))
    assert torch.allclose(pred_a[:, :, 0, 0], pred_b[:, :, 0, 0], atol=1e-6)
