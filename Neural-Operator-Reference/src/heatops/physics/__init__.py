"""heatops.physics — PDE 残差（autograd）与物理损失项。"""

from .losses import (
    LossWeights,
    bc_loss_1d,
    compute_loss_terms_1d,
    data_loss_1d,
    ic_loss_1d,
    pde_loss_1d,
    weighted_loss_1d,
)
from .residuals import heat_residual_1d

__all__ = [
    "LossWeights",
    "bc_loss_1d",
    "compute_loss_terms_1d",
    "data_loss_1d",
    "heat_residual_1d",
    "ic_loss_1d",
    "pde_loss_1d",
    "weighted_loss_1d",
]
