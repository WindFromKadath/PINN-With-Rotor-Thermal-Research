"""heatops.data — 数据集封装（torch 张量约定、显式 train/eval 划分、.pt 存取）。"""

from .datasets import Heat1DDataset, cache_key_from_params, make_xt_grid

__all__ = [
    "Heat1DDataset",
    "cache_key_from_params",
    "make_xt_grid",
]
