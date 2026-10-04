"""Heat1DDataset：构造/生成、显式 train/eval 划分、.pt 存取往返。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heatops.data import Heat1DDataset
from heatops.problems import heat_line_cooling


@pytest.fixture()
def small_dataset() -> Heat1DDataset:
    return Heat1DDataset.generate(
        params=heat_line_cooling(), nx=16, nt=8, t_max=0.05, n_samples=10, seed=7
    )


def test_generate_shapes_and_grid(small_dataset):
    """张量约定：u0 (B,nx)、u (B,nx,nt,1)、grid (1,nx,nt,2)，t[0]=0。"""
    ds = small_dataset
    assert ds.u0.shape == (10, 16)
    assert ds.u.shape == (10, 16, 8, 1)
    assert ds.u0.dtype == torch.float32 and ds.u.dtype == torch.float32
    grid = ds.grid
    assert grid.shape == (1, 16, 8, 2)
    assert torch.allclose(grid[0, :, 0, 0], ds.x.float())
    assert torch.allclose(grid[0, 0, :, 1], ds.t.float())
    assert float(ds.t[0]) == 0.0
    # 求解器约定：首帧即初始条件
    assert torch.allclose(ds.u[:, :, 0, 0], ds.u0)
    assert ds.metadata["ic"] == "sine_or_square" and ds.metadata["seed"] == 7


def test_split_disjoint_and_reproducible(small_dataset):
    """划分不重叠、覆盖全集、同种子逐点一致（修复档案「评估即训练集」缺陷）。"""
    train, eval_ds = small_dataset.split(train_ratio=0.8, seed=42)
    assert (train.n_samples, eval_ds.n_samples) == (8, 2)
    idx_tr = train.metadata["split"]["indices"]
    idx_ev = eval_ds.metadata["split"]["indices"]
    assert set(idx_tr).isdisjoint(idx_ev)
    assert set(idx_tr) | set(idx_ev) == set(range(10))

    train2, eval2 = small_dataset.split(train_ratio=0.8, seed=42)
    assert idx_tr == train2.metadata["split"]["indices"]
    assert idx_ev == eval2.metadata["split"]["indices"]
    assert torch.equal(train.u0, train2.u0) and torch.equal(eval_ds.u, eval2.u)


def test_save_load_roundtrip(small_dataset, tmp_path):
    """.pt 存取往返：张量与 metadata 一致，weights_only=True 可加载。"""
    path = small_dataset.save(tmp_path / "ds.pt")
    loaded = Heat1DDataset.load(path)
    assert torch.equal(loaded.u0, small_dataset.u0)
    assert torch.equal(loaded.u, small_dataset.u)
    assert torch.equal(loaded.x, small_dataset.x)
    assert loaded.metadata == small_dataset.metadata


def test_invalid_shapes_rejected():
    """t[0] 非 0 或形状不一致时拒绝构造（IC 约定依赖首帧）。"""
    u0 = np.zeros((2, 8))
    u = np.zeros((2, 8, 4))
    x = np.linspace(0.0, 1.0, 8)
    with pytest.raises(ValueError, match=r"t\[0\]"):
        Heat1DDataset(u0, u, x, np.linspace(0.1, 0.4, 4))
    with pytest.raises(ValueError, match="不一致"):
        Heat1DDataset(u0, np.zeros((2, 8, 5)), x, np.linspace(0.0, 0.4, 4))
