"""torch_loop：小预算训练 smoke、产物完整性、strict 加载与同种子逐点复现。"""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("neuralop")

from heatops.data import Heat1DDataset
from heatops.models.fno import HeatOperatorFNO
from heatops.physics.losses import LossWeights, compute_loss_terms_1d, weighted_loss_1d
from heatops.problems import heat_line_cooling
from heatops.training import TrainConfig, set_seed, train_operator_model

_WEIGHTS = LossWeights(data=1.0, pde=10.0, ic=200.0, bc=100.0)


def _make_dataset() -> Heat1DDataset:
    return Heat1DDataset.generate(
        params=heat_line_cooling(), nx=16, nt=16, t_max=0.02, n_samples=8, seed=11
    )


def _loss_fn(model, b_u0, b_true, b_grid):
    terms, _ = compute_loss_terms_1d(model, b_u0, b_true, b_grid, a=0.4)
    return weighted_loss_1d(terms, _WEIGHTS), terms


def _run_once(run_dir) -> dict:
    set_seed(0)  # 模型初始化随机性（训练循环内部会按 cfg.seed 再播种）
    model = HeatOperatorFNO(n_modes=(8, 8), hidden_channels=16, a=0.4)
    cfg = TrainConfig(
        run_dir=run_dir, seed=0, epochs=3, batch_size=4,
        checkpoint_every=2, log_every=99, device="cpu",
    )
    return train_operator_model(model, _make_dataset(), _loss_fn, cfg)


def test_train_loop_artifacts(tmp_path):
    """小预算训练：历史长度、checkpoint/npy/config.json/loss 曲线齐全，strict 可加载。"""
    history = _run_once(tmp_path / "run")
    assert list(history) == ["step", "total", "data", "pde", "ic", "bc"]
    assert len(history["step"]) == 4  # epochs+1（含第 0 步）
    assert np.all(np.isfinite(history["total"]))

    run_dir = tmp_path / "run"
    assert (run_dir / "checkpoints" / "checkpoint_best.pt").exists()
    assert (run_dir / "checkpoints" / "checkpoint_last.pt").exists()
    assert (run_dir / "loss_curve.png").exists()

    # loss_history.npy 为结构化数组，无需 allow_pickle
    arr = np.load(run_dir / "loss_history.npy")
    assert arr.dtype.names == ("step", "total", "data", "pde", "ic", "bc")
    assert np.allclose(arr["total"], history["total"])

    cfg_saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_saved["seed"] == 0 and cfg_saved["train"]["epochs"] == 3
    assert cfg_saved["code_revision"].startswith("heatops-")

    # 评估约定：strict=True 加载最优 checkpoint（同架构无缺失/冗余键）
    model = HeatOperatorFNO(n_modes=(8, 8), hidden_channels=16, a=0.4)
    state = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", weights_only=True)
    model.load_state_dict(state, strict=True)


def test_train_loop_seed_reproducible(tmp_path):
    """同 config 同种子两次运行：loss 历史逐点一致（H104 的测试尺度守卫）。"""
    h1 = _run_once(tmp_path / "r1")
    h2 = _run_once(tmp_path / "r2")
    assert h1["step"] == h2["step"]
    for key in ("total", "data", "pde", "ic", "bc"):
        assert np.array_equal(h1[key], h2[key]), f"{key} 历史不一致"


def test_batch_size_guard(tmp_path):
    """batch_size 超过训练样本数时明确报错。"""
    model = HeatOperatorFNO(n_modes=(8, 8), hidden_channels=16, a=0.4)
    cfg = TrainConfig(run_dir=tmp_path / "run", seed=0, epochs=1, batch_size=99)
    with pytest.raises(ValueError, match="batch_size"):
        train_operator_model(model, _make_dataset(), _loss_fn, cfg)
