"""DeepXDE 线 CLI 冒烟测试：micro 规模端到端贯通 + 全部 config 可解析。

deepxde 不可用时整文件 skip（``pytest.importorskip``）；micro 配置把网格/迭代
压到极小（单次测试约秒级），只验证管线贯通与产物完整性，不验证收敛——
收敛性验证属于对应 run_id 的实验记录，不属于单元测试。
"""

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")  # 必须先于 import deepxde

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

dde = pytest.importorskip("deepxde")
torch = pytest.importorskip("torch")

REPO_ROOT = Path(__file__).resolve().parents[1]

_PHYSICS_CARTESIAN = {
    "alpha": 0.01, "beta": 5.0, "u0": 20.0,
    "f_mean": 100.0, "f_std": 30.0, "f_max": 150.0,
}


def _write_config(tmp_path: Path, name: str, cfg: dict) -> Path:
    path = tmp_path / name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True)
    return path


def _check_common_artifacts(run_dir: Path):
    """三条线共用产物完整性检查。"""
    assert (run_dir / "config.json").exists()
    assert (run_dir / "train_log.txt").exists()
    assert (run_dir / "loss_history.npz").exists()
    assert (run_dir / "loss_curve.png").exists()
    assert list((run_dir / "checkpoints").glob("*.pt")), "checkpoint 缺失"
    assert (run_dir / "plots").is_dir() and list((run_dir / "plots").glob("*.png"))
    hist = np.load(run_dir / "loss_history.npz")
    assert hist["steps"].ndim == 1 and hist["loss_train"].shape[0] == hist["steps"].shape[0]
    assert np.all(np.isfinite(hist["loss_train"]))
    cfg_saved = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    assert cfg_saved["code_revision"].startswith("heatops-")


# ---------------------------------------------------------------- config 解析


def test_all_configs_parse_and_smoke_overrides():
    """5 个 config 均可解析，且各 CLI 的 --smoke 覆写生效。"""
    from heatops.cli.run_deeponet_2d import load_config as load_deeponet
    from heatops.cli.run_pi_deeponet import load_config as load_pi
    from heatops.cli.run_pinn_annulus import load_config as load_pinn

    cfg = load_deeponet(REPO_ROOT / "configs" / "deeponet_2d.yaml", smoke=True)
    assert cfg["experiment"] == "E111"
    assert cfg["training"]["adam_iterations"] < 10000  # smoke 覆写生效
    assert cfg["model"]["branch_layers"][0] == cfg["data"]["nx"] ** 2

    for name, geometry in (("pi_deeponet_1d.yaml", "1d"), ("pi_deeponet_2d.yaml", "2d")):
        cfg = load_pi(REPO_ROOT / "configs" / name, smoke=True)
        assert cfg["geometry"] == geometry
        assert cfg["training"]["adam_iterations"] < 10000
        assert cfg["operator"]["num_function"] < 500

    for name, mode in (("pinn_annulus_steady.yaml", "steady"), ("pinn_annulus_transient.yaml", "transient")):
        cfg = load_pinn(REPO_ROOT / "configs" / name, smoke=True)
        assert cfg["mode"] == mode
        assert cfg["training"]["adam_iterations"] < 10000
        assert cfg["collocation"]["num_domain"] < 4000


# ---------------------------------------------------------------- CLI micro 冒烟


def test_run_deeponet_2d_micro(tmp_path, monkeypatch):
    """数据生成 + TripleCartesianProd 训练 + 评估贯通；数据缓存落盘。"""
    from heatops.cli.run_deeponet_2d import main

    cfg = {
        "experiment": "T950",
        "seed": 0,
        "physics": {**_PHYSICS_CARTESIAN, "u_scale": 150.0},
        "data": {"nx": 6, "nt": 3, "t_max": 0.5, "n_samples": 6, "seed": 123,
                 "length_scale": 0.2, "split_ratio": 0.67, "split_seed": 456},
        "model": {"branch_layers": [36, 16, 8], "trunk_layers": [3, 16, 8]},
        "training": {"adam_iterations": 3, "adam_lr": 1e-3, "adam_decay": None,
                     "use_lbfgs": False, "display_every": 1, "checkpoint_period": 2},
        "zip_artifacts": True,
    }
    config_path = _write_config(tmp_path, "deeponet_2d_micro.yaml", cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "T950"
    _check_common_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_train"] == 4 and metrics["n_eval"] == 2
    assert np.isfinite(metrics["rel_l2_mean"])
    assert metrics["rmse_celsius"] >= 0.0
    # 数据缓存（键含参数哈希）与预测 npz
    assert list((tmp_path / "data" / "processed").glob("deeponet2d_*.npz"))
    assert (run_dir / "predictions_eval.npz").exists()
    assert (tmp_path / "artifacts" / "T950.zip").exists()


def test_run_pi_deeponet_1d_micro(tmp_path, monkeypatch):
    """1D PI-DeepONet：PDEOperatorCartesianProd 训练 + solve_adr_robin 对照贯通。"""
    from heatops.cli.run_pi_deeponet import main

    cfg = {
        "experiment": "T951",
        "seed": 0,
        "geometry": "1d",
        "physics": _PHYSICS_CARTESIAN,
        "collocation": {"num_domain": 30, "num_boundary": 10, "num_initial": 10,
                        "num_test_points": 20, "train_distribution": "uniform"},
        "function_space": {"length_scale": 0.3, "n_sensors": 10},
        "operator": {"num_function": 8, "num_function_test": None, "batch_size": 4},
        "model": {"branch_layers": [10, 16, 8], "trunk_layers": [2, 16, 8]},
        "training": {"adam_iterations": 3, "adam_lr": 5e-4, "adam_decay": None,
                     "use_lbfgs": False, "display_every": 1, "checkpoint_period": 2},
        "eval": {"n_test": 1, "seed": 789, "nx": 21, "nt": 11},
        "zip_artifacts": True,
    }
    config_path = _write_config(tmp_path, "pi_deeponet_1d_micro.yaml", cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "T951"
    _check_common_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["geometry"] == "1d"
    assert metrics["truth_solver"] == "solve_adr_robin"
    assert np.isfinite(metrics["rel_l2_mean"])


def test_run_pi_deeponet_2d_micro(tmp_path, monkeypatch):
    """2D PI-DeepONet：无量纲线训练 + solve_heat_2d_robin 对照贯通。"""
    from heatops.cli.run_pi_deeponet import main

    cfg = {
        "experiment": "T952",
        "seed": 0,
        "geometry": "2d",
        "physics": {**_PHYSICS_CARTESIAN, "u_scale": 80.0},
        "collocation": {"num_domain": 30, "num_boundary": 10, "num_initial": 10,
                        "train_distribution": "Hammersley"},
        "function_space": {"length_scale": 0.3, "n_sensors_per_dim": 3, "n_grid": 10},
        "operator": {"num_function": 6, "num_function_test": None, "batch_size": 3},
        "model": {"branch_layers": [9, 16, 8], "trunk_layers": [3, 16, 8]},
        "training": {"adam_iterations": 3, "adam_lr": 1e-3, "adam_decay": None,
                     "use_lbfgs": False, "display_every": 1, "checkpoint_period": 2},
        "eval": {"n_test": 1, "seed": 789, "nx": 8, "nt": 3},
        "zip_artifacts": True,
    }
    config_path = _write_config(tmp_path, "pi_deeponet_2d_micro.yaml", cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "T952"
    _check_common_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["geometry"] == "2d"
    assert metrics["truth_solver"] == "solve_heat_2d_robin"
    assert np.isfinite(metrics["rel_l2_mean"])


def test_run_pinn_annulus_steady_micro(tmp_path, monkeypatch):
    """稳态圆环 PINN：两阶段（含 L-BFGS 小预算）+ 解析解对比指标贯通。"""
    from heatops.cli.run_pinn_annulus import main

    cfg = {
        "experiment": "T953",
        "seed": 0,
        "mode": "steady",
        "physics": {},
        "collocation": {"num_domain": 60, "num_boundary": 20, "train_distribution": "Hammersley"},
        "model": {"layer_sizes": [2, 16, 16, 1]},
        "training": {"adam_iterations": 10, "adam_lr": 1e-3, "adam_decay": None,
                     "use_lbfgs": True, "lbfgs_maxiter": 3, "loss_weights": None,
                     "display_every": 5, "checkpoint_period": 5, "monitor_period": 5},
        "eval": {"n_radial": 20, "angles": [0.0]},
        "zip_artifacts": True,
    }
    config_path = _write_config(tmp_path, "pinn_annulus_steady_micro.yaml", cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "T953" / "steady"
    _check_common_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "steady"
    assert metrics["reference"] == "analytical_steady_state"
    assert np.isfinite(metrics["rel_l2_mean"])
    assert "T_inner_analytical" in metrics


def test_run_pinn_annulus_transient_micro(tmp_path, monkeypatch):
    """瞬态圆环 PINN：周期特征映射训练 + C-N 参照对比贯通。"""
    from heatops.cli.run_pinn_annulus import main

    cfg = {
        "experiment": "T954",
        "seed": 0,
        "mode": "transient",
        "physics": {},
        "domain": {"t_final": 300.0},
        "collocation": {"num_domain": 40, "num_boundary": 12, "num_initial": 12,
                        "train_distribution": "Hammersley"},
        "model": {"layer_sizes": [4, 16, 16, 1]},
        "training": {"adam_iterations": 5, "adam_lr": 1e-3, "adam_decay": None,
                     "use_lbfgs": False, "loss_weights": [1.0, 5.0, 5.0, 1.0],
                     "pde_point_resampler_period": 0,
                     "display_every": 5, "checkpoint_period": 3, "monitor_period": 5},
        "eval": {"n_times": 2, "cn_nr": 30, "cn_nt": 51},
        "zip_artifacts": True,
    }
    config_path = _write_config(tmp_path, "pinn_annulus_transient_micro.yaml", cfg)
    monkeypatch.chdir(tmp_path)
    assert main(["--config", str(config_path)]) == 0

    run_dir = tmp_path / "artifacts" / "T954" / "transient"
    _check_common_artifacts(run_dir)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["mode"] == "transient"
    assert "C-N" in metrics["reference"]
    assert np.isfinite(metrics["rel_l2_final_time"])
