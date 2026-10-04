"""1D 三模型对比线训练入口（纯数据 FNO / 纯物理 PINO / 混合 PINO，档案 R005 重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_pino_1d --config configs/pino_1d_hybrid.yaml --run-id E104
    python -m heatops.cli.run_pino_1d --config configs/pino_1d_fno.yaml --smoke --run-id E102

流程：读 config → 生成/复用 FDM 数据（缓存 ``data/processed/``，键为数据参数
哈希）→ 显式 train/eval 划分 → 训练（``heatops.training.train_operator_model``）
→ ``strict=True`` 加载最优 checkpoint → 分别在 train/eval 划分上报告 RelL2 →
三联图与末时刻剖面线 → 全部产物入 ``artifacts/<run_id>/``。

三种变体共用本入口，仅 config 的 ``weights`` 不同（补上原档案缺失的混合权重
训练脚本断链）。
"""

from __future__ import annotations

from heatops.training.run_context import managed_cli, prepare_run, run_directory, project_root, config_path_value, bind_data

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from heatops.data.datasets import Heat1DDataset, cache_key_from_params
from heatops.evaluation import batched_predict, relative_l2
from heatops.evaluation.plots import plot_field_triplet, plot_profile_final_time
from heatops.fields import (
    sample_random_sines,
    sample_single_sine,
    sample_sine_or_square,
    sample_square_wave,
)
from heatops.models.fno import HeatOperatorFNO
from heatops.physics.losses import LossWeights, compute_loss_terms_1d, weighted_loss_1d
from heatops.problems import HeatLineParams
from heatops.training.torch_loop import TrainConfig, set_seed, train_operator_model

_IC_SAMPLERS = {
    "sine_or_square": sample_sine_or_square,
    "random_sines": sample_random_sines,
    "square_wave": sample_square_wave,
    "single_sine": sample_single_sine,
}

#: --smoke 最小规模覆写（只验证管线贯通，不追求收敛）
_SMOKE_OVERRIDES = {
    ("training", "epochs"): 20,
    ("training", "batch_size"): 16,
    ("training", "checkpoint_every"): 10,
    ("data", "n_samples"): 40,
}


def load_config(path: str | Path, smoke: bool = False) -> dict:
    """读 YAML config；``smoke=True`` 时按 ``_SMOKE_OVERRIDES`` 覆写最小规模。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        for (section, key), value in _SMOKE_OVERRIDES.items():
            cfg[section][key] = value
    return cfg


def load_or_generate_dataset(cfg: dict, params: HeatLineParams, cache_dir: str | Path) -> tuple[Heat1DDataset, str]:
    """按数据参数哈希复用缓存；未命中则现场生成并缓存。

    Returns:
        ``(dataset, cache_key)``。
    """
    dcfg = cfg["data"]
    data_params = {
        "a": params.a,
        "length": params.length,
        "nx": dcfg["nx"],
        "nt": dcfg["nt"],
        "t_max": dcfg["t_max"],
        "n_samples": dcfg["n_samples"],
        "seed": dcfg["seed"],
        "ic": dcfg["ic"],
    }
    key = cache_key_from_params(data_params)
    cache_path = Path(cache_dir) / f"heat1d_{key}.pt"
    if cache_path.exists():
        print(f"[data] 命中缓存 {cache_path}")
        return Heat1DDataset.load(cache_path), key

    ic_name = dcfg["ic"]
    if ic_name not in _IC_SAMPLERS:
        raise ValueError(f"未知初始条件 {ic_name!r}，可选：{sorted(_IC_SAMPLERS)}")
    print(f"[data] 生成 {dcfg['n_samples']} 个样本（nx={dcfg['nx']}, nt={dcfg['nt']}, seed={dcfg['seed']}）...")
    dataset = Heat1DDataset.generate(
        params=params,
        nx=dcfg["nx"],
        nt=dcfg["nt"],
        t_max=dcfg["t_max"],
        n_samples=dcfg["n_samples"],
        seed=dcfg["seed"],
        ic_sampler=_IC_SAMPLERS[ic_name],
        ic_name=ic_name,
    )
    dataset.save(cache_path)
    print(f"[data] 已缓存到 {cache_path}")
    return dataset, key


def predict_fields(model: torch.nn.Module, dataset: Heat1DDataset, device: torch.device, batch_size: int = 32) -> np.ndarray:
    """对整个数据集分批前向，返回 (B, nx, nt) numpy 预测（float64）。"""
    model.eval()
    grid_base = dataset.grid.to(device)

    def _fn(u0_np: np.ndarray) -> np.ndarray:
        u0 = torch.as_tensor(u0_np, dtype=torch.float32, device=device)
        grid = grid_base.expand(u0.shape[0], -1, -1, -1)
        with torch.no_grad():
            pred = model(u0, grid)
        return pred.squeeze(-1).cpu().numpy()

    return batched_predict(_fn, dataset.u0.numpy(), batch_size=batch_size)


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（如 configs/pino_1d_hybrid.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写 epochs/n_samples 等）")
    parser.add_argument("--epochs", type=int, default=None, help="覆写训练步数（优先于 config 与 --smoke）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    run_id = args.run_id or cfg.get("experiment")
    if not run_id:
        raise ValueError("未指定 run_id：请用 --run-id 或在 config 中设 experiment 字段")
    run_dir = run_directory(args.config, run_id)
    prepare_run(run_dir, cfg, args.config)

    seed = int(cfg["seed"])
    set_seed(seed)  # 模型初始化随机性

    params = HeatLineParams(
        a=float(cfg["physics"]["a"]),
        length=float(cfg["physics"].get("length", 1.0)),
        u_scale=float(cfg["physics"].get("u_scale", 1.0)),
    )
    dataset, data_key = load_or_generate_dataset(cfg, params, project_root(args.config) / "data" / "processed")
    bind_data(project_root(args.config) / "data" / "processed" / f"heat1d_{data_key}.pt")
    train_ds, eval_ds = dataset.split(
        train_ratio=float(cfg["data"]["split_ratio"]),
        seed=int(cfg["data"]["split_seed"]),
    )
    print(f"[data] 划分：train={train_ds.n_samples} eval={eval_ds.n_samples}（不重叠，seed={cfg['data']['split_seed']}）")

    mcfg = cfg["model"]
    model = HeatOperatorFNO(
        n_modes=tuple(mcfg["n_modes"]),
        hidden_channels=int(mcfg["hidden_channels"]),
        use_ansatz=bool(mcfg.get("use_ansatz", True)),
        a=params.a,
    )
    weights = LossWeights(**{k: float(v) for k, v in cfg["weights"].items()})

    def loss_fn(model, b_u0, b_true, b_grid):
        terms, _ = compute_loss_terms_1d(model, b_u0, b_true, b_grid, a=params.a, derivative_method=tcfg.get("derivative_method", "finite_difference"), compute_pde=weights.pde != 0)
        return weighted_loss_1d(terms, weights), terms

    tcfg = cfg["training"]
    train_cfg = TrainConfig(
        run_dir=run_dir,
        seed=seed,
        epochs=int(tcfg["epochs"]),
        batch_size=int(tcfg["batch_size"]),
        lr=float(tcfg["lr"]),
        weight_decay=float(tcfg["weight_decay"]),
        checkpoint_every=int(tcfg["checkpoint_every"]),
        log_every=int(tcfg["checkpoint_every"]),
        device=str(tcfg.get("device", "auto")),
        extra={
            "config_path": str(args.config),
            "smoke": bool(args.smoke),
            "weights": {k: float(v) for k, v in cfg["weights"].items()},
            "model": {k: (list(v) if isinstance(v, tuple) else v) for k, v in mcfg.items()},
            "physics": dict(cfg["physics"]),
            "data": dict(cfg["data"]),
            "data_cache_key": data_key,
        },
    )
    train_operator_model(model, train_ds, loss_fn, train_cfg)

    # ---------------- 评估：strict=True 加载最优 checkpoint ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if tcfg.get("device", "auto") == "auto" else torch.device(tcfg["device"])
    ckpt = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)

    pred_train = predict_fields(model, train_ds, device)
    pred_eval = predict_fields(model, eval_ds, device)
    true_train = train_ds.u.squeeze(-1).numpy()
    true_eval = eval_ds.u.squeeze(-1).numpy()

    rel_l2_train = relative_l2(true_train, pred_train)
    rel_l2_eval = relative_l2(true_eval, pred_eval)
    metrics = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "rel_l2_train": rel_l2_train,
        "rel_l2_eval": rel_l2_eval,
        "n_train": train_ds.n_samples,
        "n_eval": eval_ds.n_samples,
        "data_cache_key": data_key,
        "checkpoint": "checkpoints/checkpoint_best.pt",
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] RelL2 train={rel_l2_train:.4%} eval={rel_l2_eval:.4%}")

    x_np = dataset.x.numpy()
    t_np = dataset.t.numpy()
    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        u_true=true_eval,
        u_pred=pred_eval,
        x=x_np,
        t=t_np,
    )

    sample_true = true_eval[0]
    sample_pred = pred_eval[0]
    plot_field_triplet(
        x_np, t_np, sample_true, sample_pred,
        run_dir / "plots" / "eval_sample0_triplet.png",
        model_label=f"PINO ({run_id})",
    )
    plot_profile_final_time(
        x_np, t_np, sample_true, {f"PINO ({run_id})": sample_pred},
        run_dir / "plots" / "eval_sample0_profile.png",
    )
    print(f"[done] 产物已写入 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
