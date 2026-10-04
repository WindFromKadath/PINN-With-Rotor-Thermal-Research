"""2D 直角 Robin 数据驱动 DeepONet 训练入口（档案 DeepONet 线重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_deeponet_2d --config configs/deeponet_2d.yaml --run-id E111
    python -m heatops.cli.run_deeponet_2d --config configs/deeponet_2d.yaml --smoke --run-id E111

流程：读 config → 现场生成/复用 2D FDM 数据（``heatops.solvers.solve_heat_2d_robin``
+ ``heatops.fields.GRF2DGenerator``，缓存 ``data/processed/``，键为数据参数哈希；
补上原档案数据生成脚本缺失的断链）→ 显式 train/eval 划分 →
``dde.data.TripleCartesianProd`` 纯数据训练（无 PDE 损失）→ ``strict=True``
加载最优 checkpoint → 物理量纲还原后在 eval 划分上报告 RelL2/RMSE →
全部产物入 ``artifacts/<run_id>/``。

数据格式（与档案 npz 约定一致）：分支输入为传感点热源归一化值
``f / f_max``（(N, nx·ny)），主干输入为 (x, y, t) 坐标网格（(nx·ny·nt, 3)，
``indexing='ij'`` ravel），输出为归一化温度 ``(u - u0) / u_scale``
（(N, nx·ny·nt)）；metadata 以 JSON 字符串存入 npz（避免 pickle 载荷）。
"""

from __future__ import annotations

from heatops.training.run_context import managed_cli, prepare_run, run_directory, project_root, config_path_value, bind_data

import argparse
import json
import time
from pathlib import Path

from heatops.models.deepxde_models import ensure_dde_backend_pytorch

ensure_dde_backend_pytorch()  # 必须先于 import deepxde（见 deepxde_models 模块 docstring）

import deepxde as dde  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from heatops.data.datasets import cache_key_from_params  # noqa: E402
from heatops.evaluation import relative_l2, rmse  # noqa: E402
from heatops.evaluation.deeponet_plots import plot_field_triplet_2d  # noqa: E402
from heatops.fields import GRF2DGenerator  # noqa: E402
from heatops.models.deepxde_models import make_deeponet_cartesian  # noqa: E402
from heatops.problems import CartesianHeatParams  # noqa: E402
from heatops.solvers import solve_heat_2d_robin  # noqa: E402
from heatops.training.deepxde_loop import (  # noqa: E402
    DDETrainConfig,
    find_best_checkpoint,
    load_weights_strict,
    set_seed,
    train_dde_model,
    zip_artifacts,
)

#: --smoke 最小规模覆写（只验证管线贯通，不追求收敛）
_SMOKE_OVERRIDES = {
    ("data", "nx"): 16,
    ("data", "nt"): 6,
    ("data", "n_samples"): 24,
    ("model", "branch_layers"): [256, 128, 128, 128],
    ("training", "adam_iterations"): 300,
    ("training", "use_lbfgs"): False,
    ("training", "display_every"): 40,
    ("training", "checkpoint_period"): 60,
}


def load_config(path: str | Path, smoke: bool = False) -> dict:
    """读 YAML config；``smoke=True`` 时按 ``_SMOKE_OVERRIDES`` 覆写最小规模。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        for (section, key), value in _SMOKE_OVERRIDES.items():
            cfg[section][key] = value
    return cfg


def load_or_generate_dataset(cfg: dict, params: CartesianHeatParams, cache_dir: str | Path) -> tuple[dict, str]:
    """按数据参数哈希复用缓存；未命中则现场生成并缓存。

    Returns:
        ``(payload, cache_key)``；payload 含 X_branch/X_trunk/y/x/t 与 metadata。
    """
    dcfg = cfg["data"]
    data_params = {
        "alpha": params.alpha,
        "beta": params.beta,
        "u0": params.u0,
        "f_mean": params.f_mean,
        "f_std": params.f_std,
        "f_max": params.f_max,
        "u_scale": params.u_scale,
        "nx": dcfg["nx"],
        "nt": dcfg["nt"],
        "t_max": dcfg["t_max"],
        "n_samples": dcfg["n_samples"],
        "seed": dcfg["seed"],
        "length_scale": dcfg["length_scale"],
    }
    key = cache_key_from_params(data_params)
    cache_path = Path(cache_dir) / f"deeponet2d_{key}.npz"
    if cache_path.exists():
        print(f"[data] 命中缓存 {cache_path}")
        blob = np.load(cache_path)
        payload = {k: blob[k] for k in ("X_branch", "X_trunk", "y", "x", "t")}
        payload["metadata"] = json.loads(str(blob["metadata_json"]))
        return payload, key

    n_samples = int(dcfg["n_samples"])
    nx, nt = int(dcfg["nx"]), int(dcfg["nt"])
    t_max = float(dcfg["t_max"])
    print(
        f"[data] 现场生成 {n_samples} 个样本（nx={nx}, nt={nt}, t_max={t_max}, "
        f"length_scale={dcfg['length_scale']}, seed={dcfg['seed']}）..."
    )
    gen = GRF2DGenerator(length_scale=float(dcfg["length_scale"]), n_points=nx, seed=int(dcfg["seed"]))
    f_all = gen.sample(n_samples, mean_f=params.f_mean, std_f=params.f_std, clip_negative=True)

    u_all = np.zeros((n_samples, nt, nx, nx))
    t_out = x_out = None
    for i in range(n_samples):
        out = solve_heat_2d_robin(
            alpha=params.alpha, f_grid=f_all[i], u0=params.u0,
            n=nx, T_max=t_max, nt_save=nt, beta=params.beta,
        )
        u_all[i] = out["u"]
        t_out, x_out = out["t"], out["x"]
        if (i + 1) % max(1, n_samples // 5) == 0:
            print(f"[data] 已完成 {i + 1}/{n_samples}")

    # 主干坐标网格（indexing='ij'，ravel 后 x 最慢、t 最快）
    Xg, Yg, Tg = np.meshgrid(x_out, x_out, t_out, indexing="ij")
    X_trunk = np.stack([Xg.ravel(), Yg.ravel(), Tg.ravel()], axis=1)
    # y 与主干点序一致：u (N, nt, nx, ny) → (N, nx, ny, nt) → ravel
    y = params.normalize_temperature(np.transpose(u_all, (0, 2, 3, 1))).reshape(n_samples, -1)
    metadata = {
        "generator": "solve_heat_2d_robin + GRF2DGenerator",
        "normalization": {"branch": "f / f_max", "y": "(u - u0) / u_scale"},
        "u0": params.u0,
        "u_scale": params.u_scale,
        "f_max": params.f_max,
        **{k: (list(v) if isinstance(v, tuple) else v) for k, v in data_params.items()},
    }
    payload = {
        "X_branch": params.normalize_source(f_all).reshape(n_samples, -1),
        "X_trunk": X_trunk,
        "y": y,
        "x": x_out,
        "t": t_out,
        "metadata": metadata,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X_branch=payload["X_branch"], X_trunk=payload["X_trunk"], y=payload["y"],
        x=x_out, t=t_out, metadata_json=json.dumps(metadata, ensure_ascii=False),
    )
    print(f"[data] 已缓存到 {cache_path}")
    return payload, key


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（如 configs/deeponet_2d.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写 data/training 规模）")
    parser.add_argument("--iterations", type=int, default=None, help="覆写 Adam 迭代数（优先于 config 与 --smoke）")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.iterations is not None:
        cfg["training"]["adam_iterations"] = args.iterations
    run_id = args.run_id or cfg.get("experiment")
    if not run_id:
        raise ValueError("未指定 run_id：请用 --run-id 或在 config 中设 experiment 字段")
    run_dir = run_directory(args.config, run_id)
    prepare_run(run_dir, cfg, args.config)

    seed = int(cfg["seed"])
    set_seed(seed)  # 数据构造之外的随机性（网络初始化等）

    params = CartesianHeatParams(**{k: float(v) for k, v in cfg.get("physics", {}).items()})
    payload, data_key = load_or_generate_dataset(cfg, params, project_root(args.config) / "data" / "processed")

    dcfg = cfg["data"]
    nx, nt = int(dcfg["nx"]), int(dcfg["nt"])
    X_branch, X_trunk, y = payload["X_branch"], payload["X_trunk"], payload["y"]
    n_samples = X_branch.shape[0]

    # ---------------- 显式 train/eval 划分（不重叠，同种子可复现） ----------------
    rng = np.random.default_rng(int(dcfg["split_seed"]))
    perm = rng.permutation(n_samples)
    n_train = min(max(int(n_samples * float(dcfg["split_ratio"])), 1), n_samples - 1)
    idx_train, idx_eval = perm[:n_train], perm[n_train:]
    print(f"[data] 划分：train={len(idx_train)} eval={len(idx_eval)}（seed={dcfg['split_seed']}）")

    data = dde.data.TripleCartesianProd(
        X_train=(X_branch[idx_train].astype(np.float32), X_trunk.astype(np.float32)),
        y_train=y[idx_train].astype(np.float32),
        X_test=(X_branch[idx_eval].astype(np.float32), X_trunk.astype(np.float32)),
        y_test=y[idx_eval].astype(np.float32),
    )

    # ---------------- 模型与训练 ----------------
    mcfg = cfg["model"]
    if int(mcfg["branch_layers"][0]) != nx * nx:
        raise ValueError(
            f"branch_layers[0]={mcfg['branch_layers'][0]} 与传感点数 nx²={nx * nx} 不一致"
        )
    if int(mcfg["trunk_layers"][0]) != 3:
        raise ValueError(f"trunk_layers[0] 应为 3（x, y, t），实际为 {mcfg['trunk_layers'][0]}")
    set_seed(seed)  # 网络初始化随机性
    net = make_deeponet_cartesian(
        mcfg["branch_layers"], mcfg["trunk_layers"],
        activation=str(mcfg.get("activation", "tanh")),
        kernel_initializer=str(mcfg.get("kernel_initializer", "Glorot normal")),
    )
    model = dde.Model(data, net)

    tcfg = cfg["training"]
    train_cfg = DDETrainConfig(
        run_dir=run_dir,
        seed=seed,
        adam_iterations=int(tcfg["adam_iterations"]),
        adam_lr=float(tcfg["adam_lr"]),
        adam_decay=tuple(tcfg["adam_decay"]) if tcfg.get("adam_decay") else None,
        use_lbfgs=bool(tcfg.get("use_lbfgs", True)),
        lbfgs_maxiter=int(tcfg.get("lbfgs_maxiter", 15000)),
        loss_weights=None,  # 纯数据线仅单一项 MSE
        metrics=["mean l2 relative error"],
        display_every=int(tcfg["display_every"]),
        checkpoint_period=int(tcfg["checkpoint_period"]),
        extra={
            "config_path": str(args.config),
            "smoke": bool(args.smoke),
            "physics": dict(cfg.get("physics", {})),
            "data": dict(dcfg),
            "model": dict(mcfg),
            "data_cache_key": data_key,
            "n_train": int(len(idx_train)),
            "n_eval": int(len(idx_eval)),
        },
    )
    losshistory, train_state = train_dde_model(model, train_cfg)

    # ---------------- 评估：strict=True 加载最优 checkpoint ----------------
    best_ckpt = find_best_checkpoint(run_dir / "checkpoints") or find_best_checkpoint(
        run_dir / "checkpoints", prefix="final"
    )
    load_weights_strict(model, best_ckpt)
    print(f"[eval] 已加载最优 checkpoint：{best_ckpt}")

    # 主干点对全部样本共享（CartesianProd），仅沿分支维分批以防一次性预测 OOM
    # （档案 trim_to_batch 的等价物）
    def _predict_fn(b_branch: np.ndarray) -> np.ndarray:
        return model.predict((b_branch.astype(np.float32), X_trunk.astype(np.float32)))

    preds = [
        _predict_fn(X_branch[idx_eval][s : s + 8])
        for s in range(0, len(idx_eval), 8)
    ]
    pred_norm = np.concatenate(preds, axis=0)

    true_eval = params.denormalize_temperature(y[idx_eval]).reshape(len(idx_eval), nx, nx, nt)
    pred_eval = params.denormalize_temperature(pred_norm).reshape(len(idx_eval), nx, nx, nt)

    rel_l2_per_sample = np.array([
        relative_l2(true_eval[i], pred_eval[i]) for i in range(len(idx_eval))
    ])
    metrics = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "n_train": int(len(idx_train)),
        "n_eval": int(len(idx_eval)),
        "rel_l2_mean": float(rel_l2_per_sample.mean()),
        "rel_l2_std": float(rel_l2_per_sample.std()),
        "rel_l2_per_sample": rel_l2_per_sample.tolist(),
        "rmse_celsius": rmse(true_eval, pred_eval),
        "max_abs_error_celsius": float(np.abs(true_eval - pred_eval).max()),
        "data_cache_key": data_key,
        "best_checkpoint": str(best_ckpt),
        "wall_time_s": round(time.perf_counter() - t_start, 3),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"[eval] RelL2 mean={metrics['rel_l2_mean']:.4%} ± {metrics['rel_l2_std']:.4%} | "
        f"RMSE={metrics['rmse_celsius']:.3f} °C | max|ΔT|={metrics['max_abs_error_celsius']:.3f} °C"
    )

    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        u_true=true_eval, u_pred=pred_eval,
        x=payload["x"], t=payload["t"],
        eval_indices=idx_eval,
    )
    plot_field_triplet_2d(
        payload["x"], payload["x"], true_eval[0, :, :, -1], pred_eval[0, :, :, -1],
        run_dir / "plots" / "eval_sample0_t_final_triplet.png",
        model_label=f"DeepONet ({run_id})",
        title_suffix=f"t = {payload['t'][-1]:.3g} s, eval sample 0",
    )

    if bool(cfg.get("zip_artifacts", True)):
        zip_path = zip_artifacts(run_dir)
        print(f"[done] 产物已写入 {run_dir}，打包 {zip_path}")
    else:
        print(f"[done] 产物已写入 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
