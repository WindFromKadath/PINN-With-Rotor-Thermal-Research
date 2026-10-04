"""2D 圆环瞬态 PINO（极坐标）训练入口（档案 datapino_2d_polar_train.py 重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_pino_annulus --config configs/pino_annulus_polar.yaml --run-id E108
    python -m heatops.cli.run_pino_annulus --config configs/pino_annulus_polar.yaml --smoke

流程：读 config → 构造物理参数/模型/无量纲极坐标网格 → 加载极坐标数据集
（``heatops.data.comsol`` 管线产物，**未入库，本地归档**）→ 无量纲化
（T* = (T-T_inf)/T_scale，q* = q/q_base，t* = t/t_max，r* = r/R_out，
仅缺少 t=0 时补零初始帧）→ 显式 train/eval 划分 → 训练（Adam + randperm 批，
六项损失见 ``heatops.physics.losses`` 极坐标部分）→ ``strict=True`` 加载最优
checkpoint → train/eval 划分上报告 RelL2 → 极坐标三联图与三角剖分云图 →
全部产物入 ``artifacts/<run_id>/``。

**dry-run 模式**：数据集缺失（``FileNotFoundError``）时不视为失败——验证
config/物理参数/模型/网格后明确报告「数据未入库，本地归档」，写
``dry_run_report.json`` 与 ``config.json`` 后以退出码 0 结束（E108 约定）。

与档案的差异（D103）：所有损失在无量纲域计算，换算集中在网格与数据构造；
原 CSV 列头实际含 t=0，本库按列头读取时间，保留现有初始帧；
仅输入缺少 t=0 时补零初始场。PDE/BC 使用输出场有限差分。
"""

from __future__ import annotations

from heatops.training.run_context import managed_cli, prepare_run, run_directory, project_root, config_path_value, bind_data

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import yaml

import heatops
from heatops.data.comsol import load_polar_dataset
from heatops.evaluation import batched_predict, relative_l2
from heatops.evaluation.annulus_plots import (
    plot_annulus_triangulation,
    plot_polar_frame_triplet,
)
from heatops.models.fno import HeatOperatorFNO3D
from heatops.physics.losses import (
    LossWeightsPolar,
    compute_loss_terms_polar,
    weighted_loss_polar,
)
from heatops.problems import AnnulusParams
from heatops.training.torch_loop import set_seed

_LOSS_KEYS = ("total", "data", "pde", "ic", "bc_inner", "bc_outer", "periodic")

#: --smoke 最小规模覆写（只验证管线贯通，不追求收敛）
_SMOKE_OVERRIDES = {
    ("training", "epochs"): 5,
    ("training", "checkpoint_every"): 5,
    ("training", "log_every"): 1,
}

_PHYSICS_KEYS = {"R_in", "R_out", "k", "rho", "cp", "h", "T_inf"}


def load_config(path: str | Path, smoke: bool = False) -> dict:
    """读 YAML config；``smoke=True`` 时按 ``_SMOKE_OVERRIDES`` 覆写最小规模。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        for (section, key), value in _SMOKE_OVERRIDES.items():
            cfg[section][key] = value
    return cfg


def build_params(cfg: dict) -> AnnulusParams:
    """由 config physics 段构造 AnnulusParams；``q_in`` 取参考热流 q_base。"""
    pcfg = cfg["physics"]
    kwargs = {k: float(pcfg[k]) for k in _PHYSICS_KEYS if k in pcfg}
    kwargs["q_in"] = float(pcfg["q_base"])  # 归一化参考热流 ⇒ T_scale/Bi/t_scale 一致
    return AnnulusParams(**kwargs)


def make_polar_grid(
    t_star: np.ndarray, r_star: np.ndarray, theta: np.ndarray
) -> torch.Tensor:
    """由无量纲 (t*, r*, θ) 网格构造 (1, nt, nr, ntheta, 3) float32 坐标张量（ij 索引）。"""
    T, R, Th = torch.meshgrid(
        torch.as_tensor(t_star, dtype=torch.float32),
        torch.as_tensor(r_star, dtype=torch.float32),
        torch.as_tensor(theta, dtype=torch.float32),
        indexing="ij",
    )
    return torch.stack([T, R, Th], dim=-1).unsqueeze(0)


def _save_state_dict(model: torch.nn.Module, path: Path) -> None:
    """保存剔除 ``_metadata`` 的 state_dict（neuralop 元数据项导致 weights_only 拒绝加载；
    与 ``heatops.training.torch_loop`` 同一约定）。"""
    state = {k: v for k, v in model.state_dict().items() if k != "_metadata"}
    torch.save(state, path)


def _plot_loss_curve(history: dict[str, list], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history["step"]
    plt.figure(figsize=(10, 6))
    for key in _LOSS_KEYS[1:]:
        plt.semilogy(steps, np.maximum(history[key], 1e-30), label=f"{key} loss")
    plt.title("Training Loss Convergence (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_polar(
    model: torch.nn.Module,
    q_star: torch.Tensor,
    u_star: torch.Tensor,
    grid_base: torch.Tensor,
    weights: LossWeightsPolar,
    alpha_norm: float,
    bi: float,
    cfg: dict,
    run_dir: Path,
    seed: int,
) -> dict[str, list]:
    """极坐标训练循环：Adam + randperm 批抽样 + 周期/最优 checkpoint + 产物落盘。

    与 ``heatops.training.torch_loop.train_operator_model`` 同工程约定，差异：
    Adam（档案极坐标线约定）而非 AdamW；六项损失键；batch_size=1 的显存妥协
    由 config 承载。循环开头 ``set_seed(seed)``，循环内随机序列只取决于种子。
    """
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    device_str = str(cfg.get("device", "auto"))
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_str == "auto"
        else torch.device(device_str)
    )
    set_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg["lr"]))

    q_all = q_star.to(device)
    u_all = u_star.to(device)
    grid_base = grid_base.to(device)
    n_train = q_all.shape[0]
    batch_size = int(cfg["batch_size"])
    epochs = int(cfg["epochs"])
    rate = float(cfg["pde_sample_rate"])
    if batch_size > n_train:
        raise ValueError(f"batch_size={batch_size} 超过训练样本数 {n_train}")

    history: dict[str, list] = {k: [] for k in ("step", *_LOSS_KEYS)}
    best_loss = float("inf")
    print(f"[train] device={device} n_train={n_train} epochs={epochs} batch={batch_size} pde_rate={rate}")
    for step in range(epochs + 1):
        indices = torch.randperm(n_train)[:batch_size]
        b_q = q_all[indices]
        b_true = u_all[indices]
        b_grid = grid_base.expand(batch_size, -1, -1, -1, -1)
        b_qfield = b_q.view(-1, 1, 1, 1).expand_as(b_true).contiguous()

        terms, _ = compute_loss_terms_polar(
            model, b_qfield, b_true, b_grid,
            alpha_norm=alpha_norm, bi=bi, pde_sample_rate=rate,
            derivative_method=cfg.get("derivative_method", "finite_difference"),
        )
        total = weighted_loss_polar(terms, weights)

        optimizer.zero_grad()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"非有限训练损失：step={step}")
        total.backward()
        optimizer.step()

        total_val = float(total.detach().cpu())
        history["step"].append(step)
        history["total"].append(total_val)
        for key in _LOSS_KEYS[1:]:
            history[key].append(float(terms[key].detach().cpu()))

        if total_val < best_loss:
            best_loss = total_val
            _save_state_dict(model, ckpt_dir / "checkpoint_best.pt")
        if step % int(cfg["checkpoint_every"]) == 0 or step == epochs:
            _save_state_dict(model, ckpt_dir / "checkpoint_last.pt")
        if step % int(cfg["log_every"]) == 0:
            print(
                f"[train] step {step:5d} | total {total_val:.4e} | data {history['data'][-1]:.3e} | "
                f"pde {history['pde'][-1]:.3e} | ic {history['ic'][-1]:.3e} | "
                f"bc_in {history['bc_inner'][-1]:.3e} | bc_out {history['bc_outer'][-1]:.3e} | "
                f"periodic {history['periodic'][-1]:.3e}"
            )

    history_arr = np.array(
        list(zip(*(history[k] for k in ("step", *_LOSS_KEYS)))),
        dtype=[("step", "i8"), *((k, "f8") for k in _LOSS_KEYS)],
    )
    np.save(run_dir / "loss_history.npy", history_arr)
    _plot_loss_curve(history, run_dir / "loss_curve.png")
    return history


def predict_fields(
    model: torch.nn.Module,
    q_star: torch.Tensor,
    grid_base: torch.Tensor,
    shape: tuple[int, int, int],
    device: torch.device,
    batch_size: int = 4,
) -> np.ndarray:
    """对给定样本分批前向，返回 (B, nt, nr, ntheta) numpy 预测（无量纲，float64）。"""
    model.eval()
    nt, nr, ntheta = shape

    def _fn(q_np: np.ndarray) -> np.ndarray:
        q = torch.as_tensor(q_np, dtype=torch.float32, device=device)
        q_field = q.view(-1, 1, 1, 1).expand(-1, nt, nr, ntheta).contiguous()
        grid = grid_base.expand(q.shape[0], -1, -1, -1, -1)
        with torch.no_grad():
            pred = model(q_field, grid)
        return pred.cpu().numpy()

    return batched_predict(_fn, q_star.cpu().numpy(), batch_size=batch_size)


def _write_config_json(run_dir: Path, payload: dict) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（如 configs/pino_annulus_polar.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写 epochs 等）")
    parser.add_argument("--epochs", type=int, default=None, help="覆写训练步数（优先于 config 与 --smoke）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke)
    if args.smoke:
        cfg["model"].update(n_modes=[2, 4, 4], hidden_channels=8)
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    run_id = args.run_id or cfg.get("experiment")
    if not run_id:
        raise ValueError("未指定 run_id：请用 --run-id 或在 config 中设 experiment 字段")
    run_dir = run_directory(args.config, run_id)
    prepare_run(run_dir, cfg, args.config)

    seed = int(cfg["seed"])
    set_seed(seed)  # 模型初始化随机性

    params = build_params(cfg)
    t_max = float(cfg["physics"]["t_max"])
    q_base = float(cfg["physics"]["q_base"])
    alpha_norm = params.alpha * t_max / params.R_out**2  # t* = t/t_max 缩放
    weights = LossWeightsPolar(**{k: float(v) for k, v in cfg["weights"].items()})

    mcfg = cfg["model"]
    model = HeatOperatorFNO3D(
        n_modes=tuple(int(m) for m in mcfg["n_modes"]),
        hidden_channels=int(mcfg["hidden_channels"]),
    )

    # ---------------- 数据加载（缺失 ⇒ dry-run 报告，非失败） ----------------
    data_path = config_path_value(args.config, cfg["data"].get("smoke_path", cfg["data"]["path"]) if args.smoke else cfg["data"]["path"])
    try:
        ds = load_polar_dataset(data_path)
    except FileNotFoundError as e:
        dcfg = cfg["data"]
        grid = make_polar_grid(
            np.linspace(0.0, 1.0, int(dcfg["nt"])),
            np.linspace(params.R_in_star, 1.0, int(dcfg["nr"])),
            np.linspace(0.0, 2 * np.pi, int(dcfg["ntheta"])),
        )
        report = {
            "run_id": run_id,
            "status": "dry_run_data_missing",
            "data_path": str(data_path),
            "checks": {
                "config": "ok",
                "physics": {
                    "R_in_star": params.R_in_star,
                    "Bi": params.Bi,
                    "T_scale": params.T_scale,
                    "alpha_norm": alpha_norm,
                },
                "model": f"HeatOperatorFNO3D(n_modes={list(mcfg['n_modes'])}, hidden={mcfg['hidden_channels']})",
                "grid_shape": list(grid.shape),
                "data_load": "FileNotFoundError",
            },
            "message": str(e),
            "notes": "数据未入库，本地归档；管线已验证到数据加载步骤为止",
        }
        _write_config_json(run_dir, {"config_path": str(args.config), "smoke": bool(args.smoke),
                                     "cfg": cfg, "seed": seed,
                                     "code_revision": f"heatops-{heatops.__version__}"})
        with open(run_dir / "dry_run_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"[dry-run] 数据缺失：{data_path}（未入库，本地归档）")
        print(f"[dry-run] config/物理参数/模型/网格验证通过；Bi={params.Bi:.4g} alpha_norm={alpha_norm:.4g}")
        print(f"[dry-run] 报告已写入 {run_dir / 'dry_run_report.json'}")
        return 0

    # ---------------- 无量纲化与网格（换算集中于此，D103） ----------------
    bind_data(data_path)
    q = ds["q"].to(torch.float64).numpy()
    u = ds["u"].to(torch.float64).numpy()  # (B, nt, nr, ntheta)，°C
    t = ds["t"].numpy()
    r = ds["r"].numpy()
    theta = ds["theta"].numpy()
    n_samples, nt_data, nr, ntheta = u.shape

    u_star = (u - params.T_inf) / params.T_scale
    q_star = (q / q_base).astype(np.float32)
    # 仅缺少 t=0 时补零初始帧，避免重复已有的初始时刻。
    if t[0] == 0.0:
        t_full = t
    else:
        t_full = np.concatenate([[0.0], t])
        u_star = np.concatenate([np.zeros_like(u_star[:, :1]), u_star], axis=1)
    t_star = t_full / t_max
    r_star = r / params.R_out
    grid_base = make_polar_grid(t_star, r_star, theta)

    u_star_t = torch.as_tensor(u_star, dtype=torch.float32)
    q_star_t = torch.as_tensor(q_star, dtype=torch.float32)

    # ---------------- 显式 train/eval 划分（样本级、不重叠） ----------------
    dcfg = cfg["data"]
    ratio, split_seed = float(dcfg["split_ratio"]), int(dcfg["split_seed"])
    rng = np.random.default_rng(split_seed)
    perm = rng.permutation(n_samples)
    n_train = min(max(int(n_samples * ratio), 1), n_samples - 1)
    idx_train, idx_eval = perm[:n_train], perm[n_train:]
    print(f"[data] 样本 {n_samples}（时间轴 nt={len(t_full)}，t=0 仅一帧），"
          f"划分 train={len(idx_train)} eval={len(idx_eval)}（split_seed={split_seed}）")

    tcfg = cfg["training"]
    history = train_polar(
        model,
        q_star_t[idx_train], u_star_t[idx_train], grid_base,
        weights=weights, alpha_norm=alpha_norm, bi=params.Bi,
        cfg=tcfg, run_dir=run_dir, seed=seed,
    )

    _write_config_json(run_dir, {
        "config_path": str(args.config),
        "smoke": bool(args.smoke),
        "cfg": cfg,
        "seed": seed,
        "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
        "code_revision": f"heatops-{heatops.__version__}",
        "split": {"train_ratio": ratio, "split_seed": split_seed,
                  "idx_train": idx_train.tolist(), "idx_eval": idx_eval.tolist()},
        "dimensionless": {"alpha_norm": alpha_norm, "Bi": params.Bi,
                          "T_scale": params.T_scale, "q_base": q_base, "t_max": t_max},
        "best_total_loss": min(history["total"]),
    })

    # ---------------- 评估：strict=True 加载最优 checkpoint ----------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if tcfg.get("device", "auto") == "auto" else torch.device(tcfg["device"])
    ckpt = torch.load(run_dir / "checkpoints" / "checkpoint_best.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device)

    shape = (len(t_full), nr, ntheta)
    grid_dev = grid_base.to(device)
    pred_train = predict_fields(model, q_star_t[idx_train], grid_dev, shape, device)
    pred_eval = predict_fields(model, q_star_t[idx_eval], grid_dev, shape, device)
    true_train = u_star[idx_train]
    true_eval = u_star[idx_eval]

    rel_l2_train = relative_l2(true_train, pred_train)
    rel_l2_eval = relative_l2(true_eval, pred_eval)
    metrics = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "rel_l2_train": rel_l2_train,
        "rel_l2_eval": rel_l2_eval,
        "n_train": int(len(idx_train)),
        "n_eval": int(len(idx_eval)),
        "checkpoint": "checkpoints/checkpoint_best.pt",
        "note": "RelL2 在无量纲域计算（无量纲指标与单位无关）",
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] RelL2 train={rel_l2_train:.4%} eval={rel_l2_eval:.4%}")

    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        u_star_true=true_eval, u_star_pred=pred_eval,
        q_star=q_star[idx_eval], t_star=t_star, r_star=r_star, theta=theta,
    )

    # 图：eval 首样本末时刻帧（物理单位 °C，便于读图）
    sample_true_c = params.T_inf + params.T_scale * true_eval[0, -1]
    sample_pred_c = params.T_inf + params.T_scale * pred_eval[0, -1]
    plot_polar_frame_triplet(
        r, theta, sample_true_c, sample_pred_c,
        run_dir / "plots" / "eval_sample0_final_frame.png",
        model_label=f"PINO ({run_id})",
        frame_label=f"t = {t_full[-1]:.4g} s, q* = {q_star[idx_eval][0]:.3g}",
    )
    Rm, Thm = np.meshgrid(r, theta, indexing="ij")
    plot_annulus_triangulation(
        (Rm * np.cos(Thm)).ravel(), (Rm * np.sin(Thm)).ravel(), sample_pred_c.ravel(),
        run_dir / "plots" / "eval_sample0_final_frame_tri.png",
        r_in=params.R_in, r_out=params.R_out, center=(0.0, 0.0),
        title=f"PINO ({run_id}) Prediction (triangulated)",
    )
    print(f"[done] 产物已写入 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
