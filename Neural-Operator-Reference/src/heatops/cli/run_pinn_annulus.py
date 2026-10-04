"""圆环 PINN 训练入口（稳态 / 瞬态对照，档案两条圆环 PINN 线重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_pinn_annulus --config configs/pinn_annulus_steady.yaml --run-id E110
    python -m heatops.cli.run_pinn_annulus --config configs/pinn_annulus_transient.yaml --run-id E110
    # 加 --smoke 跑最小迭代自检

两个 config 共用同一 run_id 时，产物按模式分子目录存放：
``artifacts/<run_id>/steady/`` 与 ``artifacts/<run_id>/transient/``。

- ``steady``（档案 Heat_2D_Annulus 整合版重写）：无量纲稳态 ``∇*²θ = 0``，
  几何 ``CSGDifference(Disk(1), Disk(R_in*))``，``OperatorBC`` 手写径向法向导数
  （内 ``-∂θ/∂n* = 1``、外 ``-∂θ/∂n* = Bi·θ``）。**与档案的偏差**：档案在网络
  输出端挂线性变换（θ→°C）却仍按无量纲形式书写 BC 残差，量纲混入 T_scale
  （README D103 所述量纲混用）；本实现训练全程在无量纲量上进行，仅在评估时
  经 ``AnnulusParams.to_dimensional_temperature`` 还原，与解析解
  ``analytical_steady_state`` 对比。
- ``transient``（档案 Heat_2D_Annulus_Transient 最终版重写）：极坐标无量纲瞬态
  ``∂t*θ - (θ_rr + θ_r/r + θ_φφ/r²) = 0``，几何 ``Rectangle([R_in*,0],[1,2π]) ×
  TimeDomain``，``OperatorBC`` ``θ_r + 1`` / ``θ_r + Bi·θ``，IC ``θ = 0``，
  :class:`~heatops.models.deepxde_models.PeriodicFeatureTransform` 把 φ 映射为
  (sin φ, cos φ)。**与档案的偏差**：档案保留权重为 0 的 PeriodicBC 占位项，
  本实现直接省略（周期性由特征映射兜底），loss_weights 由 [1,5,5,1,0] 变为
  [1,5,5,1]。瞬态参照为 ``heatops.solvers.solve_annulus_transient_cn``
  （问题轴对称，C-N 径向解即真值的径向剖面）。

物理参数全部来自 config 的 ``physics`` 段（缺省即 ``AnnulusParams`` 钢材基准）。
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
import torch  # noqa: E402
import yaml  # noqa: E402

from heatops.evaluation import relative_l2, rmse  # noqa: E402
from heatops.evaluation.deeponet_plots import (  # noqa: E402
    plot_annulus_field,
    plot_radial_profile_comparison,
    plot_radial_profiles_times,
)
from heatops.models.deepxde_models import PeriodicFeatureTransform, make_fnn  # noqa: E402
from heatops.problems import AnnulusParams, analytical_steady_state  # noqa: E402
from heatops.solvers import solve_annulus_transient_cn  # noqa: E402
from heatops.training.deepxde_loop import (  # noqa: E402
    DDETrainConfig,
    TrainingMonitor,
    find_best_checkpoint,
    load_weights_strict,
    set_seed,
    train_dde_model,
    zip_artifacts,
)

#: --smoke 最小规模覆写（只验证管线贯通，不追求收敛）；仅覆写 config 中已存在的键
_SMOKE_OVERRIDES = {
    ("collocation", "num_domain"): 400,
    ("collocation", "num_boundary"): 100,
    ("collocation", "num_initial"): 60,
    ("training", "adam_iterations"): 300,
    ("training", "display_every"): 100,
    ("training", "checkpoint_period"): 150,
    ("training", "monitor_period"): 100,
    ("training", "lbfgs_maxiter"): 200,
    ("training", "pde_point_resampler_period"): 0,
    ("eval", "n_radial"): 50,
    ("eval", "angles"): [0.0, 1.5707963267948966],
    ("eval", "n_times"): 3,
    ("eval", "cn_nr"): 100,
    ("eval", "cn_nt"): 401,
}


def load_config(path: str | Path, smoke: bool = False) -> dict:
    """读 YAML config；``smoke=True`` 时按 ``_SMOKE_OVERRIDES`` 覆写已存在的键。"""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if smoke:
        for (section, key), value in _SMOKE_OVERRIDES.items():
            if section in cfg and key in cfg[section]:
                cfg[section][key] = value
    return cfg


def _radial_normal_grad(theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """直角坐标输入下的径向（ outward r̂ = x/|x| ）方向导数 ∂θ/∂r̂。"""
    r = torch.norm(x, dim=1, keepdim=True)
    dtheta_dx = dde.grad.jacobian(theta, x, i=0, j=0)
    dtheta_dy = dde.grad.jacobian(theta, x, i=0, j=1)
    return (dtheta_dx * x[:, 0:1] + dtheta_dy * x[:, 1:2]) / r


def _train_and_restore(
    model,
    tcfg: dict,
    run_dir: Path,
    seed: int,
    extra: dict,
    extra_callbacks=(),
) -> Path:
    """统一训练 + ``strict=True`` 加载最优 checkpoint，返回该 checkpoint 路径。"""
    train_cfg = DDETrainConfig(
        run_dir=run_dir,
        seed=seed,
        adam_iterations=int(tcfg["adam_iterations"]),
        adam_lr=float(tcfg["adam_lr"]),
        adam_decay=tuple(tcfg["adam_decay"]) if tcfg.get("adam_decay") else None,
        use_lbfgs=bool(tcfg.get("use_lbfgs", True)),
        lbfgs_maxiter=int(tcfg.get("lbfgs_maxiter", 15000)),
        loss_weights=[float(w) for w in tcfg["loss_weights"]] if tcfg.get("loss_weights") else None,
        metrics=None,  # PDE 数据无真值 y_test，不能挂相对误差 metric
        display_every=int(tcfg["display_every"]),
        checkpoint_period=int(tcfg["checkpoint_period"]),
        extra=extra,
    )
    train_dde_model(model, train_cfg, extra_callbacks=extra_callbacks)
    best_ckpt = find_best_checkpoint(run_dir / "checkpoints") or find_best_checkpoint(
        run_dir / "checkpoints", prefix="final"
    )
    load_weights_strict(model, best_ckpt)
    print(f"[eval] 已加载最优 checkpoint：{best_ckpt}")
    return best_ckpt


def _annulus_field_grid(params: AnnulusParams, n_grid: int = 200):
    """外接正方形网格 + 圆环掩码（物理坐标），供 2D 云图预测用。"""
    x1 = np.linspace(-params.R_out, params.R_out, n_grid)
    X, Y = np.meshgrid(x1, x1, indexing="ij")
    rr = np.sqrt(X**2 + Y**2)
    mask = (rr >= params.R_in) & (rr <= params.R_out)
    return x1, X, Y, rr, mask


def run_steady(cfg: dict, params: AnnulusParams, run_dir: Path, seed: int, extra: dict) -> dict:
    """稳态线：训练 → 与 ``analytical_steady_state`` 对比（多角度径向剖面）。"""
    ri_star, bi = params.R_in_star, params.Bi
    ccfg = cfg["collocation"]

    set_seed(seed)  # 几何采点 + 网络初始化
    annulus = dde.geometry.CSGDifference(
        dde.geometry.Disk([0.0, 0.0], 1.0), dde.geometry.Disk([0.0, 0.0], ri_star)
    )

    def pde(x, theta):
        return (
            dde.grad.hessian(theta, x, i=0, j=0)
            + dde.grad.hessian(theta, x, i=1, j=1)
        )

    def bc_inner(x, theta, X):
        return -_radial_normal_grad(theta, x) - 1.0  # -∂θ/∂n* = 1（n̂ 取径向向外 r̂）

    def bc_outer(x, theta, X):
        return -_radial_normal_grad(theta, x) - bi * theta  # -∂θ/∂n* = Bi·θ

    data = dde.data.PDE(
        annulus, pde,
        [
            dde.icbc.OperatorBC(
                annulus, bc_inner,
                lambda x, on: on and np.isclose(np.linalg.norm(x), ri_star, atol=1e-4),
            ),
            dde.icbc.OperatorBC(
                annulus, bc_outer,
                lambda x, on: on and np.isclose(np.linalg.norm(x), 1.0, atol=1e-4),
            ),
        ],
        num_domain=int(ccfg["num_domain"]),
        num_boundary=int(ccfg["num_boundary"]),
        train_distribution=str(ccfg.get("train_distribution", "Hammersley")),
    )

    mcfg = cfg["model"]
    if int(mcfg["layer_sizes"][0]) != 2:
        raise ValueError(f"稳态线 layer_sizes[0] 应为 2（x*, y*），实际为 {mcfg['layer_sizes'][0]}")
    net = make_fnn(
        mcfg["layer_sizes"],
        activation=str(mcfg.get("activation", "tanh")),
        kernel_initializer=str(mcfg.get("kernel_initializer", "Glorot normal")),
    )
    model = dde.Model(data, net)

    probes = np.array([[ri_star, 0.0], [1.0, 0.0]])
    monitor = TrainingMonitor(
        period=int(cfg["training"].get("monitor_period", 0)),
        probe_points=probes,
        probe_labels=["内边界(r=R_in)", "外边界(r=R_out)"],
        probe_transform=lambda theta: params.to_dimensional_temperature(theta),
    )
    best_ckpt = _train_and_restore(
        model, cfg["training"], run_dir, seed, extra, extra_callbacks=[monitor]
    )

    # ---------------- 评估：多角度径向剖面 vs 解析解 ----------------
    ecfg = cfg["eval"]
    n_radial = int(ecfg["n_radial"])
    angles = [float(a) for a in ecfg["angles"]]
    r_star = np.linspace(ri_star, 1.0, n_radial)
    r_phys = params.to_dimensional_radius(r_star)
    T_ana = analytical_steady_state(r_phys, params)

    T_pred_per_angle = {}
    for a in angles:
        pts = np.stack([r_star * np.cos(a), r_star * np.sin(a)], axis=1)
        theta_pred = model.predict(pts.astype(np.float32)).ravel()
        T_pred_per_angle[f"PINN φ={a:.3g} rad"] = params.to_dimensional_temperature(theta_pred)

    rel_l2_per_angle = {
        label: relative_l2(T_ana, T_pred) for label, T_pred in T_pred_per_angle.items()
    }
    T_stack = np.stack(list(T_pred_per_angle.values()))
    axisym_spread = float((T_stack.max(axis=0) - T_stack.min(axis=0)).max())
    metrics = {
        "reference": "analytical_steady_state",
        "rel_l2_per_angle": rel_l2_per_angle,
        "rel_l2_mean": float(np.mean(list(rel_l2_per_angle.values()))),
        "rmse_celsius": float(np.mean([rmse(T_ana, T) for T in T_pred_per_angle.values()])),
        "max_abs_error_celsius": float(np.abs(T_stack - T_ana).max()),
        "axisymmetry_spread_celsius": axisym_spread,
        "T_inner_analytical": float(T_ana[0]),
        "T_outer_analytical": float(T_ana[-1]),
        "best_checkpoint": str(best_ckpt),
    }
    print(
        f"[eval] 稳态 RelL2 mean={metrics['rel_l2_mean']:.4%} | "
        f"max|ΔT|={metrics['max_abs_error_celsius']:.3f} °C | "
        f"轴对称散布={axisym_spread:.3e} °C"
    )

    plot_radial_profile_comparison(
        r_phys, T_ana, T_pred_per_angle,
        run_dir / "plots" / "radial_profile_vs_analytical.png",
        reference_label="Analytical",
        title="Steady Annulus PINN vs Analytical",
    )
    # 2D 云图（圆环掩码）
    x1, X, Y, _, mask = _annulus_field_grid(params)
    pts = np.stack([X[mask].ravel() / params.R_out, Y[mask].ravel() / params.R_out], axis=1)
    T_field = np.full(X.shape, np.nan)
    T_field[mask] = params.to_dimensional_temperature(
        model.predict(pts.astype(np.float32)).ravel()
    )
    plot_annulus_field(
        x1, x1, T_field, mask, params.R_in, params.R_out,
        run_dir / "plots" / "annulus_field_steady.png",
        title="Steady PINN Temperature Field",
    )
    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        r=r_phys, angles=np.asarray(angles), T_analytical=T_ana,
        T_pred=np.stack(list(T_pred_per_angle.values())),
    )
    return metrics


def run_transient(cfg: dict, params: AnnulusParams, run_dir: Path, seed: int, extra: dict) -> dict:
    """瞬态线：极坐标 PINN（sin/cos 特征映射）→ 与 C-N 轴对称参照对比。"""
    ri_star, bi = params.R_in_star, params.Bi
    t_final = float(cfg["domain"]["t_final"])
    t_star_final = float(params.to_dimensionless_time(t_final))
    ccfg = cfg["collocation"]

    set_seed(seed)  # 几何采点 + 网络初始化
    geom = dde.geometry.Rectangle([ri_star, 0.0], [1.0, 2.0 * np.pi])
    geomtime = dde.geometry.GeometryXTime(geom, dde.geometry.TimeDomain(0.0, t_star_final))

    def pde(x, theta):
        r = x[:, 0:1]
        dtheta_dt = dde.grad.jacobian(theta, x, i=0, j=2)
        dtheta_dr = dde.grad.jacobian(theta, x, i=0, j=0)
        dtheta_drr = dde.grad.hessian(theta, x, i=0, j=0)
        dtheta_dpp = dde.grad.hessian(theta, x, i=1, j=1)
        return dtheta_dt - (dtheta_drr + dtheta_dr / r + dtheta_dpp / r**2)

    def bc_inner(x, theta, X):
        return dde.grad.jacobian(theta, x, i=0, j=0) + 1.0  # ∂θ/∂r* = -1

    def bc_outer(x, theta, X):
        return dde.grad.jacobian(theta, x, i=0, j=0) + bi * theta  # ∂θ/∂r* = -Bi·θ

    data = dde.data.TimePDE(
        geomtime, pde,
        [
            dde.icbc.OperatorBC(
                geomtime, bc_inner,
                lambda x, on: on and np.isclose(x[0], ri_star),
            ),
            dde.icbc.OperatorBC(
                geomtime, bc_outer,
                lambda x, on: on and np.isclose(x[0], 1.0),
            ),
            dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial),
        ],
        num_domain=int(ccfg["num_domain"]),
        num_boundary=int(ccfg["num_boundary"]),
        num_initial=int(ccfg["num_initial"]),
        train_distribution=str(ccfg.get("train_distribution", "Hammersley")),
    )

    mcfg = cfg["model"]
    feature_transform = PeriodicFeatureTransform(theta_col=1)  # [r*, φ, t*] → [r*, sin φ, cos φ, t*]
    if int(mcfg["layer_sizes"][0]) != feature_transform.output_dim(3):
        raise ValueError(
            f"瞬态线启用周期特征映射，layer_sizes[0] 应为 {feature_transform.output_dim(3)}，"
            f"实际为 {mcfg['layer_sizes'][0]}"
        )
    net = make_fnn(
        mcfg["layer_sizes"],
        activation=str(mcfg.get("activation", "tanh")),
        kernel_initializer=str(mcfg.get("kernel_initializer", "Glorot normal")),
        feature_transform=feature_transform,
    )
    model = dde.Model(data, net)

    probes = np.array([[ri_star, 0.0, t_star_final], [1.0, 0.0, t_star_final]])
    monitor = TrainingMonitor(
        period=int(cfg["training"].get("monitor_period", 0)),
        probe_points=probes,
        probe_labels=["内边界末时刻", "外边界末时刻"],
        probe_transform=lambda theta: params.to_dimensional_temperature(theta),
    )
    extra_callbacks = [monitor]
    resampler_period = int(cfg["training"].get("pde_point_resampler_period", 0))
    if resampler_period > 0:
        extra_callbacks.append(dde.callbacks.PDEPointResampler(period=resampler_period))
    best_ckpt = _train_and_restore(
        model, cfg["training"], run_dir, seed, extra, extra_callbacks=extra_callbacks
    )

    # ---------------- 评估：径向剖面多时刻 vs C-N 参照 ----------------
    ecfg = cfg["eval"]
    cn_nr, cn_nt = int(ecfg["cn_nr"]), int(ecfg["cn_nt"])
    n_times = int(ecfg["n_times"])
    times = np.linspace(0.0, t_final, n_times + 1)[1:]  # 跳过平凡的 t=0
    cn = solve_annulus_transient_cn(
        params, nr=cn_nr, dt=t_final / (cn_nt - 1), nt=cn_nt, save_every=1
    )
    r_phys = cn["r"]
    r_star = params.to_dimensionless_radius(r_phys)

    T_cn_times, T_pinn_times = [], []
    for t_phys in times:
        idx = int(np.argmin(np.abs(cn["t"] - t_phys)))
        T_cn_times.append(cn["T_history"][idx])
        t_star = float(params.to_dimensionless_time(t_phys))
        pts = np.stack([r_star, np.zeros_like(r_star), np.full_like(r_star, t_star)], axis=1)
        T_pinn_times.append(
            params.to_dimensional_temperature(model.predict(pts.astype(np.float32)).ravel())
        )
    T_cn_times = np.asarray(T_cn_times)
    T_pinn_times = np.asarray(T_pinn_times)

    rel_l2_per_time = [
        relative_l2(T_cn_times[i], T_pinn_times[i]) for i in range(len(times))
    ]
    metrics = {
        "reference": "solve_annulus_transient_cn（轴对称 C-N）",
        "t_final_s": t_final,
        "t_star_final": t_star_final,
        "eval_times_s": times.tolist(),
        "rel_l2_per_time": rel_l2_per_time,
        "rel_l2_final_time": rel_l2_per_time[-1],
        "rel_l2_all_times": relative_l2(T_cn_times, T_pinn_times),
        "rmse_celsius": rmse(T_cn_times, T_pinn_times),
        "max_abs_error_celsius": float(np.abs(T_cn_times - T_pinn_times).max()),
        "best_checkpoint": str(best_ckpt),
    }
    print(
        f"[eval] 瞬态 RelL2（全部评估时刻）={metrics['rel_l2_all_times']:.4%} | "
        f"末时刻={metrics['rel_l2_final_time']:.4%}"
    )

    plot_radial_profiles_times(
        r_phys, times, T_cn_times, T_pinn_times,
        run_dir / "plots" / "radial_profiles_times.png",
        reference_label="C-N reference", pred_label="PINN (polar)",
        title="Transient Annulus PINN vs C-N Reference",
    )
    # 末时刻 2D 云图（极坐标输入 (r*, φ, t*)）
    x1, X, Y, rr, mask = _annulus_field_grid(params)
    pts = np.stack(
        [
            (rr[mask] / params.R_out).ravel(),
            np.mod(np.arctan2(Y[mask], X[mask]), 2.0 * np.pi).ravel(),
            np.full(int(mask.sum()), t_star_final),
        ],
        axis=1,
    )
    T_field = np.full(X.shape, np.nan)
    T_field[mask] = params.to_dimensional_temperature(model.predict(pts.astype(np.float32)).ravel())
    plot_annulus_field(
        x1, x1, T_field, mask, params.R_in, params.R_out,
        run_dir / "plots" / "annulus_field_t_final.png",
        title=f"Transient PINN Temperature Field (t = {t_final:.3g} s)",
    )
    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        r=r_phys, times=times, T_cn=T_cn_times, T_pinn=T_pinn_times,
    )
    return metrics


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（pinn_annulus_steady.yaml / pinn_annulus_transient.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写 collocation/training/eval 规模）")
    parser.add_argument("--iterations", type=int, default=None, help="覆写 Adam 迭代数（优先于 config 与 --smoke）")
    args = parser.parse_args(argv)

    t_start = time.perf_counter()
    cfg = load_config(args.config, smoke=args.smoke)
    if args.iterations is not None:
        cfg["training"]["adam_iterations"] = args.iterations
    run_id = args.run_id or cfg.get("experiment")
    if not run_id:
        raise ValueError("未指定 run_id：请用 --run-id 或在 config 中设 experiment 字段")
    mode = str(cfg.get("mode", "")).lower()
    if mode not in ("steady", "transient"):
        raise ValueError(f"config 的 mode 字段须为 'steady' 或 'transient'，实际为 {mode!r}")
    # 同一 run_id 覆盖稳态+瞬态两个 config：产物按模式分子目录，互不覆盖
    run_dir = run_directory(args.config, run_id, mode)
    prepare_run(run_dir, cfg, args.config)

    seed = int(cfg["seed"])
    params = AnnulusParams(**{k: float(v) for k, v in cfg.get("physics", {}).items()})
    extra = {
        "config_path": str(args.config),
        "smoke": bool(args.smoke),
        "mode": mode,
        "physics": dict(cfg.get("physics", {})) or "AnnulusParams 默认（钢材基准）",
        "collocation": dict(cfg["collocation"]),
        "model": dict(cfg["model"]),
        "eval": dict(cfg["eval"]),
    }
    if mode == "transient":
        extra["domain"] = dict(cfg["domain"])

    if mode == "steady":
        metrics = run_steady(cfg, params, run_dir, seed, extra)
    else:
        metrics = run_transient(cfg, params, run_dir, seed, extra)

    metrics = {
        "run_id": run_id,
        "mode": mode,
        "smoke": bool(args.smoke),
        **metrics,
        "wall_time_s": round(time.perf_counter() - t_start, 3),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if bool(cfg.get("zip_artifacts", True)):
        zip_path = zip_artifacts(run_dir)
        print(f"[done] 产物已写入 {run_dir}，打包 {zip_path}")
    else:
        print(f"[done] 产物已写入 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
