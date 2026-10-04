"""PI-DeepONet 训练入口（1D 物理量纲线 / 2D 无量纲线，档案两条 PI-DeepONet 线重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_pi_deeponet --config configs/pi_deeponet_1d.yaml --run-id E106
    python -m heatops.cli.run_pi_deeponet --config configs/pi_deeponet_2d.yaml --run-id E107
    # 加 --smoke 跑最小迭代自检

两条线共用 ``PDEOperatorCartesianProd`` 骨架，由 config 的 ``geometry`` 字段切换：

- ``1d``（档案 HeatEquation1D 线）：物理量纲，方程 ``u_t = alpha·u_xx + (v·f_std + f_mean)``，
  分支输入为原始 GRF 值 ``v = (f - f_mean)/f_std``；真值对照用
  ``heatops.solvers.solve_adr_robin``（source_scale=f_std, constant_source=f_mean 与该
  PDE 源项代数一致）。档案 ``iterations=1`` 系调试残留，本实现给正常值（config 驱动）。
- ``2d``（档案 Heat equation 线）：无量纲化 ``θ = (u - u0)/u_scale``、
  ``v_norm = f/f_max``，方程 ``θ_t = alpha·Δθ + v·(f_max/u_scale)``，IC ``θ=0``；
  分支输入为传感点归一化热源。真值对照用 ``solve_heat_2d_robin``。

与档案的偏差（修正 bug）：档案 2D 线的"Robin"边界用 ``OperatorBC`` 返回 ``-β·u``，
实际施加的是 **u=0 的 Dirichlet 约束**（OperatorBC 将返回值约束为零，不含任何
导数）；本实现改用 ``dde.icbc.RobinBC`` 施加真正的法向导数条件
``dθ/dn = -β·θ``，与 FDM 真值的边界约定一致。

物理参数全部来自 config 的 ``physics`` 段（默认值即 ``CartesianHeatParams``）。
评估函数采样由 ``eval.seed`` 独立控制，与训练采样序列互不影响。
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

from heatops.evaluation import relative_l2, rmse  # noqa: E402
from heatops.evaluation.deeponet_plots import plot_field_triplet_2d  # noqa: E402
from heatops.evaluation.plots import plot_field_triplet, plot_profile_final_time  # noqa: E402
from heatops.models.deepxde_models import make_deeponet_cartesian  # noqa: E402
from heatops.problems import CartesianHeatParams  # noqa: E402
from heatops.solvers import solve_adr_robin, solve_heat_2d_robin  # noqa: E402
from heatops.training.deepxde_loop import (  # noqa: E402
    DDETrainConfig,
    find_best_checkpoint,
    load_weights_strict,
    set_seed,
    train_dde_model,
    zip_artifacts,
)

#: --smoke 最小规模覆写（只验证管线贯通，不追求收敛）；仅覆写 config 中已存在的键
_SMOKE_OVERRIDES = {
    ("collocation", "num_domain"): 100,
    ("collocation", "num_boundary"): 40,
    ("collocation", "num_initial"): 20,
    ("collocation", "num_test_points"): 100,
    ("function_space", "n_grid"): 30,
    ("operator", "num_function"): 40,
    ("operator", "num_function_test"): 10,
    ("operator", "batch_size"): 8,
    ("training", "adam_iterations"): 500,
    ("training", "use_lbfgs"): False,
    ("training", "display_every"): 20,
    ("training", "checkpoint_period"): 30,
    ("eval", "n_test"): 2,
    ("eval", "nx"): 16,
    ("eval", "nt"): 6,
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


class NormalizedGRF2D:
    """``dde.data.GRF2D`` 的归一化包装：原始零均值场 → ``(v·f_std + f_mean)/f_max``。

    重写自档案 2D PI-DeepONet 的 ``NormalizedGRF``；映射系数全部由构造参数传入
    （来自 ``CartesianHeatParams`` / config），不在此类硬编码。

    Args:
        length_scale: GRF 相关长度。
        n_grid: GRF2D 内部协方差网格每边点数（档案默认 100 的 Cholesky 构造代价
            高；场在 length_scale≥0.2 时光滑，50 已足够，见 config）。
        mean, std, scale: 归一化映射的均值、波动与尺度（f_mean/f_std/f_max）。
    """

    def __init__(self, length_scale: float, n_grid: int, mean: float, std: float, scale: float):
        self.space = dde.data.GRF2D(length_scale=float(length_scale), N=int(n_grid))
        self.mean = float(mean)
        self.std = float(std)
        self.scale = float(scale)

    def random(self, size: int) -> np.ndarray:
        return self.space.random(size)

    def eval_batch(self, features: np.ndarray, xs: np.ndarray) -> np.ndarray:
        v_raw = self.space.eval_batch(features, xs)
        return (v_raw * self.std + self.mean) / self.scale


def _sensor_points_2d(n_per_dim: int) -> np.ndarray:
    """单位正方形 ij 网格传感点，形状 (n², 2)。"""
    xs = np.linspace(0.0, 1.0, int(n_per_dim))
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    return np.stack([X.ravel(), Y.ravel()], axis=1)


def build_data_1d(cfg: dict, params: CartesianHeatParams):
    """构造 1D 线 PDE/几何/函数空间/算子数据集（物理量纲，档案 HeatEquation1D 线）。"""
    ccfg, fcfg, ocfg = cfg["collocation"], cfg["function_space"], cfg["operator"]
    alpha, beta, u0 = params.alpha, params.beta, params.u0
    f_std, f_mean = params.f_std, params.f_mean

    def pde(x, u, v):
        du_t = dde.grad.jacobian(u, x, i=0, j=1)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        return du_t - alpha * du_xx - (v * f_std + f_mean)

    geomtime = dde.geometry.GeometryXTime(
        dde.geometry.Interval(0.0, 1.0), dde.geometry.TimeDomain(0.0, 0.5)
    )
    bc = dde.icbc.RobinBC(geomtime, lambda x, u: -beta * (u - u0), lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: u0, lambda _, on_initial: on_initial)
    pde_data = dde.data.TimePDE(
        geomtime, pde, [bc, ic],
        num_domain=int(ccfg["num_domain"]),
        num_boundary=int(ccfg["num_boundary"]),
        num_initial=int(ccfg["num_initial"]),
        num_test=int(ccfg.get("num_test_points")) if ccfg.get("num_test_points") else None,
        train_distribution=str(ccfg.get("train_distribution", "uniform")),
    )
    func_space = dde.data.GRF(length_scale=float(fcfg["length_scale"]))
    eval_pts = np.linspace(0.0, 1.0, int(fcfg["n_sensors"]))[:, None]
    data = dde.data.PDEOperatorCartesianProd(
        pde_data, func_space, eval_pts,
        int(ocfg["num_function"]),
        function_variables=[0],
        num_test=int(ocfg["num_function_test"]) if ocfg.get("num_function_test") else None,
        batch_size=int(ocfg["batch_size"]),
    )
    return data, func_space, eval_pts


def build_data_2d(cfg: dict, params: CartesianHeatParams):
    """构造 2D 线 PDE/几何/函数空间/算子数据集（无量纲 θ，档案 Heat equation 线）。"""
    ccfg, fcfg, ocfg = cfg["collocation"], cfg["function_space"], cfg["operator"]
    alpha, beta = params.alpha, params.beta
    source_factor = params.f_max / params.u_scale  # 方程换算因子 f_max/u_scale

    def pde_theta(x, u, v):
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_xx = dde.grad.hessian(u, x, i=0, j=0)
        du_yy = dde.grad.hessian(u, x, i=0, j=1)
        return du_t - alpha * (du_xx + du_yy) - v * source_factor

    geomtime = dde.geometry.GeometryXTime(
        dde.geometry.Rectangle([0.0, 0.0], [1.0, 1.0]), dde.geometry.TimeDomain(0.0, 0.5)
    )
    # 真 Robin 边界 dθ/dn = -β·θ（修正档案 OperatorBC 实为 θ=0 的偏差，见模块 docstring）
    bc = dde.icbc.RobinBC(geomtime, lambda x, u: -beta * u, lambda _, on_boundary: on_boundary)
    ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on_initial: on_initial)
    pde_data = dde.data.TimePDE(
        geomtime, pde_theta, [bc, ic],
        num_domain=int(ccfg["num_domain"]),
        num_boundary=int(ccfg["num_boundary"]),
        num_initial=int(ccfg["num_initial"]),
        train_distribution=str(ccfg.get("train_distribution", "Hammersley")),
    )
    func_space = NormalizedGRF2D(
        length_scale=float(fcfg["length_scale"]),
        n_grid=int(fcfg.get("n_grid", 50)),
        mean=params.f_mean, std=params.f_std, scale=params.f_max,
    )
    eval_pts = _sensor_points_2d(int(fcfg["n_sensors_per_dim"]))
    data = dde.data.PDEOperatorCartesianProd(
        pde_data, func_space, eval_pts,
        int(ocfg["num_function"]),
        function_variables=[0, 1],
        num_test=int(ocfg["num_function_test"]) if ocfg.get("num_function_test") else None,
        batch_size=int(ocfg["batch_size"]),
    )
    return data, func_space, eval_pts


def evaluate_1d(model, func_space, eval_pts, cfg: dict, params: CartesianHeatParams, run_dir: Path) -> dict:
    """1D 线评估：eval.seed 采样测试函数 → solve_adr_robin 真值 → 逐样本 RelL2。"""
    ecfg = cfg["eval"]
    n_test, nx, nt = int(ecfg["n_test"]), int(ecfg["nx"]), int(ecfg["nt"])
    t_max = 0.5

    set_seed(int(ecfg["seed"]))  # 评估函数采样独立于训练随机序列
    func_feats = func_space.random(n_test)
    v_sensors = func_space.eval_batch(func_feats, eval_pts)  # (n_test, n_sensors)

    x = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    Xg, Tg = np.meshgrid(x, t, indexing="ij")
    x_trunk = np.stack([Xg.ravel(), Tg.ravel()], axis=1).astype(np.float32)

    pred = model.predict((v_sensors.astype(np.float32), x_trunk)).reshape(n_test, nx, nt)

    true = np.zeros_like(pred, dtype=np.float64)
    for i in range(n_test):
        v_fine = func_space.eval_batch(func_feats[i : i + 1], x[:, None])[0]
        out = solve_adr_robin(
            xmin=0.0, xmax=1.0, tmin=0.0, tmax=t_max,
            alpha=params.alpha,
            source_values=v_fine, source_points=x,
            beta_left=params.beta, u0_left=params.u0,
            beta_right=params.beta, u0_right=params.u0,
            u_initial=params.u0,
            source_scale=params.f_std, constant_source=params.f_mean,
            nx=nx, nt=nt, interp_kind="cubic",
        )
        true[i] = out["u"]

    rel_l2_per = np.array([relative_l2(true[i], pred[i]) for i in range(n_test)])
    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        u_true=true, u_pred=pred, x=x, t=t, v_sensors=v_sensors,
    )
    plot_field_triplet(
        x, t, true[0], pred[0],
        run_dir / "plots" / "eval_sample0_triplet.png",
        model_label="PI-DeepONet 1D",
    )
    plot_profile_final_time(
        x, t, true[0], {"PI-DeepONet 1D": pred[0]},
        run_dir / "plots" / "eval_sample0_profile.png",
    )
    return {
        "rel_l2_mean": float(rel_l2_per.mean()),
        "rel_l2_std": float(rel_l2_per.std()),
        "rel_l2_per_sample": rel_l2_per.tolist(),
        "rmse_celsius": rmse(true, pred),
        "max_abs_error_celsius": float(np.abs(true - pred).max()),
        "truth_solver": "solve_adr_robin",
        "n_test": n_test,
        "eval_seed": int(ecfg["seed"]),
    }


def evaluate_2d(model, func_space, eval_pts, cfg: dict, params: CartesianHeatParams, run_dir: Path) -> dict:
    """2D 线评估：eval.seed 采样测试函数 → solve_heat_2d_robin 真值 → 物理 °C RelL2。"""
    ecfg = cfg["eval"]
    n_test, nx, nt = int(ecfg["n_test"]), int(ecfg["nx"]), int(ecfg["nt"])
    t_max = 0.5

    set_seed(int(ecfg["seed"]))  # 评估函数采样独立于训练随机序列
    func_feats = func_space.random(n_test)
    v_sensors = func_space.eval_batch(func_feats, eval_pts)  # (n_test, n_sensors²)

    x1 = np.linspace(0.0, 1.0, nx)
    t = np.linspace(0.0, t_max, nt)
    fine_pts = _sensor_points_2d(nx)  # (nx², 2)，ij 序
    Xg, Yg, Tg = np.meshgrid(x1, x1, t, indexing="ij")
    x_trunk = np.stack([Xg.ravel(), Yg.ravel(), Tg.ravel()], axis=1).astype(np.float32)

    pred_norm = model.predict((v_sensors.astype(np.float32), x_trunk)).reshape(n_test, nx, nx, nt)
    pred = params.denormalize_temperature(pred_norm)  # θ → °C

    true = np.zeros_like(pred, dtype=np.float64)
    for i in range(n_test):
        v_norm_fine = func_space.eval_batch(func_feats[i : i + 1], fine_pts)[0]
        f_phys = params.denormalize_source(v_norm_fine).reshape(nx, nx)  # ij 序 °C/s
        out = solve_heat_2d_robin(
            alpha=params.alpha, f_grid=f_phys, u0=params.u0,
            n=nx, T_max=t_max, nt_save=nt, beta=params.beta,
        )
        true[i] = np.transpose(out["u"], (1, 2, 0))  # (nt, nx, ny) → (nx, ny, nt)

    rel_l2_per = np.array([relative_l2(true[i], pred[i]) for i in range(n_test)])
    np.savez_compressed(
        run_dir / "predictions_eval.npz",
        u_true=true, u_pred=pred, x=x1, t=t, v_sensors=v_sensors,
    )
    plot_field_triplet_2d(
        x1, x1, true[0, :, :, -1], pred[0, :, :, -1],
        run_dir / "plots" / "eval_sample0_t_final_triplet.png",
        model_label="PI-DeepONet 2D",
        title_suffix=f"t = {t[-1]:.3g} s, eval sample 0",
    )
    return {
        "rel_l2_mean": float(rel_l2_per.mean()),
        "rel_l2_std": float(rel_l2_per.std()),
        "rel_l2_per_sample": rel_l2_per.tolist(),
        "rmse_celsius": rmse(true, pred),
        "max_abs_error_celsius": float(np.abs(true - pred).max()),
        "truth_solver": "solve_heat_2d_robin",
        "n_test": n_test,
        "eval_seed": int(ecfg["seed"]),
    }


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（pi_deeponet_1d.yaml / pi_deeponet_2d.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写 collocation/operator/training/eval 规模）")
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

    geometry = str(cfg.get("geometry", "")).lower()
    if geometry not in ("1d", "2d"):
        raise ValueError(f"config 的 geometry 字段须为 '1d' 或 '2d'，实际为 {geometry!r}")

    seed = int(cfg["seed"])
    set_seed(seed)  # PDE 采点 + 函数空间采样 + 网络初始化

    params = CartesianHeatParams(**{k: float(v) for k, v in cfg.get("physics", {}).items()})

    if geometry == "1d":
        data, func_space, eval_pts = build_data_1d(cfg, params)
        n_sensors = int(cfg["function_space"]["n_sensors"])
    else:
        data, func_space, eval_pts = build_data_2d(cfg, params)
        n_sensors = int(cfg["function_space"]["n_sensors_per_dim"]) ** 2

    mcfg = cfg["model"]
    if int(mcfg["branch_layers"][0]) != n_sensors:
        raise ValueError(
            f"branch_layers[0]={mcfg['branch_layers'][0]} 与传感点数 {n_sensors} 不一致"
        )
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
        loss_weights=None,  # 档案两条 PI-DeepONet 线均为等权
        metrics=None,  # PDE 数据无真值 y_test，不能挂相对误差 metric
        display_every=int(tcfg["display_every"]),
        checkpoint_period=int(tcfg["checkpoint_period"]),
        extra={
            "config_path": str(args.config),
            "smoke": bool(args.smoke),
            "geometry": geometry,
            "physics": dict(cfg.get("physics", {})),
            "collocation": dict(cfg["collocation"]),
            "function_space": dict(cfg["function_space"]),
            "operator": dict(cfg["operator"]),
            "model": dict(mcfg),
            "eval": dict(cfg["eval"]),
        },
    )
    train_dde_model(model, train_cfg)

    # ---------------- 评估：strict=True 加载最优 checkpoint ----------------
    best_ckpt = find_best_checkpoint(run_dir / "checkpoints") or find_best_checkpoint(
        run_dir / "checkpoints", prefix="final"
    )
    load_weights_strict(model, best_ckpt)
    print(f"[eval] 已加载最优 checkpoint：{best_ckpt}")

    if geometry == "1d":
        eval_metrics = evaluate_1d(model, func_space, eval_pts, cfg, params, run_dir)
    else:
        eval_metrics = evaluate_2d(model, func_space, eval_pts, cfg, params, run_dir)

    metrics = {
        "run_id": run_id,
        "geometry": geometry,
        "smoke": bool(args.smoke),
        **eval_metrics,
        "best_checkpoint": str(best_ckpt),
        "wall_time_s": round(time.perf_counter() - t_start, 3),
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(
        f"[eval] RelL2 mean={metrics['rel_l2_mean']:.4%} ± {metrics['rel_l2_std']:.4%} | "
        f"RMSE={metrics['rmse_celsius']:.3f} °C（{metrics['n_test']} 个测试函数）"
    )

    if bool(cfg.get("zip_artifacts", True)):
        zip_path = zip_artifacts(run_dir)
        print(f"[done] 产物已写入 {run_dir}，打包 {zip_path}")
    else:
        print(f"[done] 产物已写入 {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
