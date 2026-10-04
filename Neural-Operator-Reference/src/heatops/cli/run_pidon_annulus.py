"""2D 圆环 PIDON（Physics-Informed DeepONet，DeepXDE）训练入口（档案 heat_2d_annulus_pidon_polar.py 重写）。

用法（从仓库根目录）::

    python -m heatops.cli.run_pidon_annulus --config configs/pidon_annulus.yaml --run-id E109
    python -m heatops.cli.run_pidon_annulus --config configs/pidon_annulus.yaml --smoke --run-id E109

纯物理约束算子学习，**无需数据**：branch 输入为标量归一化热流 q*（自定义
``HeatFluxSpace`` 把 q* tile 到全部传感器点），trunk 输入 (r*, θ, t*)。
无量纲域（r* = r/R_out ∈ [R_in*, 1]，t* = t/t_scale，T* = (T-T_inf)/T_scale）：

- PDE：``u_t* = ∇*²u``（极坐标 Laplacian，**无源项**——热流 q 仅经内边界
  进入域内，档案设计予以保留）；
- 内边界（OperatorBC）：``du_r* + q* = 0``（-dT*/dr* = q*）；
- 外边界（OperatorBC）：``du_r* + Bi·u = 0``（-dT*/dr* = Bi·T*）；
- IC：``u(t*=0) = 0``（均匀 T_inf 起算）；
- θ 周期：``PeriodicBC(component_x=1)``。

产物入 ``artifacts/<run_id>/``：config.json、loss_history.npz、loss_curve.png、
checkpoints/（DeepXDE 格式）、sanity 径向剖面预测 npz/png、metrics.json。

DeepXDE 为延迟导入（``main()`` 内设 ``DDE_BACKEND=pytorch`` 后才 import），
模块本身无 import 副作用。
"""

from __future__ import annotations

from heatops.training.run_context import managed_cli, prepare_run, run_directory, project_root, config_path_value, bind_data

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import yaml

import heatops
from heatops.evaluation.annulus_plots import plot_radial_profiles
from heatops.problems import AnnulusParams
from heatops.training.torch_loop import set_seed

#: --smoke 最小规模覆写（CPU 分钟级；只验证管线贯通，不追求收敛）
_SMOKE_OVERRIDES = {
    ("pde", "num_domain"): 200,
    ("pde", "num_boundary"): 80,
    ("pde", "num_initial"): 80,
    ("operator", "num_function"): 16,
    ("operator", "num_test"): 8,
    ("operator", "batch_size"): 8,
    ("training", "iterations"): 60,
    ("training", "display_every"): 10,
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
    kwargs["q_in"] = float(pcfg["q_base"])
    return AnnulusParams(**kwargs)


def _plot_loss_curve(steps: np.ndarray, losses: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["pde", "bc_inner", "bc_outer", "ic", "periodic"]
    plt.figure(figsize=(10, 6))
    for i in range(losses.shape[1]):
        label = labels[i] if i < len(labels) else f"loss_{i}"
        plt.semilogy(steps, np.maximum(losses[:, i], 1e-30), label=label)
    plt.semilogy(steps, np.maximum(losses.sum(axis=1), 1e-30), "k--", label="total", alpha=0.6)
    plt.title("PIDON Training Loss (Log Scale)")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


@managed_cli
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="YAML 配置路径（如 configs/pidon_annulus.yaml）")
    parser.add_argument("--run-id", default=None, help="运行编号（默认取 config 的 experiment 字段）")
    parser.add_argument("--smoke", action="store_true", help="最小规模自检（覆写采样点数/迭代数）")
    parser.add_argument("--iterations", type=int, default=None, help="覆写训练迭代数（优先于 config 与 --smoke）")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, smoke=args.smoke)
    if args.iterations is not None:
        cfg["training"]["iterations"] = args.iterations
    run_id = args.run_id or cfg.get("experiment")
    if not run_id:
        raise ValueError("未指定 run_id：请用 --run-id 或在 config 中设 experiment 字段")
    run_dir = run_directory(args.config, run_id)
    prepare_run(run_dir, cfg, args.config)

    # DeepXDE 延迟导入：先固定后端再 import（模块级无副作用）
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    import deepxde as dde
    import torch

    seed = int(cfg["seed"])
    set_seed(seed)

    params = build_params(cfg)
    q_base = float(cfg["physics"]["q_base"])
    t_max_s = float(cfg["physics"]["t_max"])
    ri_star = params.R_in_star
    bi = params.Bi
    t_star_final = t_max_s / params.t_scale  # t* = t/t_scale ⇒ PDE 扩散系数恰为 1
    q_lo = float(cfg["space"]["q_star_min"])
    q_hi = float(cfg["space"]["q_star_max"])

    # ---------------- 算子空间与几何 ----------------
    class HeatFluxSpace(dde.data.function_spaces.FunctionSpace):
        """标量热流系数空间：q* ~ U(q_lo, q_hi)，tile 到全部采样点（档案设计）。

        branch 输入仅为空间常数 q*，传感器位置不构成信息（``eval_one`` 与 x
        无关）；传感器网格只是 DeepONet branch 的输入形状约定。
        """

        def random(self, n):
            return np.random.uniform(q_lo, q_hi, (n, 1)).astype(np.float32)

        def eval_batch(self, features, xs):
            return np.tile(features, (1, len(xs))).astype(np.float32)

        def eval_one(self, feature, x):
            return feature[0]

    geom_polar = dde.geometry.Rectangle([ri_star, 0.0], [1.0, 2 * np.pi])
    timedomain = dde.geometry.TimeDomain(0.0, t_star_final)
    geomtime = dde.geometry.GeometryXTime(geom_polar, timedomain)

    # ---------------- PDE 与边界（无量纲，无源项：q 仅经内边界进入） ----------------
    def pde(x, u, v):
        r = x[:, 0:1]
        du_t = dde.grad.jacobian(u, x, i=0, j=2)
        du_r = dde.grad.jacobian(u, x, i=0, j=0)
        du_rr = dde.grad.hessian(u, x, i=0, j=0)
        du_tt2 = dde.grad.hessian(u, x, i=1, j=1)
        laplacian = du_rr + du_r / r + du_tt2 / r**2
        return du_t - laplacian

    def bc_outer_func(x, u, v):
        du_r = dde.grad.jacobian(u, x, i=0, j=0)
        return du_r + bi * u  # -dT*/dr* = Bi·T*

    ic = dde.icbc.IC(geomtime, lambda x: 0.0, lambda _, on: on)
    from heatops.physics.operator_boundary import HeatFluxBC
    bc_in = HeatFluxBC(geomtime, lambda x, on: on and np.isclose(x[0], ri_star))
    bc_out = dde.icbc.OperatorBC(geomtime, bc_outer_func, lambda x, on: on and np.isclose(x[0], 1.0))
    bc_periodic = dde.icbc.PeriodicBC(geomtime, 1, lambda x, on: on and (np.isclose(x[1], 0.0) or np.isclose(x[1], 2 * np.pi)))

    pcfg = cfg["pde"]
    pde_data = dde.data.TimePDE(
        geomtime, pde, [bc_in, bc_out, ic, bc_periodic],
        num_domain=int(pcfg["num_domain"]),
        num_boundary=int(pcfg["num_boundary"]),
        num_initial=int(pcfg["num_initial"]),
    )

    m_r, m_theta = int(cfg["sensors"]["m_r"]), int(cfg["sensors"]["m_theta"])
    r_pts = np.linspace(ri_star, 1.0, m_r)
    theta_pts = np.linspace(0.0, 2 * np.pi, m_theta)
    rv, tv = np.meshgrid(r_pts, theta_pts)
    eval_pts = np.vstack((rv.ravel(), tv.ravel())).T

    ocfg = cfg["operator"]
    data = dde.data.PDEOperatorCartesianProd(
        pde_data, HeatFluxSpace(), eval_pts,
        num_function=int(ocfg["num_function"]),
        num_test=int(ocfg["num_test"]),
        batch_size=int(ocfg["batch_size"]),
    )

    ncfg = cfg["net"]
    net = dde.nn.DeepONetCartesianProd(
        [m_r * m_theta, *[int(w) for w in ncfg["branch_hidden"]]],
        [3, *[int(w) for w in ncfg["trunk_hidden"]]],
        str(ncfg["activation"]), "Glorot normal",
    )
    model = dde.Model(data, net)

    tcfg = cfg["training"]
    loss_weights = [float(w) for w in tcfg["loss_weights"]]
    model.compile("adam", lr=float(tcfg["lr"]), loss_weights=loss_weights)

    iterations = int(tcfg["iterations"])
    print(f"[pidon] run_id={run_id} iterations={iterations} q*~U({q_lo},{q_hi}) "
          f"t*_final={t_star_final:.4g} Bi={bi:.4g} sensors={m_r}x{m_theta}")
    t0 = time.time()
    losshistory, train_state = model.train(
        iterations=iterations, display_every=int(tcfg["display_every"])
    )
    elapsed = time.time() - t0
    if not np.all(np.isfinite(np.asarray(losshistory.loss_train))):
        raise FloatingPointError("PIDON 出现非有限损失")

    # ---------------- 产物落盘 ----------------
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model.save(str(ckpt_dir / "model"))
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    ckpt_files = sorted(p.name for p in ckpt_dir.glob("model-*"))

    steps = np.asarray(losshistory.steps, dtype=np.int64)
    loss_train = np.asarray(losshistory.loss_train, dtype=np.float64)
    np.savez_compressed(run_dir / "loss_history.npz", steps=steps, loss_train=loss_train)
    _plot_loss_curve(steps, loss_train, run_dir / "loss_curve.png")

    # sanity 预测：径向剖面（θ=0 母线），物理单位 °C（无量纲 → °C 换算集中于此）
    r_phys = np.linspace(params.R_in, params.R_out, 50)
    r_star_eval = r_phys / params.R_out
    sanity: dict[str, np.ndarray] = {}
    profiles: dict[str, np.ndarray] = {}
    for q_star_val in (1.0, 3.0):
        v_input = np.full((1, m_r * m_theta), q_star_val, dtype=np.float32)
        for frac in (0.5, 1.0):
            t_star = frac * t_star_final
            pts = np.stack(
                [r_star_eval, np.zeros_like(r_star_eval), np.full_like(r_star_eval, t_star)],
                axis=1,
            )
            u_pred = model.predict((v_input, pts)).flatten()
            T_pred = params.T_inf + params.T_scale * u_pred
            key = f"q*{q_star_val:g}_t{frac:g}"
            sanity[f"u_star_{key}"] = u_pred
            profiles[f"q*={q_star_val:g}, t={frac * t_max_s:.0f}s"] = T_pred
    np.savez_compressed(run_dir / "sanity_radial.npz", r=r_phys, **sanity)
    plot_radial_profiles(
        r_phys, profiles, run_dir / "plots" / "sanity_radial_profiles.png",
        title=f"PIDON ({run_id}) Radial Profiles", ylabel="T (°C)",
    )

    _write_config_json(run_dir, {
        "config_path": str(args.config),
        "smoke": bool(args.smoke),
        "cfg": cfg,
        "seed": seed,
        "code_revision": f"heatops-{heatops.__version__}",
        "deepxde_version": dde.__version__,
        "dimensionless": {
            "R_in_star": ri_star, "Bi": bi, "T_scale": params.T_scale,
            "t_scale": params.t_scale, "t_star_final": t_star_final, "q_base": q_base,
        },
    })
    final_losses = loss_train[-1] if loss_train.ndim == 2 else np.asarray([loss_train[-1]])
    metrics = {
        "run_id": run_id,
        "smoke": bool(args.smoke),
        "iterations": iterations,
        "elapsed_s": round(elapsed, 2),
        "final_loss_terms": [float(v) for v in final_losses],
        "final_total_loss": float(np.sum(final_losses)),
        "loss_terms_are_weighted": True,
        "best_step": int(getattr(train_state, "best_step", -1)),
        "checkpoints": ckpt_files,
        "note": "纯物理约束（无数据项）；损失项顺序 PDE/BC_in/BC_out/IC/Periodic",
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"[eval] final total loss={metrics['final_total_loss']:.4e} elapsed={elapsed:.1f}s")
    print(f"[done] 产物已写入 {run_dir}（checkpoint: {ckpt_files}）")
    return 0


def _write_config_json(run_dir: Path, payload: dict) -> None:
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    raise SystemExit(main())
