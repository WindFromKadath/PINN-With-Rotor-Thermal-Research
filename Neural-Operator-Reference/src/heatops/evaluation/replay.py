"""Strict checkpoint reconstruction and prediction replay from one run bundle."""
from pathlib import Path
import json
import numpy as np
import torch
from heatops.models.fno import HeatOperatorFNO, HeatOperatorFNO3D
from heatops.models.deepxde_models import make_deeponet_cartesian, make_fnn, PeriodicFeatureTransform
from heatops.problems import AnnulusParams


def replay_run(run_dir):
    rd = Path(run_dir).resolve()
    cfg = json.loads((rd / "config.json").read_text(encoding="utf-8"))
    metrics = json.loads((rd / "metrics.json").read_text(encoding="utf-8"))
    root = next((p for p in rd.parents if (p / "pyproject.toml").exists()), None)
    if root is None:
        raise ValueError("运行目录必须位于项目 artifacts 下")
    name = metrics.get("best_checkpoint", metrics.get("checkpoint"))
    if name:
        ckpt = Path(name)
        if not ckpt.is_absolute():
            ckpt = root / ckpt if str(name).startswith("artifacts") else rd / ckpt
        if not ckpt.exists():
            ckpt = rd / "checkpoints" / Path(name).name
    else:
        ckpt = rd / "checkpoints" / metrics["checkpoints"][-1]
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    extra = cfg.get("extra", {})
    if "cfg" in cfg:
        cc = cfg["cfg"]
        if "net" in cc:
            mc = cc["net"]
            sensors = cc["sensors"]["m_r"] * cc["sensors"]["m_theta"]
            net = make_deeponet_cartesian([sensors, *mc["branch_hidden"]], [3, *mc["trunk_hidden"]], mc["activation"])
            blob = np.load(rd / "sanity_radial.npz")
            phys = AnnulusParams(**{k: float(v) for k, v in cc["physics"].items() if k in ("R_in","R_out","k","rho","cp","h","T_inf")}, q_in=float(cc["physics"]["q_base"]))
            predictions, saved = {}, {}
            for q in (1.0, 3.0):
                for frac in (0.5, 1.0):
                    key = f"u_star_q*{q:g}_t{frac:g}"
                    pts = np.stack([blob["r"] / phys.R_out, np.zeros_like(blob["r"]), np.full_like(blob["r"], frac * cfg["dimensionless"]["t_star_final"])], axis=1)
                    net.load_state_dict(state["model_state_dict"], strict=True)
                    with torch.no_grad():
                        predictions[key] = net((torch.full((1, sensors), q), torch.tensor(pts, dtype=torch.float32))).numpy().ravel()
                    saved[key] = blob[key]
            return predictions, saved, ckpt
        from heatops.cli.run_pino_annulus import make_polar_grid, predict_fields
        blob = np.load(rd / "predictions_eval.npz")
        mc = cc["model"]
        net = HeatOperatorFNO3D(tuple(mc["n_modes"]), mc["hidden_channels"])
        net.load_state_dict(state, strict=True)
        grid = make_polar_grid(blob["t_star"], blob["r_star"], blob["theta"])
        pred = predict_fields(net, torch.tensor(blob["q_star"], dtype=torch.float32), grid, tuple(blob["u_star_pred"].shape[1:]), torch.device("cpu"))
        return {"prediction": pred}, {"prediction": blob["u_star_pred"]}, ckpt
    mc = extra["model"]
    blob = np.load(rd / "predictions_eval.npz")
    if "n_modes" in mc:
        from heatops.data.datasets import Heat1DDataset
        from heatops.cli.run_pino_1d import predict_fields
        net = HeatOperatorFNO(tuple(mc["n_modes"]), mc["hidden_channels"], use_ansatz=mc.get("use_ansatz", True), a=float(extra["physics"]["a"]))
        net.load_state_dict(state, strict=True)
        ds = Heat1DDataset.load(root / f"data/processed/heat1d_{extra['data_cache_key']}.pt")
        _, eval_ds = ds.split(extra["data"]["split_ratio"], extra["data"]["split_seed"])
        pred = predict_fields(net, eval_ds, torch.device("cpu"))
        return {"prediction": pred}, {"prediction": blob["u_pred"]}, ckpt
    if "layer_sizes" in mc:
        transient = metrics["mode"] == "transient"
        net = make_fnn(mc["layer_sizes"], mc.get("activation", "tanh"), mc.get("kernel_initializer", "Glorot normal"), feature_transform=PeriodicFeatureTransform(1) if transient else None)
        net.load_state_dict(state["model_state_dict"], strict=True)
        phys = AnnulusParams(**{k: float(v) for k,v in extra["physics"].items()}) if isinstance(extra["physics"], dict) else AnnulusParams()
        r = blob["r"] / phys.R_out
        pred = []
        for value in blob["times" if transient else "angles"]:
            pts = np.stack([r, np.zeros_like(r), np.full_like(r, value/phys.t_scale)], axis=1) if transient else np.stack([r*np.cos(value), r*np.sin(value)], axis=1)
            with torch.no_grad():
                pred.append(phys.to_dimensional_temperature(net(torch.tensor(pts,dtype=torch.float32)).numpy().ravel()))
        return {"prediction": np.asarray(pred)}, {"prediction": blob["T_pinn" if transient else "T_pred"]}, ckpt
    net = make_deeponet_cartesian(mc["branch_layers"], mc["trunk_layers"], mc.get("activation", "tanh"), mc.get("kernel_initializer", "Glorot normal"))
    net.load_state_dict(state["model_state_dict"], strict=True)
    if "v_sensors" in blob:
        sensors = blob["v_sensors"]
        axes = [blob["x"], blob["t"]] if extra["geometry"] == "1d" else [blob["x"], blob["x"], blob["t"]]
        pts = np.stack([a.ravel() for a in np.meshgrid(*axes, indexing="ij")], axis=1)
    else:
        ds = np.load(root / f"data/processed/deeponet2d_{extra['data_cache_key']}.npz")
        sensors, pts = ds["X_branch"][blob["eval_indices"]], ds["X_trunk"]
    with torch.no_grad():
        pred = net((torch.tensor(sensors,dtype=torch.float32), torch.tensor(pts,dtype=torch.float32))).numpy().reshape(blob["u_pred"].shape)
    if extra.get("geometry") != "1d":
        pred = extra["physics"]["u0"] + extra["physics"]["u_scale"] * pred
    return {"prediction": pred}, {"prediction": blob["u_pred"]}, ckpt
