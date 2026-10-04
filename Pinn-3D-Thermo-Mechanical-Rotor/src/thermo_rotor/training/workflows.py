"""Reusable short/full training workflows with independent profiles and state."""
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from ..config import Profile
from ..models import model_from_spec
from ..models.checkpoint import save, load
from ..data.comsol import sample_times
from ..data.transforms import inputs, targets
from ..geometry.sampling import mesh, volume, surface, frame, cylindrical_normals
from ..physics import thermal, mechanical
from ..coupling.temperature import TemperatureBridge, linear_iwg_temperature
from ..coupling.interface import interface_loss
from ..evaluation.predict import predict, determine_gauge
from ..evaluation.metrics import metrics
from ..plotting.figures import plot_predictions
from ..training.run import write_json, sha256
from .optimizer import optimize


def setup(config):
    rt = config["runtime"]
    torch.set_num_threads(rt["threads"])
    torch.manual_seed(rt["seed"])
    np.random.seed(rt["seed"])
    if rt["device"].startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return Profile(**config["physics"]), np.random.default_rng(rt["seed"])


def tensor(a, cfg):
    return torch.as_tensor(a, dtype=getattr(torch, cfg["runtime"]["dtype"]), device=cfg["runtime"]["device"])


def sample_geometry(cfg, p, rng):
    stl = Path(cfg["paths"]["stl"])
    n = cfg["sampling"]["domain"]
    nb = cfg["sampling"]["boundary"]
    points = volume(mesh(stl/"ring.stl"), n, rng)
    times = np.asarray(cfg["sampling"]["times"])
    dom = frame(points, rng.choice(times, n))
    initial = frame(volume(mesh(stl/"ring.stl"), cfg["sampling"]["initial"], rng), 0.)
    bc = {}
    for name in ("heating", "inner", "outer"):
        pts, normals = surface(mesh(stl/f"{name}.stl"), nb, rng, flip=name == "heating")
        bc[name] = (frame(pts, rng.choice(times, nb)), normals, cylindrical_normals(pts, normals))
    return dom, initial, bc


def _data(cfg, key, times, count, seed):
    path = cfg["paths"].get(key)
    if not path:
        raise ValueError(f"Missing required data path: {key}")
    section = cfg.get("data_units", {}).get(key, {})
    return sample_times(path, per_time=count, times=times, seed=seed, **section)


def finish(run, label, model, spec, p, data, gauge=None, source=None):
    checkpoint = run.path/f"{label}.pt"
    save(checkpoint, model, spec, p, gauge=gauge, source=source)
    data.to_csv(run.path/f"{label}_inputs.csv", index=False, float_format="%.17g")
    pred = predict(model, data, p, spec, gauge)
    reloaded, pp, meta = load(checkpoint)
    again = predict(reloaded, data, pp, spec, meta["gauge"])
    difference = float(np.abs(pred-again).max())
    # GPU/CPU use different kernels; keep the saved check on the original device.
    if run.config["runtime"]["device"] != "cpu":
        reloaded, pp, meta = load(checkpoint, device=run.config["runtime"]["device"])
        again = predict(reloaded, data, pp, spec, meta["gauge"])
        difference = float(np.abs(pred-again).max())
    if difference > (1e-5 if spec["kind"] in ("thermal", "steady") else 1e-9):
        raise ValueError("Reload prediction mismatch")
    truth = data[["T"]].to_numpy() if spec["kind"] in ("thermal", "steady") else data[["u_r", "u_z"]].to_numpy()
    report = metrics(pred, truth, spec["kind"])
    report.update(max_reload_difference=difference, checkpoint_sha256=sha256(checkpoint),
                  point_set_sha256=sha256(run.path/f"{label}_inputs.csv"),
                  scope=run.config.get("evaluation", {}).get("scope", "short workflow, no convergence claim"))
    result = data.copy()
    for j, key in enumerate(["T"] if spec["kind"] in ("thermal", "steady") else ["u_r", "u_z"]):
        result[f"{key}_prediction"] = pred[:, j].astype(np.float64)
    result.to_csv(run.path/f"{label}_predictions.csv", index=False, float_format="%.17g")
    write_json(run.path/f"{label}_metrics.json", report)
    plot_predictions(data, pred, truth, spec["kind"], run.path/"figures"/label)
    return report


def train_thermal(cfg, run, p, rng, geometry=None):
    dom, initial, bc = geometry or sample_geometry(cfg, p, rng)
    spec = dict(kind="thermal", hidden=cfg["network"]["thermal"], prefix="layers", activation="tanh")
    model = model_from_spec(spec).to(dtype=getattr(torch, cfg["runtime"]["dtype"]), device=cfg["runtime"]["device"])
    x, ic = tensor(inputs(dom, p, "thermal"), cfg), tensor(inputs(initial, p, "thermal"), cfg)
    boundaries = {k: (tensor(inputs(f, p, "thermal"), cfg), tensor(normal, cfg)) for k, (f, normal, _) in bc.items()}
    tc = cfg["training"]
    weights = tc.get("thermal_weights", dict(pde=5., ic=25., heating=5., inner=3., outer=3.))
    def loss(step):
        values = dict(pde=weights["pde"]*thermal.residual(model, x.detach().clone(), p.Fo).square().mean(),
                      ic=weights["ic"]*model(ic).square().mean())
        for name, (points, normal) in boundaries.items():
            values[name] = weights[name]*thermal.boundary_residual(model, points.detach().clone(), normal, p,
                                                        "neumann" if name == "heating" else "robin").square().mean()
        return values
    optimize([model], loss, tc.get("thermal_steps", tc.get("adam_steps", 40)), tc.get("lr", 1e-3),
             tc.get("thermal_lbfgs_steps", tc.get("lbfgs_steps", 0)), tc.get("lbfgs_max_iter", 5), run.path/"thermal_losses.json")
    if cfg["paths"].get("thermal_eval"):
        evaluation = _data(cfg, "thermal_eval", cfg["sampling"]["times"], cfg["sampling"].get("evaluation_per_time", 128), 99999)
    else:
        evaluation = dom.copy()
        evaluation["T"] = p.T0
    report = finish(run, "thermal", model, spec, p, evaluation)
    return model, geometry or (dom, initial, bc), report


def train_mechanical(cfg, run, p, rng, thermal_model=None, geometry=None):
    dom, _, bc = geometry or sample_geometry(cfg, p, rng)
    bridge_source = None
    if thermal_model is None:
        if not cfg["paths"].get("thermal_checkpoint"):
            raise ValueError("train-mechanical requires paths.thermal_checkpoint")
        ck = Path(cfg["paths"]["thermal_checkpoint"])
        thermal_model, source_profile, bridge_meta = load(ck, cfg["runtime"]["device"], expected_kind="thermal")
        thermal_model.to(dtype=getattr(torch, cfg["runtime"]["dtype"]))
        bridge_source = dict(checkpoint_sha256=sha256(ck), thermal_profile=source_profile.to_dict())
        save(run.path/"thermal_bridge.pt", thermal_model, bridge_meta["spec"], source_profile,
             source=bridge_source)
    else:
        source_profile = p
        bridge_source = dict(checkpoint="thermal.pt", checkpoint_sha256=sha256(run.path/"thermal.pt"), thermal_profile=p.to_dict())
    temp = TemperatureBridge(thermal_model, source_profile, p)
    spec = dict(kind="mechanical", hidden=cfg["network"]["mechanical"], prefix="net", activation="tanh")
    model = model_from_spec(spec).to(dtype=getattr(torch, cfg["runtime"]["dtype"]), device=cfg["runtime"]["device"])
    anchors = _data(cfg, "mechanical_train", cfg["sampling"]["times"], cfg["sampling"]["anchors_per_time"], 42)
    anchors.to_csv(run.path/"mechanical_training_anchors.csv", index=False)
    xp, yp = tensor(inputs(anchors, p, "mechanical"), cfg), tensor(targets(anchors, p, "mechanical"), cfg)
    x = tensor(inputs(dom, p, "mechanical"), cfg)
    boundaries = [(tensor(inputs(f, p, "mechanical"), cfg), tensor(nrz, cfg)) for f, _, nrz in bc.values()]
    tc = cfg["training"]
    def loss(step):
        res = mechanical.equilibrium(model, x.detach().clone(), temp, p)
        traction = sum(mechanical.traction(model, xb.detach().clone(), normals, temp, p).square().sum(dim=1).mean()
                       for xb, normals in boundaries)
        return dict(pde=tc.get("w_pde", 1.)*res.square().sum(dim=1).mean(),
                    traction=tc.get("w_traction", 1.)*traction,
                    data=tc.get("w_data", 100.)*((model(xp)-yp)**2).mean())
    before = {k: v.clone() for k, v in thermal_model.state_dict().items()}
    optimize([model], loss, tc.get("mechanical_steps", tc.get("adam_steps", 40)), tc.get("lr", 1e-3),
             tc.get("mechanical_lbfgs_steps", tc.get("lbfgs_steps", 0)), tc.get("lbfgs_max_iter", 5), run.path/"mechanical_losses.json")
    if any(not torch.equal(v, thermal_model.state_dict()[k]) for k, v in before.items()):
        raise ValueError("Frozen thermal parameters changed")
    evaluation = _data(cfg, "mechanical_eval", cfg["sampling"]["times"], cfg["sampling"].get("evaluation_per_time", 128), 99999)
    return finish(run, "mechanical", model, spec, p, evaluation, source=bridge_source)


def train_mlp(cfg, run, p):
    reports = {}
    evaluation = _data(cfg, "mechanical_eval", cfg["sampling"]["times"], cfg["sampling"].get("evaluation_per_time", 128), 99999)
    for count in cfg["training"]["label_counts"]:
        # Explicit total count, distinct from the original 1100/33000 labels.
        anchors = _data(cfg, "mechanical_train", cfg["sampling"]["times"], max(1, int(np.ceil(count/len(cfg["sampling"]["times"])))), 42).iloc[:count]
        label = f"mlp_{count}"
        anchors.to_csv(run.path/f"{label}_anchors.csv", index=False)
        spec = dict(kind="mlp", hidden=cfg["network"]["mechanical"], prefix="net")
        model = model_from_spec(spec).to(dtype=getattr(torch, cfg["runtime"]["dtype"]), device=cfg["runtime"]["device"])
        x, y = tensor(inputs(anchors, p, "mlp"), cfg), tensor(targets(anchors, p, "mlp"), cfg)
        optimize([model], lambda _: dict(data=((model(x)-y)**2).mean()), cfg["training"]["adam_steps"],
                 cfg["training"].get("lr", 1e-3), cfg["training"].get("lbfgs_steps", 0), log_path=run.path/f"{label}_losses.json")
        reports[label] = finish(run, label, model, spec, p, evaluation)
    return reports


class TrainingGauge(nn.Module):
    def __init__(self, raw, mode, anchor):
        super().__init__()
        self.raw, self.mode, self.anchor = raw, mode, anchor

    def forward(self, x):
        u = self.raw(x)
        if self.mode == "projection":
            shift = (u[:, 1:2]*x[:, 0:1]).sum()/(x[:, 0:1].sum()+1e-12)
            return torch.cat((u[:, 0:1], u[:, 1:2]-shift.detach()), dim=1)
        return u-self.raw(self.anchor).detach()


def train_iwg(cfg, run, p, rng):
    dom, _, bc = sample_geometry(cfg, p, rng)
    x = tensor(inputs(dom, p, "iwg"), cfg)
    boundaries = [(tensor(inputs(f, p, "iwg"), cfg), tensor(nrz, cfg)) for f, _, nrz in bc.values()]
    eval_frames = {mode: _data(cfg, f"iwg_{mode}", [200.], cfg["sampling"].get("evaluation_per_time", 128), 99999)
                   for mode in ("constraint", "projection")}
    reports = {}
    for mode in ("constraint", "projection"):
        spec = dict(kind="iwg", hidden=cfg["network"]["iwg"], fourier_dim=cfg["network"].get("fourier_dim", 64))
        model = model_from_spec(spec).to(dtype=x.dtype, device=x.device)
        anchor = [[p.r_min/p.L, float(x[:, 1].min())]]
        wrapped = TrainingGauge(model, mode, tensor(anchor, cfg))
        def loss(step, ramp=1.):
            temp = lambda xx: linear_iwg_temperature(xx, p)
            residual = mechanical.equilibrium(wrapped, x.detach().clone(), temp, p)
            lp = (residual[:, 0]**2+400*residual[:, 1]**2).mean()*ramp
            lb = torch.zeros((), dtype=x.dtype, device=x.device)
            for xb, nb in boundaries:
                residual_bc = mechanical.traction(wrapped, xb.detach().clone(), nb, temp, p)
                for mask, weight in ((nb[:, 0] < -.8, 10.), (nb[:, 0] > .8, 10.), (nb[:, 1] < -.3, 200.), (nb[:, 1] > .3, 200.)):
                    if mask.any():
                        lb = lb+weight*residual_bc[mask].square().sum(dim=1).mean()
            u = wrapped(x)
            z = x[:, 1]
            tol = (z.max()-z.min())*.02
            top, bot = z >= z.max()-tol, z <= z.min()+tol
            def average(mask):
                return (u[mask, 1]*x[mask, 0]).sum()/(x[mask, 0].sum()+1e-12)
            # Preserve the source axial-stretch target, including its original scale expression.
            target = p.alpha*(400-p.T0)*(float(z.max()-z.min())*p.L)/p.L/p.U
            return dict(pde=lp, boundary=lb, axial_stretch=10*(average(top)-average(bot)-target)**2)
        steps = cfg["training"]["stage_steps"]
        for stage, ramp in (("warmup", 0.), ("ramp", None), ("full", 1.)):
            adam_steps = cfg["training"].get("full_stage_adam_steps", steps) if stage == "full" else steps
            optimize([model], lambda step, rr=ramp: loss(step, step/steps if rr is None else rr), adam_steps,
                     cfg["training"].get("lr", 1e-3),
                     cfg["training"].get("lbfgs_steps", 0) if stage == "full" else 0,
                     log_path=run.path/f"iwg_{mode}_{stage}_losses.json", stage=stage,
                     accumulation=4 if mode == "projection" else 1)
        gauge = determine_gauge(model, x.detach().cpu().double().numpy(), mode, anchor[0])
        reports[mode] = finish(run, f"iwg_{mode}", model, spec, p, eval_frames[mode], gauge)
    return reports


def train_xpinn(cfg, run, p, rng):
    stl = Path(cfg["paths"]["stl"])
    region_dirs = dict(lower="Lower_ring", rib="Rib", upper="Upper_ring")
    spec = dict(kind="steady", hidden=cfg["network"]["steady"], prefix="layers")
    nets = {r: model_from_spec(spec).to(dtype=getattr(torch, cfg["runtime"]["dtype"]), device=cfg["runtime"]["device"]) for r in region_dirs}
    domains, boundaries, interfaces = {}, {}, {}
    for region, folder in region_dirs.items():
        points = volume(mesh(stl/folder/"total.stl"), cfg["sampling"]["domain"], rng)
        domains[region] = frame(points, 0.)
        names = ("outer", "side") if region == "rib" else ("inner", "outer", "top", "bottom")
        boundaries[region] = []
        for name in names:
            pts, normals = surface(mesh(stl/folder/f"{name}.stl"), cfg["sampling"]["boundary"], rng, flip=name == "inner")
            boundaries[region].append((tensor(inputs(frame(pts, 0.), p, "steady"), cfg), tensor(normals, cfg),
                                       "neumann" if region == "lower" and name == "inner" else "robin"))
    for name, source, target, file in (("rib_lower", "lower", "rib", "lower_inact.stl"),
                                       ("upper_rib", "rib", "upper", "upper_inact.stl")):
        pts, normals = surface(mesh(stl/"Rib"/file), cfg["sampling"]["boundary"], rng)
        interfaces[name] = (source, target, tensor(inputs(frame(pts, 0.), p, "steady"), cfg), tensor(normals, cfg))
    # Preserve the source's five inner-domain 380 K anchors and two successive weights of 10.
    lower_frame = domains["lower"]
    margin = (lower_frame.r.max()-lower_frame.r.min())*.1
    selected = lower_frame[lower_frame.r < lower_frame.r.min()+margin].iloc[:5]
    if selected.empty:
        raise ValueError("No inner-domain points for the source X-PINN temperature anchor")
    anchor_points = tensor(inputs(selected, p, "steady"), cfg)
    anchor_target = (380.-p.Tamb)/p.dT
    write_json(run.path/"temperature_anchor.json", dict(temperature_K=380., points=len(selected), total_weight=100.,
                protocol="first up to five points within inner radial 10 percent of sampled lower domain"))
    stages = cfg["training"].get("stages", [dict(name="A", regions=["lower"], w_bc=10., w_heat=5., w_int=0., lr=.001),
          dict(name="B", regions=["rib"], w_bc=80., w_heat=0., w_int=15., lr=.001),
          dict(name="C", regions=["upper"], w_bc=80., w_heat=0., w_int=15., lr=.001),
          dict(name="D", regions=["lower", "rib", "upper"], w_bc=10., w_heat=5., w_int=15., lr=.0005)])
    for stage in stages:
        active = stage["regions"]
        for name, net in nets.items():
            for parameter in net.parameters():
                parameter.requires_grad_(name in active)
        def loss(step):
            terms = {}
            if "lower" in active:
                terms["temperature_anchor"] = 100.*(nets["lower"](anchor_points)-anchor_target).square().mean()
            for region in active:
                net = nets[region]
                x = tensor(inputs(domains[region], p, "steady"), cfg)
                terms[f"pde_{region}"] = thermal.residual(net, x, p.Fo).square().mean()
                for j, (xb, normals, kind) in enumerate(boundaries[region]):
                    w = stage["w_heat"] if kind == "neumann" else stage["w_bc"]
                    terms[f"bc_{region}_{j}"] = w*thermal.boundary_residual(net, xb.detach().clone(), normals, p, kind).square().mean()
            for name, (source, target, x, normals) in interfaces.items():
                if target in active:
                    for key, value in interface_loss(nets[source], nets[target], x.detach().clone(), normals, p.k).items():
                        terms[f"interface_{name}_{key}"] = stage["w_int"]*value
            return terms
        optimize([nets[r] for r in active], loss, cfg["training"]["stage_steps"], stage["lr"],
                 cfg["training"].get("lbfgs_steps", 0), log_path=run.path/f"xpinn_{stage['name']}_losses.json", stage=stage["name"])
    report = {}
    for region in nets:
        d = domains[region].copy()
        # No matching steady COMSOL set was provided; this is a declared diagnostic reference.
        d["T"] = p.T0
        report[region] = finish(run, f"xpinn_{region}", nets[region], spec, p, d)
    write_json(run.path/"interface_policy.json", dict(source_flux="differentiate_then_detach", source_temperature="detach",
               direction="lower -> rib -> upper in all stages including D", steady=True))
    return report


def train(cfg, run, mode):
    p, rng = setup(cfg)
    if mode == "coupled":
        thermal_model, geom, rt = train_thermal(cfg, run, p, rng)
        rm = train_mechanical(cfg, run, p, rng, thermal_model, geom)
        reports = dict(thermal=rt, mechanical=rm)
    elif mode == "thermal":
        _, _, reports = train_thermal(cfg, run, p, rng)
    elif mode == "mechanical":
        reports = train_mechanical(cfg, run, p, rng)
    elif mode == "mlp":
        reports = train_mlp(cfg, run, p)
    elif mode == "iwg":
        reports = train_iwg(cfg, run, p, rng)
    elif mode == "xpinn":
        reports = train_xpinn(cfg, run, p, rng)
    else:
        raise ValueError(f"Unknown training mode: {mode}")
    write_json(run.path/"metrics.json", reports)
    return reports
