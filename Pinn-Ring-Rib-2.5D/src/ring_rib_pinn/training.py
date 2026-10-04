"""Explicit independent -> ring-to-rib -> alternating heat-flux training."""

from contextlib import redirect_stdout
import shutil
import time
import traceback

import numpy as np
import torch

from .artifacts import code_hash, file_hash, new_run, finish_run, write_json
from .backend import initialize
from .checkpoints import save_checkpoint, save_thickness, load_thickness
from .coupling import CouplingState
from .data import read_table, validate_frame
from .geometry import SpiralGeometry
from .materials import Material
from .networks import ThicknessNet, temperature_net
from .physics import Physics


def train_geometry(config, frames, device, dtype, records):
    t, g = config.t, config.g
    if "thickness_checkpoint" in config.values["data"]:
        return load_thickness(config.data_path("thickness_checkpoint"), config, device, dtype)
    result = {}
    for domain in ("upper", "lower"):
        raw_thickness = f"{domain}_thickness_csv"
        if raw_thickness in config.values["data"]:
            frame = validate_frame(read_table(config.data_path(raw_thickness)), config, domain,
                                   require_temperature=False)
        else:
            frame = frames[domain].drop_duplicates(["r", "theta"])
        if "thickness" not in frame:
            raise ValueError("Thickness pretraining requires a thickness column")
        net = ThicknessNet().to(device=device, dtype=dtype)
        key = f"{domain}_thickness"
        if key in config.values["data"]:
            net.load_state_dict(torch.load(config.data_path(key), map_location=device,
                                            weights_only=True), strict=True)
        else:
            # SI-scale initialization, not a new positivity transform or PDE change.
            with torch.no_grad():
                net.net[-1].weight.mul_(1e-3)
                net.net[-1].bias.fill_(float(frame.thickness.mean()))
        steps = t["thickness_steps"] if t["train_thickness"] or key not in config.values["data"] else 0
        coords = torch.as_tensor(np.column_stack([frame.r/g["radius"], frame.theta]),
                                 device=device, dtype=dtype)
        target = torch.as_tensor(frame.thickness.to_numpy(copy=True),
                                 device=device, dtype=dtype).view(-1, 1)
        physical_radius = coords[:, 0] * g["radius"]
        if domain == "upper":
            focus = (physical_radius < 0.098) | (physical_radius > 0.1235)
        else:
            focus = (physical_radius >= 0.124) | (physical_radius < 0.1175)
        weight = torch.where(focus.view(-1, 1), 10., 1.)
        optimizer = torch.optim.Adam(net.parameters(), lr=t["thickness_lr"])
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.8)
        for step in range(steps):
            if t["thickness_batch"]:
                ids = torch.randperm(len(coords), device=device)[:t["thickness_batch"]]
            else:
                ids = slice(None)
            x = coords[ids].detach().requires_grad_(True)
            optimizer.zero_grad()
            pred = net(x[:, :1], x[:, 1:2])
            mse = torch.mean(weight[ids] * (pred-target[ids])**2)
            gradient = torch.autograd.grad(pred.sum(), x, create_graph=True)[0][:, :1]
            regularization = 1e-6*gradient.square().mean()
            loss = mse+regularization
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite {domain} thickness loss")
            loss.backward()
            optimizer.step()
            scheduler.step()
            if step == 0 or (step+1) % t["display_every"] == 0 or step+1 == steps:
                records.append({"stage": "thickness", "domain": domain, "step": step+1,
                                "loss": float(loss.detach()), "mse": float(mse.detach())})
        with torch.no_grad():
            pred = net(coords[:, :1], coords[:, 1:2])
        if not torch.isfinite(pred).all() or (pred <= 0).any():
            raise FloatingPointError(f"{domain} thickness prediction is nonpositive/non-finite")
        result[domain] = net.freeze()
    return result


def make_model(dde, config, domain, net, physics, stage):
    g, t = config.g, config.t
    rib = domain == "rib"
    if rib:
        spatial = dde.geometry.Rectangle([0, 0], [g["rib_length"]/g["radius"], g["rib_count"]-1])
        def equation(x, y):
            return physics.rib(x, y, single=stage == "independent")
        length = g["rib_length"]/g["radius"]
        bounds = [(lambda x, on: on and np.isclose(x[0], 0), False),
                  (lambda x, on: on and np.isclose(x[0], length), True)]
    else:
        inner = g[f"{domain}_inner"]/g["radius"]
        spatial = dde.geometry.Rectangle([inner, 0], [1, 2*np.pi])
        def equation(x, y):
            return physics.ring(x, y, domain, coupled=stage == "coupled")
        bounds = [(lambda x, on: on and np.isclose(x[0], inner), False),
                  (lambda x, on: on and np.isclose(x[0], 1), True)]
    geomtime = dde.geometry.GeometryXTime(spatial, dde.geometry.TimeDomain(0, config.time_end_star))
    bcs = []
    for predicate, outer in bounds:
        if rib:
            def boundary(x, y, _, outer=outer):
                return physics.rib_boundary(x, y, outer, single=stage == "independent")
        else:
            def boundary(x, y, _, outer=outer):
                return physics.ring_boundary(x, y, domain, outer)
        bcs.append(dde.icbc.OperatorBC(geomtime, boundary, predicate))
    if config.case == "heating":
        bcs.append(dde.icbc.IC(geomtime, lambda x: 0., lambda _, on: on))
    prefix = "rib" if rib else "ring"
    data = dde.data.TimePDE(geomtime, equation, bcs,
                            num_domain=t[f"{prefix}_domain"],
                            num_boundary=t[f"{prefix}_boundary"],
                            num_initial=t[f"{prefix}_initial"] if config.case == "heating" else 0)
    return dde.Model(data, net)


def train_stage(dde, model, config, domain, stage, steps, records, lbfgs_steps=0):
    t = config.t
    weights = [100., 1., 1.] + ([0.1] if config.case == "heating" else [])
    independent = stage == "independent"
    lr = t["independent_lr"] if independent else t["coupling_lr"]
    decay = t["independent_decay"] if independent else t["coupling_decay"]

    class FiniteCallback(dde.callbacks.Callback):
        def on_batch_end(self):
            loss = self.model.train_state.loss_train
            if loss is not None and not np.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite {domain} {stage} loss")
            if any(not torch.isfinite(p).all() for p in self.model.net.parameters()):
                raise FloatingPointError(f"Non-finite {domain} {stage} parameters")

        def on_train_end(self):
            loss = self.model.train_state.loss_train
            if loss is not None:
                if not np.isfinite(loss).all():
                    raise FloatingPointError(f"Non-finite {domain} {stage} loss")
                records.append({"stage": stage, "domain": domain,
                                "iteration": self.model.train_state.iteration,
                                "optimizer": self.model.opt_name, "loss": loss.tolist()})

    model.compile("adam", lr=lr, decay=("step", decay, 0.5), loss_weights=weights)
    if steps:
        model.train(iterations=steps, display_every=t["display_every"], callbacks=[FiniteCallback()])
    if t["lbfgs"] and lbfgs_steps:
        dde.optimizers.config.set_LBFGS_options(maxiter=lbfgs_steps)
        model.compile("L-BFGS", loss_weights=weights)
        model.train(callbacks=[FiniteCallback()])


def snapshot_inputs(config, directory, frames, geometry):
    inputs = directory / "inputs"
    inputs.mkdir()
    saved = config.serializable()
    hashes = {}
    for key in ("upper_thickness", "lower_thickness", "upper_thickness_csv", "lower_thickness_csv",
                "thickness_checkpoint", "rib_json"):
        if key in config.values["data"]:
            hashes[key] = file_hash(config.data_path(key))
    evaluation = {}
    for domain, frame in frames.items():
        path = config.data_path(f"{domain}_csv")
        hashes[domain] = file_hash(path)
        limit = config.t["evaluation_limit"]
        # Only smoke evaluation is subsampled; full configurations retain all original points.
        points = frame.sample(n=limit, random_state=config.t["seed"]) if limit and len(frame)>limit else frame
        points.to_csv(inputs / f"{domain}.csv", index=False)
        saved["data"][f"{domain}_csv"] = f"inputs/{domain}.csv"
        evaluation[domain] = points
    for domain in ("upper", "lower"):
        key = f"{domain}_surface"
        if key in config.values["data"]:
            frame = validate_frame(read_table(config.data_path(key)), config, domain)
            limit = config.t["evaluation_limit"]
            if limit and len(frame)>limit:
                frame = frame.sample(n=limit, random_state=config.t["seed"])
            frame.to_csv(inputs / f"{domain}_surface.csv", index=False)
            saved["data"][key] = f"inputs/{domain}_surface.csv"
            hashes[key] = file_hash(config.data_path(key))
    shutil.copy2(config.data_path("rib_json"), directory / "rib_models.json")
    saved["data"]["rib_json"] = "rib_models.json"
    saved["output"]["root"] = ".."
    # All inference dependencies are in the checkpoint/inputs, not historical external paths.
    for key in list(saved["data"]):
        if key.endswith("_raw") or "thickness" in key or key == "preprocess":
            saved["data"].pop(key)
    write_json(directory / "config.json", saved)
    write_json(directory / "manifest.json", {
        "code_revision": code_hash(), "source_input_sha256": hashes,
        "evaluation_points": {key: len(frame) for key, frame in evaluation.items()},
        "evaluation_sampling": "seeded smoke subset" if config.t["evaluation_limit"] else "all original points",
        "case": config.case, "formula_profile": "legacy_heating_v6.5" if config.case == "heating"
                                           else "legacy_cooling_hardconstraint",
        "seed": config.t["seed"], "phases": geometry.phases.flatten().cpu().tolist()})
    return saved


def train(config, run_id=None, thickness_only=False):
    directory = new_run(config, run_id, "thickness" if thickness_only else "training")
    records = []
    started = time.perf_counter()
    try:
        with (directory / "training.log").open("w", encoding="utf-8") as stream, redirect_stdout(stream):
            dde, device, dtype = initialize(config)
            frames = {key: validate_frame(read_table(config.data_path(f"{key}_csv")), config, key)
                      for key in (("upper", "lower") if thickness_only else ("upper", "lower", "rib"))}
            if thickness_only:
                config.t["train_thickness"] = True
            geometry = SpiralGeometry.from_json(config, config.data_path("rib_json"), device, dtype)
            saved_config = snapshot_inputs(config, directory, frames, geometry)
            thickness = train_geometry(config, frames, device, dtype, records)
            save_thickness(directory / "thickness.pt", config, thickness)
            if not thickness_only:
                networks = {key: temperature_net(dde, key, config, geometry).to(device=device, dtype=dtype)
                            for key in ("upper", "lower", "rib")}
                physics = Physics(dde, config, geometry, Material(config, device, dtype), thickness,
                                  networks, CouplingState(geometry))
                t = config.t
                for domain in ("upper", "lower", "rib"):
                    model = make_model(dde, config, domain, networks[domain], physics, "independent")
                    steps = t["rib_steps"] if domain == "rib" else t["independent_steps"]
                    train_stage(dde, model, config, domain, "independent", steps, records,
                                t["independent_lbfgs"])
                from .config import Config
                run_config = Config(saved_config, directory)
                save_checkpoint(directory / "independent.pt", run_config, geometry, networks, thickness)
                coupled = {key: make_model(dde, config, key, networks[key], physics, "coupled")
                           for key in networks}
                for outer in range(t["outer_iterations"]):
                    print(f"Coupling round {outer+1}/{t['outer_iterations']}")
                    train_stage(dde, coupled["rib"], config, "rib", "coupled", t["coupling_steps"],
                                records, t["coupling_lbfgs"])
                    physics.coupling.update(physics, t["snapshot_s"], t["snapshot_t"])
                    for domain in ("upper", "lower"):
                        train_stage(dde, coupled[domain], config, domain, "coupled",
                                    t["coupling_steps"], records, t["coupling_lbfgs"])
                    save_checkpoint(directory / f"coupling_{outer+1}.pt", run_config, geometry,
                                    networks, thickness)
                if t["lbfgs"] and t["final_lbfgs"]:
                    for domain, model in coupled.items():
                        weights = [100., 1., 1.] + ([0.1] if config.case == "heating" else [])
                        dde.optimizers.config.set_LBFGS_options(maxiter=t["final_lbfgs"])
                        model.compile("L-BFGS", loss_weights=weights)
                        model.train()
                save_checkpoint(directory / "checkpoint.pt", run_config, geometry, networks, thickness)
            write_json(directory / "losses.json", records)
            write_json(directory / "timing.json", {"training_seconds": time.perf_counter()-started})
        finish_run(config, directory, notes="Short workflow validation; precision not validated"
                   if config.t["evaluation_limit"] else "Training completed; hypotheses require review")
        if not thickness_only:
            from .checkpoints import evaluate_run
            from .plotting import plot_run
            evaluate_run(directory, str(device))
            plot_run(directory)
            finish_run(config, directory, notes="Workflow completed with evaluation and plotting; not precision validation")
    except Exception as error:
        write_json(directory / "losses.json", records)
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        finish_run(config, directory, error)
        raise
    return directory
