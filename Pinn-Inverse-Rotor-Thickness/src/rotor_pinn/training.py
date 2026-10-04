"""Explicit state, stable DeepXDE data objects and alternating contact iterations."""

from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash, new_run, write_json
from .backend import initialize
from .config import from_dict
from .coupling import CouplingState
from .data import normalized_inputs, read_csv, spatial_split, validate_frame
from .geometry import RotorGeometry
from .materials import Material
from .networks import ThicknessNet, temperature_net
from .physics import derivative, rib_energy_residual, ring_energy_residual
from .synthetic import demo_frames


def rib_domain(dde, geometry):
    """A union of intervals indexed by discrete ids; GeometryXTime appends time last."""
    length = geometry.length / geometry.config.outer_radius
    count = geometry.config.rib_count
    class DiscreteRibs(dde.geometry.Geometry):
        def __init__(self):
            super().__init__(2, (np.array([0, 0]), np.array([length, count - 1])), length)

        def inside(self, points):
            return ((points[:, 0] >= 0) & (points[:, 0] <= length)
                    & (points[:, 1] >= 0) & (points[:, 1] < count)
                    & np.isclose(points[:, 1], np.round(points[:, 1])))

        def on_boundary(self, points):
            return self.inside(points) & (np.isclose(points[:, 0], 0)
                                          | np.isclose(points[:, 0], length))

        def boundary_normal(self, points):
            normal = np.zeros_like(points)
            normal[np.isclose(points[:, 0], 0), 0] = -1
            normal[np.isclose(points[:, 0], length), 0] = 1
            return normal

        def random_points(self, n, random="pseudo"):
            return np.column_stack([np.random.random(n) * length,
                                    np.random.randint(count, size=n)]).astype(dde.config.real(np))

        def uniform_points(self, n, boundary=True):
            return np.column_stack([np.linspace(0, length, n), np.arange(n) % count]).astype(dde.config.real(np))

        def random_boundary_points(self, n, random="pseudo"):
            return np.column_stack([np.random.randint(2, size=n) * length,
                                    np.random.randint(count, size=n)]).astype(dde.config.real(np))

        def uniform_boundary_points(self, n):
            return np.column_stack([(np.arange(n) % 2) * length,
                                    np.arange(n) % count]).astype(dde.config.real(np))
    return DiscreteRibs()


def train_thickness(frame, config, device, dtype, region, log):
    subset = frame[frame["split"] == "train"] if "split" in frame else frame
    subset = subset.drop_duplicates(["r", "theta"])
    if subset.empty:
        raise ValueError(f"No training thickness samples for {region}")
    scale = float(subset.thickness.mean())
    model = ThicknessNet(config.networks.thickness_widths, scale,
                         config.networks.minimum_thickness).to(device=device, dtype=dtype)
    coords = torch.as_tensor(np.column_stack([subset.r / config.geometry.outer_radius,
                                             subset.theta]), device=device, dtype=dtype)
    target = torch.as_tensor(subset.thickness.to_numpy(copy=True), device=device, dtype=dtype).view(-1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.training.learning_rate)
    for step in range(config.training.thickness_steps):
        ids = torch.randperm(len(coords), device=device)[:config.training.observation_batch]
        x = coords[ids].detach().requires_grad_(True)
        optimizer.zero_grad()
        prediction = model(x)
        mse = ((prediction - target[ids]) / scale).square().mean()
        gradient = derivative(prediction / scale, x)[:, 0:1]
        penalty = gradient.square().mean()
        loss = mse + config.training.thickness_gradient_weight * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite thickness loss in {region}")
        loss.backward()
        optimizer.step()
        if step % config.training.display_every == 0 or step + 1 == config.training.thickness_steps:
            log.append({"region": region, "stage": "thickness", "step": step + 1,
                        "loss": float(loss.detach()), "mse": float(mse.detach()),
                        "gradient_penalty": float(penalty.detach())})
    return model.freeze()


def make_temperature_model(dde, region, config, geometry, material, thickness,
                           coupling, frame, time_scale, device, dtype):
    g, p, t = config.geometry, config.physics, config.training
    radius = g.outer_radius
    net = temperature_net("rib" if region == "rib" else "ring", config).to(device=device, dtype=dtype)
    if region == "rib":
        spatial = rib_domain(dde, geometry)
        area = g.rib_width * g.rib_height
        scale = (p.density * p.heat_capacity_nodes[0] * area * p.temperature_scale / time_scale
                 + p.conductivity_nodes[0] * area * p.temperature_scale / radius**2
                 + area * p.rib_volume_source + p.rib_exposed_sides * g.rib_height * p.convection * p.temperature_scale)
        def equation(x, y):
            temperature = p.ambient + p.temperature_scale * y
            gain = coupling.rib_gain(x, time_scale)
            return rib_energy_residual(x, temperature, material, radius, time_scale, area,
                                       p.rib_volume_source,
                                       p.rib_exposed_sides * g.rib_height * material.surface_loss(temperature), gain) / scale
        def boundary(x, y, _):
            temperature = p.ambient + p.temperature_scale * y
            conductivity, _ = material.factors(temperature)
            return conductivity * derivative(temperature, x)[:, 0:1] / radius / (p.conductivity_nodes[0] * p.temperature_scale / radius)
        boundary_functions = [(boundary, lambda x, on: on)]
    else:
        inner = getattr(g, f"{region}_inner_radius")
        spatial = dde.geometry.Rectangle([inner / radius, 0], [1, 2 * np.pi])
        mean_h = float(thickness.scale)
        scale = (p.density * p.heat_capacity_nodes[0] * mean_h * p.temperature_scale / time_scale
                 + p.conductivity_nodes[0] * mean_h * p.temperature_scale / radius**2
                 + mean_h * p.source_strength + p.convection * p.temperature_scale)
        def equation(x, y):
            temperature = p.ambient + p.temperature_scale * y
            h = thickness(x)
            physical_r = x[:, 0:1] * radius
            source = p.source_strength * torch.exp(-(physical_r - inner) / p.source_decay)
            source = source * (physical_r <= inner + p.source_cutoff_decays * p.source_decay)
            if region == "lower":
                source = source * p.lower_source_factor
            return ring_energy_residual(x, temperature, h, material, radius, time_scale,
                                        source, p.exposed_ring_faces * material.surface_loss(temperature),
                                        coupling.ring_loss(x, region, time_scale)) / scale
        boundary_scale = p.conductivity_nodes[0] * p.temperature_scale / radius + p.convection * p.temperature_scale
        def inner_boundary(x, y, _):
            temperature = p.ambient + p.temperature_scale * y
            conductivity, _ = material.factors(temperature)
            return (conductivity * derivative(temperature, x)[:, 0:1] / radius
                    - material.surface_loss(temperature)) / boundary_scale
        def outer_boundary(x, y, _):
            temperature = p.ambient + p.temperature_scale * y
            conductivity, _ = material.factors(temperature)
            return (-conductivity * derivative(temperature, x)[:, 0:1] / radius
                    - material.surface_loss(temperature)) / boundary_scale
        boundary_functions = [(inner_boundary, lambda x, on: on and np.isclose(x[0], inner / radius)),
                              (outer_boundary, lambda x, on: on and np.isclose(x[0], 1))]
    domain = dde.geometry.GeometryXTime(spatial, dde.geometry.TimeDomain(0, 1))
    conditions = [dde.icbc.OperatorBC(domain, function, predicate)
                  for function, predicate in boundary_functions]
    weights = [1.0] * (len(conditions) + 1)
    observation = None
    if region != "rib" or t.observe_ribs:
        subset = frame[frame["split"] == "train"]
        inputs = normalized_inputs(subset, region, geometry, time_scale).astype(dde.config.real(np))
        targets = ((subset.True_T.to_numpy() - p.ambient) / p.temperature_scale).reshape(-1, 1)
        observation = dde.icbc.PointSetBC(inputs, targets.astype(dde.config.real(np)),
                                         batch_size=min(t.observation_batch, len(inputs)), shuffle=True)
        conditions.append(observation)
        weights.append(t.observation_weight)
    data = dde.data.TimePDE(domain, equation, conditions, num_domain=t.domain_points,
                            num_boundary=t.boundary_points, train_distribution="pseudo")
    model = dde.Model(data, net)
    model.compile("adam", lr=t.learning_rate, loss_weights=weights)
    return model, observation, weights


def _steps(dde, model, steps, config, batched, stage, region, log):
    if not steps:
        return
    class ObservationResampler(dde.callbacks.PDEPointResampler):
        def on_epoch_end(self):
            super().on_epoch_end()
            # DeepXDE's displayed test loss uses training points. Keep its point-set
            # targets aligned with the new batch; held-out metrics are computed separately.
            data = self.model.data
            data.test_x, data.test_y, data.test_aux_vars = None, None, None
            self.model.train_state.set_data_test(*data.test())
    callbacks = [ObservationResampler(period=1, pde_points=False, bc_points=True)] if batched else []
    history, state = model.train(iterations=steps, display_every=config.training.display_every,
                                  callbacks=callbacks)
    if not np.isfinite(state.loss_train).all():
        raise FloatingPointError(f"Non-finite {region} losses during {stage}")
    previous = max((item["step"] for item in log if item["region"] == region
                    and item["stage"] != "thickness"), default=-1)
    for step, losses in zip(history.steps, history.loss_train):
        if step > previous:
            log.append({"region": region, "stage": stage, "step": int(step),
                        "loss": [float(v) for v in losses]})
    return history


def _save_checkpoint(path, config, thicknesses, temperatures=None, time_scale=None):
    torch.save({"format_version": 1, "config": config.snapshot(),
                "thickness": {region: model.state_dict() for region, model in thicknesses.items()},
                "temperature": {region: model.state_dict() for region, model in (temperatures or {}).items()},
                "time_scale": time_scale}, path)


def load_thickness(path, config, device, dtype):
    if not Path(path).is_file():
        raise FileNotFoundError("Configured thickness checkpoint does not exist")
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported thickness checkpoint format")
    if payload["config"]["networks"] != config.snapshot()["networks"]:
        raise ValueError("Thickness checkpoint network configuration differs")
    if payload["config"]["geometry"] != config.snapshot()["geometry"]:
        raise ValueError("Thickness checkpoint geometry differs")
    result = {}
    for region, weights in payload["thickness"].items():
        model = ThicknessNet(config.networks.thickness_widths, float(weights["scale"]),
                             config.networks.minimum_thickness).to(device=device, dtype=dtype)
        model.load_state_dict(weights)
        result[region] = model.freeze()
    if set(result) != {"upper", "lower"}:
        raise ValueError("Thickness checkpoint must contain both rings")
    return result


def train_pipeline(config, run_id=None, demo=False, thickness_only=False):
    config.validate()
    with new_run(config, run_id, "thickness pretraining" if thickness_only else "coupled pipeline demo" if demo else "thermal training",
                 "synthetic pipeline functions" if demo else "user supplied SI CSV") as (directory, manifest):
        dde, device, dtype = initialize(config.training)
        manifest["device"] = str(device)
        if demo:
            frames = demo_frames(config)
        elif thickness_only:
            frames = {}
            for region in ("upper", "lower"):
                independent = getattr(config.data, f"{region}_thickness_csv")
                filename = independent or getattr(config.data, f"{region}_csv")
                frames[region] = read_csv(config.path(filename), config, region,
                                         require_temperature=False, require_thickness=True)
        else:
            frames = {region: read_csv(config.path(getattr(config.data, f"{region}_csv")), config, region)
                      for region in ("upper", "lower", "rib")}
        inputs_dir = directory / "inputs"
        inputs_dir.mkdir()
        for region, frame in frames.items():
            frame = validate_frame(frame, config, region, require_temperature=not thickness_only,
                                   require_thickness=thickness_only)
            frames[region] = spatial_split(frame, config.training)
            destination = inputs_dir / f"{region}.csv"
            frames[region].to_csv(destination, index=False)
            manifest["input_sha256"][region] = file_hash(destination)
            if not demo:
                independent = getattr(config.data, f"{region}_thickness_csv", "") if thickness_only else ""
                manifest.setdefault("original_input_sha256", {})[region] = file_hash(config.path(independent or getattr(config.data, f"{region}_csv")))
        time_scale = None if thickness_only else max(float(frame.t.max()) for frame in frames.values())
        if not thickness_only and time_scale <= 0:
            raise ValueError("Time domain must extend beyond t=0")
        geometry = RotorGeometry(config.geometry)
        log = []
        if config.data.thickness_checkpoint:
            checkpoint = config.path(config.data.thickness_checkpoint)
            thicknesses = load_thickness(checkpoint, config, device, dtype)
            manifest["input_sha256"]["thickness_checkpoint"] = file_hash(checkpoint)
        else:
            thicknesses = {}
            for region in ("upper", "lower"):
                independent = getattr(config.data, f"{region}_thickness_csv")
                if independent and not demo and not thickness_only:
                    frame = read_csv(config.path(independent), config, region,
                                     require_temperature=False, require_thickness=True)
                    frame = spatial_split(frame, config.training)
                    frame.to_csv(inputs_dir / f"{region}_thickness.csv", index=False)
                    manifest["input_sha256"][f"{region}_thickness"] = file_hash(inputs_dir / f"{region}_thickness.csv")
                else:
                    frame = validate_frame(frames[region], config, region,
                                           require_temperature=not thickness_only, require_thickness=True)
                thicknesses[region] = train_thickness(frame, config, device, dtype, region, log)
        _save_checkpoint(directory / "thickness.pt", config, thicknesses)
        manifest["thickness_checkpoint_sha256"] = file_hash(directory / "thickness.pt")
        if thickness_only:
            write_json(directory / "losses.json", log)
            return directory
        coupling = CouplingState.empty(geometry, config, time_scale)
        material = Material(config.physics)
        entries = {region: make_temperature_model(dde, region, config, geometry, material,
                                                   thicknesses.get(region), coupling, frame,
                                                   time_scale, device, dtype)
                   for region, frame in frames.items()}
        temperatures = {region: entry[0].net for region, entry in entries.items()}
        for region, (model, observation, _) in entries.items():
            _steps(dde, model, config.training.warmup_steps, config, observation is not None,
                   "independent warmup", region, log)
        diagnostics = []
        for iteration in range(config.training.outer_iterations):
            coupling.update(temperatures, thicknesses, material, config.contact.relaxation)
            diagnostics.append({"iteration": iteration + 1, "power_balance": coupling.diagnostics()})
            for region, (model, observation, _) in entries.items():
                _steps(dde, model, config.training.inner_steps, config, observation is not None,
                       f"contact round {iteration + 1}", region, log)
        if config.training.lbfgs_steps:
            # L-BFGS must see deterministic full observations throughout a line search.
            for region, (model, observation, weights) in entries.items():
                if observation is not None:
                    model.data.bcs[-1] = dde.icbc.PointSetBC(observation.points,
                                                            observation.values.cpu().numpy())
                    model.data.train_x_bc = None
                    model.data.train_x, model.data.train_y, model.data.train_aux_vars = None, None, None
                    model.data.train_next_batch()
                    model.data.test_x, model.data.test_y, model.data.test_aux_vars = None, None, None
                    model.data.test()
                dde.optimizers.config.set_LBFGS_options(maxiter=config.training.lbfgs_steps)
                options = dde.optimizers.config.LBFGS_options
                options["iter_per_step"] = min(1000, config.training.lbfgs_steps)
                options["fun_per_step"] = max(1, int(options["maxfun"] / options["maxiter"] * options["iter_per_step"]))
                model.compile("L-BFGS", loss_weights=weights)
                _steps(dde, model, 1, config, False, "L-BFGS", region, log)
        _save_checkpoint(directory / "models.pt", config, thicknesses, temperatures, time_scale)
        manifest["checkpoint_sha256"] = file_hash(directory / "models.pt")
        coupling.save(directory / "coupling.npz")
        write_json(directory / "coupling_diagnostics.json", diagnostics)
        write_json(directory / "losses.json", log)
        from .evaluation import evaluate_run, predict_frame
        evaluate_run(directory, split="all", device=str(device))
        # Compare the actual trained objects against checkpoint-backed predictions.
        reload_error = {}
        _, restored_models, _, _, _, _ = reload_models(directory, str(device))
        for region, model in temperatures.items():
            original = predict_frame(model, frames[region], region, geometry, time_scale,
                                     config.physics, device, dtype)
            restored = predict_frame(restored_models[region], frames[region], region, geometry,
                                     time_scale, config.physics, device, dtype)
            reload_error[region] = float(np.max(np.abs(original - restored)))
        write_json(directory / "reload_diagnostics.json", reload_error)
        if max(reload_error.values()) > (1e-4 if dtype == torch.float32 else 1e-9):
            raise RuntimeError("Reloaded predictions differ from trained network predictions")
        manifest["has_metrics"] = True
        from .plotting import plot_run
        plot_run(directory)
        return directory


def reload_models(run_directory, device="cpu"):
    directory = Path(run_directory)
    checkpoint = directory / "models.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError("models.pt is missing; run temperature training before evaluation")
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        import json
        expected = json.loads(manifest_path.read_text(encoding="utf-8")).get("checkpoint_sha256")
        if expected and file_hash(checkpoint) != expected:
            raise ValueError("Saved model checkpoint hash differs from run manifest")
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if payload.get("format_version") != 1:
        raise ValueError("Unsupported model checkpoint format")
    config = from_dict(payload["config"], directory)
    snapshot = directory / "config.json"
    if snapshot.exists():
        import json
        if json.loads(snapshot.read_text(encoding="utf-8")) != payload["config"]:
            raise ValueError("Run configuration differs from saved checkpoint configuration")
    config.training.device = device
    _, selected, dtype = initialize(config.training)
    thicknesses = load_thickness(checkpoint, config, selected, dtype)
    temperatures = {}
    for region, weights in payload["temperature"].items():
        model = temperature_net("rib" if region == "rib" else "ring", config).to(device=selected, dtype=dtype)
        model.load_state_dict(weights)
        temperatures[region] = model.eval()
    if set(temperatures) != {"upper", "lower", "rib"}:
        raise ValueError("Checkpoint must contain all three temperature models")
    return config, temperatures, thicknesses, float(payload["time_scale"]), selected, dtype
