"""Versioned complete model groups; never silently combine unrelated weights."""

from pathlib import Path
import json

import torch

from .backend import initialize
from .config import Config
from .geometry import SpiralGeometry
from .networks import ThicknessNet, temperature_net

FORMAT = "ring-rib-pinn/1"


def save_checkpoint(path, config, geometry, networks, thickness):
    from .artifacts import code_hash
    torch.save({"format": FORMAT, "config": config.serializable(),
                "provenance": {"code_revision": code_hash(),
                               "formula_profile": "legacy_heating_v6.5" if config.case == "heating"
                                                   else "legacy_cooling_hardconstraint"},
                "phases": geometry.phases.detach().cpu(),
                "networks": {key: net.state_dict() for key, net in networks.items()},
                "thickness": {key: net.state_dict() for key, net in thickness.items()}}, path)


def load_checkpoint(path, device="cpu", expected_config=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing model group: {path}")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != FORMAT:
        raise ValueError("Unsupported checkpoint format; use LegacyBundle for original .pth files")
    if set(saved.get("networks", {})) != {"upper", "lower", "rib"} or set(saved.get("thickness", {})) != {"upper", "lower"}:
        raise ValueError("Checkpoint must contain the complete three-domain and thickness model group")
    if expected_config is not None:
        for key in ("case", "geometry", "physics"):
            if saved["config"][key] != expected_config.values[key]:
                raise ValueError(f"Checkpoint {key} differs from expected configuration")
    saved["config"]["training"]["device"] = device
    config = Config(saved["config"], path.parent)
    dde, target, dtype = initialize(config)
    geometry = SpiralGeometry(config, saved["phases"].flatten().tolist(), target, dtype)
    networks = {}
    for domain, state in saved["networks"].items():
        net = temperature_net(dde, domain, config, geometry).to(device=target, dtype=dtype)
        net.load_state_dict(state, strict=True)
        networks[domain] = net.eval()
    thickness = {}
    for domain, state in saved["thickness"].items():
        net = ThicknessNet().to(device=target, dtype=dtype)
        net.load_state_dict(state, strict=True)
        thickness[domain] = net.freeze()
    return config, geometry, networks, thickness


def save_thickness(path, config, thickness):
    torch.save({"format": "ring-rib-thickness/1", "geometry": config.g,
                "networks": {key: net.state_dict() for key, net in thickness.items()}}, path)


def load_thickness(path, config, device, dtype):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("format") != "ring-rib-thickness/1" or saved.get("geometry") != config.g:
        raise ValueError("Thickness checkpoint format or geometry mismatch")
    result = {}
    for domain in ("upper", "lower"):
        net = ThicknessNet().to(device=device, dtype=dtype)
        net.load_state_dict(saved["networks"][domain], strict=True)
        result[domain] = net.freeze()
    return result


def evaluate_run(directory, device="cpu"):
    from .artifacts import write_json
    from .data import read_table, validate_frame
    from .evaluation import evaluate_frames
    directory = Path(directory).resolve()
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    if status["status"] != "completed":
        raise ValueError("Evaluation requires a completed run")
    config, geometry, networks, _ = load_checkpoint(directory / "checkpoint.pt", device)
    if set(networks) != {"upper", "lower", "rib"}:
        raise ValueError("Evaluation requires a complete three-domain model group")
    frames = {key: validate_frame(read_table(directory / "inputs" / f"{key}.csv"), config, key)
              for key in networks}
    metrics = {"domain_mean": evaluate_frames(networks, frames, config, geometry, directory)}
    surfaces = {key: read_table(directory / "inputs" / f"{key}_surface.csv")
                for key in ("upper", "lower")
                if (directory / "inputs" / f"{key}_surface.csv").exists()}
    if surfaces:
        metrics["surface_against_midplane"] = evaluate_frames(networks, surfaces, config, geometry,
                                                             directory, "surface")
    write_json(directory / "metrics.json", metrics)
    return metrics
