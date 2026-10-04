"""Audited original definitions as an independent compatibility oracle."""

import ast
import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash
from .backend import initialize
from .config import Config
from .geometry import SpiralGeometry
from .materials import Material
from .networks import ThicknessNet, temperature_net


def definitions(path, namespace, names=None):
    """Execute only top-level function/class definitions, never original training code."""
    parsed = ast.parse(Path(path).read_text(encoding="utf-8-sig"))
    selected = [node for node in parsed.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and (names is None or node.name in names)]
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


class LegacyBundle:
    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory).resolve()
        self.profile = json.loads((self.directory / "profile.json").read_text(encoding="utf-8"))
        if self.profile.get("format_version") != 1:
            raise ValueError("Unsupported legacy profile")
        for relative, digest in self.profile["files_sha256"].items():
            path = self.directory / relative
            if not path.is_file() or file_hash(path) != digest:
                raise ValueError(f"Legacy file missing or hash mismatch: {relative}")
        case = self.profile["case"]
        self.config = Config({"case": case,
                              "physics": {"source": 0 if case == "cooling" else 1e8},
                              "training": {"device": device}}, self.directory)
        self.dde, self.device, self.dtype = initialize(self.config)
        self.geometry = SpiralGeometry.from_json(self.config, self.directory / "rib_models.json",
                                                  self.device, self.dtype)
        self.networks, self.thickness = {}, {}
        for domain, filename in self.profile["weights"].items():
            net = temperature_net(self.dde, domain, self.config, self.geometry).to(self.device)
            net.load_state_dict(torch.load(self.directory / filename, map_location=self.device,
                                            weights_only=True), strict=True)
            self.networks[domain] = net.eval()
        for domain, filename in self.profile["thickness"].items():
            net = ThicknessNet().to(self.device)
            net.load_state_dict(torch.load(self.directory / filename, map_location=self.device,
                                            weights_only=True), strict=True)
            self.thickness[domain] = net.freeze()

    def oracle(self, source=None):
        """Bind retained source definitions to this run's inputs and tensors."""
        from scipy.integrate import quad
        cfg, geo, p, g = self.config, self.geometry, self.config.p, self.config.g
        mat = Material(cfg, self.device, self.dtype)
        ns = {"__name__": "retained_reference", "torch": torch, "nn": torch.nn,
              "np": np, "dde": self.dde, "quad": quad, "device": self.device,
              "NUM_RIBS": g["rib_count"], "R_OUT_PHYS": g["radius"],
              "T_INF_PHYS": p["ambient"], "T_REF_DELTA": p["temperature_scale"],
              "K_REF": p["k_ref"], "CP_REF": p["cp_ref"], "RHO": p["density"],
              "ALPHA_REF": cfg.alpha, "SIGMA": p["sigma"], "Bi_fixed": 10*0.225/15.1,
              "eps_fixed": 0.3, "RAD_COEF": 5.67e-8*0.225/(15.1*900),
              "RIB_WIDTH": g["rib_width"], "RIB_HEIGHT": g["rib_height"],
              "RIB_LENGTH": g["rib_length"], "RIB_WIDTH_STAR": g["rib_width"]/g["radius"],
              "RIB_HEIGHT_STAR": g["rib_height"]/g["radius"],
              "RIB_LENGTH_STAR": g["rib_length"]/g["radius"],
              "R_RIB_START_JSON": g["rib_start"], "R_RIB_END": g["rib_end"],
              "B_SPIRAL_JSON": g["rib_slope"], "PHI_STARTS": geo.phases,
              "s_grid_star": geo.s_grid, "r_grid_star": geo.r_grid,
              "T_STAR_MAX": cfg.time_end_star, "TEMP_NODES": mat.nodes,
              "KAPPA_NODES": mat.k, "CP_RATIO_NODES": mat.c, "DK_DT_PHYS": mat.dk,
              "t_model_upper": self.thickness["upper"], "t_model_lower": self.thickness["lower"],
              "net_upper": self.networks["upper"], "net_lower": self.networks["lower"],
              "net_rib": self.networks["rib"], "q_base_phys": p["source"],
              "Q_base_star": p["source"]*0.225**2/(15.1*900), "delta_phys": 0.005,
              "gap_factor": 0.3, "R_source_upper_max": 0.084+3*0.005,
              "R_source_lower_max": 0.103+3*0.005, "Q_snapshot_upper": None,
              "Q_snapshot_lower": None, "snapshot_s_grid": None, "snapshot_t_grid": None,
              "PDE_RES_SCALE": 20.0}
        return definitions(source or self.directory / "source.py", ns)

    def original_networks(self):
        oracle = self.oracle()
        if self.config.case == "heating":
            builders = {"upper": lambda: oracle["create_ring_net"](),
                        "lower": lambda: oracle["create_ring_net"](),
                        "rib": lambda: oracle["create_rib_net"]()}
        else:
            builders = {"upper": lambda: oracle["create_hard_constraint_ring_net"](
                                               oracle["ic_ring_upper_torch"]),
                        "lower": lambda: oracle["create_hard_constraint_ring_net"](
                                               oracle["ic_ring_lower_torch"]),
                        "rib": lambda: oracle["create_hard_constraint_rib_net"]()}
        result = {}
        for domain, builder in builders.items():
            net = builder().to(self.device)
            net.load_state_dict(torch.load(self.directory / self.profile["weights"][domain],
                                            map_location=self.device, weights_only=True), strict=True)
            result[domain] = net.eval()
        return result


def legacy_evaluate(config, bundle_path, run_id=None):
    from .artifacts import new_run, write_json, finish_run
    from .checkpoints import save_checkpoint
    from .data import read_table, validate_frame
    from .evaluation import evaluate_frames
    from .plotting import plot_run
    from .training import snapshot_inputs
    bundle = LegacyBundle(bundle_path, config.t["device"])
    for key in ("case", "geometry", "physics"):
        if config.values[key] != bundle.config.values[key]:
            raise ValueError(f"Legacy bundle and evaluation configuration have different {key}")
    directory = new_run(config, run_id, "legacy_replay")
    try:
        frames = {key: validate_frame(read_table(config.data_path(f"{key}_csv")), config, key)
                  for key in ("upper", "lower", "rib")}
        # Full historical points and a complete portable group, like a trained run.
        replay_values = config.serializable()
        replay_values["training"]["evaluation_limit"] = 0
        replay_config = Config(replay_values, config.directory)
        portable = snapshot_inputs(replay_config, directory, frames, bundle.geometry)
        save_checkpoint(directory/"checkpoint.pt", Config(portable, directory), bundle.geometry,
                        bundle.networks, bundle.thickness)
        metrics = {"domain_mean": evaluate_frames(bundle.networks, frames, bundle.config,
                                                  bundle.geometry, directory)}
        surfaces = {key: validate_frame(read_table(config.data_path(f"{key}_surface")), config, key)
                    for key in ("upper", "lower") if f"{key}_surface" in config.values["data"]}
        if surfaces:
            metrics["surface_against_midplane"] = evaluate_frames(
                bundle.networks, surfaces, bundle.config, bundle.geometry, directory, "surface")
        write_json(directory / "metrics.json", metrics)
        manifest = json.loads((directory/"manifest.json").read_text(encoding="utf-8"))
        manifest.update(kind="legacy_replay", profile=bundle.profile)
        write_json(directory / "manifest.json", manifest)
        plot_run(directory)
        finish_run(config, directory, notes="Historical model replay; no new training")
    except Exception as error:
        finish_run(config, directory, error)
        raise
    return directory, metrics
