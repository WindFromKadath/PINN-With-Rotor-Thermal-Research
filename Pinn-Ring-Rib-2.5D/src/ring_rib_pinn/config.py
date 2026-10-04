"""Portable TOML configurations, with the two legacy formulations kept distinct."""

from copy import deepcopy
from pathlib import Path
import json
import tomllib


DEFAULTS = {
    "case": "heating",
    "geometry": {"radius": 0.225, "upper_inner": 0.084, "lower_inner": 0.103,
                 "rib_start": 0.123, "rib_end": 0.225, "rib_length": 0.1534,
                 "rib_width": 0.016, "rib_height": 0.0094, "rib_slope": 6.4105,
                 "rib_count": 13, "mapping_points": 1000},
    "physics": {"ambient": 293.15, "temperature_scale": 900.0, "k_ref": 15.1,
                "cp_ref": 460.0, "density": 7850.0, "sigma": 5.67e-8,
                "convection": 10.0, "emissivity": 0.3, "source": 1e8,
                "source_decay": 0.005, "gap_factor": 0.3,
                "temperature_nodes": [293.15, 373.15, 573.15, 773.15],
                "conductivity_nodes": [15.1, 16.3, 18.4, 20.5],
                "capacity_nodes": [460.0, 502.0, 586.0, 670.0],
                "time_end": 100.0},
    "training": {"seed": 42, "device": "cpu", "dtype": "float32", "cpu_threads": 4,
                 "thickness_steps": 60000, "thickness_batch": 0,
                 "thickness_lr": 0.001,
                 "independent_steps": 15000, "rib_steps": 15000,
                 "outer_iterations": 5, "coupling_steps": 1000,
                 "independent_lr": 0.0015, "coupling_lr": 0.0001,
                 "independent_decay": 5000, "coupling_decay": 500,
                 "ring_domain": 20000, "ring_boundary": 7500, "ring_initial": 2500,
                 "rib_domain": 10000, "rib_boundary": 4000, "rib_initial": 1000,
                 "snapshot_s": 100, "snapshot_t": 50, "display_every": 500,
                 "lbfgs": True, "independent_lbfgs": 5000,
                 "coupling_lbfgs": 1000, "final_lbfgs": 5000,
                 "evaluation_limit": 0, "train_thickness": False},
    "data": {},
    "output": {"root": "../artifacts"},
}


def _merge(base, override):
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"Unknown configuration key: {key}")
        if isinstance(value, dict) and isinstance(base[key], dict) and key != "data":
            _merge(base[key], value)
        else:
            base[key] = deepcopy(value)


class Config:
    def __init__(self, values=None, directory=None):
        self.values = deepcopy(DEFAULTS)
        _merge(self.values, values or {})
        self.directory = Path(directory or ".").resolve()
        if self.values["case"] not in ("heating", "cooling"):
            raise ValueError("case must be heating or cooling")
        self.g = self.values["geometry"]
        self.p = self.values["physics"]
        self.t = self.values["training"]
        self.validate()

    @property
    def case(self):
        return self.values["case"]

    @property
    def alpha(self):
        return self.p["k_ref"] / (self.p["density"] * self.p["cp_ref"])

    @property
    def time_end_star(self):
        return self.p["time_end"] * self.alpha / self.g["radius"]**2

    def path(self, value):
        path = Path(value)
        return path.resolve() if path.is_absolute() else (self.directory / path).resolve()

    def data_path(self, key):
        if key not in self.values["data"]:
            raise ValueError(f"Missing data configuration: {key}")
        return self.path(self.values["data"][key])

    def validate(self):
        for key in ("radius", "upper_inner", "lower_inner", "rib_start", "rib_end",
                    "rib_length", "rib_width", "rib_height"):
            if self.g[key] <= 0:
                raise ValueError(f"geometry.{key} must be positive")
        if not (self.g["upper_inner"] < self.g["radius"]
                and self.g["lower_inner"] < self.g["radius"]
                and self.g["rib_start"] < self.g["rib_end"] <= self.g["radius"]):
            raise ValueError("Inconsistent radial geometry")
        if self.g["rib_count"] != 13:
            raise ValueError("These retained legacy profiles require 13 ribs")
        if self.g["mapping_points"] < 2:
            raise ValueError("mapping_points must be at least two")
        if self.t["dtype"] not in ("float32", "float64"):
            raise ValueError("dtype must be float32 or float64")
        if self.t["device"] not in ("cpu", "cuda", "auto"):
            raise ValueError("device must be cpu, cuda or auto")
        if self.case == "cooling" and self.p["source"] != 0:
            raise ValueError("Legacy cooling profile requires source=0")
        for key in ("snapshot_s", "snapshot_t"):
            if self.t[key] < 2:
                raise ValueError(f"{key} must be at least two")
        for key in ("thickness_steps", "independent_steps", "rib_steps", "coupling_steps",
                    "outer_iterations", "evaluation_limit", "thickness_batch"):
            if self.t[key] < 0:
                raise ValueError(f"{key} must be nonnegative")
        if self.p["time_end"] <= 0 or self.p["temperature_scale"] <= 0:
            raise ValueError("Time and temperature scales must be positive")
        nodes = self.p["temperature_nodes"]
        if len(nodes) < 2 or any(b <= a for a, b in zip(nodes, nodes[1:])):
            raise ValueError("Temperature nodes must increase")
        for key in ("conductivity_nodes", "capacity_nodes"):
            if len(self.p[key]) != len(nodes) or min(self.p[key]) <= 0:
                raise ValueError(f"Invalid physics.{key}")

    def serializable(self):
        return deepcopy(self.values)


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    return Config(values, path.parent)


def load_saved_config(path):
    path = Path(path).resolve()
    return Config(json.loads(path.read_text(encoding="utf-8")), path.parent)
