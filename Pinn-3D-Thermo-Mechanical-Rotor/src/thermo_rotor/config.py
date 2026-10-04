"""Validated physical profiles and configuration paths."""
from dataclasses import dataclass, asdict, fields
from pathlib import Path
import tomllib
import math


@dataclass(frozen=True)
class Profile:
    name: str = "little_transient"
    L: float = 0.225
    dT: float = 300.0
    tau: float = 200.0
    T0: float = 293.15
    Tamb: float = 293.15
    rho: float = 7870.0
    cp: float = 440.0
    k: float = 76.2
    q: float = 100000.0
    h: float = 20.0
    E: float = 200e9
    nu: float = 0.29
    alpha: float = 1.22e-5
    lam: float = 0.5354
    mu: float = 0.3876
    beta: float = 2.3810
    r_min: float = 0.084
    r_max: float = 0.225
    z_min: float = -0.0344
    z_max: float = 0.0399

    def __post_init__(self):
        for f in fields(self):
            if f.name != "name" and not math.isfinite(getattr(self, f.name)):
                raise ValueError(f"Non-finite profile value: {f.name}")
        for key in ("L", "dT", "tau", "rho", "cp", "k", "E", "alpha", "T0", "Tamb"):
            if getattr(self, key) <= 0:
                raise ValueError(f"Profile {key} must be positive")
        if not (-1 < self.nu < 0.5) or self.h < 0:
            raise ValueError("Invalid Poisson ratio or convection coefficient")
        if not (0 < self.r_min < self.r_max and self.z_min < self.z_max):
            raise ValueError("Invalid geometry bounds")

    @property
    def Fo(self):
        return self.k * self.tau / (self.rho * self.cp * self.L**2)

    @property
    def Bi(self):
        return self.h * self.L / self.k

    @property
    def qstar(self):
        return self.q * self.L / (self.k * self.dT)

    @property
    def U(self):
        return self.alpha * self.dT * self.L

    @property
    def stress_scale(self):
        return self.E * self.alpha * self.dT

    def to_dict(self):
        return asdict(self)


def load_config(path):
    path = Path(path).resolve()
    with path.open("rb") as f:
        config = tomllib.load(f)
    profile_path = (path.parent / config["profile"]).resolve()
    with profile_path.open("rb") as f:
        profile = Profile(**tomllib.load(f)["physics"])
    config["physics"] = profile.to_dict()
    config["config_file"] = str(path)
    config["profile_file"] = str(profile_path)
    for key, value in config.get("paths", {}).items():
        if value:
            config["paths"][key] = str((path.parent / value).resolve())
    runtime = config.setdefault("runtime", {})
    runtime.setdefault("seed", 42)
    runtime.setdefault("device", "cpu")
    runtime.setdefault("threads", 2)
    runtime.setdefault("dtype", "float32")
    if runtime["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    for key in ("adam_steps", "thermal_steps", "mechanical_steps", "stage_steps", "lbfgs_steps"):
        if config.get("training", {}).get(key, 0) < 0:
            raise ValueError(f"Negative training budget: {key}")
    return config
