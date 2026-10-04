"""Typed TOML settings. Paths are relative to the configuration file, never the cwd."""

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import tomllib


@dataclass
class Geometry:
    outer_radius: float = 0.225
    upper_inner_radius: float = 0.084
    lower_inner_radius: float = 0.103
    rib_start: float = 0.123
    rib_end: float = 0.225
    rib_slope: float = 6.4105
    rib_theta0: float = 1.2209
    rib_count: int = 13
    rib_width: float = 0.016
    rib_height: float = 0.0094


@dataclass
class Physics:
    ambient: float = 293.15
    temperature_scale: float = 900.0
    density: float = 7850.0
    temperature_nodes: list = field(default_factory=lambda: [293.15, 373.15, 573.15, 773.15])
    conductivity_nodes: list = field(default_factory=lambda: [15.1, 16.3, 18.4, 20.5])
    heat_capacity_nodes: list = field(default_factory=lambda: [460.0, 502.0, 586.0, 670.0])
    convection: float = 10.0
    emissivity: float = 0.3
    exposed_ring_faces: float = 1.0
    rib_exposed_sides: int = 2
    rib_end_condition: str = "insulated"
    ring_radial_boundary: str = "convection_radiation"
    source_strength: float = 1e8
    source_decay: float = 0.005
    source_cutoff_decays: float = 3.0
    lower_source_factor: float = 0.3
    rib_volume_source: float = 0.0
    contact_resistance_upper: float = 0.0
    contact_resistance_lower: float = 0.0


@dataclass
class Networks:
    ring_widths: list = field(default_factory=lambda: [64, 64, 64, 64])
    rib_widths: list = field(default_factory=lambda: [64, 64, 64])
    thickness_widths: list = field(default_factory=lambda: [64, 64, 32])
    initial_factor: float = 5.0
    minimum_thickness: float = 1e-6


@dataclass
class Training:
    seed: int = 42
    device: str = "auto"
    dtype: str = "float32"
    cpu_threads: int = 2
    thickness_steps: int = 2000
    warmup_steps: int = 2000
    outer_iterations: int = 5
    inner_steps: int = 1500
    lbfgs_steps: int = 5000
    learning_rate: float = 1e-3
    observation_weight: float = 100.0
    thickness_gradient_weight: float = 1e-5
    domain_points: int = 1024
    boundary_points: int = 256
    observation_batch: int = 256
    display_every: int = 100
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    observe_ribs: bool = True


@dataclass
class Contact:
    radial_nodes: int = 48
    angular_nodes: int = 128
    arc_nodes: int = 50
    time_nodes: int = 30
    gaussian_sigma: float = 0.008
    relaxation: float = 0.5


@dataclass
class Data:
    upper_csv: str = "../data/processed/upper.csv"
    lower_csv: str = "../data/processed/lower.csv"
    rib_csv: str = "../data/processed/ribs.csv"
    upper_thickness_csv: str = ""
    lower_thickness_csv: str = ""
    thickness_checkpoint: str = ""


@dataclass
class Preprocess:
    upper_input: str = "../data/raw/upper.csv"
    lower_input: str = "../data/raw/lower.csv"
    rib_input: str = "../data/raw/ribs.csv"
    encoding: str = "utf-8-sig"
    skip_rows: int = 0
    length_unit: str = "m"
    thickness_unit: str = "m"
    temperature_unit: str = "K"
    columns: dict = field(default_factory=dict)


@dataclass
class Output:
    artifacts: str = "../artifacts"


@dataclass
class Config:
    geometry: Geometry = field(default_factory=Geometry)
    physics: Physics = field(default_factory=Physics)
    networks: Networks = field(default_factory=Networks)
    training: Training = field(default_factory=Training)
    contact: Contact = field(default_factory=Contact)
    data: Data = field(default_factory=Data)
    preprocess: Preprocess = field(default_factory=Preprocess)
    output: Output = field(default_factory=Output)
    base: Path = field(default_factory=Path.cwd, repr=False)

    def path(self, value: str) -> Path:
        if not value:
            raise ValueError("An input path is empty; supply it explicitly in TOML.")
        return (self.base / value).resolve()

    def snapshot(self) -> dict:
        result = asdict(self)
        result.pop("base")
        return result

    def validate(self):
        g, p, n, t, c = self.geometry, self.physics, self.networks, self.training, self.contact
        for section in (g, p, n, t, c):
            for item in fields(section):
                value = getattr(section, item.name)
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError(f"{type(section).__name__}.{item.name} must be finite")
        if not (0 < g.upper_inner_radius < g.rib_start < g.rib_end <= g.outer_radius):
            raise ValueError("Upper annulus/rib radii are inconsistent")
        if not (0 < g.lower_inner_radius < g.rib_start):
            raise ValueError("Lower annulus/rib radii are inconsistent")
        if not isinstance(g.rib_count, int) or g.rib_count < 1:
            raise ValueError("rib_count must be a positive integer")
        if min(g.rib_width, g.rib_height, p.temperature_scale, p.density, p.source_decay) <= 0:
            raise ValueError("Geometry and physical scales must be positive")
        if min(p.convection, p.source_strength, p.rib_volume_source, p.lower_source_factor,
               p.contact_resistance_upper, p.contact_resistance_lower) < 0:
            raise ValueError("Heat-transfer coefficients, sources and resistances must be nonnegative")
        if not (0 <= p.emissivity <= 1 and 0 <= p.exposed_ring_faces <= 2):
            raise ValueError("Invalid emissivity or exposed face count")
        if p.rib_exposed_sides not in (0, 1, 2):
            raise ValueError("rib_exposed_sides must be 0, 1 or 2")
        if p.rib_end_condition != "insulated" or p.ring_radial_boundary != "convection_radiation":
            raise ValueError("This model supports insulated rib ends and convective/radiative ring edges")
        nodes = (p.temperature_nodes, p.conductivity_nodes, p.heat_capacity_nodes)
        if len(nodes[0]) < 2 or len({len(v) for v in nodes}) != 1:
            raise ValueError("Material node arrays must have equal lengths >= 2")
        if any(not math.isfinite(v) or v <= 0 for a in nodes for v in a):
            raise ValueError("Material nodes must be finite and positive")
        if any(b <= a for a, b in zip(nodes[0], nodes[0][1:])):
            raise ValueError("Temperature nodes must be strictly increasing")
        if not (0 < t.train_fraction < 1 and 0 < t.validation_fraction < 1
                and t.train_fraction + t.validation_fraction < 1):
            raise ValueError("Split fractions must leave nonempty train/validation/test groups")
        if min(c.radial_nodes, c.angular_nodes, c.arc_nodes, c.time_nodes) < 2:
            raise ValueError("Contact quadrature dimensions must be >= 2")
        for section, names in ((c, ("radial_nodes", "angular_nodes", "arc_nodes", "time_nodes")),
                               (t, ("seed", "cpu_threads", "thickness_steps", "warmup_steps",
                                    "outer_iterations", "inner_steps", "lbfgs_steps", "domain_points",
                                    "boundary_points", "observation_batch", "display_every"))):
            if any(type(getattr(section, name)) is not int for name in names):
                raise ValueError("Sampling counts, budgets and seed must be integers")
        if t.seed < 0 or p.ambient <= 0 or p.source_cutoff_decays <= 0:
            raise ValueError("Seed must be nonnegative; ambient and source cutoff must be positive")
        if not 0 < c.relaxation <= 1 or c.gaussian_sigma <= 0:
            raise ValueError("Invalid contact relaxation or Gaussian width")
        if t.device not in ("auto", "cpu", "cuda") or t.dtype not in ("float32", "float64"):
            raise ValueError("Use device=auto/cpu/cuda and dtype=float32/float64")
        if min(t.thickness_steps, t.warmup_steps, t.inner_steps, t.lbfgs_steps) < 0:
            raise ValueError("Training budgets cannot be negative")
        if min(t.outer_iterations, t.domain_points, t.boundary_points,
               t.observation_batch, t.display_every, t.cpu_threads) < 1:
            raise ValueError("Training counts must be positive")
        if min(t.learning_rate, t.observation_weight, n.initial_factor, n.minimum_thickness) <= 0:
            raise ValueError("Learning rates, weights and network scales must be positive")
        if t.thickness_gradient_weight < 0:
            raise ValueError("Gradient penalty must be nonnegative")
        for widths in (n.ring_widths, n.rib_widths, n.thickness_widths):
            if not widths or any(not isinstance(v, int) or v <= 0 for v in widths):
                raise ValueError("Hidden widths must be positive integers")
        return self


def from_dict(value: dict, base: Path | str) -> Config:
    classes = {"geometry": Geometry, "physics": Physics, "networks": Networks,
               "training": Training, "contact": Contact, "data": Data,
               "preprocess": Preprocess, "output": Output}
    unknown = set(value) - set(classes)
    if unknown:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown)}")
    kwargs = {}
    for name, values in value.items():
        expected = {v.name for v in fields(classes[name])}
        if set(values) - expected:
            raise ValueError(f"Unknown settings in {name}: {sorted(set(values) - expected)}")
        kwargs[name] = classes[name](**values)
    return Config(**kwargs, base=Path(base).resolve()).validate()


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open("rb") as stream:
        return from_dict(tomllib.load(stream), path.parent)
