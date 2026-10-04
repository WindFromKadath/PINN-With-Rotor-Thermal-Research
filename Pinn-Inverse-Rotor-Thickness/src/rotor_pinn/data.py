"""SI CSV validation, COMSOL adapters and spatial-group data partitions."""

from pathlib import Path
import re

import numpy as np
import pandas as pd

from .geometry import RotorGeometry

TEMPERATURE_COLUMNS = ["r", "theta", "t", "True_T"]


def validate_frame(frame, config, region, require_temperature=True, require_thickness=False):
    frame = frame.copy()
    required = TEMPERATURE_COLUMNS if require_temperature else ["r", "theta"]
    if require_thickness:
        required = required + ["thickness"]
    if region == "rib":
        required = required + ["rib_id"]
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{region}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{region}: empty dataset")
    checked = set(required) | ({"thickness"} if "thickness" in frame else set())
    for name in checked:
        frame[name] = pd.to_numeric(frame[name], errors="raise")
        if not np.isfinite(frame[name].to_numpy()).all():
            raise ValueError(f"{region}.{name}: non-finite values")
    if "thickness" in frame and (frame.thickness <= 0).any():
        raise ValueError(f"{region}: thickness must be strictly positive")
    frame["theta"] = frame.theta % (2 * np.pi)
    g = config.geometry
    inner = g.rib_start if region == "rib" else getattr(g, f"{region}_inner_radius")
    outer = g.rib_end if region == "rib" else g.outer_radius
    if ((frame.r < inner - 1e-8) | (frame.r > outer + 1e-8)).any():
        raise ValueError(f"{region}: radii outside geometry")
    if require_temperature:
        if (frame.t < 0).any() or (frame.True_T <= 0).any():
            raise ValueError(f"{region}: time must be nonnegative and Kelvin temperatures positive")
    if region == "rib":
        values = frame.rib_id.to_numpy()
        if ((values != np.floor(values)) | (values < 0) | (values >= g.rib_count)).any():
            raise ValueError("rib_id must be an integer in the configured range")
        frame["rib_id"] = values.astype(int)
    return frame.reset_index(drop=True)


def spatial_split(frame, training):
    frame = frame.copy()
    names = ["r", "theta"] + (["rib_id"] if "rib_id" in frame else [])
    keys = pd.MultiIndex.from_frame(frame[names].round(10))
    codes, groups = pd.factorize(keys, sort=True)
    if len(groups) < 3:
        raise ValueError("At least three distinct spatial groups are required")
    permutation = np.random.default_rng(training.seed).permutation(len(groups))
    n_train = min(max(1, int(len(groups) * training.train_fraction)), len(groups) - 2)
    n_validation = min(max(1, int(len(groups) * training.validation_fraction)),
                       len(groups) - n_train - 1)
    labels = np.full(len(groups), "test", dtype=object)
    labels[permutation[:n_train]] = "train"
    labels[permutation[n_train:n_train + n_validation]] = "validation"
    frame["split"] = labels[codes]
    return frame


def normalized_inputs(frame, region, geometry, time_scale):
    first = geometry.r_to_s(frame.r.to_numpy()) if region == "rib" else frame.r.to_numpy()
    second = frame.rib_id.to_numpy() if region == "rib" else frame.theta.to_numpy()
    return np.column_stack([first / geometry.config.outer_radius, second,
                            frame.t.to_numpy() / time_scale])


def read_csv(path, config, region, **kwargs):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing {region} dataset: {path.name}")
    return validate_frame(pd.read_csv(path), config, region, **kwargs)


def adapt_comsol(path, config, region):
    options = config.preprocess
    frame = pd.read_csv(path, encoding=options.encoding, skiprows=options.skip_rows)
    frame.columns = [str(c).strip().lstrip("%").strip() for c in frame.columns]
    frame = frame.rename(columns={source: target for target, source in options.columns.items()})
    units = {"m": 1.0, "mm": 1e-3}
    if options.length_unit not in units or options.thickness_unit not in units:
        raise ValueError("length_unit and thickness_unit must be m or mm")
    length_scale = units[options.length_unit]
    if {"x", "y"} <= set(frame):
        x, y = pd.to_numeric(frame.x), pd.to_numeric(frame.y)
        if "r" not in frame:
            frame["r"] = np.hypot(x, y)
        if "theta" not in frame:
            frame["theta"] = np.arctan2(y, x)
    if "r" in frame:
        frame["r"] = pd.to_numeric(frame.r) * length_scale
    if "thickness" in frame:
        frame["thickness"] = pd.to_numeric(frame.thickness) * units[options.thickness_unit]
    if "True_T" not in frame:
        time_columns = [c for c in frame if "@t" in c]
        if not time_columns:
            raise ValueError("No True_T column or COMSOL @t columns; configure the column mapping")
        long = []
        for name in time_columns:
            match = re.search(r"@t\s*=\s*([-+\d.eE]+)", name)
            if match is None:
                raise ValueError(f"Cannot parse time from {name!r}")
            part = frame.drop(columns=time_columns).copy()
            part["t"], part["True_T"] = float(match.group(1)), frame[name]
            long.append(part)
        frame = pd.concat(long, ignore_index=True)
    if options.temperature_unit == "C":
        frame["True_T"] = pd.to_numeric(frame.True_T) + 273.15
    elif options.temperature_unit != "K":
        raise ValueError("temperature_unit must be K or C")
    if region == "rib" and "rib_id" not in frame:
        if not {"r", "theta"} <= set(frame):
            raise ValueError("Rib assignment requires r and theta")
        frame["rib_id"] = RotorGeometry(config.geometry).assign_rib_id(frame.r, frame.theta)
    frame = validate_frame(frame, config, region)
    keep = TEMPERATURE_COLUMNS + (["thickness"] if "thickness" in frame else [])
    if region == "rib":
        keep.append("rib_id")
    return frame[keep]


def preprocess(config, overwrite=False):
    jobs = [(region, config.path(getattr(config.preprocess, f"{region}_input")),
             config.path(getattr(config.data, f"{region}_csv")))
            for region in ("upper", "lower", "rib")]
    outputs = [str(output) for _, _, output in jobs]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Preprocessing destinations must be distinct")
    inputs = {str(source) for _, source, _ in jobs}
    for _, _, output in jobs:
        if str(output) in inputs:
            raise ValueError("Preprocessing must never overwrite any source file")
        if output.exists() and not overwrite:
            raise FileExistsError(f"Destination exists: {output.name}; use --overwrite explicitly")
    frames = {region: adapt_comsol(source, config, region) for region, source, _ in jobs}
    for region, _, output in jobs:
        output.parent.mkdir(parents=True, exist_ok=True)
        frames[region].to_csv(output, index=False)
    return {region: len(frame) for region, frame in frames.items()}
