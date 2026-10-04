"""Streaming COMSOL wide tables and canonical physical long tables."""
import csv
import re
from pathlib import Path
import numpy as np
import pandas as pd

UNITS = {"m": 1., "mm": 1e-3, "um": 1e-6, "µm": 1e-6, "Pa": 1., "MPa": 1e6}


def header(path):
    comments = []
    with Path(path).open(encoding="utf-8-sig") as f:
        for i, line in enumerate(f):
            if not line.startswith("%"):
                if comments:
                    return i, next(csv.reader([comments[-1].lstrip("% ").strip()])), comments
                return 1, next(csv.reader([line.strip()])), []
            comments.append(line)
    raise ValueError("CSV contains no data")


def _unit(unit, category):
    allowed = {"length": ("m", "mm"), "displacement": ("m", "mm", "um", "µm"), "stress": ("Pa", "MPa")}
    if unit not in allowed[category]:
        raise ValueError(f"Unsupported {category} unit: {unit}")
    return UNITS[unit]


def validate(frame):
    if frame.empty:
        raise ValueError("Empty data table")
    values = frame.select_dtypes(include="number").to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Non-finite data values")
    if (frame.r <= 0).any() or (frame.t < 0).any():
        raise ValueError("Radius must be positive; time must be non-negative")
    if "T" in frame and (frame["T"] <= 0).any():
        raise ValueError("Temperature in Kelvin must be positive")
    return frame


def iter_frames(path, *, length_unit, displacement_unit="mm", temperature_unit="K", time_unit="s", chunksize=8192):
    """Yield SI frames. Units are mandatory and checked against COMSOL metadata."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Input data not found: {path.name}")
    lf, uf = _unit(length_unit, "length"), _unit(displacement_unit, "displacement")
    if temperature_unit not in ("K", "C") or time_unit != "s":
        raise ValueError("Temperature unit must be K or C, and time unit s")
    skip, cols, comments = header(path)
    if any(c in cols for c in ("r_mm", "z_mm")) and length_unit != "mm":
        raise ValueError("Long table mm coordinate headers require length_unit=mm")
    if any(c in cols for c in ("u_r_mm", "u_z_mm")) and displacement_unit != "mm":
        raise ValueError("Long table mm displacement headers require displacement_unit=mm")
    for line in comments:
        if line.startswith("% Length unit,"):
            declared = line.split(",", 1)[1].strip()
            if declared != length_unit:
                raise ValueError(f"Length unit disagrees with metadata: {declared}")
    wide = any(re.search(r"@\s*t\s*=", c) for c in cols)
    expr = {}
    if wide:
        for col in cols[3:]:
            match = re.search(r"(.+?)\s*\(([^)]+)\)\s*@\s*t\s*=\s*([+\-\d.eE]+)", col)
            if not match:
                raise ValueError(f"Unsupported COMSOL expression header: {col}")
            label, unit, t = match.groups()
            key = {"T": "T", "dr": "u_r", "w": "u_z", "solid.mises": "stress"}.get(label.strip())
            if key is None:
                raise ValueError(f"Unsupported expression: {label}")
            t = float(t)
            if t < 0 or not np.isfinite(t) or key in expr.setdefault(t, {}):
                raise ValueError("Invalid or duplicate time/expression header")
            expr[t][key] = (col, unit)
        if list(expr) != sorted(expr):
            raise ValueError("Time axis must be increasing")
        for fields in expr.values():
            if ("u_r" in fields) != ("u_z" in fields):
                raise ValueError("Displacement columns must be paired at every time")
    for raw in pd.read_csv(path, names=cols, skiprows=skip, chunksize=chunksize):
        if wide:
            xyz = raw.iloc[:, :3].to_numpy(float)*lf
            r = np.hypot(xyz[:, 0], xyz[:, 1])
            angle = np.arctan2(xyz[:, 1], xyz[:, 0])
            base = dict(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], r=r, theta=angle)
            for t, fields in expr.items():
                f = pd.DataFrame(base)
                f["t"] = t
                for key, (col, unit) in fields.items():
                    if key == "T":
                        if unit != temperature_unit:
                            raise ValueError("Temperature header unit mismatch")
                        f[key] = raw[col].to_numpy(float)+(273.15 if unit == "C" else 0.)
                    elif key == "stress":
                        f[key] = raw[col].to_numpy(float)*_unit(unit, "stress")
                    else:
                        if unit != displacement_unit:
                            raise ValueError("Displacement header unit mismatch")
                        f[key] = raw[col].to_numpy(float)*uf
                yield validate(f)
        else:
            rename = {"T_true": "T", "True_T": "T", "r_mm": "r", "z_mm": "z", "u_r_mm": "u_r", "u_z_mm": "u_z"}
            f = raw.rename(columns=rename)
            if "t" not in f:
                raise ValueError("Long table requires t in seconds")
            if "z" not in f or not ("r" in f or {"x", "y"} <= set(f)):
                raise ValueError("Long table requires z and r or x/y")
            for key in ("x", "y", "z", "r"):
                if key in f:
                    f[key] = f[key].astype(float)*lf
            if "r" not in f:
                f["r"] = np.hypot(f.x, f.y)
            if "theta" not in f:
                f["theta"] = np.arctan2(f.y, f.x) if {"x", "y"} <= set(f) else 0.
            for key in ("u_r", "u_z"):
                if key in f:
                    f[key] = f[key].astype(float)*uf
            if "T" in f and temperature_unit == "C":
                f["T"] += 273.15
            if not ({"T", "stress"} & set(f) or {"u_r", "u_z"} <= set(f)):
                raise ValueError("Missing temperature/displacement/stress values")
            yield validate(f.drop(columns=["sin_theta", "cos_theta"], errors="ignore"))


def sample_times(path, per_time=32, times=None, seed=42, **units):
    """Bounded, seeded reservoir per time; raw spatial rows stay paired across fields."""
    reservoirs, generators = {}, {}
    for frame in iter_frames(path, **units):
        for t, part in frame.groupby("t", sort=False):
            if times is not None and t not in times:
                continue
            if t not in generators:
                generators[t] = np.random.default_rng(seed+int(round(t*1000)))
            part = part.copy()
            part["_priority"] = generators[t].random(len(part))
            if t in reservoirs:
                part = pd.concat((reservoirs[t], part), ignore_index=True)
            reservoirs[t] = part.nsmallest(per_time, "_priority")
    if not reservoirs or (times is not None and set(times)-set(reservoirs)):
        raise ValueError("Requested time samples are unavailable")
    return pd.concat([reservoirs[t].drop(columns="_priority") for t in sorted(reservoirs)], ignore_index=True)


def preprocess(input_path, output_path, **units):
    output_path = Path(output_path)
    if output_path.resolve() == Path(input_path).resolve() or output_path.exists():
        raise FileExistsError("Refusing to overwrite input or existing processed data")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    for frame in iter_frames(input_path, **units):
        frame.to_csv(output_path, mode="a", header=count == 0, index=False)
        count += len(frame)
    return count
