"""Explicit SI data contracts and COMSOL integral-temperature conversion."""

from io import StringIO
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def read_table(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Missing data file: {path}")
    # Skip metadata as bytes: legacy COMSOL comments may contain mixed encodings.
    # Numeric tables/headers are still strictly decoded; malformed values are not hidden.
    lines = path.read_bytes().removeprefix(b"\xef\xbb\xbf").splitlines()
    start = 0
    # COMSOL metadata precedes a commented coordinate header.
    while start < len(lines) and lines[start].lstrip().startswith(b"%"):
        header = lines[start].lstrip().removeprefix(b"%").strip()
        if header.lower().startswith((b"x,", b"x (", b"x(", b"r,")):
            lines[start] = header
            break
        start += 1
    frame = pd.read_csv(StringIO(b"\n".join(lines[start:]).decode("utf-8")))
    frame.columns = [str(c).strip().removeprefix("%").strip() for c in frame.columns]
    return frame


def validate_frame(frame, config, domain, require_temperature=True):
    required = ["r", "theta", "t"] if require_temperature else ["r", "theta", "thickness"]
    if require_temperature:
        required.append("True_T")
    if domain == "rib":
        required.append("rib_id")
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{domain}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{domain}: empty data")
    numeric = frame[required].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError(f"{domain}: input must be finite")
    g = config.g
    inner = g["rib_start"] if domain == "rib" else g[f"{domain}_inner"]
    if ((frame.r < inner - 1e-5) | (frame.r > g["radius"] + 1e-5)).any():
        raise ValueError(f"{domain}: radius outside domain; expected meters")
    if (np.abs(frame.theta) > 2*np.pi + 1e-6).any():
        raise ValueError("Angles must be radians")
    if "thickness" in frame:
        if not np.isfinite(frame.thickness).all() or (frame.thickness <= 0).any():
            raise ValueError("Thickness must be finite and positive meters")
    if require_temperature:
        if (frame.t < 0).any() or (frame.True_T <= 0).any():
            raise ValueError("Time must be nonnegative seconds; temperature must be positive Kelvin")
    if domain == "rib":
        ids = frame.rib_id
        if ((ids % 1 != 0) | (ids < 0) | (ids >= g["rib_count"])).any():
            raise ValueError("rib_id must be an integer in 0..12")
    return frame


def validate_prediction_frame(frame, config, domain):
    temp = frame.copy()
    if "True_T" not in temp:
        temp["True_T"] = config.p["ambient"]
    return validate_frame(temp, config, domain)


def normalized_inputs(frame, config, geometry, domain):
    validate_prediction_frame(frame, config, domain)
    time = frame.t.to_numpy() * config.alpha / config.g["radius"]**2
    if domain == "rib":
        if "s" in frame:
            arcs = frame.s.to_numpy(dtype=float)
            if not np.isfinite(arcs).all() or (arcs < -1e-5).any():
                raise ValueError("Arc coordinates must be finite nonnegative meters")
        else:
            radii, indices = np.unique(frame.r.to_numpy(), return_inverse=True)
            arcs = np.asarray([geometry.arc(r) for r in radii])[indices]
        return np.column_stack([arcs / config.g["radius"], frame.rib_id, time])
    return np.column_stack([frame.r / config.g["radius"], frame.theta, time])


def sparsify_legacy(frame, eps=0.005):
    """Original greedy cKDTree metric on (r in m, theta in rad), not physical distance."""
    groups = []
    for _, group in frame.groupby("t", sort=True):
        points = group[["r", "theta"]].to_numpy()
        tree = cKDTree(points)
        keep = np.ones(len(points), dtype=bool)
        for i in range(len(points)):
            if keep[i]:
                neighbors = [n for n in tree.query_ball_point(points[i], eps) if n > i]
                keep[neighbors] = False
        groups.append(group.iloc[np.flatnonzero(keep)])
    return pd.concat(groups, ignore_index=True)


def convert_table(frame, *, thickness=None, length_unit="m", thickness_unit="m",
                  temperature_unit="K", representation="temperature", columns=None,
                  sparsify=False, eps=0.005):
    if length_unit not in ("m", "mm") or thickness_unit not in ("m", "mm"):
        raise ValueError("Length/thickness units must be m or mm")
    if temperature_unit not in ("K", "C"):
        raise ValueError("Temperature units must be K or C")
    if representation not in ("temperature", "integral"):
        raise ValueError("representation must be temperature or integral (m*K)")
    frame = frame.copy().rename(columns={v: k for k, v in (columns or {}).items()})
    if thickness is not None:
        thickness = thickness.copy()
        if "thickness" not in thickness:
            thickness = thickness.rename(columns={thickness.columns[-1]: "thickness"})
        frame = frame.merge(thickness[["x", "y", "z", "thickness"]],
                            on=["x", "y", "z"], how="left", validate="many_to_one")
    wide = [c for c in frame if re.search(r"@\s*t\s*=", c)]
    if wide:
        coords = [c for c in frame if c not in wide]
        parts = []
        for column in wide:
            time = float(re.search(r"@\s*t\s*=\s*([\d.eE+-]+)", column).group(1))
            part = frame[coords].copy()
            part["t"], part["T"] = time, frame[column].to_numpy()
            parts.append(part)
        frame = pd.concat(parts, ignore_index=True)
    for column in ("x", "y", "z", "r", "s"):
        if column in frame and length_unit == "mm":
            frame[column] = frame[column] / 1000
    if "thickness" in frame:
        if thickness_unit == "mm":
            frame["thickness"] = frame.thickness / 1000
        # Raw COMSOL merge includes zero-thickness nodes, as in the original pipeline.
        if wide:
            frame = frame[frame.thickness > 0].copy()
    if "r" not in frame:
        frame["r"] = np.hypot(frame.x, frame.y)
    if "theta" not in frame:
        frame["theta"] = np.arctan2(frame.y, frame.x)
    if "True_T" not in frame:
        if "T" not in frame:
            raise ValueError("Expected True_T, T, or COMSOL time columns")
        if representation == "integral":
            if temperature_unit != "K" or "thickness" not in frame:
                raise ValueError("Integral temperature requires m*K values and thickness")
            if (frame.thickness <= 0).any():
                raise ValueError("Integral temperature requires positive thickness")
            frame["True_T"] = frame["T"] / frame.thickness
        else:
            frame["True_T"] = frame["T"] + (273.15 if temperature_unit == "C" else 0)
    # True_T already uses Kelvin; it is never divided or shifted a second time.
    return sparsify_legacy(frame, eps) if sparsify else frame.reset_index(drop=True)


def preprocess(config):
    from .artifacts import file_hash, write_json
    results = {}
    options = config.values["data"].get("preprocess", {})
    for domain in ("upper", "lower", "rib"):
        raw_key = f"{domain}_raw"
        if raw_key not in config.values["data"]:
            raise ValueError(f"Missing {raw_key}")
        original = config.data_path(raw_key)
        per_domain = dict(options.get("common", {}))
        per_domain.update(options.get(domain, {}))
        thick = per_domain.pop("thickness_csv", None)
        frame = convert_table(read_table(original),
                              thickness=read_table(config.path(thick)) if thick else None,
                              **per_domain)
        if domain == "rib" and "rib_id" not in frame:
            # Classification matches the known JSON spiral phases; no speculative clustering.
            import json
            phases = json.loads(config.data_path("rib_json").read_text(encoding="utf-8"))
            centers = np.array([phases[str(i)]["phi_start_rad"] for i in range(13)])
            difference = (frame.theta.to_numpy()[:, None] - centers[None, :]
                          - config.g["rib_slope"]*(frame.r.to_numpy()[:, None]
                                                  - config.g["rib_start"]) + np.pi)
            frame["rib_id"] = np.argmin(np.abs(difference % (2*np.pi) - np.pi), axis=1)
        validate_frame(frame, config, domain)
        destination = config.data_path(f"{domain}_csv")
        destination.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(destination, index=False)
        results[domain] = {"rows": len(frame), "source_sha256": file_hash(original),
                           "processed_sha256": file_hash(destination), "options": per_domain}
    write_json(config.data_path("upper_csv").parent / "preprocess.json", results)
    return results
