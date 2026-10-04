"""Partition-aware Kelvin metrics and batched predictions from portable run snapshots."""

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .artifacts import file_hash, write_json
from .data import normalized_inputs, validate_frame
from .geometry import RotorGeometry


def compute_metrics(true, predicted):
    true, predicted = np.asarray(true), np.asarray(predicted)
    if true.shape != predicted.shape or not true.size:
        raise ValueError("Metric inputs must be nonempty and have equal shapes")
    if not np.isfinite(true).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric inputs must be finite")
    error = predicted - true
    norm = float(np.linalg.norm(true))
    return {"count": int(true.size), "rmse_k": float(np.sqrt(np.mean(error**2))),
            "mae_k": float(np.mean(np.abs(error))), "max_k": float(np.max(np.abs(error))),
            "l2_relative_absolute_kelvin": float(np.linalg.norm(error) / norm) if norm else None}


def predict_frame(model, frame, region, geometry, time_scale, physics, device, dtype, batch=2048):
    inputs = normalized_inputs(frame, region, geometry, time_scale)
    predictions = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch):
            x = torch.as_tensor(inputs[start:start + batch], device=device, dtype=dtype)
            predictions.append((physics.ambient + physics.temperature_scale * model(x)).cpu().numpy().ravel())
    return np.concatenate(predictions) if predictions else np.array([])


def evaluate_run(run_directory, split="test", device="cpu"):
    from .training import reload_models
    directory = Path(run_directory)
    config, models, _, time_scale, selected, dtype = reload_models(directory, device)
    geometry = RotorGeometry(config.geometry)
    partitions = ["train", "validation", "test"] if split == "all" else [split]
    if any(part not in ("train", "validation", "test") for part in partitions):
        raise ValueError("Use train, validation, test or all")
    predictions = {}
    output = {"l2_definition": "norm(predicted_K - true_K) / norm(true_K)", "partitions": {}}
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        import json
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {}
    for region, model in models.items():
        path = directory / "inputs" / f"{region}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved {region} input snapshot: {path.name}")
        expected = manifest.get("input_sha256", {}).get(region)
        if expected and file_hash(path) != expected:
            raise ValueError(f"Saved {region} input snapshot hash differs from training input")
        frame = validate_frame(pd.read_csv(path), config, region)
        if "split" not in frame or not frame["split"].isin(["train", "validation", "test"]).all():
            raise ValueError(f"Saved {region} input has invalid split labels")
        frame["predicted_T"] = predict_frame(model, frame, region, geometry, time_scale,
                                              config.physics, selected, dtype)
        frame["error_K"] = frame.predicted_T - frame.True_T
        frame.to_csv(directory / f"predictions_{region}.csv", index=False, float_format="%.17g")
        predictions[region] = frame
    for partition in partitions:
        regional = {}
        true, predicted = [], []
        for region, frame in predictions.items():
            part = frame[frame["split"] == partition]
            regional[region] = compute_metrics(part.True_T.to_numpy(), part.predicted_T.to_numpy())
            true.extend(part.True_T.to_numpy())
            predicted.extend(part.predicted_T.to_numpy())
        regional["overall"] = compute_metrics(np.array(true), np.array(predicted))
        output["partitions"][partition] = regional
    # Keep prior partition results when an individual partition is reevaluated.
    destination = directory / "metrics.json"
    if destination.exists() and split != "all":
        import json
        prior = json.loads(destination.read_text(encoding="utf-8"))
        prior["partitions"].update(output["partitions"])
        output = prior
    write_json(destination, output)
    return output
