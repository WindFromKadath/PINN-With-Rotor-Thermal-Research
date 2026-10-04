"""Read-only reuse of the PINN_Inverse_fin_pls models_final inference profile.

Only audited network definitions are read from pinn_mse_v3.py. Its top-level
data loading, interactive prompts, saving and training scripts are not executed.
Historical scales are deliberately kept separate from rotor_pinn checkpoints.
"""

import argparse
import ast
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd
from scipy.integrate import quad
import torch

from rotor_pinn.artifacts import file_hash, new_run, write_json
from rotor_pinn.backend import initialize
from rotor_pinn.config import Config
from rotor_pinn.evaluation import compute_metrics


DEFINITIONS = {"SimpleThicknessNet", "create_2d_net", "create_1d_rib_net", "r_to_s_phys"}


def load_definitions(path, dde):
    """Use the historical network transforms exactly, without importing the script."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    selected = [node for node in tree.body
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in DEFINITIONS]
    if len(selected) != len(DEFINITIONS) or {node.name for node in selected} != DEFINITIONS:
        raise ValueError("Unsupported historical script: expected network/arc definitions are missing")
    namespace = {"dde": dde, "torch": torch, "nn": torch.nn, "np": np, "quad": quad,
                 "R_RIB_START": 0.1200, "R_RIB_END": 0.2250, "b_spiral": 6.4105}
    module = ast.Module(body=selected, type_ignores=[])
    exec(compile(module, str(path), "exec"), namespace)
    return namespace


def load_weight(model, path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return {"sha256": file_hash(path), "tensor_shapes": {key: list(value.shape)
                                                          for key, value in state.items()}}


def predict(model, frame, region, definitions, batch_size):
    radius_scale = 0.225
    alpha_ref = 15.1 / (7850 * 460)
    time = frame.t.to_numpy() * alpha_ref / radius_scale**2
    if region == "rib":
        unique, inverse = np.unique(frame.r.to_numpy(), return_inverse=True)
        # Reuse the source's quad and tolerance exactly, once per distinct radius.
        arc = np.asarray([definitions["r_to_s_phys"](r) for r in unique])[inverse]
        keep = np.isfinite(arc)
        frame = frame.loc[keep].reset_index(drop=True)
        inputs = np.column_stack([arc[keep] / radius_scale, time[keep], frame.rib_id])
    else:
        inputs = np.column_stack([frame.r / radius_scale, frame.theta, time])
    result = []
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            x = torch.tensor(inputs[start:start + batch_size], dtype=torch.float32, device="cpu")
            result.append(model(x).numpy().ravel() * 900 + 293.15)
    return frame, np.concatenate(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--run-id")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--dataset-kind", choices=("outer", "domain"), default="outer")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    source = Path(args.source_dir).resolve()
    script = source / "pinn_mse_v3.py"
    cfg = Config(base=Path(__file__).resolve().parents[1])
    cfg.output.artifacts = "artifacts"
    cfg.geometry.rib_start = 0.1200
    cfg.networks.initial_factor = 10
    cfg.training.device = "cpu"
    cfg.training.dtype = "float32"
    names = {"outer": ("T_d3_outer_long.csv", "T_d1_outer_long.csv", "T_d2_outer_long_with_rib.csv"),
             "domain": ("domain_data_3.csv", "domain_data_1.csv", "domain_data_2_with_rib.csv")}
    for region, filename in zip(("upper", "lower", "rib"), names[args.dataset_kind]):
        setattr(cfg.data, f"{region}_csv", str(source / "Dataset" / filename))
    cfg.validate()
    with new_run(cfg, args.run_id, "historical saved-weight inference; no retraining or new-model validation",
                 f"historical {args.dataset_kind} tables; full selected population") as (directory, manifest):
        manifest["adapter_sha256"] = file_hash(__file__)
        manifest["legacy_source_sha256"] = file_hash(script)
        dde, _, _ = initialize(cfg.training)
        definitions = load_definitions(script, dde)
        profile = {"source_script": str(script), "source_script_sha256": file_hash(script),
                   "ring_input": "[r/0.225,theta,t*alpha_ref/0.225**2]",
                   "rib_input": "[s/0.225,t*alpha_ref/0.225**2,rib_id]",
                   "alpha_ref_m2_s": 15.1 / (7850 * 460), "temperature_scale_k": 900,
                   "ambient_k": 293.15, "output_factor": "1-exp(-10*t_star)",
                   "arc_start_m": 0.1200, "arc_end_m": 0.2250,
                   "dataset_kind": args.dataset_kind,
                   "comparison_scope": "historical full population; no held-out split assertion",
                   "new_checkpoint_format_compatible": False, "weights": {}}
        results = {"scope": profile["comparison_scope"], "temperature": {}, "thickness": {}}
        all_true, all_predicted = [], []
        for region in ("upper", "lower", "rib"):
            stem = {"upper": "upper_ring", "lower": "lower_ring", "rib": "rib_net"}[region]
            weight = source / "models_final" / f"{stem}_bidirectional_final.pth"
            model = definitions["create_1d_rib_net" if region == "rib" else "create_2d_net"]().cpu()
            profile["weights"][region] = {"path": str(weight), **load_weight(model, weight)}
            manifest["input_sha256"][f"weight_{region}"] = file_hash(weight)
            path = Path(getattr(cfg.data, f"{region}_csv"))
            manifest["input_sha256"][f"data_{region}"] = file_hash(path)
            frame = pd.read_csv(path)
            required = ["r", "theta", "t", "True_T"] + (["rib_id"] if region == "rib" else [])
            if set(required) - set(frame):
                raise ValueError(f"Missing historical {region} input columns")
            if not np.isfinite(frame[required].to_numpy()).all():
                raise ValueError(f"Non-finite historical {region} input")
            if region == "rib" and ((frame.rib_id % 1 != 0) | (frame.rib_id < 0) | (frame.rib_id >= 13)).any():
                raise ValueError("Invalid historical rib id")
            original_count = len(frame)
            frame, predicted = predict(model, frame, region, definitions, args.batch_size)
            results["temperature"][region] = compute_metrics(frame.True_T.to_numpy(), predicted)
            results["temperature"][region]["excluded_arc_rows"] = original_count - len(frame)
            all_true.append(frame.True_T.to_numpy())
            all_predicted.append(predicted)
            print(f"Loaded {region}: {len(frame)} rows; RMSE {results['temperature'][region]['rmse_k']:.4f} K")
        results["temperature"]["overall"] = compute_metrics(np.concatenate(all_true), np.concatenate(all_predicted))
        for region, domain in (("upper", 3), ("lower", 1)):
            path = source / "Dataset" / f"domain_data_{domain}.csv"
            manifest["input_sha256"][f"thickness_data_{region}"] = file_hash(path)
            frame = pd.read_csv(path, usecols=["r", "theta", "thickness"]).drop_duplicates(["r", "theta"])
            x = torch.tensor(np.column_stack([frame.r / 0.225, frame.theta]), dtype=torch.float32)
            for suffix in ("_reg", ""):
                weight = source / "thickness" / f"{region}_thickness_net{suffix}.pth"
                model = definitions["SimpleThicknessNet"]().cpu()
                key = f"{region}_thickness{suffix}"
                profile["weights"][key] = {"path": str(weight), **load_weight(model, weight)}
                manifest["input_sha256"][key] = file_hash(weight)
                with torch.no_grad():
                    raw = model(x[:, 0:1], x[:, 1:2]).numpy().ravel()
                if not np.isfinite(raw).all():
                    raise FloatingPointError("Non-finite historical thickness prediction")
                results["thickness"][key] = {"raw_min": float(raw.min()), "raw_max": float(raw.max()),
                                            "target_min": float(frame.thickness.min()),
                                            "target_max": float(frame.thickness.max()),
                                            "scale_note": "reg uses raw column units; unregularized uses training DELTA_REF; no new-format conversion"}
        report = source / "error_report.txt"
        if report.is_file():
            manifest["input_sha256"]["historical_report"] = file_hash(report)
            original = report.read_text(encoding="gb18030")
            values = [float(v) for v in re.findall(r"RMSE:\s*([0-9.]+)", original)]
            if len(values) == 4:
                actual = [results["temperature"][r]["rmse_k"] for r in ("upper", "lower", "rib", "overall")]
                results["historical_report_comparison"] = {
                    "reported_rmse_k": values, "recomputed_rmse_k": actual,
                    "matches_reported_four_decimal_rmse": all(f"{a:.4f}" == f"{b:.4f}" for a, b in zip(actual, values))}
        write_json(directory / "legacy_profile.json", profile)
        write_json(directory / "metrics.json", results)
        manifest["has_metrics"] = True
        manifest["legacy_profile_sha256"] = file_hash(directory / "legacy_profile.json")
        print(json.dumps(results["temperature"]["overall"], indent=2))
        print(directory)


if __name__ == "__main__":
    main()
