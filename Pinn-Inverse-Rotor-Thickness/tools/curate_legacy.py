"""Compare complete audited rotor families and preserve the best few locally.

Selection is an exploratory saved-weight comparison on one historical population,
not a held-out generalization test or reproduction of the old published RMSE.
Original code, data and weights are always read-only.
"""

import argparse
import ast
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import torch

from rotor_pinn.artifacts import file_hash, new_run, write_json
from rotor_pinn.backend import initialize
from rotor_pinn.config import Config
from rotor_pinn.evaluation import compute_metrics
from rotor_pinn.legacy import LegacyBundle, verify_bundle


DEFINITIONS = {"SimpleThicknessNet", "create_2d_net", "create_1d_rib_net"}
IMPORTS = "import torch\nimport torch.nn as nn\nimport numpy as np\nimport deepxde as dde\nNUM_RIBS = 13\n\n"


def candidates(root):
    fin = root / "PINN_Inverse_fin_pls"
    again = root / "InversePINN_Again"
    specs = []
    for version, directory, suffix, training, predictor, start in (
        ("fin_final", "models_final", "_bidirectional_final", "pinn_v3.2.py", "pinn_mse_v3.py", .120),
        ("fin_v3_1", "models_v3.1", "_bidirectional", "pinn_v3.1.py", "pinn_mse_v3.py", .120),
        ("fin_v3", "models_v3", "_bidirectional", "pinn_v3.py", "pinn_mse_v3.py", .120),
        ("fin_v2_4", "models_v2.4", "_final", "pinn_v2.4.py", "pinn_pre_v2.4.py", .123),
    ):
        specs.append({"version_id": version, "source": fin, "directory": directory,
                      "training": training, "predictor": predictor, "factory_source": predictor if version == "fin_final" else training,
                      "weights": {"upper": f"upper_ring{suffix}.pth", "lower": f"lower_ring{suffix}.pth",
                                  "rib": f"rib_net{suffix}.pth"}, "rib_start_m": start,
                      "rib_input_order": ["s_star", "Fo", "rib_id"], "rib_id_conditioned": True,
                      "physical_thickness": True})
    for version, directory, suffix, training, predictor in (
        ("again_fixed_a", "model_v3.1", "fixed", "inverse_pinn_v3.1.py", "inverse_pinn_predict_v3.py"),
        ("again_fixed_b", "models_v3.1", "fixed", "inverse_pinn_v3.1.py", "inverse_pinn_predict_v3.py"),
        ("again_v3", "models_v3.1", "final", "inverse_pinn_v3.py", "inverse_pinn_predict.py"),
        ("again_base", "models", "final", "inverse_pinn.py", "inverse_pinn_predict.py"),
    ):
        specs.append({"version_id": version, "source": again, "directory": directory,
                      "training": training, "predictor": predictor, "factory_source": training,
                      "weights": {region: f"{region}_{suffix}.pth" for region in ("upper", "lower", "rib")},
                      "rib_start_m": .123, "rib_input_order": ["s_star", "Fo"],
                      "rib_id_conditioned": False, "physical_thickness": False})
    return specs


def extract_factory(path):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
             and n.name in DEFINITIONS]
    if len(nodes) != len(DEFINITIONS) or {n.name for n in nodes} != DEFINITIONS:
        raise ValueError(f"Unsupported historical network definitions in {path}")
    return IMPORTS + "\n\n".join(ast.unparse(n) for n in nodes) + "\n"


def profile_for(spec):
    return {"format_version": 1, "version_id": spec["version_id"],
            "radius_scale_m": .225, "ring_inner_m": {"upper": .084, "lower": .103},
            "rib_start_m": spec["rib_start_m"], "rib_end_m": .225, "rib_slope_rad_m": 6.4105,
            "alpha_ref_m2_s": 15.1 / (7850 * 460), "temperature_scale_k": 900., "ambient_k": 293.15,
            "ring_input_order": ["r_star", "theta", "Fo"], "rib_input_order": spec["rib_input_order"],
            "rib_id_conditioned": spec["rib_id_conditioned"], "weights": {}, "thickness": {},
            "new_training_checkpoint_compatible": False,
            "original_paths_are_provenance_only": True,
            "training_source_provenance": "Matching definitions and available source snapshot; original checkpoint-to-source revision was not saved",
            "files_sha256": {}}


def audited_model(spec):
    path = spec["source"] / spec["factory_source"]
    factory = extract_factory(path)
    namespace = {"__name__": "audited_legacy_factory"}
    exec(compile(factory, str(path), "exec"), namespace)
    model = object.__new__(LegacyBundle)
    model.profile = profile_for(spec)
    model.device = torch.device("cpu")
    model.models = {}
    hashes = {}
    for region, filename in spec["weights"].items():
        weight = spec["source"] / spec["directory"] / filename
        net = namespace["create_1d_rib_net" if region == "rib" else "create_2d_net"]().cpu()
        state = torch.load(weight, map_location="cpu", weights_only=True)
        net.load_state_dict(state, strict=True)
        model.models[region] = net.eval()
        hashes[region] = {"path": str(weight), "sha256": file_hash(weight),
                          "tensor_shapes": {key: list(value.shape) for key, value in state.items()}}
    return model, factory, hashes


def examples(profile, region):
    # Generated physical inputs, no historical temperature observations needed.
    if region == "rib":
        radii = np.linspace(profile["rib_start_m"], .225, 5)
        rows = [(r, (1.2209 + 6.4105 * (r - profile["rib_start_m"]) + 2 * np.pi * i / 13) % (2 * np.pi), t, i)
                for i in range(13) for r in radii for t in (0., 300., 1200.)]
        return pd.DataFrame(rows, columns=["r", "theta", "t", "rib_id"])
    rows = [(r, theta, t) for r in np.linspace(profile["ring_inner_m"][region] + .001, .224, 5)
            for theta in np.linspace(0., 2 * np.pi, 13, endpoint=False) for t in (0., 300., 1200.)]
    return pd.DataFrame(rows, columns=["r", "theta", "t"])


def preserve(spec, model, factory, hashes, metrics, destination):
    folder = destination / spec["version_id"]
    folder.mkdir(exist_ok=False)
    (folder / "weights").mkdir()
    (folder / "source").mkdir()
    (folder / "examples").mkdir()
    (folder / "factory.py").write_text(factory, encoding="utf-8")
    profile = profile_for(spec)
    profile["original_source_directory"] = str(spec["source"])
    profile["original_weights"] = hashes
    profile["selection_metrics"] = metrics
    for region, filename in spec["weights"].items():
        relative = f"weights/{region}.pth"
        shutil.copy2(spec["source"] / spec["directory"] / filename, folder / relative)
        profile["weights"][region] = relative
    for filename in dict.fromkeys((spec["training"], spec["predictor"], spec["factory_source"])):
        shutil.copy2(spec["source"] / filename, folder / "source" / filename)
    if spec["physical_thickness"]:
        for region in ("upper", "lower"):
            relative = f"weights/{region}_thickness_m.pth"
            original = spec["source"] / "thickness" / f"{region}_thickness_net_reg.pth"
            shutil.copy2(original, folder / relative)
            profile["thickness"][region] = {"weight": relative, "output_scale_m": 1.,
                                             "original_path": str(original), "units": "m"}
        for filename in ("thickness_train_org.py", "thickness_reg_vision.py"):
            shutil.copy2(spec["source"] / "thickness" / filename, folder / "source" / filename)
    else:
        profile["thickness_note"] = "Historical naked dimensionless thickness weights omit DELTA_REF; no physical-thickness conversion claimed"
    for region in model.models:
        frame = examples(profile, region)
        frame["predicted_T"] = model.predict(frame, region)
        frame.to_csv(folder / "examples" / f"{region}.csv", index=False, float_format="%.17g")
    profile["files_sha256"] = {str(path.relative_to(folder)).replace("\\", "/"): file_hash(path)
                                 for path in sorted(folder.rglob("*")) if path.is_file()}
    write_json(folder / "profile.json", profile)
    check = verify_bundle(folder)
    write_json(folder / "verification.json", check)
    return folder, check


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--keep", type=int, default=3)
    parser.add_argument("--run-id")
    parser.add_argument("--output", type=Path, default=Path("local_reference_versions"))
    args = parser.parse_args()
    if args.keep < 1 or args.keep > 8:
        parser.error("keep must be between 1 and 8")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite a preserved collection: {args.output}")
    source = args.source_root.resolve()
    destination = args.output.resolve()
    if destination == source or source in destination.parents:
        raise ValueError("Preserved collection must be outside the read-only historical root")
    cfg = Config(base=Path(__file__).resolve().parents[1])
    cfg.output.artifacts = "artifacts"
    cfg.training.device = "cpu"
    cfg.training.dtype = "float32"
    with new_run(cfg, args.run_id, "local top-three legacy inference selection; user accepts usable code/models without exact historical metrics",
                 "one common historical outer population; not held out") as (run, manifest):
        manifest["curation_tool_sha256"] = file_hash(__file__)
        initialize(cfg.training)
        inventory = []
        for path in sorted(source.rglob("*")):
            if path.is_file() and path.suffix.lower() in (".pt", ".pth"):
                inventory.append({"path": str(path.relative_to(source)), "bytes": path.stat().st_size})
        write_json(run / "weight_inventory.json", inventory)
        frames = {}
        source_data = source / "PINN_Inverse_fin_pls" / "Dataset"
        (run / "inputs").mkdir()
        for region, filename in zip(("upper", "lower", "rib"),
                                    ("T_d3_outer_long.csv", "T_d1_outer_long.csv", "T_d2_outer_long_with_rib.csv")):
            path = source_data / filename
            original = pd.read_csv(path)
            required = ["r", "theta", "t", "True_T"] + (["rib_id"] if region == "rib" else [])
            if set(required) - set(original) or not np.isfinite(original[required].to_numpy()).all():
                raise ValueError(f"Invalid common comparison table: {path}")
            frame = original[required].copy()
            frame["source_row"] = np.arange(len(frame))
            # Intersection of audited historical rib geometries, fixed before ranking.
            if region == "rib":
                frame = frame[(frame.r >= .123) & (frame.r <= .225)]
            frame = frame.reset_index(drop=True)
            frames[region] = frame
            frame.to_csv(run / "inputs" / f"{region}.csv", index=False, float_format="%.17g")
            manifest["input_sha256"][f"original_{region}"] = file_hash(path)
            manifest["input_sha256"][f"common_{region}"] = file_hash(run / "inputs" / f"{region}.csv")
        scope = {"ranking": "aggregate RMSE_K on exactly the same rows, smaller first; no retraining",
                 "counts": {r: len(f) for r, f in frames.items()}, "rib_common_radius_m": [.123, .225],
                 "scope": "Audited complete upper/lower/rib temperature sets only, no global optimum claim",
                 "held_out": False,
                 "excluded": [
                     {"family": "PINN_Inverse_fin_pls/models_v2 and models_test_v1.1", "reason": "Duplicate temperature weights; training prepare_obs_1d order conflicts with feature/output transforms"},
                     {"family": "InversePINN_Again/models_v1.1", "reason": "Training v2 time square-root transform differs from existing v2 predictor; exact source revision not saved"},
                     {"family": "InversePINN_Again/model_v3.1/*_final", "reason": "Same three state_dict hashes as models_v3.1/*_final; duplicate"},
                     {"family": "remaining two-ring checkpoints", "reason": "Missing a compatible complete rib set"},
                     {"family": "LetsFuckingGo and InversePINN POD/MoE/other inverse models", "reason": "Different simplified domain or inverse task; not comparable to two-ring/13-rib profile"}
                 ]}
        records, eligible = [], {}
        for spec in candidates(source):
            try:
                model, factory, hashes = audited_model(spec)
                regional, all_true, all_pred = {}, [], []
                for region, frame in frames.items():
                    pred = model.predict(frame, region)
                    regional[region] = compute_metrics(frame.True_T.to_numpy(), pred)
                    all_true.append(frame.True_T.to_numpy())
                    all_pred.append(pred)
                metrics = {"regions": regional, "overall": compute_metrics(np.concatenate(all_true), np.concatenate(all_pred))}
                record = {"version_id": spec["version_id"], "status": "usable", "metrics": metrics,
                          "factory_source": str(spec["source"] / spec["factory_source"]), "weights": hashes,
                          "factory_source_sha256": file_hash(spec["source"] / spec["factory_source"])}
                eligible[spec["version_id"]] = spec, model, factory, hashes, metrics
                print(f"{spec['version_id']}: {metrics['overall']['rmse_k']:.6f} K; strict three-model load and finite inference")
            except (OSError, ValueError, RuntimeError, FloatingPointError) as error:
                record = {"version_id": spec["version_id"], "status": "unusable", "error": f"{type(error).__name__}: {error}"}
                print(f"{spec['version_id']}: {record['error']}")
            records.append(record)
            write_json(run / "candidate_results.json", records)
        ranked = sorted((record for record in records if record["status"] == "usable"),
                        key=lambda record: record["metrics"]["overall"]["rmse_k"])
        if len(ranked) < args.keep:
            raise RuntimeError(f"Only {len(ranked)} complete usable versions; cannot retain {args.keep}")
        destination.mkdir(parents=True, exist_ok=False)
        selections = []
        for rank, record in enumerate(ranked[:args.keep], 1):
            spec, model, factory, hashes, metrics = eligible[record["version_id"]]
            folder, check = preserve(spec, model, factory, hashes, metrics, destination)
            selections.append({"rank": rank, "version_id": record["version_id"], "bundle": folder.name,
                               "profile_sha256": file_hash(folder / "profile.json"),
                               "rmse_k": metrics["overall"]["rmse_k"], "verification": check})
        collection = {"format_version": 1, "run_id": run.name, "scope": scope, "selected": selections,
                      "local_only": True, "source_root": str(source), "source_modified": False,
                      "ranking_evidence": str(run), "curation_tool_sha256": file_hash(__file__),
                      "uv_lock_sha256": manifest["uv_lock_sha256"], "source_sha256": manifest["source_sha256"]}
        write_json(destination / "index.json", collection)
        pd.DataFrame([{"rank": row["rank"], "version_id": row["version_id"], "rmse_k": row["rmse_k"],
                       "profile": f"{row['bundle']}/profile.json", "verified": True} for row in selections]).to_csv(
                           destination / "index.csv", index=False)
        write_json(run / "metrics.json", {"selection_scope": scope, "candidates": records, "selected": selections})
        manifest["has_metrics"] = True
        manifest["preserved_collection"] = str(destination)
        manifest["collection_index_sha256"] = file_hash(destination / "index.json")
        print(json.dumps([{key: row[key] for key in ("rank", "version_id", "rmse_k")} for row in selections], indent=2))
        print(f"Preserved {len(selections)} verified versions in {destination}")


if __name__ == "__main__":
    main()
