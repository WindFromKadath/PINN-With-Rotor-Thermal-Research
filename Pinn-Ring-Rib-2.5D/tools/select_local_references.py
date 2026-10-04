"""Bounded screening and local retention of complete mature reference versions."""

import argparse
import csv
import json
from pathlib import Path
import shutil

from ring_rib_pinn.artifacts import file_hash, new_run, finish_run, write_json
from ring_rib_pinn.config import Config
from ring_rib_pinn.data import read_table
from ring_rib_pinn.evaluation import compute_metrics
from ring_rib_pinn.reference import SourceReference, FORMAT

ROOT = Path(__file__).resolve().parents[1]
SPECS = [
    ("heating_v65", "heating", "models_heating", "pinn_v6.5.py", "standard",
     ["upper_ring_flux_coupled_lam.pth", "lower_ring_flux_coupled_lam.pth", "rib_net_flux_coupled_lam.pth"], False, None),
    ("heating_v64", "heating", "models_heating", "pinn_v6.4.py", "standard",
     ["upper_ring_flux_coupled_lambda.pth", "lower_ring_flux_coupled_lambda.pth", "rib_net_flux_coupled_lambda.pth"], False, None),
    ("heating_with_data", "heating", "models_learnable_linux", "pinn_mse_v6_variable.py", "standard",
     ["upper_ring_learnable_data_lam.pth", "lower_ring_learnable_data_lam.pth", "rib_net_learnable_data_lam.pth"], True, None),
    ("cooling_hardconstraint", "cooling", "models_colding_linux", "pinn_v6_colding_hardcore.py", "hard_initial",
     ["upper_ring_hardconstraint.pth", "lower_ring_hardconstraint.pth", "rib_hardconstraint.pth"], False, None),
    ("cooling_raw", "cooling", "models_colding_linux", "pinn_mse_v6_colding.py", "standard",
     ["upper_ring_flux_coupled_colding.pth", "lower_ring_flux_coupled_colding.pth", "rib_net_flux_coupled_colding.pth"], False, "pinn_v6_colding.py"),
    ("cooling_with_data", "cooling", "models_learnable_colding", "pinn_mse_learnable_colding.py", "hard_initial",
     ["upper_learnable_colding_final.pth", "lower_learnable_colding_final.pth", "rib_learnable_colding_final.pth"], True, "pinn_v6_colding_with_data.py"),
]


def import_candidate(source, folder, spec):
    name, case, weights_folder, script, factory, files, coefficients, training = spec
    folder.mkdir()
    hashes, sources, weights = {}, {}, {}

    def copy(relative, destination):
        original, target = source/relative, folder/destination
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original, target)
        digest = file_hash(original)
        if file_hash(target) != digest:
            raise RuntimeError(f"Copy differs: {relative}")
        hashes[destination] = digest
        sources[destination] = relative

    copy(f"Re-PINN/{script}", "source.py")
    if training:
        copy(f"Re-PINN/{training}", "training_source.py")
    copy("Re-PINN/Dataset/rib_models.json", "rib_models.json")
    for domain, filename in zip(("upper", "lower", "rib"), files):
        relative = f"weights/{domain}.pth"
        copy(f"Re-PINN/{weights_folder}/{filename}", relative)
        weights[domain] = relative
    for domain in ("upper", "lower"):
        copy(f"Re-PINN/ring_thickness/{domain}_thickness_oscillation_reg.pth", f"weights/{domain}_thickness.pth")
    heads = {domain: ["T_star"] for domain in ("upper", "lower", "rib")}
    if coefficients:
        heads.update(upper=["T_star", "h_multiplier", "emissivity_multiplier"],
                     lower=["T_star", "h_multiplier", "emissivity_multiplier"],
                     rib=["T_star", "contact_resistance_multiplier"])
    write_json(folder/"profile.json", {
        "format": FORMAT, "version": name, "case": case, "factory": factory,
        "weights": weights, "output_heads": heads, "clip_rib_radius": True,
        "files_sha256": hashes, "source_files": sources,
        "scope": "original model inference; additional versions are not retrained",
        "use": "tools/predict_reference.py --bundle this_folder --domain upper/lower/rib --input file.csv --output prediction.csv"})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--sample", type=int, default=512)
    args = parser.parse_args()
    if args.sample < 1:
        raise ValueError("sample must be positive")
    source = args.source.resolve()
    cfg = Config({"output": {"root": str(ROOT/"artifacts")}})
    directory = new_run(cfg, args.run_id, "local_reference_selection")
    try:
        frames = {case: {domain: read_table(ROOT/"data/processed"/case/f"{domain}_domain.csv")
                                .sample(n=args.sample, random_state=42)
                        for domain in ("upper", "lower", "rib")}
                  for case in ("heating", "cooling")}
        for case, points in frames.items():
            for domain, frame in points.items():
                target = directory/"inputs"/case/f"{domain}.csv"
                target.parent.mkdir(parents=True, exist_ok=True)
                frame.to_csv(target, index=False)
        candidates = directory/"candidates"
        candidates.mkdir()
        scores = []
        for spec in SPECS:
            name, case = spec[:2]
            folder = candidates/name
            try:
                import_candidate(source, folder, spec)
                bundle = SourceReference(folder)
                metrics, all_true, all_pred = {}, [], []
                for domain, frame in frames[case].items():
                    predicted = bundle.predict(frame, domain)
                    predicted.to_csv(folder/f"{domain}_screen_prediction.csv", index=False)
                    metrics[domain] = compute_metrics(frame.True_T.to_numpy(), predicted.Pred_T.to_numpy())
                    all_true.extend(frame.True_T)
                    all_pred.extend(predicted.Pred_T)
                metrics["all_points"] = compute_metrics(all_true, all_pred)
                scores.append({"version": name, "case": case, "status": "usable",
                               "rmse_K": metrics["all_points"]["rmse_K"], "metrics": metrics})
            except (RuntimeError, ValueError, FileNotFoundError, FloatingPointError) as error:
                scores.append({"version": name, "case": case, "status": "unavailable", "reason": str(error)})
        selected = []
        for case in ("heating", "cooling"):
            ranked = sorted([s for s in scores if s["case"] == case and s["status"] == "usable"], key=lambda s: s["rmse_K"])
            if len(ranked) < 2:
                raise RuntimeError(f"Fewer than two usable complete references: {case}")
            selected.extend(ranked[:2])
        rows, imported = [], []
        for result in selected:
            name, case = result["version"], result["case"]
            original = candidates/name
            retained = ROOT/"local_reference_versions/selected"/name
            if retained.exists():
                raise FileExistsError(f"Selected version already exists: {name}")
            retained.parent.mkdir(parents=True, exist_ok=True)
            profile = json.loads((original/"profile.json").read_text(encoding="utf-8"))
            shutil.copytree(original, retained)
            profile["selection"] = {"run_id": args.run_id, "sample_per_domain": args.sample,
                                    "seed": 42, "rmse_K": result["rmse_K"],
                                    "criterion": "top two usable versions per case among six complete mature candidates"}
            write_json(retained/"profile.json", profile)
            SourceReference(retained)  # Check the retained copy, not merely the candidate.
            for relative, digest in profile["files_sha256"].items():
                if file_hash(retained/relative) != digest:
                    raise RuntimeError("Retained copy mismatch")
                imported.append((profile["source_files"][relative], (retained/relative).relative_to(ROOT).as_posix()))
            rows.append({"version": name, "case": case, "bundle": retained.relative_to(ROOT).as_posix(),
                         "status": "usable", "selection_run": args.run_id, "sample_rmse_K": result["rmse_K"]})
        index = ROOT/"local_reference_versions/selected_versions.csv"
        with index.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        manifest_path = ROOT/"local_reference_versions/source_manifest.csv"
        with manifest_path.open(encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            fields, manifest = reader.fieldnames, list(reader)
        by_source = {row["source_path"]: row for row in manifest}
        for original, retained in imported:
            row = by_source[original]
            if retained not in row["copied_to"].split(";"):
                row["copied_to"] = ";".join(filter(None, [row["copied_to"], retained]))
        with manifest_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(manifest)
        report = {"completion_criterion": "usable code and models plus retained local references; metric reproduction not required",
                  "seed": 42, "sample_per_domain": args.sample, "candidates": scores, "selected": rows,
                  "scope": "bounded sample screening; not a global optimum claim or precision reproduction",
                  "input_sha256": {f"{case}/{domain}": file_hash(ROOT/"data/processed"/case/f"{domain}_domain.csv")
                                   for case in frames for domain in frames[case]}}
        write_json(directory/"selection.json", report)
        finish_run(cfg, directory, notes="Four usable original versions retained locally; no training or Git upload")
        print(json.dumps({"selected": rows, "candidates": [{k:v for k,v in s.items() if k != "metrics"} for s in scores]}, indent=2))
    except Exception as error:
        finish_run(cfg, directory, error)
        raise


if __name__ == "__main__":
    main()
