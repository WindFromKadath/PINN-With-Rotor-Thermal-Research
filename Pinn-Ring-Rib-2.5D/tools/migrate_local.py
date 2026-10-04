"""One-time local import. Source trees are read-only and every copy is hashed."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError("Source and destination must be independent trees")
    for folder in ("src/ring_rib_pinn", "tests", "tools", "configs", "data/raw",
                   "data/processed", "artifacts", "local_reference_versions/source"):
        (out / folder).mkdir(parents=True, exist_ok=True)
    manifest = []
    excluded = {".git", ".venv", "__pycache__", ".vscode", "pinn-ring-rib-2.5d"}
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if not path.is_file() or excluded.intersection(relative.parts):
            continue
        manifest.append({"source_path": relative.as_posix(), "size_bytes": path.stat().st_size,
                         "sha256": digest(path), "copied_to": ""})
    by_source = {row["source_path"]: row for row in manifest}

    def copy(relative, destination):
        original, target = source / relative, out / destination
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and digest(target) != digest(original):
            raise FileExistsError(f"Refusing to overwrite a different imported file: {destination}")
        if not target.exists():
            shutil.copy2(original, target)
        if digest(original) != digest(target):
            raise RuntimeError(f"Copy hash mismatch: {relative}")
        row = by_source.get(relative)
        if row is not None:
            row["copied_to"] = ";".join(filter(None, [row["copied_to"], destination]))
        return digest(target)

    # Preserve original source code/documents as local evidence, never execute scripts wholesale.
    for row in manifest:
        rel = row["source_path"]
        if Path(rel).suffix.lower() in (".py", ".md", ".log"):
            copy(rel, f"local_reference_versions/source/{rel}")
    for case, folder, suffix in (("heating", "Dataset", ""),
                                ("cooling", "Dataset_copy", "_colding")):
        for domain, number in (("upper", 3), ("lower", 1), ("rib", 2)):
            label = "_auto_labeled" if domain == "rib" else ""
            filename = f"domain_data_{number}{suffix}{label}.csv"
            copy(f"Re-PINN/{folder}/{filename}", f"data/raw/{case}/{domain}_domain.csv")
            copy(f"Re-PINN/{folder}/{filename}", f"data/processed/{case}/{domain}_domain.csv")
    for domain, number in (("upper", 3), ("lower", 1)):
        copy(f"Re-PINN/Dataset/T_d{number}_outer_long.csv",
             f"data/processed/heating/{domain}_surface.csv")
    copy("Re-PINN/Dataset/rib_models.json", "data/processed/rib_models.json")
    for row in list(manifest):
        rel = row["source_path"]
        if rel.startswith("lestfuckinggo/Data_process/UnprocessData") and rel.endswith(".csv"):
            copy(rel, f"data/raw/comsol/{rel.split('Data_process/', 1)[1]}")
    for domain, number in (("upper", 3), ("lower", 1)):
        copy(f"ThicknesNet/Dataset/thickness_d{number}_1_processed.csv",
             f"data/raw/thickness/{domain}.csv")

    profiles = [
        ("heating_coupled", "heating", "models_heating", "pinn_v6.5.py",
         ["upper_ring_flux_coupled_lam.pth", "lower_ring_flux_coupled_lam.pth",
          "rib_net_flux_coupled_lam.pth"]),
        ("cooling_hardconstraint", "cooling", "models_colding_linux",
         "pinn_v6_colding_hardcore.py",
         ["upper_ring_hardconstraint.pth", "lower_ring_hardconstraint.pth",
          "rib_hardconstraint.pth"]),
    ]
    index = []
    for bundle, case, folder, script, files in profiles:
        hashes, weights, thickness = {}, {}, {}
        for domain, filename in zip(("upper", "lower", "rib"), files):
            relative = f"{bundle}/weights/{domain}.pth"
            hashes[f"weights/{domain}.pth"] = copy(f"Re-PINN/{folder}/{filename}",
                                                    f"local_reference_versions/{relative}")
            weights[domain] = f"weights/{domain}.pth"
        for domain in ("upper", "lower"):
            hashes[f"weights/{domain}_thickness.pth"] = copy(
                f"Re-PINN/ring_thickness/{domain}_thickness_oscillation_reg.pth",
                f"local_reference_versions/{bundle}/weights/{domain}_thickness.pth")
            thickness[domain] = f"weights/{domain}_thickness.pth"
        hashes["source.py"] = copy(f"Re-PINN/{script}",
                                     f"local_reference_versions/{bundle}/source.py")
        hashes["rib_models.json"] = copy("Re-PINN/Dataset/rib_models.json",
                                          f"local_reference_versions/{bundle}/rib_models.json")
        write_json(out / f"local_reference_versions/{bundle}/profile.json", {
            "format_version": 1, "case": case, "source_script": f"Re-PINN/{script}",
            "weights": weights, "thickness": thickness, "files_sha256": hashes,
            "radius_scale_m": 0.225, "temperature_scale_K": 900.0,
            "ambient_K": 293.15, "alpha_ref_m2_s": 15.1 / (7850 * 460),
            "input_order": {"ring": ["r_star", "theta", "t_star"],
                            "rib": ["s_star", "rib_id", "t_star"]},
            "status": "copied_hash_verified_prediction_not_yet_validated"})
        index.append({"bundle_id": bundle, "case": case,
                      "profile": f"{bundle}/profile.json", "source": f"Re-PINN/{script}"})
    for name, case, script in (
        ("ntk_archive", "heating", "Re-PINN/pinn_no_rib_data_v6.5_NTK.py"),
        ("moe_archive", "geometry", "ThicknesNet/data_surface_train_moe.py"),
        ("soft_moe_archive", "geometry", "ThicknesNet/data_surface_train_moe_soft.py"),
        ("learnable_archive", "cooling", "Re-PINN/pinn_mse_learnable_colding.py"),
        ("surface_correction_archive", "heating", "Re-PINN/pinn_no_rib_data_v7_fixed.py")):
        index.append({"bundle_id": name, "case": case, "profile": "", "source": script})
    for filename, rows, fields in (
        ("source_manifest.csv", manifest, ["source_path", "size_bytes", "sha256", "copied_to"]),
        ("index.csv", index, ["bundle_id", "case", "profile", "source"])):
        with (out / "local_reference_versions" / filename).open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    old = source / "pinn-ring-rib-2.5d"
    for filename in ("hypotheses.csv", "experiments.csv"):
        destination = out / filename
        if not destination.exists():
            shutil.copy2(old / filename, destination)
    agent = out / "AGENTS.md"
    if not agent.exists():
        shutil.copy2(args.template / "AGENTS.md", agent)
    print(json.dumps({"indexed_files": len(manifest), "bundles": [r[0] for r in profiles],
                      "copied_files": sum(bool(r["copied_to"]) for r in manifest)}))


if __name__ == "__main__":
    main()
