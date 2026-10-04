"""Read-only source migration. Pass the complete outer source directory explicitly."""
from pathlib import Path
import argparse
import hashlib
import shutil
import csv
import json


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    src = args.source_root.resolve()
    root = Path(__file__).resolve().parents[1]
    archive = root/"local_reference_versions"
    manifest = archive/"source_manifest.csv"
    if manifest.exists():
        raise FileExistsError("Source manifest already exists; migration is not overwritten")
    rows = []
    selected_data = {"HeatPINNCheck.csv", "HeatPINNCheck_long.csv", "ElsPINNTrain.csv", "ElsPINNCheck_long.csv"}
    little_weights = {
        "training/models/best_model.pth": "little/thermal.pth",
        "mechanical_transient/models/best_model_transient.pth": "little/mechanical.pth",
        "experiment_mlp_1100_unified/best_model.pth": "mlp_1100/mechanical.pth",
        "experiment_mlp_33k_unified/best_model.pth": "mlp_33k/mechanical.pth",
        "iwwg_comparison/constraint/best_model.pth": "iwg_constraint/mechanical.pth",
        "iwwg_comparison/projection/best_model.pth": "iwg_projection/mechanical.pth",
    }
    fxx_weights = {"training/models/best_model.pth": "fxx/thermal.pth",
                   "mech_transient/models/best_model_transient.pth": "fxx/mechanical.pth",
                   "mech_transient_ntk/models/best_model_transient.pth": "fxx_ntk/mechanical.pth"}
    for path in sorted(src.rglob("*")):
        if not path.is_file() or any(v in path.parts for v in (".git", ".venv", "__pycache__", ".cache")):
            continue
        rel = path.relative_to(src).as_posix()
        target = None
        if path.suffix.lower() in (".py", ".yaml", ".toml", ".md", ".tex", ".txt"):
            target = archive/"source"/rel
        if rel.startswith("FXXKINGGOD/PINN_transient/Dataset/") and path.name in selected_data:
            target = root/"data/raw"/path.name
        if rel.startswith("LittleTinyTest/IWG_Comparison/data/") and path.suffix == ".csv":
            target = root/"data/raw/iwg"/path.name
        if rel.startswith("LittleTinyTest/STL/"):
            target = root/"data/raw/STL/main"/path.name
        if "/3D_PINN/X_PINN_Spilt_v2/STL/" in rel:
            target = root/"data/raw/STL/xpinn"/rel.split("/STL/", 1)[1]
        if "/3D_PINN/PINN_FrameWork_Full/STL/" in rel:
            target = root/"data/raw/STL/formal"/rel.split("/STL/", 1)[1]
        if rel.startswith("LittleTinyTest/output/"):
            chosen = little_weights.get(rel.removeprefix("LittleTinyTest/output/"))
            if chosen:
                target = archive/"packages"/chosen
        if rel.startswith("FXXKINGGOD/PINN_transient/output/"):
            chosen = fxx_weights.get(rel.removeprefix("FXXKINGGOD/PINN_transient/output/"))
            if chosen:
                target = archive/"packages"/chosen
        if path.suffix == ".json" and "output" in path.parts:
            target = archive/"historical_metrics"/rel
        hashed = digest(path)
        copied = ""
        if target is not None:
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and digest(target) != hashed:
                raise FileExistsError(f"Different migration destination: {target.name}")
            if not target.exists():
                shutil.copy2(path, target)
            if digest(target) != hashed:
                raise ValueError(f"Copied hash mismatch: {rel}")
            copied = target.relative_to(root).as_posix()
        rows.append(dict(source=rel, size=path.stat().st_size, sha256=hashed, copied_to=copied))
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=("source", "size", "sha256", "copied_to"))
        writer.writeheader()
        writer.writerows(rows)
    report = dict(source_files=len(rows), copied_files=sum(bool(r["copied_to"]) for r in rows), hash_mismatches=[])
    (archive/"migration.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
