"""Exclusive run directories, durable status and portable experiment evidence."""

from datetime import datetime
import csv
import hashlib
import importlib.metadata
import json
from pathlib import Path
import re
import sys
import zipfile


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                    encoding="utf-8")


def code_hash():
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return "sha256:" + digest.hexdigest()


def environment():
    packages = {}
    for name in ("deepxde", "torch", "numpy", "pandas", "scipy", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    return {"python": sys.version.split()[0], "packages": packages}


def register_run(config, directory, status, notes="", metrics=False):
    project = Path(__file__).resolve().parents[2]
    index = project / "experiments.csv"
    if not index.exists():
        return
    with index.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames
        rows = list(reader)
    row = {key: "" for key in fields}
    relative = directory.relative_to(project).as_posix() if directory.is_relative_to(project) else ""
    row.update(run_id=directory.name, hypothesis_id="", mode="exploratory",
               code_revision=code_hash(), data_version="input_sha256 in manifest.json",
               config_path=f"{relative}/config.json" if relative else "",
               seed=str(config.t["seed"]), environment_ref=f"{relative}/environment.json",
               status=status, metrics_path=f"{relative}/metrics.json" if metrics else "",
               artifacts_path=relative, notes=notes)
    rows = [old for old in rows if old["run_id"] != directory.name] + [row]
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def new_run(config, run_id=None, kind="training"):
    run_id = run_id or f"R_{config.case}_{datetime.now():%Y%m%d_%H%M%S_%f}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", run_id):
        raise ValueError("run_id must be a simple unique identifier")
    directory = config.path(config.values["output"]["root"]) / run_id
    directory.mkdir(parents=True, exist_ok=False)
    write_json(directory / "config.json", config.serializable())
    write_json(directory / "environment.json", environment())
    with zipfile.ZipFile(directory / "code.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for file in sorted(Path(__file__).parent.glob("*.py")):
            archive.write(file, "ring_rib_pinn/" + file.name)
        project = Path(__file__).resolve().parents[2]
        for filename in ("pyproject.toml", "uv.lock", ".python-version"):
            if (project/filename).exists():
                archive.write(project/filename, filename)
    write_json(directory / "status.json", {"status": "running", "kind": kind})
    register_run(config, directory, "running", kind)
    return directory


def finish_run(config, directory, error=None, notes=""):
    status = "failed" if error else "completed"
    previous = json.loads((directory/"status.json").read_text(encoding="utf-8"))
    previous.update(status=status, error=str(error) if error else None)
    write_json(directory / "status.json", previous)
    register_run(config, directory, status, notes if not error else str(error),
                 (directory / "metrics.json").exists())
