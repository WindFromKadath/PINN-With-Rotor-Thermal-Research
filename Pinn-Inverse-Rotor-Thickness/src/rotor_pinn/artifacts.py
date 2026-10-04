"""Independent run folders and reproducible, privacy-conscious provenance."""

from contextlib import contextmanager, redirect_stdout, redirect_stderr
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import re
import sys
import traceback
import uuid


EXPERIMENT_FIELDS = ["run_id", "hypothesis_id", "mode", "code_revision", "data_version",
                     "config_path", "seed", "environment_ref", "status", "metrics_path",
                     "artifacts_path", "notes"]


class _Tee:
    def __init__(self, original, log):
        self.original, self.log = original, log

    def write(self, value):
        self.log.write(value)
        self.log.flush()
        return self.original.write(value)

    def flush(self):
        self.log.flush()
        self.original.flush()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False),
                          encoding="utf-8")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_hash():
    root = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ("torch", "deepxde", "numpy", "pandas", "scipy", "matplotlib"):
        packages[name] = importlib.metadata.version(name)
    import torch
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "cuda_runtime": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}


def record_run(root, manifest):
    index = root / "experiments.csv"
    rows = []
    if index.exists():
        with index.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
    run_id = manifest["run_id"]
    if any(row["run_id"] == run_id for row in rows):
        raise FileExistsError(f"Experiment id already recorded: {run_id}")
    relative = f"artifacts/{run_id}"
    rows.append({"run_id": run_id, "hypothesis_id": "", "mode": "exploratory",
                 "code_revision": "sha256:" + manifest["source_sha256"],
                 "data_version": manifest["data_kind"], "config_path": relative + "/config.json",
                 "seed": manifest["seed"], "environment_ref": relative + "/environment.json",
                 "status": manifest["status"],
                 "metrics_path": relative + "/metrics.json" if manifest.get("has_metrics") else "",
                 "artifacts_path": relative,
                 "notes": manifest["purpose"] + "; not evidence of COMSOL accuracy"})
    with index.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=EXPERIMENT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


@contextmanager
def new_run(config, run_id, purpose, data_kind):
    root = config.path(config.output.artifacts)
    root.mkdir(parents=True, exist_ok=True)
    run_id = run_id or (datetime.now(timezone.utc).strftime("R%Y%m%dT%H%M%SZ_") + uuid.uuid4().hex[:6])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", run_id):
        raise ValueError("Run id must be a short ASCII name without path separators")
    directory = root / run_id
    directory.mkdir(exist_ok=False)
    manifest = {"format_version": 1, "run_id": run_id, "purpose": purpose,
                "data_kind": data_kind, "status": "running", "seed": config.training.seed,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "source_sha256": source_hash(), "input_sha256": {}, "has_metrics": False}
    lock = root.parent / "uv.lock"
    manifest["uv_lock_sha256"] = file_hash(lock) if lock.exists() else None
    write_json(directory / "manifest.json", manifest)
    try:
        write_json(directory / "config.json", config.snapshot())
        manifest["config_sha256"] = file_hash(directory / "config.json")
        write_json(directory / "environment.json", environment())
        with (directory / "console.log").open("w", encoding="utf-8") as log:
            with redirect_stdout(_Tee(sys.stdout, log)), redirect_stderr(_Tee(sys.stderr, log)):
                yield directory, manifest
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = f"{type(error).__name__}: {error}"
        (directory / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    else:
        manifest["status"] = "completed"
    finally:
        manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_json(directory / "manifest.json", manifest)
        record_run(root.parent, manifest)
