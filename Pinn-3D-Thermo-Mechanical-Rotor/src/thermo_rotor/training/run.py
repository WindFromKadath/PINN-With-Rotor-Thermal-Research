"""Exclusive run directories, provenance and failure retention."""
from pathlib import Path
import hashlib
import json
import platform
import importlib.metadata
import time
import traceback


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


class Run:
    def __init__(self, root, run_id, config, command):
        import re
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", run_id):
            raise ValueError("run_id must be a simple safe identifier")
        self.path = Path(root)/run_id
        self.path.mkdir(parents=True, exist_ok=False)
        self.config = config
        self.command = command
        self.started = time.time()
        write_json(self.path/"config.json", config)
        self._status("running")
        env = {"python": platform.python_version(), "platform": platform.platform()}
        env["packages"] = {v: importlib.metadata.version(v) for v in
                            ("torch", "numpy", "pandas", "scipy", "matplotlib", "trimesh", "rtree")}
        write_json(self.path/"environment.json", env)
        source = Path(__file__).resolve().parents[1]
        code_hash = {f.relative_to(source).as_posix(): sha256(f) for f in sorted(source.rglob("*.py"))}
        project = source.parents[1]
        inputs = {}
        for key, value in config.get("paths", {}).items():
            p = Path(value)
            if p.is_file():
                inputs[key] = sha256(p)
        write_json(self.path/"manifest.json", dict(command=command, code_version="thermo-rotor-0.1.0",
                    code_sha256=code_hash, data_sha256=inputs,
                    uv_lock_sha256=sha256(project/"uv.lock") if (project/"uv.lock").exists() else None,
                    seed=config.get("runtime", {}).get("seed", 42), git_actions=[]))

    def _status(self, status, **extra):
        write_json(self.path/"status.json", dict(status=status, command=self.command,
                   elapsed_seconds=time.time()-self.started, **extra))

    def __enter__(self):
        return self

    def __exit__(self, typ, value, tb):
        if typ:
            (self.path/"failure.txt").write_text("".join(traceback.format_exception(typ, value, tb)), encoding="utf-8")
            self._status("failed", error_type=typ.__name__, error=str(value))
        else:
            self._status("completed")
        return False
