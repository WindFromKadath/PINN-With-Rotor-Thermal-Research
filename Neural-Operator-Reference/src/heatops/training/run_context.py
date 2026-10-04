"""Independent run directories, portable provenance, logs, and failure records."""
from __future__ import annotations
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
from pathlib import Path
import platform
import re
import sys
import traceback
from datetime import datetime, timezone
import importlib.metadata as md

_active: ContextVar[Path | None] = ContextVar("heatops_active_run", default=None)


def project_root(config_path):
    parent = Path(config_path).resolve().parent
    return parent.parent if parent.name == "configs" else parent


def config_path_value(config_path, value):
    p = Path(value)
    return p if p.is_absolute() else (Path(config_path).resolve().parent / p).resolve()


def run_directory(config_path, run_id, subdir=None):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(run_id)) or run_id in (".", ".."):
        raise ValueError("非法运行编号：只能含英文字母、数字、下划线、点和连字符")
    rd = project_root(config_path) / "artifacts" / run_id
    return rd / subdir if subdir else rd


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def prepare_run(run_dir, cfg, config_path):
    rd = Path(run_dir)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", str(cfg.get("experiment", rd.name))):
        raise ValueError("运行编号只能含英文字母、数字、下划线、点和连字符")
    try:
        rd.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        raise FileExistsError(f"运行目录已存在，拒绝覆盖：{rd.name}；请使用新的 --run-id") from None
    _active.set(rd)
    versions = {}
    for package in ("heatops", "torch", "neuraloperator", "deepxde", "numpy", "scipy", "pyyaml"):
        try:
            versions[package] = md.version(package)
        except md.PackageNotFoundError:
            versions[package] = "not-installed"
    source_root = Path(__file__).resolve().parents[1]
    manifest = {str(p.relative_to(source_root)).replace("\\", "/"): file_hash(p)
                for p in sorted(source_root.rglob("*.py"))}
    write_json(rd / "run.json", {"status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "config_file": Path(config_path).name, "seed": int(cfg["seed"]), "configuration": cfg,
        "python": platform.python_version(), "platform": platform.system(), "environment": versions,
        "source_sha256": manifest})
    return rd


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


def bind_data(path):
    rd = _active.get()
    if rd is not None:
        manifest = rd / "run.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.setdefault("data_sha256", {})[Path(path).name] = file_hash(path)
        write_json(manifest, payload)


class _Log:
    def __init__(self, stream):
        self.stream = stream
    def write(self, message):
        self.stream.write(message)
        rd = _active.get()
        if rd is not None:
            with (rd / "console.log").open("a", encoding="utf-8") as f:
                f.write(message)
    def flush(self):
        self.stream.flush()
    def __getattr__(self, name):
        return getattr(self.stream, name)


def managed_cli(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        token = _active.set(None)
        stdout = sys.stdout
        sys.stdout = _Log(stdout)
        try:
            result = fn(*args, **kwargs)
            rd = _active.get()
            if rd is not None:
                payload = json.loads((rd / "run.json").read_text(encoding="utf-8"))
                payload.update(status="dry_run_data_missing" if (rd / "dry_run_report.json").exists() else "completed",
                               finished_at=datetime.now(timezone.utc).isoformat(), exit_code=result)
                write_json(rd / "run.json", payload)
            return result
        except BaseException as exc:
            rd = _active.get()
            if rd is not None:
                payload = json.loads((rd / "run.json").read_text(encoding="utf-8"))
                payload.update(status="failed", error_type=type(exc).__name__, error=str(exc))
                write_json(rd / "run.json", payload)
                (rd / "failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        finally:
            sys.stdout = stdout
            _active.reset(token)
    return wrapped
