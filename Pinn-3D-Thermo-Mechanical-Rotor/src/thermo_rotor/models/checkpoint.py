"""Versioned weights plus immutable model/profile/gauge binding."""
from pathlib import Path
import torch
from ..config import Profile
from ..training.run import sha256, write_json
from .networks import model_from_spec

FORMAT_VERSION = 1


def save(path, model, spec, profile, *, gauge=None, source=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.with_suffix(".json").exists():
        raise FileExistsError("Checkpoint already exists")
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise FloatingPointError("Non-finite checkpoint weights")
    torch.save({"format_version": FORMAT_VERSION, "state_dict": state}, path)
    write_json(path.with_suffix(".json"), dict(format_version=FORMAT_VERSION, spec=spec,
               profile=profile.to_dict(), gauge=gauge or {"mode": "raw"}, source=source,
               weights_sha256=sha256(path), input_units="m,s,rad", output_units="K or m"))


def load(path, device="cpu", expected_profile=None, expected_kind=None):
    import json
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    if meta["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported checkpoint format version")
    if sha256(path) != meta["weights_sha256"]:
        raise ValueError("Checkpoint weight hash mismatch")
    p = Profile(**meta["profile"])
    if expected_profile is not None and p.to_dict() != expected_profile.to_dict():
        raise ValueError("Checkpoint physical profile mismatch")
    if expected_kind is not None and meta["spec"]["kind"] != expected_kind:
        raise ValueError("Checkpoint model kind mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["format_version"] != FORMAT_VERSION:
        raise ValueError("Checkpoint payload version mismatch")
    model = model_from_spec(meta["spec"])
    state = payload["state_dict"]
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Non-finite model weights")
    dtype = next(iter(state.values())).dtype
    model.to(dtype=dtype).load_state_dict(state, strict=True)
    return model.to(device).eval(), p, meta


def legacy_state(path, spec):
    """Only tensor/native metadata pickle types are accepted; archived scripts are not executed."""
    c = torch.load(path, map_location="cpu", weights_only=True)
    state = c.get("model_state_dict", c.get("state_dict", c))
    model = model_from_spec(spec)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as e:
        raise ValueError(f"Historical network/weights mismatch: {Path(path).name}") from e
    if not all(torch.isfinite(t).all() for t in state.values()):
        raise ValueError("Historical weights contain non-finite values")
    return model.eval(), c
