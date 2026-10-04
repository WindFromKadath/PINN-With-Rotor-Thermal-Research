"""Load converted local historical models using their original transforms."""
import json
from pathlib import Path
import torch
from .fno import HeatOperatorFNO, HeatOperatorFNO3D
from .deepxde_models import make_deeponet_cartesian, make_fnn, PeriodicFeatureTransform
from heatops.training.run_context import file_hash


def load_legacy_model(package, model_name=None):
    root = Path(package)
    spec = json.loads((root / "model_spec.json").read_text(encoding="utf-8"))
    if spec["format_version"] != 1:
        raise ValueError("不支持的历史模型包格式")
    record = spec["models"][model_name or next(iter(spec["models"]))]
    path = root / record["checkpoint"]
    if file_hash(path) != record["sha256"]:
        raise ValueError("模型包权重哈希不匹配")
    family, mc = spec["family"], record["network"]
    if family == "fno_1d":
        net = HeatOperatorFNO(tuple(mc["n_modes"]), mc["hidden_channels"], use_ansatz=True, a=float(mc["a"]))
    elif family == "fno_polar":
        net = HeatOperatorFNO3D(tuple(mc["n_modes"]), mc["hidden_channels"])
    elif family == "pidon":
        net = make_deeponet_cartesian(mc["branch_layers"], mc["trunk_layers"])
    elif family == "pinn_polar":
        net = make_fnn(mc["layer_sizes"], feature_transform=PeriodicFeatureTransform(1) if mc.get("periodic_features") else None)
    else:
        raise ValueError(f"未知模型类型：{family}")
    bundle = torch.load(path, map_location="cpu", weights_only=True)
    if bundle["format_version"] != 1 or bundle["family"] != family:
        raise ValueError("权重格式与模型定义不匹配")
    net.load_state_dict(bundle["model_state_dict"], strict=True)
    net.eval()
    return net, spec, record
