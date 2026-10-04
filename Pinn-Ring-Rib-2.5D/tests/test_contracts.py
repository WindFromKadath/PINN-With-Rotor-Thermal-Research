import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from ring_rib_pinn.config import Config
from ring_rib_pinn.data import convert_table, normalized_inputs, read_table, validate_frame
from ring_rib_pinn.checkpoints import load_checkpoint, save_checkpoint
from ring_rib_pinn.geometry import interpolate
from ring_rib_pinn.networks import initial_temperature


def test_imports_have_no_io(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Module import performed file/model I/O")
    monkeypatch.setattr(Path, "read_text", fail)
    monkeypatch.setattr(torch, "load", fail)
    for name in ("training", "cli", "checkpoints", "plotting"):
        importlib.reload(importlib.import_module(f"ring_rib_pinn.{name}"))


def test_si_roundtrip_and_periodicity(bundle):
    frame = pd.DataFrame({"r": [.15, .2], "theta": [.2, .5], "t": [10., 100.], "True_T": [400., 600.]})
    x = normalized_inputs(frame, bundle.config, bundle.geometry, "upper")
    np.testing.assert_allclose(x[:, 0]*.225, frame.r)
    np.testing.assert_allclose(x[:, 2]*.225**2/bundle.config.alpha, frame.t)
    a = torch.tensor(x, dtype=torch.float32)
    b = a.clone()
    b[:, 1] += 2*np.pi
    with torch.no_grad():
        torch.testing.assert_close(bundle.networks["upper"](a), bundle.networks["upper"](b),
                                   rtol=1e-5, atol=1e-6)


def test_arc_mapping_and_thickness_derivative(bundle):
    geo = bundle.geometry
    r = torch.tensor([.14, .18, .21])/.225
    s = interpolate(r, geo.r_grid, geo.s_grid)
    np.testing.assert_allclose(s.numpy(), [geo.arc(v*.225)/.225 for v in r.numpy()], rtol=2e-5)
    back = interpolate(s, geo.s_grid, geo.r_grid)
    torch.testing.assert_close(back, r, rtol=1e-5, atol=1e-6)
    x = torch.tensor([[.7, .8]], requires_grad=True)
    h = bundle.thickness["upper"](x[:, :1], x[:, 1:2])
    gradient = torch.autograd.grad(h.sum(), x)[0]
    assert torch.isfinite(gradient).all() and gradient.abs().max() > 0
    assert all(not p.requires_grad for p in bundle.thickness["upper"].parameters())


def test_initial_condition(bundle):
    for domain in ("upper", "lower", "rib"):
        x = torch.tensor([[.65, 1., 0.], [.75, 4., 0.]])
        if domain == "rib":
            r, _ = bundle.geometry.rib_coordinates(x[:, :1], x[:, 1:2])
        else:
            r = x[:, :1]
        expected = initial_temperature(r, bundle.config) if bundle.config.case == "cooling" else torch.zeros_like(r)
        with torch.no_grad():
            torch.testing.assert_close(bundle.networks[domain](x), expected)


def test_data_temperature_contract():
    f = pd.DataFrame({"r": [.15], "theta": [0.], "t": [0.], "T": [4.], "thickness": [.01], "True_T": [400.]})
    assert convert_table(f, representation="integral").True_T.iloc[0] == 400.
    assert convert_table(f.drop(columns="True_T"), representation="integral").True_T.iloc[0] == 400.
    raw = pd.DataFrame({"x": [150.], "y": [0.], "z": [0.], "thickness": [10.],
                        "T_d1 (m*K) @ t=0": [4.], "T_d1 (m*K) @ t=10": [5.]})
    result = convert_table(raw, length_unit="mm", thickness_unit="mm", representation="integral")
    np.testing.assert_allclose(result.True_T, [400, 500])
    np.testing.assert_allclose(result.r, .15)


@pytest.mark.parametrize("field,value,match", [("r", 150., "meters"), ("theta", 360., "radians"),
    ("thickness", 0., "positive"), ("True_T", float("nan"), "finite"),
    ("rib_id", 13., "0..12"), ("rib_id", 1.5, "integer")])
def test_reject_invalid_data(field, value, match):
    f = pd.DataFrame({"r": [.15], "theta": [0.], "t": [0.], "True_T": [400.],
                      "thickness": [.01], "rib_id": [0]})
    f[field] = value
    with pytest.raises(ValueError, match=match):
        validate_frame(f, Config(), "rib")


def test_missing_data_and_invalid_units(tmp_path):
    with pytest.raises(FileNotFoundError, match="Missing data"):
        read_table(tmp_path/"missing.csv")
    with pytest.raises(ValueError, match="units"):
        convert_table(pd.DataFrame(), length_unit="cm")


def test_mixed_metadata_encoding(tmp_path):
    path = tmp_path/"comsol.csv"
    path.write_bytes(b"% Model,\xD6\xD0\xCE\xC4\n%x,y,z,r,theta,thickness\n0.15,0,0,0.15,0,0.01\n")
    f = read_table(path)
    assert f.thickness.iloc[0] == .01


def test_checkpoint_group_and_mismatch(bundle, tmp_path):
    path = tmp_path/"model.pt"
    save_checkpoint(path, bundle.config, bundle.geometry, bundle.networks, bundle.thickness)
    _, _, nets, _ = load_checkpoint(path)
    x = torch.tensor([[.7, 1., .003]])
    for domain, net in nets.items():
        torch.testing.assert_close(net(x), bundle.networks[domain](x), rtol=0, atol=0)
    changed = bundle.config.serializable()
    changed["physics"]["temperature_scale"] = 901.
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(path, expected_config=Config(changed))
    torch.save({"format": "bad"}, tmp_path/"bad.pt")
    with pytest.raises(ValueError, match="format"):
        load_checkpoint(tmp_path/"bad.pt")


def test_run_exclusivity_and_failure(tmp_path, monkeypatch):
    import ring_rib_pinn.artifacts as artifacts
    monkeypatch.setattr(artifacts, "register_run", lambda *args, **kwargs: None)
    cfg = Config({"output": {"root": str(tmp_path)}})
    directory = artifacts.new_run(cfg, "test-run", "test")
    with pytest.raises(FileExistsError):
        artifacts.new_run(cfg, "test-run", "test")
    artifacts.finish_run(cfg, directory, FloatingPointError("Non-finite loss"))
    status = json.loads((directory/"status.json").read_text())
    assert status["status"] == "failed"


def test_nonfinite_training_loss_rejected():
    from types import SimpleNamespace
    from ring_rib_pinn.backend import initialize
    from ring_rib_pinn.training import train_stage
    cfg = Config()
    dde, _, _ = initialize(cfg)

    class BrokenModel:
        net = torch.nn.Linear(3, 1)
        train_state = SimpleNamespace(loss_train=np.array([float("nan")]))

        def compile(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            callback = kwargs["callbacks"][0]
            callback.set_model(self)
            callback.on_batch_end()

    with pytest.raises(FloatingPointError, match="Non-finite"):
        train_stage(dde, BrokenModel(), cfg, "upper", "independent", 1, [])
