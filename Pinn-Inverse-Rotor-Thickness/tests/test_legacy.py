import json

import numpy as np
import pandas as pd
import pytest
import torch

from rotor_pinn.artifacts import file_hash, write_json
from rotor_pinn.legacy import LegacyBundle


@pytest.fixture
def bundle_dir(tmp_path):
    # Tiny deterministic saved models isolate scale/order and corruption handling.
    factory = tmp_path / "factory.py"
    factory.write_text("""import torch
def create_2d_net():
    return torch.nn.Linear(3, 1)
def create_1d_rib_net():
    return torch.nn.Linear(3, 1)
class SimpleThicknessNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 1)
    def forward(self, r, theta):
        return self.net(torch.cat([r, theta], dim=1))
""", encoding="utf-8")
    weights = {}
    for region in ("upper", "lower", "rib"):
        relative = f"{region}.pth"
        torch.save({"weight": torch.tensor([[1., .1, 2.]], dtype=torch.float32),
                    "bias": torch.tensor([0.], dtype=torch.float32)}, tmp_path / relative)
        weights[region] = relative
    torch.save({"net.weight": torch.tensor([[.005, .0001]], dtype=torch.float32),
                "net.bias": torch.tensor([.01], dtype=torch.float32)}, tmp_path / "thickness.pth")
    profile = {"format_version": 1, "version_id": "fixture", "weights": weights,
               "radius_scale_m": .225, "ring_inner_m": {"upper": .084, "lower": .103},
               "rib_start_m": .12, "rib_end_m": .225, "rib_slope_rad_m": 6.4105,
               "alpha_ref_m2_s": 15.1 / (7850 * 460), "temperature_scale_k": 900.,
               "ambient_k": 293.15, "rib_input_order": ["s_star", "Fo", "rib_id"],
               "original_source_directory": str(tmp_path / "unavailable_original_source"),
               "thickness": {"upper": {"weight": "thickness.pth", "output_scale_m": 1.}},
               "files_sha256": {p.name: file_hash(p) for p in tmp_path.iterdir()}}
    write_json(tmp_path / "profile.json", profile)
    return tmp_path


def test_si_scales_batch_reload_and_autograd_without_source(bundle_dir):
    model = LegacyBundle(bundle_dir)
    frame = pd.DataFrame({"r": [.15, .18], "theta": [.4, 1.2], "t": [1200., 0.]})
    fo = 1200 * (15.1 / (7850 * 460)) / .225**2
    expected = 293.15 + 900 * np.array([.15 / .225 + .1 * .4 + 2 * fo, .18 / .225 + .1 * 1.2])
    actual = model.predict(frame, "upper", batch_size=1)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
    np.testing.assert_array_equal(actual, LegacyBundle(bundle_dir).predict(frame, "upper"))
    x = torch.tensor(model.normalized_inputs(frame, "upper"), dtype=torch.float32, requires_grad=True)
    gradient = torch.autograd.grad(model.temperature_tensor("upper", x).sum(), x)[0]
    torch.testing.assert_close(gradient, torch.tensor([[900., 90., 1800.]]).expand(2, -1))
    thickness = model.predict_thickness(frame, "upper")
    np.testing.assert_allclose(thickness, .01 + .005 * frame.r / .225 + .0001 * frame.theta, atol=1e-8)
    tx = torch.tensor([[.5, .4]], requires_grad=True)
    derivative = torch.autograd.grad(model.thickness_tensor("upper", tx).sum(), tx)[0]
    torch.testing.assert_close(derivative, torch.tensor([[.005, .0001]]))


def test_rib_order_comes_from_preserved_profile(bundle_dir):
    frame = pd.DataFrame({"r": [.17], "theta": [.5], "t": [600.], "rib_id": [7]})
    before = LegacyBundle(bundle_dir).normalized_inputs(frame, "rib")
    profile = json.loads((bundle_dir / "profile.json").read_text(encoding="utf-8"))
    profile["rib_input_order"] = ["s_star", "rib_id", "Fo"]
    write_json(bundle_dir / "profile.json", profile)
    after = LegacyBundle(bundle_dir).normalized_inputs(frame, "rib")
    np.testing.assert_array_equal(before[:, [0, 2, 1]], after)
    assert before[0, 2] == 7 and before[0, 1] < 1


@pytest.mark.parametrize("change, message", [
    ({"rib_id": .5}, "integer"), ({"rib_id": 13}, "integer"),
    ({"r": .01}, "geometry"), ({"t": -1}, "nonnegative"),
    ({"theta": np.nan}, "finite"),
])
def test_invalid_inputs_fail_clearly(bundle_dir, change, message):
    frame = pd.DataFrame({"r": [.17], "theta": [.5], "t": [600.], "rib_id": [7]})
    for column, value in change.items():
        frame[column] = value
    with pytest.raises(ValueError, match=message):
        LegacyBundle(bundle_dir).predict(frame, "rib")


def test_missing_and_corrupt_preserved_files_fail_clearly(bundle_dir):
    weight = bundle_dir / "rib.pth"
    original = weight.read_bytes()
    weight.unlink()
    with pytest.raises(FileNotFoundError, match="Missing preserved model file"):
        LegacyBundle(bundle_dir)
    weight.write_bytes(original + b"corrupted")
    with pytest.raises(ValueError, match="hash mismatch"):
        LegacyBundle(bundle_dir)


def test_missing_profile_and_input_columns_fail_clearly(bundle_dir, tmp_path):
    with pytest.raises(FileNotFoundError, match="model profile"):
        LegacyBundle(tmp_path / "absent")
    model = LegacyBundle(bundle_dir)
    with pytest.raises(ValueError, match="input columns"):
        model.predict(pd.DataFrame({"r": [.17]}), "rib")
