import json
import os
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch

from rotor_pinn.artifacts import file_hash, new_run, write_json
from rotor_pinn.backend import initialize
from rotor_pinn.evaluation import compute_metrics, evaluate_run, predict_frame
from rotor_pinn.geometry import RotorGeometry
from rotor_pinn.networks import ThicknessNet, temperature_net
from rotor_pinn.plotting import plot_run
from rotor_pinn.synthetic import demo_frames
from rotor_pinn.training import _save_checkpoint, reload_models, train_pipeline


def test_four_metrics_kelvin_denominator():
    result = compute_metrics(np.array([300, 400]), np.array([303, 396]))
    assert result["mae_k"] == 3.5 and result["max_k"] == 4
    assert result["rmse_k"] == pytest.approx(np.sqrt(12.5))
    assert result["l2_relative_absolute_kelvin"] == pytest.approx(5 / 500)


def test_import_has_no_backend_or_plotting_side_effects(tmp_path):
    code = "import sys; import rotor_pinn.training, rotor_pinn.plotting; assert 'deepxde' not in sys.modules; assert 'matplotlib.pyplot' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True,
                   env={**os.environ, "DDE_BACKEND": "pytorch"}, capture_output=True)
    assert not list(tmp_path.iterdir())


def test_checkpoint_reloaded_prediction_matches(config, tmp_path):
    initialize(config.training)
    thickness = {r: ThicknessNet([64, 64, 32], 0.01).double().freeze() for r in ("upper", "lower")}
    models = {r: temperature_net("rib" if r == "rib" else "ring", config).double()
              for r in ("upper", "lower", "rib")}
    _save_checkpoint(tmp_path / "models.pt", config, thickness, models, 1200)
    _, restored, _, scale, device, dtype = reload_models(tmp_path, device="auto")
    geometry = RotorGeometry(config.geometry)
    for region, frame in demo_frames(config).items():
        before = predict_frame(models[region], frame.iloc[:40], region, geometry, 1200,
                               config.physics, next(models[region].parameters()).device, dtype)
        after = predict_frame(restored[region], frame.iloc[:40], region, geometry, scale,
                              config.physics, device, dtype)
        np.testing.assert_allclose(before, after, atol=1e-10, rtol=1e-10)
    with pytest.raises(FileNotFoundError, match="input snapshot"):
        evaluate_run(tmp_path)
    input_dir = tmp_path / "inputs"
    input_dir.mkdir()
    frame = demo_frames(config)["upper"].copy()
    frame["split"] = "train"
    input_path = input_dir / "upper.csv"
    frame.to_csv(input_path, index=False)
    checkpoint = tmp_path / "models.pt"
    write_json(tmp_path / "manifest.json", {"checkpoint_sha256": file_hash(checkpoint),
                                           "input_sha256": {"upper": file_hash(input_path)}})
    with input_path.open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="input snapshot hash"):
        evaluate_run(tmp_path)
    with checkpoint.open("ab") as stream:
        stream.write(b"corrupted")
    with pytest.raises(ValueError, match="checkpoint hash"):
        reload_models(tmp_path)


def test_run_id_refuses_overwrite_and_keeps_failure(config):
    config.output.artifacts = "artifacts"
    with new_run(config, "TEST", "unit check", "synthetic") as (directory, _):
        (directory / "sentinel").write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        with new_run(config, "TEST", "duplicate", "synthetic"):
            pass
    assert (directory / "sentinel").read_text(encoding="utf-8") == "original"
    with pytest.raises(RuntimeError, match="deliberate"):
        with new_run(config, "FAIL", "error check", "synthetic"):
            raise RuntimeError("deliberate")
    manifest = json.loads((directory.parent / "FAIL" / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert (directory.parent / "FAIL" / "failure.txt").is_file()


def test_missing_models_and_predictions_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="models.pt"):
        reload_models(tmp_path)
    with pytest.raises(FileNotFoundError, match="predictions"):
        plot_run(tmp_path)


def test_thickness_only_needs_no_temperature_data(config, tmp_path):
    config.output.artifacts = "artifacts"
    config.training.thickness_steps = 2
    for region in ("upper", "lower"):
        path = tmp_path / f"{region}-thickness.csv"
        demo_frames(config)[region][["r", "theta", "thickness"]].drop_duplicates().to_csv(path, index=False)
        setattr(config.data, f"{region}_thickness_csv", path.name)
    directory = train_pipeline(config, "THICKNESS", thickness_only=True)
    assert (directory / "thickness.pt").is_file()
    assert not (directory / "models.pt").exists()


def test_tiny_pipeline_with_lbfgs(config):
    config.output.artifacts = "artifacts"
    t = config.training
    t.thickness_steps = t.warmup_steps = t.inner_steps = 2
    t.outer_iterations = 1
    t.lbfgs_steps = 3
    t.domain_points = 12
    t.boundary_points = 16
    t.observation_batch = 16
    t.display_every = 2
    config.networks.ring_widths = config.networks.rib_widths = [8, 8]
    config.networks.thickness_widths = [8]
    directory = train_pipeline(config, "LBFGS_PIPELINE", demo=True)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    assert np.isfinite(metrics["partitions"]["test"]["overall"]["rmse_k"])
    for region in ("upper", "lower", "rib"):
        predictions = pd.read_csv(directory / f"predictions_{region}.csv")
        assert np.isfinite(predictions.predicted_T).all()
        assert (directory / "figures" / f"{region}_temperature.png").is_file()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA extra and hardware required")
def test_cuda_tensor_and_autodiff():
    x = torch.tensor([0.2, 0.5, 0.9], device="cuda", dtype=torch.float64, requires_grad=True)
    y = (x.sin() * x**2).sum()
    gradient = torch.autograd.grad(y, x)[0]
    expected = 2 * x * x.sin() + x**2 * x.cos()
    torch.testing.assert_close(gradient, expected, atol=1e-12, rtol=1e-12)
    assert torch.isfinite(torch.randn(32, 32, device="cuda") @ torch.randn(32, 32, device="cuda")).all()
