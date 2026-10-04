"""Portable inference for explicitly preserved historical model bundles.

Bundles contain audited network definitions, original state_dicts and SI scales.
They remain separate from the new full-length-contact training checkpoints.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.integrate import quad
import torch

from .artifacts import file_hash
from .backend import initialize
from .config import Training


class LegacyBundle:
    """Load a local trusted-code bundle; never access its historical source paths."""

    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory).resolve()
        path = self.directory / "profile.json"
        if not path.is_file():
            raise FileNotFoundError(f"Missing preserved model profile: {path}")
        self.profile = json.loads(path.read_text(encoding="utf-8"))
        if self.profile.get("format_version") != 1:
            raise ValueError("Unsupported historical bundle format")
        for relative, digest in self.profile["files_sha256"].items():
            saved = self.directory / relative
            if not saved.is_file():
                raise FileNotFoundError(f"Missing preserved model file: {saved}")
            if file_hash(saved) != digest:
                raise ValueError(f"Preserved file hash mismatch: {relative}")
        _, self.device, _ = initialize(Training(device=device, dtype="float32"))
        # factory.py has only the audited definitions and their explicit imports.
        # It is local Python source, not deserialized code from a model pickle.
        factory = self.directory / "factory.py"
        namespace = {"__name__": "preserved_legacy_factory"}
        exec(compile(factory.read_text(encoding="utf-8"), str(factory), "exec"), namespace)
        self.models = {}
        for region in ("upper", "lower", "rib"):
            builder = namespace["create_1d_rib_net" if region == "rib" else "create_2d_net"]
            model = builder().to(self.device)
            model.load_state_dict(torch.load(self.directory / self.profile["weights"][region],
                                             map_location=self.device, weights_only=True), strict=True)
            self.models[region] = model.eval()
        self.thickness_models = {}
        for region, info in self.profile.get("thickness", {}).items():
            model = namespace["SimpleThicknessNet"]().to(self.device)
            model.load_state_dict(torch.load(self.directory / info["weight"],
                                             map_location=self.device, weights_only=True), strict=True)
            self.thickness_models[region] = model.eval()

    def _frame(self, frame, region):
        if region not in self.models:
            raise ValueError("Region must be upper, lower or rib")
        required = ["r", "theta", "t"] + (["rib_id"] if region == "rib" else [])
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing SI input columns: {sorted(missing)}")
        if frame.empty or not np.isfinite(frame[required].to_numpy(dtype=float)).all():
            raise ValueError("Prediction input must be nonempty and finite")
        if (frame.t < 0).any():
            raise ValueError("Time must be nonnegative seconds")
        scale = self.profile["radius_scale_m"]
        inner = self.profile["ring_inner_m"][region] if region != "rib" else self.profile["rib_start_m"]
        if ((frame.r < inner - 1e-5) | (frame.r > scale + 1e-5)).any():
            raise ValueError(f"Radius outside preserved {region} geometry, in meters")
        if region == "rib" and ((frame.rib_id % 1 != 0) | (frame.rib_id < 0)
                                 | (frame.rib_id >= 13)).any():
            raise ValueError("rib_id must be an integer from 0 through 12")
        return frame

    def normalized_inputs(self, frame, region):
        """Keep each historical order and scale; input columns use m/rad/s."""
        frame = self._frame(frame, region)
        scale = self.profile["radius_scale_m"]
        time = frame.t.to_numpy() * self.profile["alpha_ref_m2_s"] / scale**2
        if region != "rib":
            return np.column_stack([frame.r / scale, frame.theta, time])
        unique, inverse = np.unique(frame.r.to_numpy(), return_inverse=True)
        start, end = self.profile["rib_start_m"], self.profile["rib_end_m"]
        slope = self.profile["rib_slope_rad_m"]
        arc = np.asarray([quad(lambda r: np.sqrt(1 + (slope * r)**2),
                               start, np.clip(radius, start, end))[0] for radius in unique])[inverse]
        columns = {"s_star": arc / scale, "Fo": time, "rib_id": frame.rib_id.to_numpy()}
        return np.column_stack([columns[key] for key in self.profile["rib_input_order"]])

    def temperature_tensor(self, region, normalized_inputs):
        """Physical Kelvin output with autograd retained for future reuse."""
        output = self.models[region](normalized_inputs)[:, :1]
        return output * self.profile["temperature_scale_k"] + self.profile["ambient_k"]

    def predict(self, frame, region, batch_size=4096):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        inputs = self.normalized_inputs(frame, region)
        result = []
        with torch.no_grad():
            for start in range(0, len(inputs), batch_size):
                x = torch.tensor(inputs[start:start + batch_size], dtype=torch.float32,
                                 device=self.device)
                result.append(self.temperature_tensor(region, x).cpu().numpy().ravel())
        predicted = np.concatenate(result)
        if not np.isfinite(predicted).all():
            raise FloatingPointError("Non-finite historical temperature prediction")
        return predicted

    def thickness_tensor(self, region, normalized_radius_theta):
        """Physical thickness with coordinate autograd retained (r/R, theta)."""
        if region not in self.thickness_models:
            raise ValueError(f"No physical thickness model preserved for {region}")
        model = self.thickness_models[region]
        return model(normalized_radius_theta[:, :1], normalized_radius_theta[:, 1:2]) * self.profile["thickness"][region]["output_scale_m"]

    def predict_thickness(self, frame, region):
        self._frame(frame.assign(t=0.0), region)
        scale = self.profile["radius_scale_m"]
        x = torch.tensor(np.column_stack([frame.r / scale, frame.theta]),
                         device=self.device, dtype=torch.float32)
        with torch.no_grad():
            output = self.thickness_tensor(region, x).cpu().numpy().ravel()
        if not np.isfinite(output).all() or (output <= 0).any():
            raise ValueError("Historical thickness prediction is nonpositive or non-finite")
        return output


def verify_bundle(directory, device="cpu"):
    """Strict loading, local-only smoke inference, reload identity and derivatives."""
    bundle = LegacyBundle(directory, device)
    output = {"version_id": bundle.profile["version_id"], "device": str(bundle.device),
              "regions": {}, "thickness": {}}
    predictions = {}
    for region in bundle.models:
        frame = pd.read_csv(bundle.directory / "examples" / f"{region}.csv")
        actual = bundle.predict(frame, region)
        expected = frame.predicted_T.to_numpy()
        error = float(np.max(np.abs(actual - expected)))
        if not np.allclose(actual, expected, rtol=0, atol=1e-3):
            raise AssertionError(f"Preserved prediction differs for {region}: {error} K")
        x = torch.tensor(bundle.normalized_inputs(frame.iloc[:3], region),
                         device=bundle.device, dtype=torch.float32, requires_grad=True)
        derivative = torch.autograd.grad(bundle.temperature_tensor(region, x).sum(), x)[0]
        if not torch.isfinite(derivative).all():
            raise AssertionError(f"Non-finite {region} coordinate derivative")
        output["regions"][region] = {"count": len(frame), "finite": True,
                                      "example_max_difference_k": error, "autograd_finite": True}
        predictions[region] = actual
    for region in bundle.thickness_models:
        frame = pd.read_csv(bundle.directory / "examples" / f"{region}.csv")
        thickness = bundle.predict_thickness(frame, region)
        x = torch.tensor(np.column_stack([frame.r.iloc[:3] / bundle.profile["radius_scale_m"],
                                          frame.theta.iloc[:3]]), device=bundle.device,
                         dtype=torch.float32, requires_grad=True)
        derivative = torch.autograd.grad(bundle.thickness_tensor(region, x).sum(), x)[0]
        if not torch.isfinite(derivative).all():
            raise AssertionError(f"Non-finite {region} thickness derivative")
        output["thickness"][region] = {"min_m": float(thickness.min()),
                                        "max_m": float(thickness.max()), "positive_finite": True,
                                        "autograd_finite": True}
    reloaded = LegacyBundle(directory, device)
    for region in bundle.models:
        frame = pd.read_csv(bundle.directory / "examples" / f"{region}.csv")
        difference = float(np.max(np.abs(predictions[region] - reloaded.predict(frame, region))))
        if difference != 0:
            raise AssertionError("Repeated loading changed predictions")
        output["regions"][region]["reload_max_difference_k"] = difference
    return output
