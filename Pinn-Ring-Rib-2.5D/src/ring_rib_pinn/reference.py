"""Prediction-only access to locally retained original model definitions."""

import json
from pathlib import Path

import numpy as np
import torch

from .artifacts import file_hash
from .backend import initialize
from .config import Config
from .data import normalized_inputs
from .geometry import SpiralGeometry
from .legacy import definitions

FORMAT = "retained-source-predict/1"


class SourceReference:
    def __init__(self, directory, device="cpu"):
        self.directory = Path(directory).resolve()
        self.profile = json.loads((self.directory/"profile.json").read_text(encoding="utf-8"))
        if self.profile.get("format") != FORMAT:
            raise ValueError("Expected a selected original reference profile")
        for relative, expected in self.profile["files_sha256"].items():
            if file_hash(self.directory/relative) != expected:
                raise ValueError(f"Reference file hash mismatch: {relative}")
        case = self.profile["case"]
        self.config = Config({"case": case, "physics": {"source": 0 if case == "cooling" else 1e8},
                              "training": {"device": device}}, self.directory)
        dde, self.device, self.dtype = initialize(self.config)
        self.geometry = SpiralGeometry.from_json(self.config, self.directory/"rib_models.json",
                                                  self.device, self.dtype)
        geo = self.geometry
        ns = {"__name__": "retained_reference", "torch": torch, "nn": torch.nn,
              "np": np, "dde": dde, "NUM_RIBS": 13, "R_OUT_PHYS": .225,
              "T_INF_PHYS": 293.15, "T_REF_DELTA": 900.,
              "s_grid_star": geo.s_grid, "r_grid_star": geo.r_grid,
              "PHI_STARTS": geo.phases, "R_RIB_START_JSON": .123,
              "R_RIB_START": .123, "B_SPIRAL_JSON": 6.4105}
        names = {"create_ring_net", "create_rib_net", "create_hard_constraint_ring_net",
                 "create_hard_constraint_rib_net", "ic_ring_upper_torch", "ic_ring_lower_torch",
                 "ic_rib_torch", "torch_interp"}
        definitions(self.directory/"source.py", ns, names)
        if self.profile["factory"] == "hard_initial":
            builders = {"upper": lambda: ns["create_hard_constraint_ring_net"](ns["ic_ring_upper_torch"]),
                        "lower": lambda: ns["create_hard_constraint_ring_net"](ns["ic_ring_lower_torch"]),
                        "rib": lambda: ns["create_hard_constraint_rib_net"]()}
        else:
            builders = {"upper": ns["create_ring_net"], "lower": ns["create_ring_net"],
                        "rib": ns["create_rib_net"]}
        self.networks = {}
        for domain, builder in builders.items():
            net = builder().to(device=self.device, dtype=self.dtype)
            state = torch.load(self.directory/self.profile["weights"][domain],
                               map_location=self.device, weights_only=True)
            if any(not torch.isfinite(value).all() for value in state.values()):
                raise FloatingPointError(f"Non-finite reference weights: {domain}")
            net.load_state_dict(state, strict=True)
            self.networks[domain] = net.eval()

    def predict(self, frame, domain, batch_size=4096):
        if domain not in self.networks or batch_size < 1:
            raise ValueError("Expected upper/lower/rib domain and positive batch_size")
        points = frame.copy()
        if domain == "rib" and self.profile.get("clip_rib_radius", True):
            # Validate before clipping: only tolerated COMSOL boundary noise is clipped.
            from .data import validate_prediction_frame
            validate_prediction_frame(points, self.config, domain)
            points["r"] = points.r.clip(.123, .225)
        values = normalized_inputs(points, self.config, self.geometry, domain)
        output = []
        with torch.no_grad():
            for start in range(0, len(values), batch_size):
                x = torch.as_tensor(values[start:start+batch_size], device=self.device, dtype=self.dtype)
                result = self.networks[domain](x)
                if not torch.isfinite(result).all():
                    raise FloatingPointError(f"Non-finite reference output: {domain}")
                output.append(result.cpu().numpy().astype(np.float64))
        result = np.concatenate(output)
        predicted = frame.copy()
        # Convert in the network dtype, preserving the original inference calculation.
        raw = result[:, 0].astype(np.float32)
        predicted["Pred_T"] = (raw*900.+293.15).astype(np.float64)
        for index, name in enumerate(self.profile["output_heads"][domain][1:], start=1):
            predicted[name] = result[:, index]
        return predicted
