"""Save numerical or CUDA evidence with the same run provenance as training."""

import argparse
from pathlib import Path
import shutil

import numpy as np
from scipy.integrate import quad
import torch

from rotor_pinn.artifacts import file_hash, new_run, write_json
from rotor_pinn.backend import initialize
from rotor_pinn.config import load_config
from rotor_pinn.coupling import ContactProjection
from rotor_pinn.geometry import RotorGeometry
from rotor_pinn.materials import Material
from rotor_pinn.physics import derivative, ring_energy_residual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/demo.toml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--pytest-report")
    args = parser.parse_args()
    config = load_config(args.config)
    config.training.device = "cuda" if args.cuda else "cpu"
    config.training.dtype = "float64"
    with new_run(config, args.run_id, "CUDA hardware/autodiff check" if args.cuda else "analytic and discrete conservation checks",
                 "analytic functions; no COMSOL data") as (directory, manifest):
        manifest["verification_script_sha256"] = file_hash(__file__)
        _, device, dtype = initialize(config.training)
        x = torch.tensor([[0.6, 0.4, 0.1], [0.8, 1.3, 0.2]], device=device,
                         dtype=dtype, requires_grad=True)
        p = config.physics
        # A separate material avoids changing the saved configuration.
        from dataclasses import replace
        constant = replace(p, density=2, temperature_nodes=[100, 2000],
                           conductivity_nodes=[3, 3], heat_capacity_nodes=[4, 4])
        material = Material(constant)
        radius, time = config.geometry.outer_radius, 1200
        r, theta = x[:, 0:1] * radius, x[:, 1:2]
        h = 0.01 + 0.02 * r
        temperature = 293 + r**2 + theta.sin() + x[:, 2:3] * time
        result = ring_energy_residual(x, temperature, h, material, radius, time)
        expected = 8 * h - 3 * (0.04 + 0.12 * r - h * theta.sin() / r**2)
        error = float(torch.max(torch.abs((result - expected) / expected)).detach().cpu())
        gradient_error = float(torch.max(torch.abs(derivative(h, x)[:, 0] - 0.02 * radius)).detach().cpu())
        geometry = RotorGeometry(config.geometry)
        g = config.geometry
        reference, _ = quad(lambda rr: np.sqrt(1 + (rr * g.rib_slope)**2), g.rib_start, g.rib_end)
        diagnostics = {"device": str(device), "dtype": str(dtype),
                       "ring_residual_max_relative_error": error,
                       "thickness_derivative_max_absolute_error": gradient_error,
                       "rib_length_m": geometry.length,
                       "arc_length_absolute_error_m": abs(geometry.length - reference),
                       "power_balance": {}}
        for region in ("upper", "lower"):
            projection = ContactProjection(geometry, config, region)
            power = np.random.default_rng(42).normal(0, 300, len(projection.ids))
            diagnostic = projection.residual(power)
            diagnostics["power_balance"][region] = diagnostic
            if diagnostic["relative_imbalance"] >= 1e-6:
                raise AssertionError("Discrete exchange power is not conserved")
        if args.cuda:
            z = torch.tensor([0.2, 0.5, 0.9], device=device, dtype=dtype, requires_grad=True)
            grad = torch.autograd.grad((z.sin() * z**2).sum(), z)[0]
            expected_grad = 2 * z * z.sin() + z**2 * z.cos()
            diagnostics["cuda_gradient_max_absolute_error"] = float((grad - expected_grad).abs().max().detach().cpu())
            product = torch.randn(32, 32, device=device) @ torch.randn(32, 32, device=device)
            diagnostics["cuda_matrix_product_finite"] = bool(torch.isfinite(product).all())
            torch.cuda.synchronize()
        if error >= 1e-6 or gradient_error >= 1e-12:
            raise AssertionError("Analytic derivative or residual check failed")
        write_json(directory / "metrics.json", diagnostics)
        manifest["has_metrics"] = True
        if args.pytest_report:
            shutil.copyfile(Path(args.pytest_report), directory / "pytest.xml")
        print(directory)


if __name__ == "__main__":
    main()
