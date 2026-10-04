import numpy as np
import pytest
import torch

from rotor_pinn.backend import initialize
from rotor_pinn.coupling import ContactProjection, CouplingState, contact_line_power
from rotor_pinn.geometry import RotorGeometry
from rotor_pinn.materials import Material
from rotor_pinn.networks import ThicknessNet, temperature_net
from rotor_pinn.physics import derivative, rib_energy_residual, ring_energy_residual
from rotor_pinn.training import rib_domain


def constant_material(config):
    p = config.physics
    p.density = 2
    p.temperature_nodes = [100, 2000]
    p.conductivity_nodes = [3, 3]
    p.heat_capacity_nodes = [4, 4]
    return Material(p)


def test_ring_analytic_variable_thickness_residual(config):
    material = constant_material(config)
    radius, time = 0.225, 1200
    x = torch.tensor([[0.6, 0.4, 0.1], [0.8, 1.3, 0.2]], dtype=torch.float64, requires_grad=True)
    r, angle, physical_t = x[:, 0:1] * radius, x[:, 1:2], x[:, 2:3] * time
    a, b = 0.01, 0.02
    h = a + b * r
    temperature = 293 + r**2 + angle.sin() + physical_t
    residual = ring_energy_residual(x, temperature, h, material, radius, time)
    expected = 8 * h - 3 * (4 * a + 6 * b * r - h * angle.sin() / r**2)
    relative = torch.max(torch.abs(residual - expected) / torch.abs(expected)).item()
    assert relative < 1e-6
    np.testing.assert_allclose(derivative(h, x)[:, 0].detach(), b * radius, atol=1e-12)
    gradients = derivative(temperature, x)
    np.testing.assert_allclose(gradients[:, 2].detach(), time, rtol=1e-12)
    np.testing.assert_allclose(derivative(gradients[:, 0:1], x)[:, 0].detach(), 2 * radius**2, rtol=1e-12)


def test_temperature_dependent_conductivity_in_divergence(config):
    material = constant_material(config)
    config.physics.conductivity_nodes = [1, 20]  # k(T) = 0.01 T
    radius, time, h = 0.225, 30, 0.01
    x = torch.tensor([[0.6, 0.2, 0.1], [0.8, 1.2, 0.2]], dtype=torch.float64, requires_grad=True)
    r = x[:, 0:1] * radius
    temperature = 300 + r**2
    residual = ring_energy_residual(x, temperature, torch.ones_like(r) * h,
                                    material, radius, time)
    expected = -h * (0.04 * temperature + 0.04 * r**2)
    torch.testing.assert_close(residual, expected, rtol=1e-6, atol=1e-12)


def test_rib_analytic_time_index_and_zero_second_derivative(config):
    material = constant_material(config)
    radius, time, area = 0.225, 1200, 0.00015
    x = torch.tensor([[0.2, 0, 0.1], [0.4, 12, 0.2]], dtype=torch.float64, requires_grad=True)
    temperature = 293 + (x[:, 0:1] * radius)**2 + x[:, 2:3] * time
    result = rib_energy_residual(x, temperature, material, radius, time, area)
    torch.testing.assert_close(result, torch.full_like(result, (8 - 6) * area), rtol=1e-6, atol=1e-12)
    affine = 2 * x[:, 0:1] + 4 * x[:, 2:3]
    torch.testing.assert_close(derivative(derivative(affine, x)[:, 0:1], x), torch.zeros_like(x))


def test_frozen_thickness_positive_periodic_and_differentiable(config):
    torch.manual_seed(42)
    net = ThicknessNet([8, 8], 0.01).double().freeze()
    x = torch.tensor([[0.6, 0.4, 0.1]], dtype=torch.float64, requires_grad=True)
    h = net(x)
    assert h.item() > 0
    grad = derivative(h, x)
    assert torch.isfinite(grad).all() and abs(grad[0, 0].item()) > 1e-12
    assert all(not parameter.requires_grad for parameter in net.parameters())
    periodic = x.detach().clone()
    periodic[:, 1] += 2 * np.pi
    torch.testing.assert_close(h, net(periodic), atol=1e-12, rtol=1e-12)


def test_discrete_rib_sampling_and_temperature_initial_condition(config):
    dde, _, _ = initialize(config.training)
    domain = rib_domain(dde, RotorGeometry(config.geometry))
    for points in (domain.random_points(100), domain.random_boundary_points(100)):
        np.testing.assert_array_equal(points[:, 1], np.round(points[:, 1]))
        assert domain.inside(points).all()
    for kind in ("ring", "rib"):
        net = temperature_net(kind, config)
        x = torch.tensor([[0.6, 0.4, 0.0], [0.8, 1.2, 0.0]], dtype=torch.float64)
        torch.testing.assert_close(net(x), torch.zeros((2, 1), dtype=torch.float64))


def test_contact_resistance_zero_and_reversed_sign():
    q = contact_line_power(350, 300, 0.01, 10, 20, 0.0094, 0.016, 0)
    expected = 0.016 * 50 / (0.01 / 20 + 0.0094 / 40)
    assert q == pytest.approx(expected)
    assert contact_line_power(300, 350, 0.01, 10, 20, 0.0094, 0.016, 0) == -q
    assert contact_line_power(300, 300, 0.01, 10, 20, 0.0094, 0.016, 0) == 0
    assert contact_line_power(350, 300, 0.01, 10, 20, 0.0094, 0.016, 0.001) < q


@pytest.mark.parametrize("region", ["upper", "lower"])
def test_projection_conservation_and_constant_average(config, region):
    geometry = RotorGeometry(config.geometry)
    projection = ContactProjection(geometry, config, region)
    area = np.pi * (config.geometry.outer_radius**2 - getattr(config.geometry, f"{region}_inner_radius")**2)
    assert projection.area_weights.sum() == pytest.approx(area, rel=1e-12)
    np.testing.assert_allclose(projection.average(np.full(len(projection.area_weights), 350)), 350, rtol=1e-12)
    rng = np.random.default_rng(42)
    line = rng.normal(0, 300, len(projection.ids))
    diagnostic = projection.residual(line)
    assert diagnostic["relative_imbalance"] < 1e-6
    state = CouplingState.empty(geometry, config, 1200)
    getattr(state, f"{region}_power")[:] = line
    x = torch.tensor(np.column_stack([projection.ring_points[:, 0] / config.geometry.outer_radius,
                                      projection.ring_points[:, 1],
                                      np.full(len(projection.ring_points), 0.5)]), dtype=torch.float64)
    loss = state.ring_loss(x, region, 1200).numpy().ravel()
    np.testing.assert_allclose(loss, projection.project(line), atol=1e-10, rtol=1e-10)
    assert abs(projection.area_weights @ loss - projection.weights @ line) / (projection.weights @ abs(line)) < 1e-6


def test_coupling_saved_state_and_rib_quadrature(config, tmp_path):
    geometry = RotorGeometry(config.geometry)
    state = CouplingState.empty(geometry, config, 1200)
    state.upper_power[:] = np.arange(len(state.upper.ids))
    state.lower_power[:] = -0.2 * state.upper_power
    path = tmp_path / "coupling.npz"
    state.save(path)
    restored = CouplingState.load(path, geometry, config, 1200)
    np.testing.assert_array_equal(restored.upper_power, state.upper_power)
    x = torch.tensor(np.column_stack([state.upper.arc / config.geometry.outer_radius,
                                      state.upper.ids, np.full(len(state.upper.ids), 0.5)]), dtype=torch.float64)
    gain = restored.rib_gain(x, 1200).numpy().ravel()
    np.testing.assert_allclose(gain, state.upper_power[1] + state.lower_power[1], atol=1e-12)
    with pytest.raises(ValueError, match="rib id"):
        bad = x.clone()
        bad[0, 1] = 0.5
        restored.rib_gain(bad, 1200)
