"""SI energy balances differentiated with respect to normalized network coordinates."""

import torch


def derivative(value, inputs):
    """Retain higher derivatives, including exactly zero derivatives of affine fields."""
    if not value.requires_grad:
        return inputs * 0
    gradient = torch.autograd.grad(value, inputs, grad_outputs=torch.ones_like(value),
                                   create_graph=True, retain_graph=True, allow_unused=True)[0]
    return inputs * 0 if gradient is None else gradient + inputs * 0


def ring_energy_residual(inputs, temperature, thickness, material, radius_scale, time_scale,
                         volume_source=0.0, surface_sink=0.0, contact_loss=0.0):
    """W/m²: rho c H T_t - div(k H grad T) - H q + surface + contact."""
    conductivity, capacity = material.factors(temperature)
    r = inputs[:, 0:1] * radius_scale
    grad_t = derivative(temperature, inputs)
    flux_r = conductivity * thickness * grad_t[:, 0:1] / radius_scale
    radial_div = derivative(flux_r, inputs)[:, 0:1] / radius_scale + flux_r / r
    angular_flux = conductivity * thickness * grad_t[:, 1:2]
    angular_div = derivative(angular_flux, inputs)[:, 1:2] / r**2
    storage = material.physics.density * capacity * thickness * grad_t[:, 2:3] / time_scale
    return storage - radial_div - angular_div - thickness * volume_source + surface_sink + contact_loss


def rib_energy_residual(inputs, temperature, material, radius_scale, time_scale, area,
                        volume_source=0.0, surface_sink=0.0, contact_gain=0.0):
    """W/m: rho c A T_t - (k A T_s)_s - A q + side loss - contact gain."""
    conductivity, capacity = material.factors(temperature)
    gradient = derivative(temperature, inputs)
    axial_flux = conductivity * area * gradient[:, 0:1] / radius_scale
    axial_div = derivative(axial_flux, inputs)[:, 0:1] / radius_scale
    storage = material.physics.density * capacity * area * gradient[:, 2:3] / time_scale
    return storage - axial_div - area * volume_source + surface_sink - contact_gain
