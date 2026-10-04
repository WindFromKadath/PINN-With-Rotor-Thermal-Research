"""Piecewise linear material properties, constant outside the supplied node range."""

import torch


class Material:
    def __init__(self, physics):
        self.physics = physics

    def factors(self, temperature):
        p = self.physics
        nodes = torch.as_tensor(p.temperature_nodes, device=temperature.device,
                                dtype=temperature.dtype)
        value = temperature.clamp(nodes[0], nodes[-1])
        index = torch.searchsorted(nodes, value.contiguous()).clamp(1, len(nodes) - 1)
        fraction = (value - nodes[index - 1]) / (nodes[index] - nodes[index - 1])
        def interpolate(values):
            table = torch.as_tensor(values, device=value.device, dtype=value.dtype)
            return table[index - 1] + fraction * (table[index] - table[index - 1])
        return interpolate(p.conductivity_nodes), interpolate(p.heat_capacity_nodes)

    def surface_loss(self, temperature):
        p = self.physics
        return (p.convection * (temperature - p.ambient)
                + p.emissivity * 5.670374419e-8 * (temperature**4 - p.ambient**4))
