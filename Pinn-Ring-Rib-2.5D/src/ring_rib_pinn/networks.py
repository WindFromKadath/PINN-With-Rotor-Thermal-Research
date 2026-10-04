"""Legacy-compatible parameter names and explicit feature/output transforms."""

import torch
from torch import nn


class ThicknessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(3, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU(),
                                 nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU(),
                                 nn.Linear(32, 1))

    def forward(self, radius_star, theta):
        return self.net(torch.cat([radius_star, torch.sin(theta), torch.cos(theta)], dim=1))

    def freeze(self):
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        return self


def initial_temperature(x, config, geometry=None):
    """Physical radius, Kelvin IC from the retained cooling scripts."""
    if geometry is None:
        radius = x[:, :1] * config.g["radius"]
    else:
        radius, _ = geometry.rib_coordinates(x[:, :1], x[:, 1:2])
        radius = radius * config.g["radius"]
    kelvin = 793.15 * (0.225 - (radius - 0.084)) / 0.225
    kelvin = kelvin.clamp(config.p["ambient"], 1200)
    return (kelvin - config.p["ambient"]) / config.p["temperature_scale"]


def temperature_net(dde, region, config, geometry):
    is_rib = region == "rib"
    widths = [4, 64, 64, 64, 1] if is_rib else [4, 64, 64, 64, 64, 1]
    base = dde.nn.FNN(widths, "tanh", "Glorot normal")

    def features(x):
        coord, angle, time = x[:, :1], x[:, 1:2], x[:, 2:3]
        if is_rib:
            angle = 2 * torch.pi * angle / config.g["rib_count"]
        return torch.cat([coord, torch.sin(angle), torch.cos(angle),
                          torch.log(1 + 10 * time)], dim=1)

    base.apply_feature_transform(features)
    if config.case == "heating":
        def output(x, y):
            if is_rib:
                y = torch.nn.functional.softplus(y)
            return (1 - torch.exp(-50 * x[:, 2:3])) * y
        base.apply_output_transform(output)
        return base

    class HardConstraint(nn.Module):
        def __init__(self, base_net):
            super().__init__()
            self.base_net = base_net

        def forward(self, x):
            ic = initial_temperature(x, config, geometry if is_rib else None)
            return ic + x[:, 2:3] * self.base_net(x)

    class Wrapped(dde.nn.NN):
        def __init__(self):
            super().__init__()
            self.hard_net = HardConstraint(base)
            self.input_dim, self.output_dim = 3, 1

        def forward(self, x):
            return self.hard_net(x)

    return Wrapped()
