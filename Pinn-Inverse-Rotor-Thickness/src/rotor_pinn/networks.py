"""Network factories shared by training and reload."""

import math
import os

import torch
from torch import nn


class ThicknessNet(nn.Module):
    def __init__(self, widths, scale, minimum=1e-6):
        super().__init__()
        layers = []
        for left, right in zip([3] + list(widths), list(widths) + [1]):
            layers.append(nn.Linear(left, right))
            if right != 1:
                layers.append(nn.SiLU())
        self.net = nn.Sequential(*layers)
        self.register_buffer("scale", torch.tensor(float(scale)))
        self.minimum = minimum

    def forward(self, inputs):
        r, theta = inputs[:, 0:1], inputs[:, 1:2]
        features = torch.cat([r, theta.sin(), theta.cos()], dim=1)
        return self.minimum + self.scale * torch.nn.functional.softplus(self.net(features))

    def freeze(self):
        self.eval()
        self.requires_grad_(False)
        return self


def temperature_net(kind, config):
    os.environ.setdefault("DDE_BACKEND", "pytorch")
    import deepxde as dde
    if dde.backend.backend_name != "pytorch":
        raise RuntimeError("Initialize the PyTorch backend before creating networks")
    widths = config.networks.rib_widths if kind == "rib" else config.networks.ring_widths
    net = dde.nn.FNN([4] + list(widths) + [1], "tanh", "Glorot normal")
    def features(inputs):
        position = inputs[:, 0:1]
        angle = inputs[:, 1:2]
        if kind == "rib":
            angle = angle * (2 * math.pi / config.geometry.rib_count)
        time = torch.log1p(10 * inputs[:, 2:3])
        return torch.cat([position, angle.sin(), angle.cos(), time], dim=1)
    def output(inputs, raw):
        return -torch.expm1(-config.networks.initial_factor * inputs[:, 2:3]) * raw
    net.apply_feature_transform(features)
    net.apply_output_transform(output)
    return net
