"""Small explicit architectures compatible with selected historical state dicts."""
import torch
from torch import nn
import math


class DenseNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden, prefix="net", activation="tanh", init="normal"):
        super().__init__()
        activations = {"tanh": nn.Tanh, "silu": nn.SiLU, "relu": nn.ReLU,
                       "sigmoid": nn.Sigmoid, "gelu": nn.GELU}
        if activation.lower() not in activations or not hidden or min(hidden) < 1:
            raise ValueError("Invalid activation or hidden layers")
        self.prefix = prefix
        layers = []
        for width in hidden:
            layer = nn.Linear(input_dim, width)
            layers.extend((layer, activations[activation.lower()]()))
            input_dim = width
        layers.append(nn.Linear(input_dim, output_dim))
        setattr(self, prefix, nn.Sequential(*layers))
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                if init == "uniform":
                    nn.init.xavier_uniform_(layer.weight)
                else:
                    nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, x, *rest):
        if rest:
            x = torch.cat((x, *rest), dim=1)
        return getattr(self, self.prefix)(x)


class FourierFeatures(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.register_buffer("B", torch.randn(2, dim))

    def forward(self, x):
        a = 2 * math.pi * x @ self.B
        return torch.cat((a.sin(), a.cos()), dim=1)


class IWGNetwork(nn.Module):
    """Raw Fourier network; reference frame is applied explicitly outside forward."""
    def __init__(self, hidden=(64, 64, 64, 64), fourier_dim=64):
        super().__init__()
        self.fourier = FourierFeatures(fourier_dim)
        body = DenseNetwork(2 * fourier_dim, 2, hidden)
        self.net = body.net

    def forward(self, x, *rest):
        if rest:
            x = torch.cat((x, *rest), dim=1)
        return self.net(self.fourier(x))


def model_from_spec(spec):
    kind = spec["kind"]
    if kind == "iwg":
        return IWGNetwork(spec["hidden"], spec.get("fourier_dim", 64))
    expected = {"thermal": (5, 1), "mechanical": (3, 2), "mlp": (3, 2), "steady": (4, 1)}
    if kind not in expected:
        raise ValueError(f"Unsupported model kind: {kind}")
    ni, no = expected[kind]
    return DenseNetwork(ni, no, spec["hidden"], spec.get("prefix", "layers" if no == 1 else "net"),
                        spec.get("activation", "tanh"), "uniform" if no == 1 else "normal")
