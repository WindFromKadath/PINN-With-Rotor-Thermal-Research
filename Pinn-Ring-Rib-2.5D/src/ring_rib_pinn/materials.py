"""Piecewise linear legacy material factors, including endpoint saturation."""

import torch


class Material:
    def __init__(self, config, device, dtype):
        p = config.p
        self.nodes = torch.tensor(p["temperature_nodes"], device=device, dtype=dtype)
        self.k = torch.tensor(p["conductivity_nodes"], device=device, dtype=dtype) / p["k_ref"]
        self.c = torch.tensor(p["capacity_nodes"], device=device, dtype=dtype) / p["cp_ref"]
        self.dk = (self.k[1:] - self.k[:-1]) / (self.nodes[1:] - self.nodes[:-1])

    def factors(self, kelvin):
        shape = kelvin.shape
        temp = kelvin.flatten().contiguous()
        idx = (torch.searchsorted(self.nodes, temp) - 1).clamp(0, len(self.nodes) - 2)
        weight = ((temp - self.nodes[idx]) / (self.nodes[idx+1] - self.nodes[idx] + 1e-12))
        weight = weight.clamp(0, 1)
        k = self.k[idx] + weight * (self.k[idx+1] - self.k[idx])
        c = self.c[idx] + weight * (self.c[idx+1] - self.c[idx])
        return k.reshape(shape), c.reshape(shape), self.dk[idx].reshape(shape), idx.reshape(shape)
