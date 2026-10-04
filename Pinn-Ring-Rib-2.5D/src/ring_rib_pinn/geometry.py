"""Original 1000-node spiral mapping and normalized Gaussian projection."""

import json
import numpy as np
import torch
from scipy.integrate import quad


def interpolate(x, xp, fp):
    idx = torch.searchsorted(xp, x.contiguous())
    idx = torch.clamp(idx, 1, len(xp) - 1)
    weight = (x - xp[idx - 1]) / (xp[idx] - xp[idx - 1] + 1e-12)
    return fp[idx - 1] + weight * (fp[idx] - fp[idx - 1])


class SpiralGeometry:
    def __init__(self, config, phases, device="cpu", dtype=torch.float32):
        self.config = config
        g = config.g
        self.phases = torch.as_tensor(phases, device=device, dtype=dtype).view(-1, 1)
        if len(phases) != g["rib_count"] or not torch.isfinite(self.phases).all():
            raise ValueError("Expected 13 finite rib start phases")
        radii = np.linspace(g["rib_start"], g["rib_end"], g["mapping_points"])
        arcs = np.asarray([self.arc(r) for r in radii])
        self.r_numpy, self.s_numpy = radii, arcs
        self.r_grid = torch.as_tensor(radii / g["radius"], device=device, dtype=dtype)
        self.s_grid = torch.as_tensor(arcs / g["radius"], device=device, dtype=dtype)

    @classmethod
    def from_json(cls, config, path, device="cpu", dtype=torch.float32):
        data = json.loads(path.read_text(encoding="utf-8"))
        phases = [data[str(i)]["phi_start_rad"] for i in range(config.g["rib_count"])]
        return cls(config, phases, device, dtype)

    def arc(self, radius):
        g = self.config.g
        return quad(lambda r: np.sqrt(1 + (r * g["rib_slope"])**2),
                    g["rib_start"], radius)[0]

    def rib_coordinates(self, s_star, rib_id):
        g = self.config.g
        radius_star = interpolate(s_star.flatten(), self.s_grid, self.r_grid).view(-1, 1)
        phase = self.phases[rib_id.long().flatten()]
        theta = (phase + g["rib_slope"] * (radius_star * g["radius"] - g["rib_start"]))
        return radius_star, theta % (2 * np.pi)

    def project(self, radius_star, theta):
        g = self.config.g
        radius = radius_star * g["radius"]
        centers = (self.phases.T + g["rib_slope"] * (radius - g["rib_start"])) % (2*np.pi)
        difference = (theta - centers + np.pi) % (2*np.pi) - np.pi
        distance = torch.abs(radius * difference)
        sigma = g["rib_width"] / 4
        scaling = g["rib_width"] / (sigma * np.sqrt(2 * np.pi))
        masks = scaling * torch.exp(-distance**2 / (2 * sigma**2))
        mask, rib_id = torch.max(masks, dim=1, keepdim=True)
        radial_mask = (radius >= g["rib_start"]) & (radius <= g["rib_end"])
        mask = mask * radial_mask.float().view(-1, 1)
        arc_star = interpolate(radius_star.flatten(), self.r_grid, self.s_grid).view(-1, 1)
        return mask, arc_star, rib_id.float()
