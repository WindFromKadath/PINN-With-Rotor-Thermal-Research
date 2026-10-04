"""Per-run heat-flux snapshots; preserve original searchsorted lookup semantics."""

import torch


class CouplingState:
    def __init__(self, geometry):
        self.geometry = geometry
        self.flux = {}
        self.s_grid = self.t_grid = None

    def update(self, physics, ns, nt):
        cfg = physics.cfg
        g = cfg.g
        device, dtype = self.geometry.phases.device, self.geometry.phases.dtype
        self.s_grid = torch.linspace(0, g["rib_length"]/g["radius"], ns,
                                     device=device, dtype=dtype)
        self.t_grid = torch.linspace(0, cfg.time_end_star, nt, device=device, dtype=dtype)
        ids = torch.arange(g["rib_count"], device=device, dtype=dtype)
        s, t, rid = torch.meshgrid(self.s_grid, self.t_grid, ids, indexing="ij")
        s, t, rid = [v.flatten().unsqueeze(1) for v in (s, t, rid)]
        radius, theta = self.geometry.rib_coordinates(s, rid)
        with torch.no_grad():
            ring_x = torch.cat([radius, theta, t], dim=1)
            upper = physics.networks["upper"](ring_x)
            lower = physics.networks["lower"](ring_x)
            rib = physics.networks["rib"](torch.cat([s, rid, t], dim=1))
            qu, ql = physics.contact(radius, theta, upper, lower, rib)
        self.flux = {"upper": qu.reshape(ns, nt, g["rib_count"]).detach(),
                     "lower": ql.reshape(ns, nt, g["rib_count"]).detach()}

    def lookup(self, radius, theta, time, domain):
        if not self.flux:
            return torch.zeros_like(radius)
        mask, s, rid = self.geometry.project(radius, theta)
        si = torch.searchsorted(self.s_grid, s.flatten().contiguous()).clamp(0, len(self.s_grid)-1)
        ti = torch.searchsorted(self.t_grid, time.flatten().contiguous()).clamp(0, len(self.t_grid)-1)
        ri = rid.long().flatten().clamp(0, self.geometry.config.g["rib_count"]-1)
        return self.flux[domain][si, ti, ri].view_as(radius) * mask
