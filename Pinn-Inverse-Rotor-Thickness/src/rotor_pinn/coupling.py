"""Conservative discrete contact: matched averaging/projection and SI line powers."""

from dataclasses import dataclass

import numpy as np
import torch

from .geometry import gauss_interval, ring_quadrature


def contact_line_power(ring_temperature, rib_temperature, thickness, ring_k, rib_k,
                       rib_height, contact_width, interface_resistance):
    resistance = thickness / (2 * ring_k) + rib_height / (2 * rib_k) + interface_resistance
    return contact_width * (ring_temperature - rib_temperature) / resistance


class ContactProjection:
    def __init__(self, geometry, config, region):
        self.geometry, self.config, self.region = geometry, config, region
        g, c = geometry.config, config.contact
        self.arc_nodes, arc_weights = gauss_interval(c.arc_nodes, 0, geometry.length)
        rr = geometry.s_to_r(self.arc_nodes)
        self.ids = np.repeat(np.arange(g.rib_count), c.arc_nodes)
        self.arc = np.tile(self.arc_nodes, g.rib_count)
        self.weights = np.tile(arc_weights, g.rib_count)
        self.radius = np.tile(rr, g.rib_count)
        self.theta = geometry.theta(self.radius, self.ids)
        self.centers = np.column_stack([self.radius * np.cos(self.theta),
                                        self.radius * np.sin(self.theta)])
        self.ring_points, self.area_weights = ring_quadrature(
            getattr(g, f"{region}_inner_radius"), g.outer_radius, c.radial_nodes, c.angular_nodes)
        xy = np.column_stack([self.ring_points[:, 0] * np.cos(self.ring_points[:, 1]),
                              self.ring_points[:, 0] * np.sin(self.ring_points[:, 1])])
        raw = self._raw_kernel(xy)
        self.normalization = self.area_weights @ raw
        if np.any(self.normalization < 1e-20):
            raise ValueError("Contact quadrature misses the Gaussian support; refine the grid")
        self.kernel = raw / self.normalization

    def _raw_kernel(self, xy):
        distance2 = ((xy[:, None, :] - self.centers[None, :, :])**2).sum(axis=2)
        return np.exp(-distance2 / self.config.contact.gaussian_sigma**2)

    def average(self, values):
        return (self.kernel * self.area_weights[:, None]).T @ values

    def project(self, line_power):
        return self.kernel @ (self.weights * line_power)

    def residual(self, line_power):
        ring_power = float(self.area_weights @ self.project(line_power))
        rib_power = float(self.weights @ line_power)
        magnitude = max(float(self.weights @ np.abs(line_power)), 1e-12)
        return {"ring_loss_w": ring_power, "rib_gain_w": rib_power,
                "relative_imbalance": abs(ring_power - rib_power) / magnitude}


@dataclass
class CouplingState:
    """Both ring and rib PDEs consume the same lagged contact powers in each round."""
    upper: ContactProjection
    lower: ContactProjection
    time: np.ndarray
    upper_power: np.ndarray
    lower_power: np.ndarray

    @classmethod
    def empty(cls, geometry, config, time_scale):
        upper = ContactProjection(geometry, config, "upper")
        lower = ContactProjection(geometry, config, "lower")
        times = np.linspace(0, time_scale, config.contact.time_nodes)
        zeros = np.zeros((len(times), len(upper.ids)))
        return cls(upper, lower, times, zeros.copy(), zeros.copy())

    def update(self, temperatures, thicknesses, material, relaxation):
        new = {}
        device, dtype = next(temperatures["rib"].parameters()).device, next(temperatures["rib"].parameters()).dtype
        for region in ("upper", "lower"):
            projection = getattr(self, region)
            powers = []
            for time in self.time:
                ring_points = np.column_stack([projection.ring_points[:, 0] / projection.geometry.config.outer_radius,
                                               projection.ring_points[:, 1],
                                               np.full(len(projection.ring_points), time / self.time[-1])])
                rib_points = np.column_stack([projection.arc / projection.geometry.config.outer_radius,
                                              projection.ids,
                                              np.full(len(projection.ids), time / self.time[-1])])
                xr = torch.as_tensor(ring_points, device=device, dtype=dtype)
                xb = torch.as_tensor(rib_points, device=device, dtype=dtype)
                with torch.no_grad():
                    p = material.physics
                    tr = p.ambient + p.temperature_scale * temperatures[region](xr)
                    tb = p.ambient + p.temperature_scale * temperatures["rib"](xb)
                    h = thicknesses[region](xr)
                    average_t = torch.as_tensor(projection.average(tr.cpu().numpy().ravel()), device=device, dtype=dtype).view(-1, 1)
                    average_h = torch.as_tensor(projection.average(h.cpu().numpy().ravel()), device=device, dtype=dtype).view(-1, 1)
                    kr, _ = material.factors(average_t)
                    kb, _ = material.factors(tb)
                    resistance = getattr(p, f"contact_resistance_{region}")
                    line = contact_line_power(average_t, tb, average_h, kr, kb,
                                              projection.geometry.config.rib_height,
                                              projection.geometry.config.rib_width, resistance)
                powers.append(line.cpu().numpy().ravel())
            previous = getattr(self, f"{region}_power")
            new[region] = (1 - relaxation) * previous + relaxation * np.asarray(powers)
        self.upper_power, self.lower_power = new["upper"], new["lower"]

    def _at_time(self, time, region):
        grid = torch.as_tensor(self.time, device=time.device, dtype=time.dtype)
        powers = torch.as_tensor(getattr(self, f"{region}_power"), device=time.device, dtype=time.dtype)
        index = torch.searchsorted(grid, time.contiguous()).clamp(1, len(grid) - 1)
        weight = ((time - grid[index - 1]) / (grid[index] - grid[index - 1])).clamp(0, 1)
        return powers[index - 1] + weight[:, None] * (powers[index] - powers[index - 1])

    def ring_loss(self, inputs, region, time_scale):
        projection = getattr(self, region)
        g = projection.geometry.config
        r, theta = inputs[:, 0:1] * g.outer_radius, inputs[:, 1:2]
        xy = torch.cat([r * theta.cos(), r * theta.sin()], dim=1)
        centers = torch.as_tensor(projection.centers, device=inputs.device, dtype=inputs.dtype)
        norm = torch.as_tensor(projection.normalization, device=inputs.device, dtype=inputs.dtype)
        arc_weights = torch.as_tensor(projection.weights, device=inputs.device, dtype=inputs.dtype)
        d2 = ((xy[:, None, :] - centers[None, :, :])**2).sum(dim=2)
        kernel = torch.exp(-d2 / projection.config.contact.gaussian_sigma**2) / norm
        line = self._at_time(inputs[:, 2] * time_scale, region)
        return (kernel * line * arc_weights).sum(dim=1, keepdim=True)

    def rib_gain(self, inputs, time_scale):
        projection = self.upper
        arc = torch.as_tensor(projection.arc_nodes, device=inputs.device, dtype=inputs.dtype)
        s = inputs[:, 0] * projection.geometry.config.outer_radius
        ids = inputs[:, 1].round().long()
        if (torch.any(inputs[:, 1] != ids) or torch.any(ids < 0)
                or torch.any(ids >= projection.geometry.config.rib_count)):
            raise ValueError("Invalid rib id in coupling evaluation")
        index = torch.searchsorted(arc, s.contiguous()).clamp(1, len(arc) - 1)
        weight = ((s - arc[index - 1]) / (arc[index] - arc[index - 1])).clamp(0, 1)
        powers = self._at_time(inputs[:, 2] * time_scale, "upper") + self._at_time(inputs[:, 2] * time_scale, "lower")
        offset = ids * len(arc)
        row = torch.arange(len(inputs), device=inputs.device)
        return (powers[row, offset + index - 1] * (1 - weight)
                + powers[row, offset + index] * weight).view(-1, 1)

    def diagnostics(self):
        return {region: [getattr(self, region).residual(row)
                         for row in getattr(self, f"{region}_power")]
                for region in ("upper", "lower")}

    def save(self, path):
        np.savez_compressed(path, time=self.time, upper_power=self.upper_power,
                            lower_power=self.lower_power)

    @classmethod
    def load(cls, path, geometry, config, time_scale):
        state = cls.empty(geometry, config, time_scale)
        with np.load(path, allow_pickle=False) as data:
            if not np.array_equal(data["time"], state.time):
                raise ValueError("Saved coupling time grid differs from configuration")
            for region in ("upper", "lower"):
                values = data[f"{region}_power"]
                if values.shape != state.upper_power.shape or not np.isfinite(values).all():
                    raise ValueError("Invalid saved contact powers")
                setattr(state, f"{region}_power", values.copy())
        return state
