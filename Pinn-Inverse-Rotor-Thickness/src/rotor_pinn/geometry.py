"""Analytic theta(r), exact arc integration and integer rib assignments."""

import numpy as np
from scipy.optimize import brentq

from .config import Geometry


class RotorGeometry:
    def __init__(self, config: Geometry):
        self.config = config
        self.length = float(self.r_to_s(config.rib_end))

    def _primitive(self, r):
        b = self.config.rib_slope
        r = np.asarray(r, dtype=float)
        if abs(b) < 1e-14:
            return r
        return (r * np.sqrt(1 + (b * r) ** 2) + np.arcsinh(b * r) / b) / 2

    def r_to_s(self, r):
        r = np.asarray(r, dtype=float)
        g = self.config
        if np.any(~np.isfinite(r)) or np.any((r < g.rib_start - 1e-9) | (r > g.rib_end + 1e-9)):
            raise ValueError("Rib radius outside configured range")
        return self._primitive(np.clip(r, g.rib_start, g.rib_end)) - self._primitive(g.rib_start)

    def s_to_r(self, s):
        g = self.config
        values = np.asarray(s, dtype=float)
        if np.any(~np.isfinite(values)) or np.any((values < -1e-9) | (values > self.length + 1e-9)):
            raise ValueError("Arc coordinate outside rib length")
        flat = np.clip(values.reshape(-1), 0, self.length)
        result = [brentq(lambda r: float(self.r_to_s(r)) - v, g.rib_start, g.rib_end)
                  for v in flat]
        return np.asarray(result).reshape(values.shape)

    def theta(self, r, rib_id):
        g = self.config
        return np.mod(g.rib_theta0 + g.rib_slope * (np.asarray(r) - g.rib_start)
                      + 2 * np.pi * np.asarray(rib_id) / g.rib_count, 2 * np.pi)

    def assign_rib_id(self, r, theta):
        r, theta = np.broadcast_arrays(np.asarray(r), np.asarray(theta))
        centers = self.theta(r[..., None], np.arange(self.config.rib_count))
        distance = np.abs(np.angle(np.exp(1j * (theta[..., None] - centers))))
        return distance.argmin(axis=-1)


def gauss_interval(count, start, end):
    nodes, weights = np.polynomial.legendre.leggauss(count)
    return start + (nodes + 1) * (end - start) / 2, weights * (end - start) / 2


def ring_quadrature(inner, outer, radial_nodes, angular_nodes):
    radius, radial_weights = gauss_interval(radial_nodes, inner, outer)
    theta = np.arange(angular_nodes) * (2 * np.pi / angular_nodes)
    rr, tt = np.meshgrid(radius, theta, indexing="ij")
    area = np.repeat(radial_weights * radius * (2 * np.pi / angular_nodes), angular_nodes)
    return np.column_stack([rr.ravel(), tt.ravel()]), area
