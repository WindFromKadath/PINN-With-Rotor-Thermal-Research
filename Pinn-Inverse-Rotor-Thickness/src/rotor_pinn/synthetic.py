"""Smooth synthetic observations for exercising I/O; not physical ground truth."""

import numpy as np
import pandas as pd

from .geometry import RotorGeometry


def demo_frames(config):
    g, p = config.geometry, config.physics
    geometry = RotorGeometry(g)
    times = np.linspace(0, 1200, 9)
    result = {}
    for region in ("upper", "lower"):
        inner = getattr(g, f"{region}_inner_radius")
        rr, th, tt = np.meshgrid(np.linspace(inner, g.outer_radius, 8),
                                 np.arange(16) * 2 * np.pi / 16, times, indexing="ij")
        r, theta, time = rr.ravel(), th.ravel(), tt.ravel()
        amplitude = 100 if region == "upper" else 60
        temperature = (p.ambient + amplitude * (1 - np.exp(-time / 300))
                       * (0.6 + 0.4 * np.exp(-((r - inner) / 0.06)**2))
                       * (1 + 0.1 * np.cos(3 * theta)))
        thickness = ((0.009 if region == "upper" else 0.012)
                     * (1 + 0.1 * np.sin(theta) + 0.04 * r / g.outer_radius))
        result[region] = pd.DataFrame({"r": r, "theta": theta, "t": time,
                                       "True_T": temperature, "thickness": thickness})
    ss, ids, tt = np.meshgrid(np.linspace(0, geometry.length, 10),
                              np.arange(g.rib_count), times, indexing="ij")
    s, rib_ids, time = ss.ravel(), ids.ravel(), tt.ravel()
    radius = geometry.s_to_r(s)
    result["rib"] = pd.DataFrame({"r": radius, "theta": geometry.theta(radius, rib_ids),
                                  "t": time, "rib_id": rib_ids,
                                  "True_T": p.ambient + (60 + 20 * np.cos(2 * np.pi * rib_ids / g.rib_count))
                                  * (1 - np.exp(-time / 350))
                                  * (0.85 + 0.15 * np.cos(np.pi * s / geometry.length))})
    return result
