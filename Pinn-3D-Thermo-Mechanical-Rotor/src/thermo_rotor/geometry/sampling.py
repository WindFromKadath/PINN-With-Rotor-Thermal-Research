"""STL sampling in metres; caller owns all random state."""
from pathlib import Path
import numpy as np
import trimesh


def mesh(path, unit="mm"):
    if unit not in ("m", "mm"):
        raise ValueError("STL length unit must be m or mm")
    result = trimesh.load_mesh(Path(path), process=True)
    if not isinstance(result, trimesh.Trimesh) or len(result.faces) == 0:
        raise ValueError("STL must contain a nonempty triangle mesh")
    result.apply_scale(1e-3 if unit == "mm" else 1.)
    if not np.isfinite(result.vertices).all() or result.area <= 0:
        raise ValueError("Invalid STL geometry")
    return result


def surface(m, n, rng, flip=False):
    triangles = rng.choice(len(m.faces), n, p=m.area_faces/m.area)
    uv = rng.random((n, 2))
    root = np.sqrt(uv[:, 0:1])
    weights = np.column_stack((1-root, root*(1-uv[:, 1:2]), root*uv[:, 1:2]))
    points = (m.triangles[triangles]*weights[:, :, None]).sum(axis=1)
    normals = m.face_normals[triangles].copy()
    return points, -normals if flip else normals


def volume(m, n, rng, max_rounds=200):
    if not m.is_watertight:
        raise ValueError("Volume sampling requires a watertight STL")
    pieces, remaining = [], n
    for _ in range(max_rounds):
        candidates = rng.uniform(m.bounds[0], m.bounds[1], (max(256, remaining*4), 3))
        accepted = candidates[m.contains(candidates)][:remaining]
        pieces.append(accepted)
        remaining -= len(accepted)
        if remaining == 0:
            return np.concatenate(pieces)
    raise RuntimeError("Insufficient interior points in bounded rejection sampling")


def frame(points, time):
    import pandas as pd
    return pd.DataFrame(dict(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                             r=np.hypot(points[:, 0], points[:, 1]),
                             theta=np.arctan2(points[:, 1], points[:, 0]), t=time))


def cylindrical_normals(points, normals):
    angle = np.arctan2(points[:, 1], points[:, 0])
    nr = normals[:, 0]*np.cos(angle)+normals[:, 1]*np.sin(angle)
    return np.column_stack((nr, normals[:, 2]))
