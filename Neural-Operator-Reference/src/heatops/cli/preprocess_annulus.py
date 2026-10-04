"""Assemble local COMSOL CSVs into portable physical-unit polar datasets."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import yaml
from heatops.data.comsol import assemble_cartesian_4d, cartesian_to_polar, save_polar_dataset
from heatops.training.run_context import config_path_value, file_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    src = config_path_value(args.config, cfg["source"])
    dest = config_path_value(args.config, cfg["smoke_output"] if args.smoke else cfg["output"])
    if dest.exists():
        raise FileExistsError(f"处理后数据已存在，拒绝覆盖：{dest.name}")
    paths = sorted(src.glob("q_*.csv"))
    if len(paths) != int(cfg["samples"]):
        raise ValueError(f"需要 {cfg['samples']} 个 CSV，实际找到 {len(paths)} 个")
    q = np.linspace(float(cfg["q_min"]), float(cfg["q_max"]), len(paths))
    indices = np.linspace(0, len(paths) - 1, 4, dtype=int) if args.smoke else np.arange(len(paths))
    nr, nth = (16, 16) if args.smoke else (int(cfg["nr"]), int(cfg["ntheta"]))
    r = np.linspace(float(cfg["r_in"]), float(cfg["r_out"]), nr)
    theta = np.linspace(0, 2 * np.pi, nth)
    fields, frames, source_hashes = [], None, {}
    for index in indices:
        ds = assemble_cartesian_4d([paths[index]], q[index:index + 1], length_unit=cfg["length_unit"])
        selected = np.linspace(0, len(ds["t"]) - 1, 10, dtype=int) if args.smoke else np.arange(len(ds["t"]))
        frames = ds["t"][selected]
        fields.append(cartesian_to_polar(ds["u"][:, selected], ds["x"], ds["y"], r, theta,
                                       center=tuple(cfg["center"]), mask=ds["mask"])[0].astype(np.float32))
        source_hashes[paths[index].name] = file_hash(paths[index])
    save_polar_dataset(dest, q[indices], np.stack(fields), frames, r, theta,
        metadata={"format_version": 1, "units": {"u": "°C", "q": "W/m²", "t": "s", "r": "m", "theta": "rad"},
                  "source_sha256": source_hashes, "sample_indices": indices.tolist(),
                  "time_source": "CSV @ t header", "boundary_interpolation": "nearest valid domain fill, then bilinear; out-of-grid query uses nearest valid point", "smoke": args.smoke})
    print(json.dumps({"output": dest.name, "shape": list(np.stack(fields).shape), "time_range": frames[[0, -1]].tolist(), "sha256": file_hash(dest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
