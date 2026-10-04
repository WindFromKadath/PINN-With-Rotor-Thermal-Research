"""Noninteractive file-backed figures; no plotting takes place during import."""

import json
from pathlib import Path

import numpy as np
import pandas as pd


def plot_run(run_directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    directory = Path(run_directory)
    figures = directory / "figures"
    figures.mkdir(exist_ok=True)
    for region in ("upper", "lower", "rib"):
        path = directory / f"predictions_{region}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing predictions: {path.name}; evaluate the run before plotting")
        frame = pd.read_csv(path)
        part = frame[np.isclose(frame.t, frame.t.max())]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
        x, y = part.r * np.cos(part.theta), part.r * np.sin(part.theta)
        for axis, column in zip(axes, ("True_T", "predicted_T", "error_K")):
            plot = axis.scatter(x, y, c=part[column], s=12, cmap="coolwarm")
            axis.set(title=f"{region}: {column}", xlabel="x [m]", ylabel="y [m]", aspect="equal")
            fig.colorbar(plot, ax=axis, label="K")
        fig.savefig(figures / f"{region}_temperature.png", dpi=140)
        plt.close(fig)
    records = json.loads((directory / "losses.json").read_text(encoding="utf-8"))
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for region in ("upper", "lower", "rib"):
        entries = [r for r in records if r["region"] == region and r["stage"] != "thickness"]
        if entries:
            axis.semilogy([r["step"] for r in entries], [np.sum(r["loss"]) for r in entries], "o-", label=region)
    axis.set(xlabel="optimizer step", ylabel="sum of raw loss components", title="Recorded training losses")
    axis.legend()
    fig.savefig(figures / "losses.png", dpi=140)
    plt.close(fig)
    return figures
