"""Standalone figures regenerated from saved predictions, with no interactive UI."""

from pathlib import Path


def plot_run(directory):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    directory = Path(directory)
    files = sorted(directory.glob("domain_*_predictions.csv"))
    if not files:
        raise FileNotFoundError("No saved domain predictions; run evaluate first")
    figure, axes = plt.subplots(len(files), 2, figsize=(10, 3.2*len(files)), squeeze=False)
    for row, path in enumerate(files):
        data = pd.read_csv(path)
        label = path.stem.removeprefix("domain_").removesuffix("_predictions")
        sampled = data.sample(n=min(len(data), 3000), random_state=42)
        axes[row, 0].scatter(sampled.True_T, sampled.Pred_T, s=4, alpha=0.4)
        limits = [min(data.True_T.min(), data.Pred_T.min()), max(data.True_T.max(), data.Pred_T.max())]
        axes[row, 0].plot(limits, limits, "k--")
        axes[row, 0].set(xlabel="Reference temperature [K]", ylabel="Prediction [K]", title=label)
        averaged = data.groupby("t")[["True_T", "Pred_T"]].mean()
        axes[row, 1].plot(averaged.index, averaged.True_T, label="reference mean")
        axes[row, 1].plot(averaged.index, averaged.Pred_T, label="prediction mean")
        axes[row, 1].set(xlabel="Time [s]", ylabel="Temperature [K]")
        axes[row, 1].legend()
    figure.tight_layout()
    figure.savefig(directory / "temperature_comparison.png", dpi=150)
    plt.close(figure)
    figure, axes = plt.subplots(1, len(files), figsize=(4*len(files), 3.3), squeeze=False)
    for ax, path in zip(axes[0], files):
        error = pd.read_csv(path).error_K
        ax.hist(error, bins=50)
        ax.set(xlabel="Prediction - reference [K]", ylabel="Count",
               title=path.stem.removeprefix("domain_").removesuffix("_predictions"))
        ax.axvline(np.mean(error), color="red")
    figure.tight_layout()
    figure.savefig(directory / "error_histograms.png", dpi=150)
    plt.close(figure)
    return directory / "temperature_comparison.png"
