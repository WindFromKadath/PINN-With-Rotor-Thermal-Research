from pathlib import Path
import numpy as np


def plot_predictions(frame, prediction, truth, kind, path):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    labels = ["Temperature / K"] if kind in ("thermal", "steady") else ["Radial displacement / um", "Axial displacement / um"]
    factor = 1 if kind in ("thermal", "steady") else 1e6
    figures = []
    for j, label in enumerate(labels):
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True)
        axes[0].scatter(truth[:, j]*factor, prediction[:, j]*factor, s=3, alpha=.5)
        limits = [min(truth[:, j].min(), prediction[:, j].min())*factor,
                  max(truth[:, j].max(), prediction[:, j].max())*factor]
        axes[0].plot(limits, limits, color="black", lw=.8)
        axes[0].set(xlabel="Reference", ylabel="Prediction", title=label)
        latest = frame.t == frame.t.max()
        s = axes[1].scatter(frame.loc[latest, "r"]*1e3, frame.loc[latest, "z"]*1e3,
                            c=(prediction[latest, j]-truth[latest, j])*factor, s=6, cmap="coolwarm")
        axes[1].set(xlabel="r / mm", ylabel="z / mm", title="Error at final sample time")
        fig.colorbar(s, ax=axes[1])
        f = path/f"{kind}_{j}.png"
        fig.savefig(f, dpi=160)
        plt.close(fig)
        figures.append(f.name)
    return figures
