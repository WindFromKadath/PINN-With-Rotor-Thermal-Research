"""matplotlib 绘图辅助（Agg 后端，函数内延迟导入，无 import 副作用）。

三联热图 / 末时刻剖面线的风格沿用原研究档案 DataPINO 线的
``data_compare.py`` 与 ``final_compare.py``。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_field_triplet(
    x: np.ndarray,
    t: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    path: str | Path,
    model_label: str = "Model",
) -> Path:
    """单样本三联图：真值 / 预测 / 绝对误差（u 形状 (nx, nt)）。

    热图按档案约定 ``imshow(u.T, extent=[x0, x1, t0, t1], origin='lower')``，
    横轴 x、纵轴 t。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    error = np.abs(u_true - u_pred)
    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
    panels = [
        (u_true, "Numerical Truth", "magma", ""),
        (u_pred, f"{model_label} Prediction", "magma", ""),
        (error, "Absolute Error", "Reds", f" (Max: {error.max():.2e})"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (data, title, cmap, suffix) in zip(axes, panels):
        im = ax.imshow(data.T, extent=extent, origin="lower", aspect="auto", cmap=cmap)
        ax.set_title(title + suffix)
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
        fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_profile_final_time(
    x: np.ndarray,
    t: np.ndarray,
    u_true: np.ndarray,
    preds: dict[str, np.ndarray],
    path: str | Path,
) -> Path:
    """末时刻（t=T_max）温度剖面线对比：真值实线 + 各模型预测。

    Args:
        x, t: 网格坐标（(nx,) 与 (nt,)）。
        u_true: 真值场 (nx, nt)。
        preds: ``{标签: 预测场 (nx, nt)}``。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = ["r--", "g-.", "b:", "m--", "c-."]
    plt.figure(figsize=(10, 6))
    plt.plot(x, u_true[:, -1], "k-", linewidth=2, label=f"Truth (t={t[-1]:.3g})")
    for i, (label, pred) in enumerate(preds.items()):
        plt.plot(x, pred[:, -1], styles[i % len(styles)], label=f"{label} (t={t[-1]:.3g})")
    plt.title("Temperature Profile at Final Time")
    plt.xlabel("x (m)")
    plt.ylabel("u (°C)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_multi_field_comparison(
    x: np.ndarray,
    t: np.ndarray,
    u_true: np.ndarray,
    preds: dict[str, np.ndarray],
    rel_l2: dict[str, float],
    path: str | Path,
) -> Path:
    """多模型四联热图：真值 + 各模型预测（标题含 RelL2），档案 final_compare 风格。"""
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    extent = [float(x[0]), float(x[-1]), float(t[0]), float(t[-1])]
    n_panels = 1 + len(preds)
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 5))
    im = axes[0].imshow(u_true.T, extent=extent, origin="lower", aspect="auto", cmap="magma")
    axes[0].set_title("Ground Truth (Numerical)")
    fig.colorbar(im, ax=axes[0])
    for ax, (label, pred) in zip(axes[1:], preds.items()):
        im = ax.imshow(pred.T, extent=extent, origin="lower", aspect="auto", cmap="magma")
        ax.set_title(f"{label}\nRelL2: {rel_l2[label]:.2%}")
        fig.colorbar(im, ax=ax)
    for ax in axes:
        ax.set_xlabel("x (m)")
        ax.set_ylabel("t (s)")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
