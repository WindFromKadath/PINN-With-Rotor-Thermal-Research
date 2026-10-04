"""DeepXDE 线专属可视化（Agg 后端，函数内延迟导入，无 import 副作用）。

覆盖 ``heatops.evaluation.plots``（1D 时空三联图）未包含的 DeepXDE 线图形：
deepxde LossHistory 收敛曲线、2D 空间场三联图、圆环径向剖面对比与 2D 云图。
网格约定 ``indexing='ij'``（数组第一维为 x / r），绘图时转置。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_loss_history(
    loss_history,
    path: str | Path,
    title: str = "Training Loss Convergence",
) -> Path:
    """deepxde LossHistory 半对数收敛曲线：train 各分项 + test 总计。

    Args:
        loss_history: ``dde.model.LossHistory``（鸭子类型：需有 ``steps``、
            ``loss_train``、``loss_test`` 属性，均为逐步记录列表）。
        path: 输出 png 路径。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    steps = np.asarray(loss_history.steps)
    loss_train = np.asarray(loss_history.loss_train, dtype=np.float64)
    loss_test = np.asarray(loss_history.loss_test, dtype=np.float64)

    plt.figure(figsize=(10, 6))
    n_terms = loss_train.shape[1] if loss_train.ndim == 2 else 1
    for j in range(n_terms):
        plt.semilogy(steps, np.maximum(loss_train[:, j], 1e-30), label=f"train loss[{j}]")
    if loss_test.size:
        plt.semilogy(
            steps, np.maximum(loss_test.sum(axis=1), 1e-30),
            "k--", linewidth=1.5, label="test loss (sum)",
        )
    plt.title(title)
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_field_triplet_2d(
    x: np.ndarray,
    y: np.ndarray,
    u_true: np.ndarray,
    u_pred: np.ndarray,
    path: str | Path,
    model_label: str = "DeepONet",
    value_label: str = "T (°C)",
    title_suffix: str = "",
) -> Path:
    """单时刻 2D 场三联图：真值 / 预测 / 绝对误差（u 形状 (nx, ny)，ij 约定）。

    热图按档案约定 ``imshow(u.T, extent=[x0, x1, y0, y1], origin='lower')``，
    横轴 x、纵轴 y。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    error = np.abs(u_true - u_pred)
    extent = [float(x[0]), float(x[-1]), float(y[0]), float(y[-1])]
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
        ax.set_ylabel("y (m)")
        fig.colorbar(im, ax=ax, label=value_label)
    if title_suffix:
        fig.suptitle(title_suffix)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_radial_profile_comparison(
    r: np.ndarray,
    T_reference: np.ndarray,
    preds: dict[str, np.ndarray],
    path: str | Path,
    reference_label: str = "Analytical",
    title: str = "Radial Temperature Distribution",
) -> Path:
    """径向温度剖面对比（双面板：剖面线 + 绝对误差），圆环稳态评估用。

    Args:
        r: 物理半径 (nr,)，单位 m。
        T_reference: 参照剖面 (nr,)，单位 °C。
        preds: ``{标签: 预测剖面 (nr,)}``（可多条，如多个方位角）。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    styles = ["--", "-.", ":", "--"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(r, T_reference, "k-", linewidth=2, label=reference_label)
    for i, (label, pred) in enumerate(preds.items()):
        style = styles[i % len(styles)]
        ax1.plot(r, pred, style, linewidth=1.5, label=label)
        ax2.plot(r, np.abs(pred - T_reference), style, linewidth=1.5, label=label)
    ax1.set_xlabel("Radius r (m)")
    ax1.set_ylabel("Temperature T (°C)")
    ax1.set_title(title)
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax2.set_xlabel("Radius r (m)")
    ax2.set_ylabel("Absolute Error (°C)")
    ax2.set_title("Absolute Error vs Reference")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path


def plot_radial_profiles_times(
    r: np.ndarray,
    times: np.ndarray,
    T_reference: np.ndarray,
    T_pred: np.ndarray,
    path: str | Path,
    reference_label: str = "C-N reference",
    pred_label: str = "PINN",
    title: str = "Radial Temperature Evolution",
) -> Path:
    """多时刻径向温度分布对比：参照实线 vs 预测虚线（圆环瞬态评估用）。

    Args:
        r: 物理半径 (nr,)，单位 m。
        times: 快照时刻 (n_times,)，单位 s。
        T_reference, T_pred: 形状 (n_times, nr) 的剖面序列，单位 °C。
    """
    plt = _plt()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    colors = cmap(np.linspace(0.0, 0.9, len(times)))
    plt.figure(figsize=(10, 6))
    for i, (t, color) in enumerate(zip(times, colors)):
        plt.plot(r, T_reference[i], "-", color=color, linewidth=2,
                 label=f"{reference_label} t={t:.3g} s")
        plt.plot(r, T_pred[i], "--", color=color, linewidth=1.5,
                 label=f"{pred_label} t={t:.3g} s")
    plt.title(title)
    plt.xlabel("Radius r (m)")
    plt.ylabel("Temperature T (°C)")
    plt.legend(fontsize=8, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def plot_annulus_field(
    x: np.ndarray,
    y: np.ndarray,
    field: np.ndarray,
    mask: np.ndarray,
    r_in: float,
    r_out: float,
    path: str | Path,
    value_label: str = "T (°C)",
    title: str = "2D Temperature Distribution",
) -> Path:
    """圆环域 2D 云图（contourf + 内外边界圆），域外按 mask 留空。

    Args:
        x, y: 覆盖外接正方形的网格坐标 (nx,)、(ny,)。
        field: 温度场 (nx, ny)（ij 约定；域外取值任意，绘制前被掩码）。
        mask: 布尔数组 (nx, ny)，True 表示圆环域内。
        r_in, r_out: 内外半径（m），用于画边界圆。
    """
    plt = _plt()
    from matplotlib.patches import Circle

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    field_masked = np.where(mask, field, np.nan)
    X, Y = np.meshgrid(x, y, indexing="ij")
    fig, ax = plt.subplots(figsize=(7, 6))
    levels = np.linspace(np.nanmin(field_masked), np.nanmax(field_masked), 50)
    im = ax.contourf(X, Y, field_masked, levels=levels, cmap="jet", extend="both")
    ax.add_patch(Circle((0.0, 0.0), r_in, fill=False, color="red", linewidth=2))
    ax.add_patch(Circle((0.0, 0.0), r_out, fill=False, color="black", linewidth=2))
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label=value_label)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return path
