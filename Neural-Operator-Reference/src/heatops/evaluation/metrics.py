"""评估指标与分批预测工具（纯 numpy 实现）。

约定：所有指标函数把输入展平为向量计算；输入可以是任意可被
``np.asarray`` 转换的对象（包括 CPU 上的 torch 张量）。
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np


def relative_l2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """相对 L2 误差 ``||y_true - y_pred||₂ / ||y_true||₂``（无量纲）。"""
    yt = np.asarray(y_true, dtype=np.float64).ravel()
    yp = np.asarray(y_pred, dtype=np.float64).ravel()
    return float(np.linalg.norm(yt - yp) / np.linalg.norm(yt))


def mean_relative_error(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """平均逐点相对误差 ``mean(|y_pred - y_true| / (|y_true| + eps))``（无量纲）。

    ``eps`` 为防止除零的小量；对真值接近零的点，分母由 eps 兜底。
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(yp - yt) / (np.abs(yt) + eps)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """均方根误差 ``sqrt(mean((y_true - y_pred)²))``（与输入同单位）。"""
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def batched_predict(
    model_fn: Callable,
    X: np.ndarray | Sequence[np.ndarray],
    batch_size: int = 1024,
) -> np.ndarray:
    """分批预测包装：沿第 0 维分批调用 ``model_fn`` 并拼接，防止一次性预测 OOM。

    Args:
        model_fn: 预测函数，接受与 ``X`` 同结构的分批输入
            （单个数组或数组元组），返回可被 ``np.asarray`` 转换的预测值。
        X: 输入；单个形状 ``(n_samples, ...)`` 的数组，或共享同一
            ``n_samples`` 的数组元组（如 DeepONet 的 ``(branch, trunk)`` 输入）。
        batch_size: 每批样本数，必须为正整数。

    Returns:
        沿第 0 维拼接后的预测数组，与一次性调用 ``model_fn(X)`` 的结果一致。
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size 必须为正整数，实际为 {batch_size}")

    is_tuple = isinstance(X, (tuple, list))
    parts = [np.asarray(x) for x in X] if is_tuple else [np.asarray(X)]
    n_samples = parts[0].shape[0]
    if any(p.shape[0] != n_samples for p in parts):
        raise ValueError("输入各数组的第 0 维（样本数）不一致")

    preds = []
    for start in range(0, n_samples, batch_size):
        sl = slice(start, min(start + batch_size, n_samples))
        batch = tuple(p[sl] for p in parts) if is_tuple else parts[0][sl]
        preds.append(np.asarray(model_fn(batch)))
    return np.concatenate(preds, axis=0)


def windowed_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_windows: int,
    axis: int = -1,
    metric_fn: Callable[[np.ndarray, np.ndarray], float] = relative_l2,
) -> np.ndarray:
    """分段窗口指标：沿指定轴等分为若干窗口，逐窗口计算指标。

    Args:
        y_true, y_pred: 形状相同的真值与预测数组。
        n_windows: 窗口数（正整数）。
        axis: 分段轴，默认最后一维（通常为时间轴）。
        metric_fn: 窗口指标函数 ``(y_true_w, y_pred_w) -> float``。

    Returns:
        形状 (n_windows,) 的指标值数组；不能整除时靠前的窗口略大
        （``np.array_split`` 约定）。
    """
    yt = np.asarray(y_true, dtype=np.float64)
    yp = np.asarray(y_pred, dtype=np.float64)
    if yt.shape != yp.shape:
        raise ValueError(f"y_true 与 y_pred 形状不一致：{yt.shape} vs {yp.shape}")
    if n_windows <= 0:
        raise ValueError(f"n_windows 必须为正整数，实际为 {n_windows}")

    true_windows = np.array_split(yt, n_windows, axis=axis)
    pred_windows = np.array_split(yp, n_windows, axis=axis)
    return np.asarray(
        [metric_fn(tw, pw) for tw, pw in zip(true_windows, pred_windows)]
    )
