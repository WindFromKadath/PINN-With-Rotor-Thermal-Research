"""evaluation.metrics 校核：已知向量手算值对照、分批预测一致性、窗口指标。"""

import numpy as np
import pytest

from heatops.evaluation import (
    batched_predict,
    mean_relative_error,
    relative_l2,
    rmse,
    windowed_metrics,
)


def test_relative_l2_known_values():
    """RelL2 手算对照：零误差为 0；||y||=5 时零预测得 1；正交单位向量得 √2。"""
    assert relative_l2(np.array([3.0, 4.0]), np.array([3.0, 4.0])) == 0.0
    assert np.isclose(relative_l2(np.array([3.0, 4.0]), np.zeros(2)), 1.0, rtol=1e-15)
    # y_true=[1,0]（范数 1），y_pred=[0,1]，差向量范数 √2
    assert np.isclose(relative_l2(np.array([1.0, 0.0]), np.array([0.0, 1.0])), np.sqrt(2.0), rtol=1e-15)


def test_mean_relative_error_known_values():
    """MRE 手算对照：|1-2|/2=0.5 与 0 的均值为 0.25（eps 默认 1e-12，可忽略）。"""
    val = mean_relative_error(np.array([2.0, 4.0]), np.array([1.0, 4.0]))
    assert np.isclose(val, 0.25, rtol=0, atol=1e-11)


def test_rmse_known_values():
    """RMSE 手算对照：差 [0, 2] → sqrt((0+4)/2) = √2。"""
    assert np.isclose(rmse(np.array([1.0, 2.0]), np.array([1.0, 4.0])), np.sqrt(2.0), rtol=1e-15)


def test_batched_predict_matches_single_call():
    """分批预测与一次性预测结果一致（含不能整除的批大小与元组输入）。"""
    rng = np.random.default_rng(0)
    n, d_in, d_out = 10, 6, 4
    X = rng.normal(size=(n, d_in))
    W = rng.normal(size=(d_in, d_out))
    b = rng.normal(size=d_out)

    def model_fn(xb):
        return xb @ W + b

    expected = model_fn(X)
    for batch_size in (1, 3, 4, n, 100):
        out = batched_predict(model_fn, X, batch_size=batch_size)
        # 分批 GEMM 与一次性 GEMM 走不同 BLAS 路径，允许末位 ulp 级差异
        assert np.allclose(out, expected, rtol=1e-12, atol=1e-13)

    # DeepONet 风格的 (branch, trunk) 元组输入
    X_tr = rng.normal(size=(n, 2))

    def deeponet_like(pair):
        xb, xt = pair
        return xb @ W + xt[:, :1] * b

    expected_tuple = deeponet_like((X, X_tr))
    out_tuple = batched_predict(deeponet_like, (X, X_tr), batch_size=4)
    assert np.allclose(out_tuple, expected_tuple, rtol=1e-12, atol=1e-13)


def test_batched_predict_invalid_batch_size():
    with pytest.raises(ValueError):
        batched_predict(lambda x: x, np.zeros((4, 2)), batch_size=0)


def test_windowed_metrics():
    """窗口指标：仅中间窗口被扰动时，其余窗口指标为零、该窗口为手算值。"""
    y_true = np.ones((4, 10))
    y_pred = y_true.copy()
    # 第 2 个窗口（共 5 窗，每窗 2 列）内预测减半：RelL2 = ||t/2||/||t|| = 0.5
    y_pred[:, 4:6] = 0.5

    values = windowed_metrics(y_true, y_pred, n_windows=5, axis=-1)
    assert values.shape == (5,)
    assert np.allclose(values[[0, 1, 3, 4]], 0.0, atol=1e-15)
    assert np.isclose(values[2], 0.5, rtol=1e-12)

    # 不能整除时窗口数仍正确（np.array_split 约定）
    values3 = windowed_metrics(y_true, y_pred, n_windows=3, axis=-1)
    assert values3.shape == (3,)
    assert np.all(values3 >= 0.0)


def test_windowed_metrics_shape_mismatch():
    with pytest.raises(ValueError):
        windowed_metrics(np.ones((2, 4)), np.ones((2, 5)), n_windows=2)
