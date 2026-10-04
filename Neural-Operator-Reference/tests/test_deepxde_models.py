"""DeepXDE 线模型工厂 / 周期特征映射 / grid_sample 插值 / 训练运维件的单元测试。

deepxde 后端约定：``DDE_BACKEND`` 必须在首次 import deepxde 前固定为 pytorch
（本文件顶部 setdefault；测试进程若已显式指向其他后端则由
``ensure_dde_backend_pytorch`` 报错而非静默切换）。
"""

import os

os.environ.setdefault("DDE_BACKEND", "pytorch")  # 必须先于 import deepxde

import numpy as np
import pytest

torch = pytest.importorskip("torch")
dde = pytest.importorskip("deepxde")

from heatops.models.deepxde_models import (
    PeriodicFeatureTransform,
    ensure_dde_backend_pytorch,
    make_deeponet_cartesian,
    make_fnn,
    make_linear_output_transform,
)
from heatops.physics.interpolation import make_grid_field_sampler
from heatops.training.deepxde_loop import (
    find_best_checkpoint,
    load_weights_strict,
    set_seed,
    tee_stdout,
    zip_artifacts,
)


# ---------------------------------------------------------------- 后端守卫


def test_ensure_backend_idempotent():
    """后端守卫幂等：重复调用不报错且环境变量保持 pytorch。"""
    ensure_dde_backend_pytorch()
    ensure_dde_backend_pytorch()
    assert os.environ["DDE_BACKEND"] == "pytorch"
    assert dde.backend.backend_name == "pytorch"


# ---------------------------------------------------------------- 周期特征映射


def test_periodic_feature_transform_identity():
    """恒等性：θ 与 θ+2πk 的映射结果逐位一致；非角度列保持原位。"""
    tfm = PeriodicFeatureTransform(theta_col=1)
    x = torch.tensor([[0.7, 0.3, 0.1], [0.5, 1.0, 0.9], [0.9, -2.0, 0.4]], dtype=torch.float64)
    y0 = tfm(x)
    x_shifted = x.clone()
    x_shifted[:, 1] += 2.0 * np.pi * torch.tensor([1.0, -3.0, 5.0], dtype=torch.float64)
    y1 = tfm(x_shifted)
    assert y0.shape == (3, 4)  # n 列 → n+1 列
    assert tfm.output_dim(3) == 4
    assert torch.allclose(y0, y1, atol=1e-12), "θ+2πk 应与 θ 映射一致"
    # 非角度列原位保留：第 0 列与最后列不变
    assert torch.allclose(y0[:, 0], x[:, 0])
    assert torch.allclose(y0[:, -1], x[:, -1])
    # 中间两列为 (sin θ, cos θ)
    assert torch.allclose(y0[:, 1], torch.sin(x[:, 1]))
    assert torch.allclose(y0[:, 2], torch.cos(x[:, 1]))


def test_periodic_feature_transform_column_placement_and_grad():
    """theta_col=0 时首列被替换；变换对输入可微（供 PDE 残差链式求导）。"""
    tfm = PeriodicFeatureTransform(theta_col=0)
    x = torch.tensor([[0.3, 2.0]], dtype=torch.float32, requires_grad=True)
    y = tfm(x)
    assert y.shape == (1, 3)
    assert torch.allclose(y[:, 2], x[:, 1])  # 原第 1 列移到最后
    y.sum().backward()
    assert x.grad is not None and torch.all(torch.isfinite(x.grad))
    assert float(x.grad[0, 1]) == 1.0  # 非角度列梯度为 1


# ---------------------------------------------------------------- 线性输出变换


def test_linear_output_transform_affine():
    """y_phys = offset + scale·y_norm 精确成立，且可挂到 FNN 的输出端。"""
    transform = make_linear_output_transform(offset=20.0, scale=294.0)
    x = torch.zeros(3, 2)
    y_norm = torch.tensor([[0.0], [1.0], [-0.5]])
    y_phys = transform(x, y_norm)
    assert torch.allclose(y_phys, torch.tensor([[20.0], [314.0], [-127.0]]))

    torch.manual_seed(0)
    net = make_fnn([2, 8, 1], output_transform=transform)
    raw = make_fnn([2, 8, 1], output_transform=None)
    raw.load_state_dict(net.state_dict(), strict=True)  # 同构网络权重一致
    xin = torch.rand(4, 2)
    assert torch.allclose(net(xin), 20.0 + 294.0 * raw(xin), atol=1e-6)


# ---------------------------------------------------------------- 网络工厂


def test_make_deeponet_cartesian_output_shape():
    """DeepONetCartesianProd 前向输出 (n_branch, n_trunk)；末层宽度不一致时报错。"""
    torch.manual_seed(0)
    net = make_deeponet_cartesian([9, 16, 8], [3, 16, 8])
    branch = torch.rand(4, 9)
    trunk = torch.rand(7, 3)
    out = net((branch, trunk))
    assert out.shape == (4, 7)

    with pytest.raises(ValueError, match="末层宽度"):
        make_deeponet_cartesian([9, 16, 8], [3, 16, 4])


def test_make_fnn_with_feature_transform():
    """挂周期映射的 FNN：输入 3 列、网络首维 4，前向形状正确。"""
    torch.manual_seed(0)
    net = make_fnn([4, 8, 8, 1], feature_transform=PeriodicFeatureTransform(theta_col=1))
    out = net(torch.rand(5, 3))
    assert out.shape == (5, 1)
    # θ 与 θ+2π 前向结果一致（周期性的端到端验证）
    x = torch.rand(5, 3)
    x2 = x.clone()
    x2[:, 1] += 2.0 * np.pi
    assert torch.allclose(net(x), net(x2), atol=1e-6)


# ---------------------------------------------------------------- grid_sample 插值


def test_grid_sampler_nodes_exact_and_plane_bilinear():
    """格点取值精确还原；双线性场（平面）内部插值精确。"""
    x1 = np.linspace(0.0, 1.0, 6)
    y1 = np.linspace(0.0, 1.0, 5)
    X, Y = np.meshgrid(x1, y1, indexing="ij")
    f = 2.0 * X + 3.0 * Y  # 双线性可精确表示的平面场
    sampler = make_grid_field_sampler(f)

    # 格点精确
    pts = np.stack([X.ravel(), Y.ravel()], axis=1)
    val = sampler(torch.as_tensor(pts, dtype=torch.float64))
    assert val.shape == (pts.shape[0], 1)
    assert np.allclose(val.detach().numpy().ravel(), f.ravel(), atol=1e-5)

    # 内部随机点双线性精确（平面场）
    rng = np.random.default_rng(0)
    pts_rand = rng.random((17, 2))
    val_rand = sampler(torch.as_tensor(pts_rand, dtype=torch.float64)).detach().numpy().ravel()
    assert np.allclose(val_rand, 2.0 * pts_rand[:, 0] + 3.0 * pts_rand[:, 1], atol=1e-5)


def test_grid_sampler_border_padding_and_extra_columns():
    """域外按边界常数外延（border）；多于两列的输入（如 t 列）被忽略。"""
    f = np.arange(12.0).reshape(4, 3)
    sampler = make_grid_field_sampler(f, x_range=(0.0, 3.0), y_range=(0.0, 2.0))
    # 域外查询 → 最近边界值
    out = sampler(torch.tensor([[-1.0, 1.0], [99.0, 2.0]]))
    assert np.isclose(float(out[0]), f[0, 1], atol=1e-5)   # x<0 → x=0 边
    assert np.isclose(float(out[1]), f[-1, -1], atol=1e-5)  # 右上角外延
    # 第三列（t）不影响取值
    pts2 = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 123.0]])
    out2 = sampler(pts2)
    assert torch.allclose(out2[0], out2[1])


def test_grid_sampler_rejects_bad_field():
    """非二维场或非法区间明确报错。"""
    with pytest.raises(ValueError, match="二维"):
        make_grid_field_sampler(np.zeros((3, 3, 3)))
    with pytest.raises(ValueError, match="递增"):
        make_grid_field_sampler(np.zeros((3, 3)), x_range=(1.0, 1.0))


# ---------------------------------------------------------------- 训练运维件


def test_set_seed_reproducible():
    """set_seed 后 numpy/torch/deepxde 函数空间采样序列可复现。"""
    set_seed(7)
    a_np = np.random.rand(4)
    a_torch = torch.rand(4)
    grf = dde.data.GRF(length_scale=0.3, N=50)
    a_grf = grf.random(2)
    set_seed(7)
    assert np.array_equal(a_np, np.random.rand(4))
    assert torch.equal(a_torch, torch.rand(4))
    assert np.array_equal(a_grf, grf.random(2))


def test_tee_stdout_and_zip_artifacts(tmp_path):
    """tee 日志双写且不泄漏；zip 打包包含全部产物且排除 .zip 本身。"""
    run_dir = tmp_path / "artifacts" / "T900"
    run_dir.mkdir(parents=True)
    (run_dir / "dummy.txt").write_text("data", encoding="utf-8")
    with tee_stdout(run_dir / "train_log.txt"):
        print("hello-tee")
    assert "hello-tee" in (run_dir / "train_log.txt").read_text(encoding="utf-8")

    zip_path = zip_artifacts(run_dir)
    assert zip_path.exists()
    import zipfile

    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
    assert any(name.endswith("dummy.txt") for name in names)
    assert any(name.endswith("train_log.txt") for name in names)
    assert not any(name.endswith(".zip") for name in names)


def test_find_best_checkpoint(tmp_path):
    """取步号最大的 best-*.pt；无候选时返回 None。"""
    for step in (0, 150, 40):
        (tmp_path / f"best-{step}.pt").touch()
    assert find_best_checkpoint(tmp_path) == tmp_path / "best-150.pt"
    assert find_best_checkpoint(tmp_path / "empty") is None


def test_load_weights_strict_roundtrip(tmp_path):
    """严格加载：同构网络权重往返一致；架构不匹配时 RuntimeError（strict=True）。"""
    torch.manual_seed(0)
    net = make_deeponet_cartesian([4, 8, 8], [2, 8, 8])
    data = dde.data.TripleCartesianProd(
        X_train=(np.random.rand(3, 4).astype(np.float32), np.random.rand(5, 2).astype(np.float32)),
        y_train=np.random.rand(3, 5).astype(np.float32),
        X_test=(np.random.rand(2, 4).astype(np.float32), np.random.rand(5, 2).astype(np.float32)),
        y_test=np.random.rand(2, 5).astype(np.float32),
    )
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-3)

    ckpt_path = tmp_path / "best-0.pt"
    torch.save({"model_state_dict": net.state_dict()}, ckpt_path)
    with torch.no_grad():
        for p in net.parameters():
            p.add_(1.0)  # 打乱权重
    load_weights_strict(model, ckpt_path)
    torch.manual_seed(0)
    ref = make_deeponet_cartesian([4, 8, 8], [2, 8, 8])
    for p_loaded, p_ref in zip(model.net.parameters(), ref.parameters()):
        assert torch.equal(p_loaded, p_ref), "strict 加载后权重应与保存时一致"

    # 架构不匹配：strict=True 必须报错
    model_bad = dde.Model(data, make_deeponet_cartesian([4, 8, 9], [2, 8, 9]))
    model_bad.compile("adam", lr=1e-3)
    with pytest.raises(RuntimeError):
        load_weights_strict(model_bad, ckpt_path)
