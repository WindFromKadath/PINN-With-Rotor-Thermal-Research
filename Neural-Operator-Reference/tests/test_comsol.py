"""comsol：CSV 读取（mm→m、comment='%'）、4D 装配、极坐标往返、重采样与数据集存取。"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from heatops.data.comsol import (
    assemble_cartesian_4d,
    cartesian_to_polar,
    load_polar_dataset,
    polar_to_cartesian,
    read_comsol_csv,
    resample_spatial,
    save_polar_dataset,
)
from heatops.data.comsol_batch import main as comsol_batch_main


def _write_synthetic_csv(path, x_mm, y_mm, temps, comments=("% COMSOL 导出", "% x, y, T1..Tn")):
    """写合成 COMSOL CSV：注释行 + ``x[mm], y[mm], T1..Tn`` 数据行。

    temps: 形状 (npts, nt) 数组；np.nan 表示域外点。
    """
    lines = list(comments)
    for (x, y), row in zip([(x, y) for x in x_mm for y in y_mm], temps):
        vals = ", ".join(f"{v}" if not np.isnan(v) else "nan" for v in row)
        lines.append(f"{x}, {y}, {vals}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_comsol_csv_mm_to_m_and_comments(tmp_path):
    """读取合成 CSV：注释行跳过、x/y 由 mm 换算为 m、时刻列齐全。"""
    x_mm = [0.0, 100.0, 200.0]
    y_mm = [0.0, 100.0]
    temps = 20.0 + np.arange(6 * 3, dtype=float).reshape(6, 3)
    path = tmp_path / "q_000.csv"
    _write_synthetic_csv(path, x_mm, y_mm, temps)

    df = read_comsol_csv(path, length_unit="mm")
    assert list(df.columns) == ["x", "y", "T_0", "T_1", "T_2"]
    assert df.shape == (6, 5)
    assert df["x"].max() == pytest.approx(0.2)  # 200 mm → 0.2 m
    assert df["T_2"].iloc[-1] == pytest.approx(temps[-1, 2])

    df_m = read_comsol_csv(path, length_unit="m")
    assert df_m["x"].max() == pytest.approx(200.0)  # m 模式不换算

    with pytest.raises(FileNotFoundError, match="未入库，本地归档"):
        read_comsol_csv(tmp_path / "missing.csv")


def test_assemble_cartesian_4d_grid_and_mask(tmp_path):
    """两个 CSV 装配为 (2, nt, nx, ny)；NaN 提取掩码后填 0；单位保持物理量。"""
    x_mm = [0.0, 100.0, 200.0]
    y_mm = [0.0, 100.0]
    temps_a = 20.0 + np.arange(6 * 2, dtype=float).reshape(6, 2)
    temps_b = temps_a + 100.0
    temps_b[0] = np.nan  # 首样本之外：第二个 CSV 的 (0,0) 点域外
    paths = []
    for i, temps in enumerate((temps_a, temps_b)):
        p = tmp_path / f"q_{i:03d}.csv"
        _write_synthetic_csv(p, x_mm, y_mm, temps)
        paths.append(p)

    out = assemble_cartesian_4d(paths, q_values=np.array([1e5, 2e5]), t_values=np.array([1.0, 2.0]))
    assert out["u"].shape == (2, 2, 3, 2)  # (样本, 时刻, nx, ny)
    assert out["mask"].shape == (3, 2)
    assert out["mask"].all()  # 掩码取自首样本首帧（无 NaN）
    assert out["u"][1, 0, 0, 0] == 0.0  # 域外 NaN 已填 0
    assert out["u"][0, 1, 2, 1] == pytest.approx(temps_a[5, 1])  # 物理单位 °C 不归一化
    assert np.allclose(out["q"], [1e5, 2e5])
    assert np.allclose(out["x"], [0.0, 0.1, 0.2])  # mm → m

    with pytest.raises(ValueError, match="不一致"):
        assemble_cartesian_4d(paths, q_values=np.array([1e5]))
    with pytest.raises(FileNotFoundError, match="未入库"):
        assemble_cartesian_4d([tmp_path / "missing.csv"], q_values=np.array([1e5]))


def test_cartesian_polar_roundtrip_mean_conserved():
    """Cartesian → 极坐标 → 回原点的往返：均值守恒且内部点逐点吻合（光滑场）。

    圆环域内点的两次双线性插值误差应远小于场量级；判据：域内样本点往返
    均值相对偏差 <1%，逐点绝对偏差 <0.05（场 O(1)）。
    """
    nx = ny = 80
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = np.sin(2 * np.pi * X) * np.cos(2 * np.pi * Y) + 0.5 * X  # 光滑 O(1) 场

    center = (0.5, 0.5)
    r_in, r_out = 0.2, 0.4
    r_grid = np.linspace(r_in, r_out, 24)
    theta_grid = np.linspace(0.0, 2 * np.pi, 48)  # 含两端点（周期母线）
    u_polar = cartesian_to_polar(field, x, y, r_grid, theta_grid, center=center)
    assert u_polar.shape == (24, 48)

    # 域内采样点（避开极坐标网格边界带一拍）
    rng = np.random.default_rng(0)
    n_pts = 500
    rq = rng.uniform(r_in + 0.02, r_out - 0.02, n_pts)
    tq = rng.uniform(0.0, 2 * np.pi, n_pts)
    xq = center[0] + rq * np.cos(tq)
    yq = center[1] + rq * np.sin(tq)
    orig = np.sin(2 * np.pi * xq) * np.cos(2 * np.pi * yq) + 0.5 * xq
    back = polar_to_cartesian(u_polar, r_grid, theta_grid, xq, yq, center=center)
    assert not np.isnan(back).any()  # 域内点不触发 NaN 填充
    assert abs(back.mean() - orig.mean()) / max(abs(orig).max(), 1e-12) < 0.01  # 均值守恒
    assert np.abs(back - orig).max() < 0.05


def test_resample_spatial_shapes_and_modes():
    """重采样形状正确；constant 场 bilinear 保持常数；nearest 保持 0/1 掩码属性。"""
    const = np.full((2, 3, 8, 8), 7.0)
    out = resample_spatial(const, (4, 4), mode="bilinear")
    assert out.shape == (2, 3, 4, 4)
    assert np.allclose(out, 7.0)

    mask = np.zeros((8, 8))
    mask[2:6, 2:6] = 1.0
    out_m = resample_spatial(mask, (16, 16), mode="nearest")
    assert out_m.shape == (16, 16)
    assert set(np.unique(out_m)) <= {0.0, 1.0}


def test_polar_dataset_save_load_roundtrip(tmp_path):
    """极坐标数据集 .pt 往返：张量与 metadata 保持；缺失文件抛未入库提示。"""
    q = np.array([1e5, 2e5])
    u = np.random.default_rng(0).normal(size=(2, 3, 4, 5))
    t = np.arange(1, 4, dtype=float)
    r = np.linspace(0.05, 0.147, 4)
    theta = np.linspace(0.0, 2 * np.pi, 5)
    path = save_polar_dataset(tmp_path / "polar.pt", q, u, t, r, theta,
                              metadata={"generator": "test", "units": {"u": "°C"}})

    ds = load_polar_dataset(path)
    assert torch.allclose(ds["u"], torch.as_tensor(u, dtype=torch.float32))
    assert torch.allclose(ds["q"], torch.as_tensor(q, dtype=torch.float32))
    assert ds["metadata"]["generator"] == "test"

    with pytest.raises(FileNotFoundError, match="未入库，本地归档"):
        load_polar_dataset(tmp_path / "missing.pt")


def test_comsol_batch_without_mph_or_model(tmp_path):
    """mph 未装或模型缺失时清晰报错（退出码 2），不产生部分产物。"""
    rc = comsol_batch_main([
        "--mph-path", str(tmp_path / "model.mph"),
        "--out-dir", str(tmp_path / "out"),
        "--n-q", "3",
    ])
    assert rc == 2
    assert not list(tmp_path.rglob("q_*.csv"))
