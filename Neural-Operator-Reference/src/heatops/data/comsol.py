"""COMSOL 数据管线：CSV 读取、4D 装配、Cartesian→极坐标插值、重采样与极坐标数据集存取。

重写自原研究档案的四个脚本（按本库接口重写，非逐段拷贝）：

- ``pino_dataset_pro.py``：COMSOL CSV（x, y, T1..Tn）→ 4D 张量，NaN 提取几何掩码；
- ``cartesian_2_polar_dataset.py``：笛卡尔→极坐标；传入 mask 时按最近有效点填补域外值；
- ``resample_dataset.py``：空间重采样（场 bilinear / 掩码 nearest）；
- ``link_test.py`` 的导出约定：每 CSV 含全部时刻长表。

与档案的关键差异（D103 量纲统一）：档案在装配时做 min-max 归一化并把 q 归一化
广播为输入场；本模块**一律保持物理单位**（u: °C、q: W/m²、t: s、x/y/r: m、
θ: rad），无量纲化集中在 ``heatops.problems.annulus`` 与训练 CLI 中完成，
``info`` 以 metadata 形式随数据集存取。

数据可得性：COMSOL 原始 CSV（约 298MB）**未入库，本地归档**；所有读取入口在
文件缺失时抛 ``FileNotFoundError`` 并注明该事实。
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.interpolate import RegularGridInterpolator

#: 数据缺失统一提示（隐私约定：本地数据位置不写盘符路径，统一「未入库，本地归档」）
MISSING_DATA_HINT = "COMSOL 数据未入库，本地归档"


def _require_exists(path: str | Path) -> Path:
    """校验数据路径存在；缺失时抛 FileNotFoundError 并提示「未入库，本地归档」。"""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"数据文件不存在：{path}（{MISSING_DATA_HINT}；"
            "可用 heatops.data.comsol_batch 在本机 COMSOL 环境重新生成，"
            "或由归档 CSV 经 heatops.data.comsol 装配）"
        )
    return path


def read_comsol_csv(path: str | Path, length_unit: str = "mm") -> pd.DataFrame:
    """读取单个 COMSOL 导出 CSV：注释行以 ``%`` 开头，数据列 ``x, y, T1..Tn``。

    Args:
        path: CSV 路径。
        length_unit: x/y 列的长度单位；``"mm"`` 时换算为 m（×1e-3），``"m"`` 不变。
            档案 COMSOL 模型以 mm 导出。

    Returns:
        DataFrame：列 ``x``、``y``（单位 m，float64）与 ``T_0..T_{n-1}``（°C）。
    """
    path = _require_exists(path)
    if length_unit == "mm":
        scale = 1e-3
    elif length_unit == "m":
        scale = 1.0
    else:
        raise ValueError(f"length_unit 仅支持 'mm'/'m'，实际为 {length_unit!r}")
    headers = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            if not line.startswith("%"):
                break
            headers.append(line)
    times = []
    for line in headers:
        if "@ t=" in line:
            times = [float(v) for v in re.findall(r"@ t=([\d.eE+\-]+)", line)]
            if "degC" not in line and "°C" not in line:
                raise ValueError("CSV 温度单位必须为 degC/°C；不允许静默混用 Kelvin")
    declared_length = next((line.split(",", 1)[1].strip() for line in headers if "Length unit," in line), None)
    if declared_length is not None and declared_length != length_unit:
        raise ValueError(f"CSV 长度单位 {declared_length} 与 length_unit={length_unit} 不匹配")
    df = pd.read_csv(path, comment="%", header=None)
    if df.shape[1] < 3:
        raise ValueError(f"{path} 应至少含 x, y, T1 三列，实际 {df.shape[1]} 列")
    out = df.astype(np.float64)
    out.columns = ["x", "y", *[f"T_{k}" for k in range(df.shape[1] - 2)]]
    out[["x", "y"]] *= scale
    out.attrs["time_values"] = times
    return out


def assemble_cartesian_4d(
    csv_paths: list[str | Path],
    q_values: np.ndarray,
    t_values: np.ndarray | None = None,
    length_unit: str = "mm",
) -> dict:
    """CSV 列表 → 4D Cartesian 数据集（物理单位），NaN 提取几何掩码后填 0。

    与档案 ``reshape(num_t, res, res)`` 的差异：档案假设 CSV 行序恰好按网格
    排列；本实现按 x/y 列的唯一值建立规则网格索引再填充（对 COMSOL 行序
    不敏感），并要求各 CSV 共享同一网格。

    Args:
        csv_paths: COMSOL CSV 路径列表（每个对应一个热流工况）。
        q_values: 各工况内边界热流密度 (n,)，单位 W/m²，与 csv_paths 一一对应。
        t_values: 时刻网格 (nt,)，单位 s；默认读取 CSV 的 @t= 列头。
            列头缺失时必须显式提供时刻，不猜测时间轴。
        length_unit: 传给 ``read_comsol_csv``。

    Returns:
        dict：``u`` (n, nt, nx, ny) float64（°C，域外 NaN 已填 0）、
        ``mask`` (nx, ny) bool（True=域内，由首样本首帧 NaN 提取）、
        ``q`` (n,)、``t`` (nt,)、``x`` (nx,)、``y`` (ny,) 与 ``metadata``。
    """
    paths = [_require_exists(p) for p in csv_paths]
    q_values = np.asarray(q_values, dtype=np.float64)
    if len(paths) != q_values.shape[0]:
        raise ValueError(f"CSV 数 {len(paths)} 与 q_values 数 {q_values.shape[0]} 不一致")

    samples: list[np.ndarray] = []
    x_grid = y_grid = None
    header_times = None
    for path in paths:
        df = read_comsol_csv(path, length_unit=length_unit)
        if header_times is None:
            header_times = df.attrs["time_values"]
        elif header_times != df.attrs["time_values"]:
            raise ValueError("CSV 时刻列头不一致")
        x_unique, x_inv = np.unique(df["x"].to_numpy(), return_inverse=True)
        y_unique, y_inv = np.unique(df["y"].to_numpy(), return_inverse=True)
        nx, ny = len(x_unique), len(y_unique)
        if len(df) != nx * ny:
            raise ValueError(f"{path} 数据点数 {len(df)} 不是完整规则网格（{nx}×{ny}）")
        if x_grid is None:
            x_grid, y_grid = x_unique, y_unique
        elif not (np.array_equal(x_unique, x_grid) and np.array_equal(y_unique, y_grid)):
            raise ValueError(f"{path} 的网格与首个 CSV 不一致")
        vals = df[[c for c in df.columns if c.startswith("T_")]].to_numpy()  # (npts, nt)
        u = np.full((vals.shape[1], nx, ny), np.nan)
        u[:, x_inv, y_inv] = vals.T
        samples.append(u)

    u_all = np.stack(samples, axis=0)  # (n, nt, nx, ny)
    mask = ~np.isnan(u_all[0, 0])
    u_all = np.nan_to_num(u_all, nan=0.0)

    nt = u_all.shape[1]
    if t_values is None:
        if not header_times:
            raise ValueError("CSV 未提供时刻列头，请显式传入 t_values；不再猜测 1..nt")
        t_values = np.asarray(header_times, dtype=np.float64)
    t_values = np.asarray(t_values, dtype=np.float64)
    if t_values.shape[0] != nt:
        raise ValueError(f"t_values 长度 {t_values.shape[0]} 与 CSV 时刻数 {nt} 不一致")
    if not np.all(np.isfinite(t_values)) or t_values[0] < 0 or not np.all(np.diff(t_values) > 0):
        raise ValueError("时刻必须非负、有限且严格递增")

    return {
        "u": u_all,
        "mask": mask,
        "q": q_values,
        "t": t_values,
        "x": x_grid,
        "y": y_grid,
        "metadata": {
            "generator": "heatops.data.comsol.assemble_cartesian_4d",
            "n_csv": len(paths),
            "length_unit": length_unit,
            "units": {"u": "°C", "q": "W/m²", "t": "s", "x/y": "m"},
        },
    }


def cartesian_to_polar(
    u: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    center: tuple[float, float] | None = None,
    fill_value: float = 0.0,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Cartesian 场 → 极坐标网格插值（``RegularGridInterpolator``，域外填 ``fill_value``）。

    Args:
        u: Cartesian 场，形状 (..., nx, ny)（最后两维 ``indexing='ij'``）。
        x, y: Cartesian 网格坐标 (nx,)、(ny,)，单位 m。
        r_grid, theta_grid: 目标极坐标网格 (nr,)、(ntheta,)，单位 m / rad。
        center: 圆心；默认取 x/y 范围中点（假设圆环居中于数据域，档案约定）。
        fill_value: 目标点落在 Cartesian 域外时的填充值（档案取 0）。

    Returns:
        形状 (..., nr, ntheta) 的插值场。
    """
    u = np.asarray(u, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    cx, cy = center if center is not None else (0.5 * (x[0] + x[-1]), 0.5 * (y[0] + y[-1]))
    R, Theta = np.meshgrid(r_grid, theta_grid, indexing="ij")
    pts = np.stack(
        [(cx + R * np.cos(Theta)).ravel(), (cy + R * np.sin(Theta)).ravel()], axis=1
    )
    nr, ntheta = len(r_grid), len(theta_grid)
    flat = u.reshape(-1, u.shape[-2], u.shape[-1])
    out = np.empty((flat.shape[0], nr, ntheta))
    for i, frame in enumerate(flat):
        if mask is None:
            interp = RegularGridInterpolator((x, y), frame, bounds_error=False, fill_value=fill_value)
            values = interp(pts)
        else:
            from scipy.interpolate import NearestNDInterpolator
            from scipy.ndimage import distance_transform_edt
            valid = np.asarray(mask, dtype=bool)
            if valid.shape != frame.shape or not valid.any():
                raise ValueError("几何掩码尺寸不匹配或没有有效点")
            nearest_indices = distance_transform_edt(~valid, return_distances=False, return_indices=True)
            repaired = frame[tuple(nearest_indices)]
            interp = RegularGridInterpolator((x, y), repaired, bounds_error=False, fill_value=np.nan)
            values = interp(pts)
            missing = ~np.isfinite(values)
            if missing.any():
                xx, yy = np.meshgrid(x, y, indexing="ij")
                nearest = NearestNDInterpolator(np.stack([xx[valid], yy[valid]], axis=1), frame[valid])
                values[missing] = nearest(pts[missing])
        out[i] = values.reshape(nr, ntheta)
    return out.reshape(*u.shape[:-2], nr, ntheta)


def polar_to_cartesian(
    u_polar: np.ndarray,
    r_grid: np.ndarray,
    theta_grid: np.ndarray,
    x_pts: np.ndarray,
    y_pts: np.ndarray,
    center: tuple[float, float] | None = None,
) -> np.ndarray:
    """极坐标场 → 任意 Cartesian 点插值（``cartesian_to_polar`` 的反向，供往返校验）。

    查询点换算 ``r = hypot(x-cx, y-cy)``、``θ = arctan2(y-cy, x-cx) mod 2π``；
    约定 ``theta_grid`` 含两端点 0 与 2π（周期母线两端各一列）。圆环域外
    查询点填 NaN（bounds_error=False 的外推值）。

    Args:
        u_polar: 极坐标场，形状 (..., nr, ntheta)。
        r_grid, theta_grid: 极坐标网格 (nr,)、(ntheta,)。
        x_pts, y_pts: 查询点坐标 (npts,)，单位 m。
        center: 圆心；默认取 ``(0, 0)``（调用方多使用以圆心为原点的坐标）。

    Returns:
        形状 (..., npts) 的插值。
    """
    u_polar = np.asarray(u_polar, dtype=np.float64)
    cx, cy = center if center is not None else (0.0, 0.0)
    x_pts = np.asarray(x_pts, dtype=np.float64)
    y_pts = np.asarray(y_pts, dtype=np.float64)
    rq = np.hypot(x_pts - cx, y_pts - cy)
    tq = np.mod(np.arctan2(y_pts - cy, x_pts - cx), 2 * np.pi)
    pts = np.stack([rq, tq], axis=1)
    flat = u_polar.reshape(-1, u_polar.shape[-2], u_polar.shape[-1])
    out = np.empty((flat.shape[0], pts.shape[0]))
    for i, frame in enumerate(flat):
        interp = RegularGridInterpolator(
            (r_grid, theta_grid), frame, bounds_error=False, fill_value=np.nan
        )
        out[i] = interp(pts)
    return out.reshape(*u_polar.shape[:-2], pts.shape[0])


def resample_spatial(
    u: np.ndarray,
    target_shape: tuple[int, int],
    mode: str = "bilinear",
) -> np.ndarray:
    """空间重采样：最后两维 (H, W) → ``target_shape``（档案 ``resample_dataset.py`` 语义）。

    场用 ``bilinear``（``align_corners=True``），几何掩码用 ``nearest``
    （保持布尔属性）。内部经 torch ``F.interpolate`` 实现。
    """
    u_arr = np.asarray(u, dtype=np.float64)
    t = torch.from_numpy(u_arr.reshape(-1, 1, *u_arr.shape[-2:]))
    kwargs = {} if mode == "nearest" else {"align_corners": True}
    out = F.interpolate(t, size=target_shape, mode=mode, **kwargs)
    return out.numpy().reshape(*u_arr.shape[:-2], *target_shape)


def save_polar_dataset(
    path: str | Path,
    q: np.ndarray,
    u: np.ndarray,
    t: np.ndarray,
    r: np.ndarray,
    theta: np.ndarray,
    metadata: dict | None = None,
) -> Path:
    """保存极坐标数据集 .pt（纯 torch 张量 + 原生类型 metadata，可 ``weights_only=True`` 加载）。

    张量约定（物理单位，无量纲化由训练 CLI 经 ``problems.annulus`` 完成）：
    ``q`` (B,) W/m²；``u`` (B, nt, nr, ntheta) °C；``t`` (nt,) s；
    ``r`` (nr,) m；``theta`` (ntheta,) rad（含两端点）。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "q": torch.as_tensor(np.asarray(q), dtype=torch.float32),
            "u": torch.as_tensor(np.asarray(u), dtype=torch.float32),
            "t": torch.as_tensor(np.asarray(t), dtype=torch.float64),
            "r": torch.as_tensor(np.asarray(r), dtype=torch.float64),
            "theta": torch.as_tensor(np.asarray(theta), dtype=torch.float64),
            "metadata": dict(metadata) if metadata else {},
        },
        path,
    )
    return path


def load_polar_dataset(path: str | Path) -> dict:
    """加载极坐标数据集 .pt（``weights_only=True``）；缺失时抛 FileNotFoundError（未入库提示）。"""
    path = _require_exists(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    validate_polar_dataset(payload)
    return payload


def validate_polar_dataset(ds: dict) -> None:
    required = {"q", "u", "t", "r", "theta"}
    if not required <= ds.keys():
        raise ValueError(f"极坐标数据缺字段：{sorted(required - ds.keys())}")
    if ds["u"].ndim != 4:
        raise ValueError("温度张量必须为 (B,nt,nr,ntheta)")
    b, nt, nr, nth = ds["u"].shape
    if tuple(ds["q"].shape) != (b,) or tuple(ds["t"].shape) != (nt,) or tuple(ds["r"].shape) != (nr,) or tuple(ds["theta"].shape) != (nth,):
        raise ValueError("极坐标数据形状不匹配")
    if not all(bool(torch.isfinite(ds[k]).all()) for k in required):
        raise ValueError("数据中含非有限值")
    if not bool((ds["q"] > 0).all()) or not bool((ds["r"] > 0).all()) or ds["t"][0] < 0:
        raise ValueError("热流和半径必须为正，时刻必须非负")
    for name in ("t", "r", "theta"):
        if not bool((torch.diff(ds[name]) > 0).all()):
            raise ValueError(f"{name} 必须严格递增")
    units = ds.get("metadata", {}).get("units", {})
    expected = {"u": "°C", "q": "W/m²", "t": "s", "r": "m", "theta": "rad"}
    if any(k in units and units[k] != v for k, v in expected.items()):
        raise ValueError("数据单位不符合物理量契约（°C、W/m²、s、m、rad）")
