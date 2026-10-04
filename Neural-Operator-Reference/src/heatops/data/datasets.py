"""1D 热传导算子学习数据集：torch 张量封装、显式 train/eval 划分与 .pt 存取。

张量约定（与 ``heatops.models`` / ``heatops.physics`` 一致）：

- ``u0``: (B, nx) float32，初始温度场，单位 °C；
- ``u``: (B, nx, nt, 1) float32，时空温度场，末维为通道维，单位 °C；
- ``x``: (nx,) float64，空间网格，单位 m；``t``: (nt,) float64，时间网格，单位 s；
- ``grid`` 属性: (1, nx, nt, 2) float32，(x, t) 坐标网格，``indexing='ij'``，
  末维通道顺序为 (x, t)，与原研究档案 DataPINO 线一致。

随机性约定：数据生成由求解器 seed 控制；train/eval 划分由独立的 split seed
控制（``numpy.random.Generator.permutation``），同参数同种子完全可复现。
显式划分修复了原档案「评估集即训练集」的缺陷（见 README D105）。

``.pt`` 载荷为纯 torch 张量 + 原生类型 metadata dict，可用
``torch.load(..., weights_only=True)`` 安全加载。npz 互操作约定暂留 TODO。
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch

from heatops.fields import sample_sine_or_square
from heatops.problems import HeatLineParams
from heatops.solvers import solve_heat_1d_explicit

_ICSampler = Callable[[np.ndarray, np.random.Generator], np.ndarray]

#: 划分索引下界保护：训练集与评估集至少各 1 个样本
_MIN_SPLIT = 1


def make_xt_grid(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """由 (nx,) 与 (nt,) 网格构造 (1, nx, nt, 2) 的 (x, t) 坐标张量（``indexing='ij'``）。

    输出 dtype 为 float32（模型输入约定），与坐标张量的存储 dtype 无关。
    """
    X, T = torch.meshgrid(x, t, indexing="ij")
    grid = torch.stack([X, T], dim=-1)
    return grid.to(torch.float32).unsqueeze(0)


def cache_key_from_params(params: dict) -> str:
    """数据参数的短哈希缓存键：JSON 规范序列化（键排序）后取 sha256 前 12 位。

    数据语义参数（物理、网格、样本数、种子、初始条件名）任一变化都会改变键，
    从而触发重新生成而非误用旧缓存。
    """
    blob = json.dumps(params, sort_keys=True, ensure_ascii=True, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


class Heat1DDataset:
    """1D 热传导算子学习数据集（u0 → u(x, t) 映射）。

    Args:
        u0: 初始温度场，形状 (B, nx)，单位 °C。
        u: 时空温度场，形状 (B, nx, nt) 或 (B, nx, nt, 1)，单位 °C；
            内部统一存储为 (B, nx, nt, 1)。
        x: 空间网格 (nx,)，单位 m。
        t: 时间网格 (nt,)，单位 s（t[0] 必须为 0，首帧即初始条件）。
        metadata: 溯源信息（生成参数、种子、划分记录等），随 .pt 一起存取。
    """

    def __init__(
        self,
        u0: np.ndarray | torch.Tensor,
        u: np.ndarray | torch.Tensor,
        x: np.ndarray | torch.Tensor,
        t: np.ndarray | torch.Tensor,
        metadata: dict | None = None,
    ):
        u0_t = torch.as_tensor(np.asarray(u0), dtype=torch.float32)
        u_t = torch.as_tensor(np.asarray(u), dtype=torch.float32)
        if u_t.dim() == 3:
            u_t = u_t.unsqueeze(-1)
        x_t = torch.as_tensor(np.asarray(x), dtype=torch.float64)
        t_t = torch.as_tensor(np.asarray(t), dtype=torch.float64)

        if u0_t.dim() != 2:
            raise ValueError(f"u0 应为 (B, nx)，实际形状 {tuple(u0_t.shape)}")
        if u_t.shape[:3] != (u0_t.shape[0], u0_t.shape[1], t_t.shape[0]):
            raise ValueError(
                f"u 形状 {tuple(u_t.shape)} 与 u0/x/t 不一致"
                f"（期望 (B, nx, nt, 1) = ({u0_t.shape[0]}, {u0_t.shape[1]}, {t_t.shape[0]}, 1)）"
            )
        if u_t.shape[-1] != 1:
            raise ValueError(f"u 的通道维应为 1，实际形状 {tuple(u_t.shape)}")
        if x_t.shape[0] != u0_t.shape[1]:
            raise ValueError(f"x 长度 {x_t.shape[0]} 与 nx={u0_t.shape[1]} 不一致")
        if float(t_t[0]) != 0.0:
            raise ValueError("t[0] 必须为 0（首帧即初始条件，IC 损失依赖此约定）")

        self.u0 = u0_t
        self.u = u_t
        self.x = x_t
        self.t = t_t
        self.metadata: dict = dict(metadata) if metadata else {}

    # ------------------------------------------------------------------ 构造
    @classmethod
    def from_solver_output(cls, out: dict, metadata: dict | None = None) -> Heat1DDataset:
        """由 ``heatops.solvers.solve_heat_1d_explicit`` 的输出 dict 构造。"""
        return cls(out["u0"], out["u"], out["x"], out["t"], metadata=metadata)

    @classmethod
    def generate(
        cls,
        params: HeatLineParams,
        nx: int,
        nt: int,
        t_max: float,
        n_samples: int,
        seed: int,
        ic_sampler: _ICSampler = sample_sine_or_square,
        ic_name: str = "sine_or_square",
    ) -> Heat1DDataset:
        """现场生成散热线数据（FDM 数值参照），物理参数只来自 ``problems`` 的 params。

        Args:
            params: 1D 问题参数（只用 ``a``；网格/时长/样本数由本函数显式传入）。
            nx, nt: 空间/时间网格点数。
            t_max: 物理时长，单位 s。
            n_samples: 样本数。
            seed: 数据生成种子（求解器内部 ic 采样序列由它控制）。
            ic_sampler: 初始条件采样器 ``(x, rng) -> (nx,)``。
            ic_name: 采样器名（写入 metadata 供缓存键与溯源）。
        """
        out = solve_heat_1d_explicit(
            a=params.a,
            nx=nx,
            nt=nt,
            T_max=t_max,
            ic_sampler=ic_sampler,
            n_samples=n_samples,
            seed=seed,
        )
        metadata = {
            "generator": "solve_heat_1d_explicit",
            "a": params.a,
            "length": params.length,
            "u_scale": params.u_scale,
            "nx": nx,
            "nt": nt,
            "t_max": t_max,
            "n_samples": n_samples,
            "seed": seed,
            "ic": ic_name,
            "bc": "dirichlet0",
        }
        return cls.from_solver_output(out, metadata=metadata)

    # ------------------------------------------------------------------ 属性
    @property
    def n_samples(self) -> int:
        """样本数 B。"""
        return int(self.u0.shape[0])

    @property
    def nx(self) -> int:
        """空间网格点数。"""
        return int(self.u0.shape[1])

    @property
    def nt(self) -> int:
        """时间帧数（含 t=0）。"""
        return int(self.u.shape[2])

    @property
    def grid(self) -> torch.Tensor:
        """(1, nx, nt, 2) float32 坐标网格，批维为 1，供 ``expand`` 到批大小。"""
        return make_xt_grid(self.x, self.t)

    def __len__(self) -> int:
        return self.n_samples

    def __repr__(self) -> str:
        return (
            f"Heat1DDataset(n_samples={self.n_samples}, nx={self.nx}, nt={self.nt}, "
            f"ic={self.metadata.get('ic', '?')!r}, split={self.metadata.get('split', {}).get('role', '-')})"
        )

    # ------------------------------------------------------------------ 划分
    def split(self, train_ratio: float, seed: int) -> tuple[Heat1DDataset, Heat1DDataset]:
        """显式 train/eval 划分（样本级、不重叠、同种子可复现）。

        Args:
            train_ratio: 训练集占比 (0, 1)。
            seed: 划分种子，独立于数据生成种子；同一种子划分结果逐点一致。

        Returns:
            ``(train, eval)`` 两个新数据集，共享同一 x/t 网格；各自的
            ``metadata['split']`` 记录比例、种子、角色与样本索引。
        """
        if not 0.0 < train_ratio < 1.0:
            raise ValueError(f"train_ratio 须在 (0, 1) 内，实际为 {train_ratio}")
        rng = np.random.default_rng(seed)
        perm = rng.permutation(self.n_samples)
        n_train = int(self.n_samples * train_ratio)
        n_train = min(max(n_train, _MIN_SPLIT), self.n_samples - _MIN_SPLIT)
        idx_train = perm[:n_train]
        idx_eval = perm[n_train:]

        def _subset(idx: np.ndarray, role: str) -> Heat1DDataset:
            idx_t = torch.as_tensor(idx, dtype=torch.long)
            meta = dict(self.metadata)
            meta["split"] = {
                "role": role,
                "train_ratio": float(train_ratio),
                "seed": int(seed),
                "indices": idx.tolist(),
            }
            return Heat1DDataset(
                self.u0[idx_t], self.u[idx_t], self.x, self.t, metadata=meta
            )

        return _subset(idx_train, "train"), _subset(idx_eval, "eval")

    # ------------------------------------------------------------------ 存取
    def save(self, path: str | Path) -> Path:
        """保存为 .pt（纯张量 + 原生类型 metadata，可 ``weights_only=True`` 加载）。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"u0": self.u0, "u": self.u, "x": self.x, "t": self.t, "metadata": self.metadata},
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> Heat1DDataset:
        """加载 .pt 数据集（``weights_only=True``，拒绝任意 pickle 载荷）。"""
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        return cls(
            payload["u0"], payload["u"], payload["x"], payload["t"],
            metadata=payload.get("metadata"),
        )
