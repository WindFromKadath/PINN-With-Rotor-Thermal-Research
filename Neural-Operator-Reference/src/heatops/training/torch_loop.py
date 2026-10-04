"""通用 torch 算子模型训练循环（1D 三模型对比线共用）。

重写自原研究档案 DataPINO 线 ``datapino_train.py`` 的训练骨架（工程约定沿用：
AdamW、``torch.randperm`` 批抽样、周期 checkpoint、config.json + loss 历史 +
loss 曲线），并修复其系统性缺陷：显式种子、train/eval 划分由数据集对象承载、
评估权重以 ``strict=True`` 加载（见 README D105）。

产物布局（``run_dir = artifacts/<run_id>/``）：

- ``checkpoints/checkpoint_best.pt``：训练 total loss 最优时的 state_dict；
- ``checkpoints/checkpoint_last.pt``：最近一次周期 checkpoint（含末步）；
- ``loss_history.npy``：结构化数组（step/total/data/pde/ic/bc，无需 pickle 加载）；
- ``config.json``：训练超参 + 种子 + 代码版本 + 调用方附加配置；
- ``loss_curve.png``：各项 loss 的半对数收敛曲线。
"""

from __future__ import annotations

import json
import random
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

import heatops
from heatops.data.datasets import Heat1DDataset

_LossFn = Callable[
    [torch.nn.Module, torch.Tensor, torch.Tensor, torch.Tensor],
    tuple[torch.Tensor, dict[str, torch.Tensor]],
]

_LOSS_KEYS = ("total", "data", "pde", "ic", "bc")


def _save_state_dict(model: torch.nn.Module, path: Path) -> None:
    """保存剔除 ``_metadata`` 的 state_dict。

    neuralop 的 FNO 在 state_dict 中写入含 ``gelu`` 函数与 ``SpectralConv`` 类
    引用的 ``_metadata`` 项，导致 ``torch.load(weights_only=True)`` 拒绝加载；
    剔除后 checkpoint 为纯张量字典，可安全加载且 ``strict=True`` 不受影响
    （``_metadata`` 仅承载版本提示，加载端不强制要求）。
    """
    state = {k: v for k, v in model.state_dict().items() if k != "_metadata"}
    torch.save(state, path)


def set_seed(seed: int) -> None:
    """显式设定全部随机源种子（random / numpy / torch / cuda）。

    同时设 ``cudnn.deterministic=True`` 并关闭 ``benchmark`` 以获得逐位可复现
    的卷积算法选择；注意这不保证跨硬件/跨版本一致，且个别 CUDA 算子仍有
    非确定性实现（CPU 短训不受此影响）。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass(frozen=True)
class TrainConfig:
    """训练循环配置。

    Attributes:
        run_dir: 产物目录（通常为 ``artifacts/<run_id>/``）。
        seed: 训练种子（模型初始化之外的循环内随机性由它控制；模型初始化
            种子由调用方在构造模型前设定，二者通常取同一值）。
        epochs: 迭代步数（每步一个 randperm 批；含第 0 步，共 epochs+1 次记录）。
        batch_size: 批大小。
        lr: AdamW 学习率。
        weight_decay: AdamW 权重衰减。
        checkpoint_every: 周期 checkpoint 间隔（步）；末步必存。
        log_every: 标准输出日志间隔（步）。
        device: ``"auto"``（有 CUDA 用 CUDA，否则 CPU）或显式 ``"cpu"``/``"cuda"``。
        extra: 写入 config.json 的附加信息（如完整实验配置、数据缓存键）。
    """

    run_dir: str | Path
    seed: int
    epochs: int = 3000
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-5
    checkpoint_every: int = 500
    log_every: int = 500
    device: str = "auto"
    extra: dict | None = field(default=None, compare=False)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _plot_loss_curve(history: dict[str, list], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = history["step"]
    plt.figure(figsize=(10, 6))
    for key in ("total", "data", "pde", "ic", "bc"):
        plt.semilogy(steps, np.maximum(history[key], 1e-30), label=f"{key} loss")
    plt.title("Training Loss Convergence (Log Scale)")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def train_operator_model(
    model: torch.nn.Module,
    dataset: Heat1DDataset,
    loss_fn: _LossFn,
    cfg: TrainConfig,
) -> dict[str, list]:
    """训练循环：AdamW + randperm 批抽样 + 周期/最优 checkpoint + 产物落盘。

    Args:
        model: 待训练模型 ``(u0, grid) -> pred``。
        dataset: **训练划分**数据集（评估划分不得传入本函数）。
        loss_fn: ``(model, batch_u0, batch_u_true, batch_grid) -> (total, terms)``，
            total 为加权总损失（标量），terms 为各分项标量字典
            （键含 data/pde/ic/bc，记入历史）。
        cfg: 训练配置。

    Returns:
        loss 历史 dict：键 ``step/total/data/pde/ic/bc``，值为长度 epochs+1 的列表。

    可复现性：函数开头调用 ``set_seed(cfg.seed)``，使循环内随机序列只取决于
    cfg.seed，与模型初始化消耗了多少随机数无关；同一 config 同种子两次运行
    的 loss 历史在 CPU 上逐点一致（H104）。
    """
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(cfg.device)
    set_seed(cfg.seed)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    u0_all = dataset.u0.to(device)
    u_all = dataset.u.to(device)
    grid_base = dataset.grid.to(device)  # (1, nx, nt, 2)
    n_train = dataset.n_samples
    if cfg.batch_size > n_train:
        raise ValueError(f"batch_size={cfg.batch_size} 超过训练样本数 {n_train}")

    history: dict[str, list] = {k: [] for k in ("step", *_LOSS_KEYS)}
    best_loss = float("inf")

    print(f"[train] device={device} n_train={n_train} epochs={cfg.epochs} batch={cfg.batch_size}")
    for step in range(cfg.epochs + 1):
        indices = torch.randperm(n_train)[: cfg.batch_size]
        b_u0 = u0_all[indices]
        b_true = u_all[indices]
        b_grid = grid_base.expand(cfg.batch_size, -1, -1, -1)

        total, terms = loss_fn(model, b_u0, b_true, b_grid)

        optimizer.zero_grad()
        if not bool(torch.isfinite(total)):
            raise FloatingPointError(f"非有限训练损失：step={step}")
        total.backward()
        optimizer.step()

        total_val = float(total.detach().cpu())
        history["step"].append(step)
        history["total"].append(total_val)
        for key in _LOSS_KEYS[1:]:
            history[key].append(float(terms[key].detach().cpu()))

        if total_val < best_loss:
            best_loss = total_val
            _save_state_dict(model, ckpt_dir / "checkpoint_best.pt")
        if step % cfg.checkpoint_every == 0 or step == cfg.epochs:
            _save_state_dict(model, ckpt_dir / "checkpoint_last.pt")
        if step % cfg.log_every == 0:
            print(
                f"[train] step {step:5d} | total {total_val:.4e} | "
                f"data {history['data'][-1]:.3e} | pde {history['pde'][-1]:.3e} | "
                f"ic {history['ic'][-1]:.3e} | bc {history['bc'][-1]:.3e}"
            )

    history_arr = np.array(
        list(zip(*(history[k] for k in ("step", *_LOSS_KEYS)))),
        dtype=[("step", "i8"), *((k, "f8") for k in _LOSS_KEYS)],
    )
    np.save(run_dir / "loss_history.npy", history_arr)

    config_payload = {
        "train": {k: v for k, v in asdict(cfg).items() if k not in ("extra",)},
        "seed": cfg.seed,
        "device": str(device),
        "code_revision": f"heatops-{heatops.__version__}",
        "best_total_loss": best_loss,
        "extra": cfg.extra,
    }
    config_payload["train"]["run_dir"] = str(run_dir)
    with open(run_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config_payload, f, ensure_ascii=False, indent=2)

    _plot_loss_curve(history, run_dir / "loss_curve.png")
    return history
