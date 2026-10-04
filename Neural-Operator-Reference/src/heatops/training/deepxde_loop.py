"""DeepXDE 统一两阶段训练循环（Adam → 可选 L-BFGS）与运维件。

三条 DeepXDE 线（数据驱动 DeepONet / PI-DeepONet / 圆环 PINN）共用本模块，
重写自原研究档案各脚本的训练骨架并修复其系统性缺陷：

- 显式种子（档案仅个别脚本播种 numpy）；
- checkpoint 只存更优模型且评估端以 ``strict=True`` 权重加载（README D105）；
- loss 历史、config.json、训练日志统一落盘到 ``artifacts/<run_id>/``。

产物布局（``run_dir = artifacts/<run_id>/``）：

- ``checkpoints/best-<step>.pt``：train loss 更优时的周期 checkpoint
  （``ModelCheckpoint(save_better_only=True)``，deepxde 命名自带步号后缀）；
- ``checkpoints/final-<step>.pt``：训练结束时的最终模型；
- ``loss_history.npz``：steps / loss_train / loss_test（/ metrics_test）数组；
- ``config.json``：训练超参 + 种子 + 代码版本 + 调用方附加配置；
- ``loss_curve.png``：各项 loss 的半对数收敛曲线；
- ``train_log.txt``：训练全程 stdout（tee 同时到终端）。
"""

from __future__ import annotations

import contextlib
import json
import sys
import zipfile
from collections.abc import Callable, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path

from heatops.models.deepxde_models import ensure_dde_backend_pytorch

ensure_dde_backend_pytorch()  # 必须先于 import deepxde（见 deepxde_models 模块 docstring）

import deepxde as dde  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import heatops  # noqa: E402
from heatops.evaluation.deeponet_plots import plot_loss_history  # noqa: E402
from heatops.training.torch_loop import set_seed as _torch_set_seed  # noqa: E402


def set_seed(seed: int) -> None:
    """显式设定全部随机源种子（random / numpy / torch / cuda / deepxde）。

    deepxde 的几何采点、函数空间（GRF）采样与批抽样均走 numpy 全局 RNG，
    网络初始化与训练走 torch RNG；本函数在 ``torch_loop.set_seed`` 之外再调用
    deepxde 官方入口 ``dde.config.set_random_seed`` 兜底，保证同种子两次运行
    的采点与初始化序列一致。
    """
    _torch_set_seed(int(seed))
    dde.config.set_random_seed(int(seed))


class TeeLogger:
    """stdout 分流器：消息同时写终端与日志文件（AutoDL 运维件）。

    重写自原研究档案 PI-DeepONet 线的 ``Logger`` 类；与 :func:`tee_stdout`
    配合使用，进程退出前由上下文管理器负责关闭与恢复。
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.terminal = sys.stdout
        self._file = open(self.path, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self._file.write(message)

    def flush(self) -> None:
        self.terminal.flush()
        self._file.flush()

    def close(self) -> None:
        self._file.close()


@contextlib.contextmanager
def tee_stdout(log_path: str | Path) -> Iterator[TeeLogger]:
    """把 stdout 同时写入 ``log_path`` 的上下文管理器；退出时恢复原 stdout。"""
    logger = TeeLogger(log_path)
    old_stdout = sys.stdout
    sys.stdout = logger
    try:
        yield logger
    finally:
        sys.stdout = old_stdout
        logger.close()


def zip_artifacts(run_dir: str | Path, zip_path: str | Path | None = None) -> Path:
    """把 ``run_dir`` 的全部产物打包为 zip（AutoDL 下载运维件）。

    重写自原研究档案 PI-DeepONet 线的 ``zip_results``。zip 内路径以 run_dir
    目录名为根（``<run_id>/...``）；默认输出到 run_dir 同级的 ``<name>.zip``；
    排除既有 ``.zip`` 与 ``__pycache__``。
    """
    run_dir = Path(run_dir)
    out = Path(zip_path) if zip_path is not None else run_dir.parent / f"{run_dir.name}.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(run_dir.rglob("*")):
            if path.is_file() and path.suffix != ".zip" and "__pycache__" not in path.parts:
                zf.write(path, path.relative_to(run_dir.parent))
    return out


class TrainingMonitor(dde.callbacks.Callback):
    """周期性打印损失与探测点预测的训练监控回调。

    重写自原研究档案稳态圆环 PINN 的 ``TrainingMonitor``（去掉其温度阈值
    分级打印，保留损失 + 探测点输出）。

    Args:
        period: 打印间隔（迭代步）；<= 0 时关闭。
        probe_points: 可选探测点数组，形状 (P, dim)，随损失一起打印各点输出。
        probe_labels: 探测点标签，与 probe_points 等长。
        probe_transform: 可选输出换算 ``(P, 1) -> (P,)``（如无量纲温度 → °C）。
    """

    def __init__(
        self,
        period: int = 1000,
        probe_points: np.ndarray | None = None,
        probe_labels: Sequence[str] | None = None,
        probe_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ):
        super().__init__()
        self.period = int(period)
        self.probe_points = None if probe_points is None else np.asarray(probe_points)
        self.probe_labels = list(probe_labels) if probe_labels is not None else None
        self.probe_transform = probe_transform

    def on_epoch_end(self) -> None:
        if self.period <= 0:
            return
        iteration = int(self.model.train_state.iteration)
        if iteration == 0 or iteration % self.period != 0:
            return
        loss = np.atleast_1d(np.asarray(self.model.train_state.loss_train, dtype=float))
        loss_str = ", ".join(f"{v:.3e}" for v in loss)
        line = f"[monitor] iter {iteration} | train loss [{loss_str}] | total {loss.sum():.3e}"
        if self.probe_points is not None:
            pred = np.asarray(self.model.predict(self.probe_points)).reshape(len(self.probe_points), -1)[:, 0]
            if self.probe_transform is not None:
                pred = np.asarray(self.probe_transform(pred), dtype=float)
            labels = self.probe_labels or [f"probe{i}" for i in range(len(pred))]
            line += " | " + ", ".join(f"{lab}={val:.4g}" for lab, val in zip(labels, pred))
        print(line)


@dataclass(frozen=True)
class DDETrainConfig:
    """DeepXDE 两阶段训练配置。

    Attributes:
        run_dir: 产物目录（通常为 ``artifacts/<run_id>/``）。
        seed: 显式种子（数据构造与模型初始化由调用方在各自步骤前播种；
            本循环在训练前再播种一次，覆盖 BatchSampler 等训练期随机性）。
        adam_iterations: 阶段 1 Adam 迭代数；0 表示跳过阶段 1。
        adam_lr: Adam 学习率。
        adam_decay: deepxde 学习率衰减三元组（如 ``("step", 2000, 0.8)``，
            即 StepLR(step_size, gamma)）；None 表示不衰减。
        use_lbfgs: 是否在 Adam 后进行阶段 2 L-BFGS 精调。
        lbfgs_maxiter: L-BFGS 最大迭代数（经 ``dde.optimizers.set_LBFGS_options``
            设置，并同步修正 deepxde 在 import 时固化的 ``iter_per_step``/
            ``fun_per_step``；注意这些设置是 deepxde 的进程级全局项）。
        loss_weights: 各损失项权重（顺序与 data 的损失项一致：PDE 在前，
            其后按 ic_bcs 列表顺序）；None 表示等权。
        metrics: deepxde 评估指标名列表（如 ``("mean l2 relative error",)``）；
            PDE 类数据无真值 y_test，必须为 None。
        display_every: deepxde 训练日志打印间隔（步）。
        checkpoint_period: 最优 checkpoint 检查间隔（步）。
        extra: 写入 config.json 的附加信息（完整实验配置、数据缓存键等）。
    """

    run_dir: str | Path
    seed: int
    adam_iterations: int = 10000
    adam_lr: float = 1e-3
    adam_decay: tuple | list | None = ("step", 2000, 0.8)
    use_lbfgs: bool = True
    lbfgs_maxiter: int = 15000
    loss_weights: list[float] | None = None
    metrics: tuple[str, ...] | list[str] | None = None
    display_every: int = 500
    checkpoint_period: int = 2000
    extra: dict | None = field(default=None, compare=False)


def find_best_checkpoint(ckpt_dir: str | Path, prefix: str = "best") -> Path | None:
    """在 checkpoint 目录中找最新保存的 ``{prefix}-<step>.pt``。

    ``ModelCheckpoint(save_better_only=True)`` 仅在监控量改善时保存，故步号
    最大的文件即训练全程最优模型；若无任何 best checkpoint（如迭代数小于
    检查间隔），返回 None，由调用方回退到 final checkpoint。
    """
    ckpt_dir = Path(ckpt_dir)
    candidates = []
    for path in ckpt_dir.glob(f"{prefix}-*.pt"):
        try:
            step = int(path.stem.rsplit("-", 1)[1])
        except (ValueError, IndexError):
            continue
        candidates.append((step, path))
    if not candidates:
        return None
    return max(candidates)[1]


def load_weights_strict(model: dde.Model, ckpt_path: str | Path, device: str = "cpu") -> None:
    """以 ``strict=True`` 把 deepxde checkpoint 的网络权重加载到 ``model.net``。

    只加载 ``model_state_dict``（不碰优化器状态），因此可安全地跨优化器
    阶段使用（如最优权重存于 Adam 阶段、评估时模型已编译为 L-BFGS）。
    """
    ckpt = torch.load(Path(ckpt_path), map_location=torch.device(device), weights_only=True)
    model.net.load_state_dict(ckpt["model_state_dict"], strict=True)


def train_dde_model(
    model: dde.Model,
    cfg: DDETrainConfig,
    extra_callbacks: Sequence[dde.callbacks.Callback] = (),
) -> tuple[dde.model.LossHistory, dde.model.TrainState]:
    """统一两阶段训练：Adam（可跳过）→（可选）L-BFGS，带 checkpoint 与产物落盘。

    Args:
        model: 已绑定 data 与 net 的 ``dde.Model``（本函数负责 compile/train）。
        cfg: 训练配置。
        extra_callbacks: 附加回调（如 :class:`TrainingMonitor`、
            ``dde.callbacks.PDEPointResampler``），排在最优 checkpoint 回调之后。

    Returns:
        ``(losshistory, train_state)``（即 deepxde 训练返回对象本身，含两阶段
        累积历史）。

    可复现性：函数开头调用 ``set_seed(cfg.seed)``；deepxde 的采点在 data 构造时
    完成（由调用方在构造前播种），训练期随机性（批抽样、dropout 等）由这里的
    播种控制。
    """
    run_dir = Path(cfg.run_dir)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    set_seed(cfg.seed)
    metrics = list(cfg.metrics) if cfg.metrics else None

    with tee_stdout(run_dir / "train_log.txt"):
        checker = dde.callbacks.ModelCheckpoint(
            str(ckpt_dir / "best"),
            verbose=1,
            save_better_only=True,
            period=cfg.checkpoint_period,
        )
        callbacks = [checker, *extra_callbacks]

        if cfg.adam_iterations > 0:
            print(f"[train] 阶段 1：Adam（{cfg.adam_iterations} iters, lr={cfg.adam_lr}）")
            model.compile(
                optimizer="adam",
                lr=cfg.adam_lr,
                decay=cfg.adam_decay,
                loss_weights=cfg.loss_weights,
                metrics=metrics,
            )
            model.train(
                iterations=cfg.adam_iterations,
                display_every=cfg.display_every,
                callbacks=callbacks,
            )
        else:
            print("[train] 阶段 1：adam_iterations=0，跳过")

        if cfg.use_lbfgs:
            print(f"[train] 阶段 2：L-BFGS（maxiter={cfg.lbfgs_maxiter}）")
            dde.optimizers.set_LBFGS_options(maxiter=cfg.lbfgs_maxiter)
            # deepxde 的 iter_per_step/fun_per_step 在 optimizers.config 导入时按默认
            # maxiter=15000 固化，set_LBFGS_options 不更新它们——不同步修正则单个
            # 外层步最多跑 1000 次内层迭代，lbfgs_maxiter 形同一虚设
            dde.optimizers.LBFGS_options["iter_per_step"] = min(1000, cfg.lbfgs_maxiter)
            dde.optimizers.LBFGS_options["fun_per_step"] = (
                dde.optimizers.LBFGS_options["maxfun"]
                * dde.optimizers.LBFGS_options["iter_per_step"]
                // cfg.lbfgs_maxiter
            )
            # L-BFGS 外层步数远少于 Adam 迭代数，检查点间隔须改为 1，否则阶段内
            # 永远不会保存（ModelCheckpoint 按 epoch 计数而非全局 iteration）
            checker.period = 1
            model.compile(optimizer="L-BFGS", loss_weights=cfg.loss_weights, metrics=metrics)
            model.train(display_every=cfg.display_every, callbacks=callbacks)
        else:
            print("[train] 阶段 2：use_lbfgs=false，跳过")

        final_path = model.save(str(ckpt_dir / "final"), verbose=1)
        print(f"[train] 最终模型已保存：{final_path}")
        losshistory, train_state = model.losshistory, model.train_state

        steps = np.asarray(losshistory.steps, dtype=np.int64)
        loss_train = np.asarray(losshistory.loss_train, dtype=np.float64)
        loss_test = np.asarray(losshistory.loss_test, dtype=np.float64)
        if not np.all(np.isfinite(loss_train)):
            raise FloatingPointError("DeepXDE 出现非有限训练损失")
        payload = {"steps": steps, "loss_train": loss_train, "loss_test": loss_test}
        metrics_test = np.asarray(losshistory.metrics_test, dtype=np.float64)
        if metrics_test.size > 0:
            payload["metrics_test"] = metrics_test
        np.savez(run_dir / "loss_history.npz", **payload)

        config_payload = {
            "train": {k: v for k, v in asdict(cfg).items() if k != "extra"},
            "seed": cfg.seed,
            "code_revision": f"heatops-{heatops.__version__}",
            "backend": f"deepxde-{dde.__version__}/pytorch-{torch.__version__}",
            "best_train_loss": float(np.min(loss_train.sum(axis=1))) if loss_train.size else None,
            "extra": cfg.extra,
        }
        config_payload["train"]["run_dir"] = str(run_dir)
        with open(run_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_payload, f, ensure_ascii=False, indent=2)

        plot_loss_history(losshistory, run_dir / "loss_curve.png")

    return losshistory, train_state
