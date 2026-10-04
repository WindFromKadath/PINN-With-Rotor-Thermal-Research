"""DeepXDE 网络工厂与特征/输出变换（三条 DeepXDE 线共用）。

重写自原研究档案三条 DeepXDE 线的网络构造片段（DeepONet 分支/主干 FNN、
圆环 PINN 的 FNN + sin/cos 特征映射、稳态线的线性输出还原），网络维度、激活
与初始化全部配置驱动，物理参数不在本模块出现。

后端约定：deepxde 在首次 ``import deepxde`` 时按 ``DDE_BACKEND`` 环境变量选定
计算后端且进程内不可更改，因此本模块在导入 deepxde 之前把后端固定为
``pytorch``（与本库 torch 实现一致）。这是对「无 import 副作用」约定的唯一
显式例外：设置动作幂等，仅在该变量未设置时生效；若用户已显式指向其他后端
则报错而非静默覆盖。
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence


def ensure_dde_backend_pytorch() -> None:
    """在 ``import deepxde`` 之前把 ``DDE_BACKEND`` 固定为 ``pytorch``（幂等）。

    仅在环境变量未设置时写入；已设置为其他后端时抛出 RuntimeError，
    避免 deepxde 静默选中与本库 torch 张量不兼容的后端。
    """
    existing = os.environ.get("DDE_BACKEND")
    if existing is None:
        os.environ["DDE_BACKEND"] = "pytorch"
    elif existing != "pytorch":
        raise RuntimeError(
            f"DDE_BACKEND 已设置为 {existing!r}，本库 DeepXDE 线仅支持 'pytorch' 后端；"
            "请取消该环境变量或改为 'pytorch' 后再导入。"
        )


ensure_dde_backend_pytorch()

import deepxde as dde  # noqa: E402
import torch  # noqa: E402


def make_deeponet_cartesian(
    branch_layers: Sequence[int],
    trunk_layers: Sequence[int],
    activation: str = "tanh",
    kernel_initializer: str = "Glorot normal",
) -> dde.nn.pytorch.DeepONetCartesianProd:
    """构造 CartesianProd 布局的 DeepONet（分支/主干均为 FNN）。

    Args:
        branch_layers: 分支网络层宽，首维为传感点数（如 32×32=1024）。
        trunk_layers: 主干网络层宽，首维为坐标维数（如 (x, y, t) 为 3）。
        activation: 激活函数名（deepxde 注册名，如 ``"tanh"``）。
        kernel_initializer: 权重初始化（如 ``"Glorot normal"``）。

    Returns:
        ``dde.nn.pytorch.DeepONetCartesianProd`` 实例；前向输入
        ``(branch (B, m), trunk (N2, d))``，输出形状 ``(B, N2)``。

    Raises:
        ValueError: 分支/主干输出维（末层宽度）不一致时。
    """
    if int(branch_layers[-1]) != int(trunk_layers[-1]):
        raise ValueError(
            f"分支/主干末层宽度必须一致，实际为 {branch_layers[-1]} vs {trunk_layers[-1]}"
        )
    return dde.nn.pytorch.DeepONetCartesianProd(
        list(map(int, branch_layers)),
        list(map(int, trunk_layers)),
        activation,
        kernel_initializer,
    )


def make_fnn(
    layer_sizes: Sequence[int],
    activation: str = "tanh",
    kernel_initializer: str = "Glorot normal",
    feature_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    output_transform: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
) -> dde.nn.pytorch.FNN:
    """构造 FNN 并可选用特征/输出变换（PINN 线用）。

    Args:
        layer_sizes: 层宽列表；启用 ``feature_transform`` 时首维为**变换后**
            的输入维数（如周期映射使 3 维输入变 4 维）。
        activation: 激活函数名。
        kernel_initializer: 权重初始化。
        feature_transform: 可选输入变换（如 :class:`PeriodicFeatureTransform`），
            经 ``net.apply_feature_transform`` 挂载。
        output_transform: 可选输出变换 ``(x, y) -> y'``（如
            :func:`make_linear_output_transform` 的产物），经
            ``net.apply_output_transform`` 挂载。

    Returns:
        ``dde.nn.pytorch.FNN`` 实例。
    """
    net = dde.nn.pytorch.FNN(list(map(int, layer_sizes)), activation, kernel_initializer)
    if feature_transform is not None:
        net.apply_feature_transform(feature_transform)
    if output_transform is not None:
        net.apply_output_transform(output_transform)
    return net


class PeriodicFeatureTransform:
    """角度坐标的周期特征映射：``θ`` 列替换为 ``(sin θ, cos θ)`` 两列。

    重写自原研究档案圆环瞬态 PINN 的 ``feature_transform``（把 [r, θ, t] 映射为
    [r, sin θ, cos θ, t]）。θ 与 θ+2πk 的映射结果**代数恒等**，从输入端消除
    0/2π 接缝处的数值跳变，无需再以 PeriodicBC 损失兜底（档案中该 loss 权重
    已为 0，本库直接省略该约束项）。

    输入 n 列 → 输出 n+1 列；``theta_col`` 之外各列保持原位与顺序。
    变换可微（autograd 经 sin/cos 链式求导），可作 deepxde 的
    ``apply_feature_transform`` 回调。

    Args:
        theta_col: 角度列在输入张量中的列号（弧度），默认 1（对应 [r, θ, t]）。
    """

    def __init__(self, theta_col: int = 1):
        self.theta_col = int(theta_col)

    def output_dim(self, input_dim: int) -> int:
        """变换后维数：``input_dim + 1``。"""
        return int(input_dim) + 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        theta = x[:, self.theta_col : self.theta_col + 1]
        return torch.cat(
            [
                x[:, : self.theta_col],
                torch.sin(theta),
                torch.cos(theta),
                x[:, self.theta_col + 1 :],
            ],
            dim=1,
        )


def make_linear_output_transform(
    offset: float, scale: float
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    """线性输出还原变换工厂：``y_phys = offset + scale · y_norm``。

    重写自原研究档案稳态圆环 PINN 的 ``OutputTransform``（去掉其 softplus 版本
    的非负约束）。返回的 callable 签名 ``(x, y_norm) -> y_phys``，可直接挂到
    ``net.apply_output_transform``。

    注意：挂载后 PDE/BC 残差中的 ``y`` 即为物理量，残差表达式的常数须按同一
    量纲书写；本库三条 CLI 线为保持损失项量纲一致（README D103），训练均在
    无量纲量上进行并在评估时还原，故未在 CLI 中挂载本变换，留作下游工具的
    公开 API。
    """
    offset = float(offset)
    scale = float(scale)

    def _transform(x: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        return offset + scale * y_norm

    return _transform
