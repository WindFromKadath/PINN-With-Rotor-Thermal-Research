"""直角坐标热传导问题：2D 单位正方形与 1D 单位区间的参数定义。

2D 问题定义（长度单位 m，温度单位 °C，其余 SI）：

- 域：单位正方形 ``[0, 1]²``，网格约定 ``indexing='ij'``（第一维为 x）；
- 控制方程：``u_t = alpha * Δu + f(x, y)``，热源 f 单位 °C/s；
- 边界：四边 Robin/Newton 冷却 ``du/dn = -beta * (u - u0)``，
  其中 ``beta = h/k`` 单位 1/m，n 为外法向；
- 初始条件：均匀 ``u0``。

热源采样约定：GRF 均值 ``f_mean``、波动 ``f_std``，归一化尺度 ``F_MAX``。

1D 问题（``HeatLineParams``）：单位区间 ``[0, 1]``，方程 ``u_t = a * u_xx (+ f)``，
Dirichlet 零边界；含两条原研究档案线：散热线 ``a=0.4`` 与带源加热线 ``a=0.1``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CartesianHeatParams:
    """2D 单位正方形 Robin 热传导参数。

    Attributes:
        alpha: 热扩散系数，单位 m²/s。
        beta: Robin 系数 h/k，单位 1/m。
        u0: 初始温度与环境温度，单位 °C。
        f_mean: 热源 GRF 采样均值，单位 °C/s。
        f_std: 热源 GRF 采样波动（标准差），单位 °C/s。
        f_max: 热源归一化尺度 F_MAX，单位 °C/s；归一化约定 ``f_norm = f / F_MAX``。
        u_scale: 温度归一化尺度，单位 °C；归一化约定 ``u_norm = (u - u0) / u_scale``。
            默认与 F_MAX 同值仅为数值尺度约定，二者物理单位不同（°C 与 °C/s）。
    """

    alpha: float = 0.01
    beta: float = 5.0
    u0: float = 20.0
    f_mean: float = 100.0
    f_std: float = 30.0
    f_max: float = 150.0
    u_scale: float = 150.0

    def normalize_temperature(self, u: np.ndarray | float) -> np.ndarray:
        """物理温度 u（°C）→ 归一化温度 (u - u0)/u_scale。"""
        return (np.asarray(u, dtype=np.float64) - self.u0) / self.u_scale

    def denormalize_temperature(self, u_norm: np.ndarray | float) -> np.ndarray:
        """归一化温度 → 物理温度 u0 + u_scale·u_norm（°C），与上一函数互逆。"""
        return self.u0 + self.u_scale * np.asarray(u_norm, dtype=np.float64)

    def normalize_source(self, f: np.ndarray | float) -> np.ndarray:
        """物理热源 f（°C/s）→ 归一化热源 f/F_MAX。"""
        return np.asarray(f, dtype=np.float64) / self.f_max

    def denormalize_source(self, f_norm: np.ndarray | float) -> np.ndarray:
        """归一化热源 → 物理热源 F_MAX·f_norm（°C/s），与上一函数互逆。"""
        return self.f_max * np.asarray(f_norm, dtype=np.float64)


@dataclass(frozen=True)
class HeatLineParams:
    """1D 单位区间热传导参数（Dirichlet 零边界）。

    Attributes:
        a: 热扩散系数，单位 m²/s。
        length: 区间长度，单位 m（档案线为 1 m）。
        u_scale: 温度归一化尺度，单位 °C；档案线初始条件本身为 O(1) 量级，默认 1.0。
    """

    a: float = 0.4
    length: float = 1.0
    u_scale: float = 1.0

    @property
    def t_char(self) -> float:
        """扩散特征时间 length²/a，单位 s。"""
        return self.length**2 / self.a

    def normalize_temperature(self, u: np.ndarray | float) -> np.ndarray:
        """物理温度 u（°C）→ 归一化温度 u/u_scale。"""
        return np.asarray(u, dtype=np.float64) / self.u_scale

    def denormalize_temperature(self, u_norm: np.ndarray | float) -> np.ndarray:
        """归一化温度 → 物理温度 u_scale·u_norm（°C），与上一函数互逆。"""
        return self.u_scale * np.asarray(u_norm, dtype=np.float64)

    def normalize_time(self, t: np.ndarray | float) -> np.ndarray:
        """物理时间 t（s）→ 无量纲时间 t/t_char。"""
        return np.asarray(t, dtype=np.float64) / self.t_char

    def denormalize_time(self, t_star: np.ndarray | float) -> np.ndarray:
        """无量纲时间 → 物理时间 t_char·t*（s），与上一函数互逆。"""
        return self.t_char * np.asarray(t_star, dtype=np.float64)


def heat_line_cooling() -> HeatLineParams:
    """散热线参数（原档案 DataPINO 散热基准线）：a=0.4 m²/s，无热源。"""
    return HeatLineParams(a=0.4)


def heat_line_heating() -> HeatLineParams:
    """带源加热线参数（原档案 Heat_1D_Heating 线）：a=0.1 m²/s，配高斯热源。"""
    return HeatLineParams(a=0.1)
