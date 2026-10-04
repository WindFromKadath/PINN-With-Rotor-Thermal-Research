"""轴对称圆环热传导问题：物理参数、无量纲化与稳态解析解。

问题定义（长度单位 m，温度单位 °C，其余 SI）：

- 几何：内半径 ``R_in``、外半径 ``R_out`` 的圆环（取单位轴向长度），
  轴对称使问题退化为径向一维；
- 控制方程（瞬态）：``rho*cp * ∂T/∂t = (1/r) * ∂/∂r(k * r * ∂T/∂r)``，
  稳态退化为 ``(r * dT/dr)' = 0``；
- 内边界 ``r=R_in``：Neumann 热流边界 ``-k * dT/dr = q_in``
  （``q_in > 0`` 表示热量沿 +r 方向流入域内，单位 W/m²）；
- 外边界 ``r=R_out``：Robin 对流边界 ``-k * dT/dr = h * (T - T_inf)``。

稳态解析解为 ``T(r) = C1 * ln(r) + C2``，由两个边界条件定常数。

无量纲化约定（``*`` 上标）：

- ``r* = r / R_out``，``t* = t / t_scale``，``T* = (T - T_inf) / T_scale``；
- 此时内边界为 ``dT*/dr* = -1``，外边界为 ``-dT*/dr* = Bi * T*``，
  稳态解 ``T* = C1* * ln(r*) + C2*``，``C1* = -R_in*``，``C2* = -C1* / Bi``。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AnnulusParams:
    """轴对称圆环热传导物理参数（默认值为钢材基准工况）。

    Attributes:
        R_in: 内半径，单位 m。
        R_out: 外半径，单位 m。
        k: 热导率，单位 W/(m·K)。
        rho: 密度，单位 kg/m³。
        cp: 比热容，单位 J/(kg·K)。
        h: 外边界对流换热系数，单位 W/(m²·K)。
        T_inf: 环境温度，单位 °C。
        q_in: 内边界热流密度，单位 W/m²。
    """

    R_in: float = 0.05
    R_out: float = 0.147
    k: float = 50.0
    rho: float = 7850.0
    cp: float = 500.0
    h: float = 10.0
    T_inf: float = 20.0
    q_in: float = 1e5

    @property
    def alpha(self) -> float:
        """热扩散系数 k/(rho·cp)，单位 m²/s（默认参数下约 1.274e-5）。"""
        return self.k / (self.rho * self.cp)

    @property
    def T_scale(self) -> float:
        """温度尺度 q_in·R_out/k，单位 °C（默认参数下 294）。"""
        return self.q_in * self.R_out / self.k

    @property
    def t_scale(self) -> float:
        """扩散时间尺度 R_out²/alpha，单位 s（默认参数下约 1696）。"""
        return self.R_out**2 / self.alpha

    @property
    def Bi(self) -> float:
        """外半径定义的 Biot 数 h·R_out/k（默认参数下 0.0294）。"""
        return self.h * self.R_out / self.k

    @property
    def R_in_star(self) -> float:
        """无量纲内半径 R_in/R_out（默认参数下约 0.3401）。"""
        return self.R_in / self.R_out

    @property
    def c1(self) -> float:
        """物理形式稳态常数 C1 = -q_in·R_in/k，单位 °C。"""
        return -self.q_in * self.R_in / self.k

    @property
    def c2(self) -> float:
        """物理形式稳态常数 C2，由外边界 Robin 条件确定，单位 °C。"""
        return self.T_inf - self.c1 * np.log(self.R_out) - self.k * self.c1 / (self.h * self.R_out)

    @property
    def c1_star(self) -> float:
        """无量纲稳态常数 C1* = -R_in*。"""
        return -self.R_in_star

    @property
    def c2_star(self) -> float:
        """无量纲稳态常数 C2* = -C1*/Bi。"""
        return -self.c1_star / self.Bi

    # ---------- 无量纲化换算（往返互逆） ----------

    def to_dimensionless_temperature(self, T: np.ndarray | float) -> np.ndarray:
        """物理温度 T（°C）→ 无量纲温度 T* = (T - T_inf)/T_scale。"""
        return (np.asarray(T, dtype=np.float64) - self.T_inf) / self.T_scale

    def to_dimensional_temperature(self, T_star: np.ndarray | float) -> np.ndarray:
        """无量纲温度 T* → 物理温度 T = T_inf + T_scale·T*（°C）。"""
        return self.T_inf + self.T_scale * np.asarray(T_star, dtype=np.float64)

    def to_dimensionless_radius(self, r: np.ndarray | float) -> np.ndarray:
        """物理半径 r（m）→ 无量纲半径 r* = r/R_out。"""
        return np.asarray(r, dtype=np.float64) / self.R_out

    def to_dimensional_radius(self, r_star: np.ndarray | float) -> np.ndarray:
        """无量纲半径 r* → 物理半径 r = R_out·r*（m）。"""
        return self.R_out * np.asarray(r_star, dtype=np.float64)

    def to_dimensionless_time(self, t: np.ndarray | float) -> np.ndarray:
        """物理时间 t（s）→ 无量纲时间 t* = t/t_scale。"""
        return np.asarray(t, dtype=np.float64) / self.t_scale

    def to_dimensional_time(self, t_star: np.ndarray | float) -> np.ndarray:
        """无量纲时间 t* → 物理时间 t = t_scale·t*（s）。"""
        return self.t_scale * np.asarray(t_star, dtype=np.float64)


def analytical_steady_state(r: np.ndarray | float, params: AnnulusParams) -> np.ndarray:
    """稳态解析解 T(r) = C1·ln(r) + C2，单位 °C。

    边界残差恒为零（连续意义下）：内边界 ``-k·dT/dr = q_in``、
    外边界 ``-k·dT/dr = h·(T - T_inf)`` 均严格满足。
    """
    r = np.asarray(r, dtype=np.float64)
    return params.c1 * np.log(r) + params.c2


def analytical_steady_state_dimensionless(
    r_star: np.ndarray | float, params: AnnulusParams
) -> np.ndarray:
    """无量纲稳态解析解 T*(r*) = C1*·ln(r*) + C2*。

    与物理形式的关系：``T_inf + T_scale * T*(r/R_out) == T(r)``（浮点精度内）。
    """
    r_star = np.asarray(r_star, dtype=np.float64)
    return params.c1_star * np.log(r_star) + params.c2_star
