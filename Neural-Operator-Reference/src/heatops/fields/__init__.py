"""heatops.fields — 随机场与初始条件采样（全部 seed 可控）。"""

from .grf import GRF1DGenerator, GRF2DGenerator
from .initial_conditions import (
    gaussian_heat_source,
    sample_random_sines,
    sample_single_sine,
    sample_sine_or_square,
    sample_square_wave,
)

__all__ = [
    "GRF1DGenerator",
    "GRF2DGenerator",
    "gaussian_heat_source",
    "sample_random_sines",
    "sample_single_sine",
    "sample_sine_or_square",
    "sample_square_wave",
]
