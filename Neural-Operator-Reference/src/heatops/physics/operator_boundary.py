"""DeepXDE operator boundary adapter that receives auxiliary function values."""
import deepxde as dde
import torch


class HeatFluxBC(dde.icbc.BC):
    """du/dr + q* = 0; q* is the operator's aux_var, never its coordinates."""
    def __init__(self, geom, on_boundary):
        super().__init__(geom, on_boundary, component=0)

    def error(self, X, inputs, outputs, beg, end, aux_var=None):
        if aux_var is None:
            raise ValueError("HeatFluxBC 需要算子的 aux_var 热流值")
        values = torch.as_tensor(aux_var, dtype=outputs.dtype, device=outputs.device)
        du_r = dde.grad.jacobian(outputs, inputs, i=0, j=0)
        return (du_r + values.reshape(-1, 1))[beg:end]
