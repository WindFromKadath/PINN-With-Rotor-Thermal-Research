import numpy as np
import torch
from ..data.transforms import inputs, physical


def determine_gauge(model, x, mode, anchor=None, batch_size=4096):
    """One global reference shift, saved with the checkpoint; independent of inference batches."""
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    if mode == "constraint":
        if anchor is None:
            raise ValueError("Constraint gauge requires an anchor")
        with torch.no_grad():
            value = model(torch.as_tensor([anchor], dtype=dtype, device=device)).cpu().numpy()[0]
        return dict(mode="fixed", offset=value.tolist(), reference="constraint_anchor", anchor=anchor)
    if mode == "projection":
        total, weight = 0., 0.
        with torch.no_grad():
            for chunk in np.array_split(x, max(1, int(np.ceil(len(x)/batch_size)))):
                a = torch.as_tensor(chunk, dtype=dtype, device=device)
                u = model(a).cpu().double().numpy()
                total += float((u[:, 1]*chunk[:, 0]).sum())
                weight += float(chunk[:, 0].sum())
        return dict(mode="fixed", offset=[0., total/(weight+1e-12)], reference="global_radius_weighted",
                    points=len(x))
    if mode != "raw":
        raise ValueError(f"Unknown alignment: {mode}")
    return {"mode": "raw"}


def predict(model, frame, profile, spec, gauge=None, batch_size=4096):
    if batch_size < 1 or frame.empty:
        raise ValueError("Prediction needs data and a positive batch size")
    kind = spec["kind"]
    x = inputs(frame, profile, kind)
    dtype, device = next(model.parameters()).dtype, next(model.parameters()).device
    values = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            y = model(torch.as_tensor(x[start:start+batch_size], dtype=dtype, device=device)).cpu().numpy()
            if gauge and gauge.get("mode") == "fixed":
                y = y-np.asarray(gauge["offset"], dtype=y.dtype)
            values.append(physical(y, profile, kind))
    output = np.concatenate(values)
    if not np.isfinite(output).all():
        raise FloatingPointError("Non-finite predictions")
    return output
