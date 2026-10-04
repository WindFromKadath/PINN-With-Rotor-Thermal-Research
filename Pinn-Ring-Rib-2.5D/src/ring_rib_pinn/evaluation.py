"""Physical-temperature prediction and metrics on explicitly named point sets."""

import numpy as np
import torch

from .data import normalized_inputs


def predict(net, frame, config, geometry, domain, batch_size=4096):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    inputs = normalized_inputs(frame, config, geometry, domain)
    parameter = next(net.parameters())
    outputs = []
    net.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            tensor = torch.as_tensor(inputs[start:start+batch_size], device=parameter.device,
                                     dtype=parameter.dtype)
            output = net(tensor)[:, 0] * config.p["temperature_scale"] + config.p["ambient"]
            outputs.append(output.cpu().numpy())
    # Preserve the computed float32 value when pandas serializes decimal CSVs.
    # Widening here changes neither inference nor its Kelvin conversion.
    result = np.concatenate(outputs).astype(np.float64)
    if not np.isfinite(result).all():
        raise FloatingPointError("Prediction contains non-finite temperatures")
    return result


def compute_metrics(true, prediction):
    true, prediction = np.asarray(true, dtype=np.float64), np.asarray(prediction, dtype=np.float64)
    if true.shape != prediction.shape or true.size == 0:
        raise ValueError("Metric inputs must have equal nonempty shapes")
    if not np.isfinite(true).all() or not np.isfinite(prediction).all():
        raise FloatingPointError("Metrics require finite temperatures")
    error = prediction - true
    return {"count": int(true.size), "mse_K2": float(np.mean(error**2)),
            "rmse_K": float(np.sqrt(np.mean(error**2))),
            "mae_K": float(np.mean(np.abs(error))), "max_absolute_K": float(np.max(np.abs(error))),
            "relative_l2_kelvin": float(np.linalg.norm(error)/(np.linalg.norm(true)+1e-12))}


def evaluate_frames(networks, frames, config, geometry, output, prefix="domain"):
    metrics, errors = {}, []
    for domain, frame in frames.items():
        pred = predict(networks[domain], frame, config, geometry, domain)
        saved = frame.copy()
        saved["Pred_T"] = pred
        saved["error_K"] = pred-frame.True_T.to_numpy()
        saved.to_csv(output / f"{prefix}_{domain}_predictions.csv", index=False)
        metrics[domain] = compute_metrics(frame.True_T.to_numpy(), pred)
        errors.append((frame.True_T.to_numpy(), pred))
    metrics["all_points"] = compute_metrics(np.concatenate([a for a, _ in errors]),
                                            np.concatenate([b for _, b in errors]))
    return metrics
