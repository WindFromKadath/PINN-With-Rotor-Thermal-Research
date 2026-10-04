import numpy as np


def metrics(prediction, truth, kind):
    if prediction.shape != truth.shape or not np.isfinite(prediction).all() or not np.isfinite(truth).all():
        raise ValueError("Metrics require matching finite prediction/target arrays")
    error = prediction-truth
    denominator = float(np.linalg.norm(truth))
    result = dict(mae=float(np.abs(error).mean()), rmse=float(np.sqrt((error**2).sum(axis=1).mean())),
                  max_absolute_error=float(np.abs(error).max()),
                  relative_l2=float(np.linalg.norm(error)/denominator) if denominator > 1e-12 else None,
                  count=len(error), units="K" if kind in ("thermal", "steady") else "m",
                  relative_l2_denominator="absolute Kelvin" if kind in ("thermal", "steady") else "displacement norm")
    if error.shape[1] == 2:
        result["component_rmse_m"] = np.sqrt((error**2).mean(axis=0)).tolist()
        result["rmse_um"] = result["rmse"]*1e6
    return result
