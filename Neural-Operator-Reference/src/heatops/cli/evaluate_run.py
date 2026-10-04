"""Replay exactly one existing model bundle, without training or replacing metrics."""
import argparse
from pathlib import Path
import numpy as np
from heatops.evaluation.replay import replay_run
from heatops.training.run_context import write_json, file_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True, help="必须使用不存在的新目录")
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        predictions, saved, checkpoint = replay_run(args.run_dir)
        differences = {k: float(np.max(np.abs(v-saved[k]))) for k,v in predictions.items()}
        if not all(np.isfinite(v).all() for v in predictions.values()):
            raise FloatingPointError("重载预测出现非有限值")
        np.savez_compressed(out / "predictions.npz", **predictions)
        write_json(out / "metrics.json", {"checkpoint_sha256": file_hash(checkpoint), "max_abs_reload_difference": differences, "scope": "checkpoint replay; no retraining or accuracy claim"})
        print(differences)
        return 0
    except Exception as exc:
        write_json(out / "failure.json", {"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise
