"""Small local legacy inference check; no old training script is executed."""
import argparse
from pathlib import Path
import numpy as np
import torch
from heatops.models.legacy import load_legacy_model
from heatops.training.run_context import write_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)
    try:
        net, spec, record = load_legacy_model(args.package, args.model)
        family = spec["family"]
        with torch.no_grad():
            if family == "fno_1d":
                x, t = torch.linspace(0,1,32), torch.linspace(0,0.5,32)
                grid = torch.stack(torch.meshgrid(x,t,indexing="ij"),-1)[None]
                prediction = net(torch.sin(torch.pi*x)[None],grid).numpy()
            elif family == "fno_polar":
                axes = [torch.linspace(0,1,8),torch.linspace(0,1,24),torch.linspace(0,1,24)]
                grid = torch.stack(torch.meshgrid(*axes,indexing="ij"),-1)[None]
                pred_norm = net(torch.ones(1,8,24,24)*0.5,grid).numpy()
                lo, hi = spec["transforms"]["temperature_range"]
                prediction = lo + (hi-lo)*pred_norm
            else:
                tf = spec["transforms"]
                pts = torch.stack([torch.linspace(tf["r_min_star"],1,50),torch.zeros(50),torch.full((50,),tf["t_final_star"])],1)
                prediction = net((torch.ones(1,record["network"]["branch_layers"][0]),pts)).numpy() if family=="pidon" else net(pts).numpy()
                prediction = tf["T_inf"] + tf["T_scale"]*prediction
        if not np.isfinite(prediction).all():
            raise FloatingPointError("历史模型预测含非有限值")
        np.savez_compressed(out/"predictions.npz",prediction=prediction)
        write_json(out/"metrics.json",{"family":family,"model":args.model or next(iter(spec["models"])),"shape":list(prediction.shape),"finite":True,"source_checkpoint":record["source_checkpoint"],"scope":"historical model inference, not generalization validation"})
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.plot(prediction.reshape(-1))
        plt.xlabel("sample index"); plt.ylabel("historical model output")
        plt.tight_layout(); plt.savefig(out/"prediction.png"); plt.close()
        return 0
    except Exception as exc:
        write_json(out/"failure.json",{"status":"failed","error_type":type(exc).__name__,"error":str(exc)})
        raise
