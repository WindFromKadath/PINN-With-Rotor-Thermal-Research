"""Check completed smoke runs from another cwd and portable model directories."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

from ring_rib_pinn.artifacts import new_run, finish_run, write_json, file_hash
from ring_rib_pinn.checkpoints import load_checkpoint
from ring_rib_pinn.config import Config
from ring_rib_pinn.evaluation import predict


def command(directory, *args):
    cli = Path(sys.executable).parent/"ring-rib-pinn.exe"
    result = subprocess.run([str(cli), *map(str, args)], cwd=directory,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout)
    return result.stdout


def check_run(source, destination, cwd):
    status = json.loads((source/"status.json").read_text(encoding="utf-8"))
    assert status["status"] == "completed"
    config, geometry, nets, _ = load_checkpoint(source/"checkpoint.pt")
    t = config.t
    assert (t["seed"], t["thickness_steps"], t["independent_steps"], t["rib_steps"],
            t["outer_iterations"], t["coupling_steps"], t["lbfgs"]) == (42, 100, 40, 40, 2, 40, False)
    destination.mkdir()
    for name in ("checkpoint.pt", "config.json", "status.json", "rib_models.json"):
        shutil.copy2(source/name, destination/name)
    shutil.copytree(source/"inputs", destination/"inputs")
    # Stage 2 and final weights must agree for the no-LBFGS fixed short budget.
    _, _, stage, _ = load_checkpoint(source/"coupling_2.pt")
    for domain in nets:
        for key, state in nets[domain].state_dict().items():
            torch.testing.assert_close(state, stage[domain].state_dict()[key], rtol=0, atol=0)
    log = []
    # Fresh cwd and moved run directory: all evaluation inputs are local to the run.
    log.append(command(cwd, "evaluate", "--run-dir", destination))
    max_difference = {}
    for domain, net in nets.items():
        frame = pd.read_csv(destination/"inputs"/f"{domain}.csv")
        before = predict(net, frame, config, geometry, domain)
        path = destination/f"custom_{domain}.csv"
        log.append(command(cwd, "predict", "--run-dir", destination, "--domain", domain,
                           "--input", destination/"inputs"/f"{domain}.csv", "--output", path))
        after = pd.read_csv(path).Pred_T.to_numpy()
        assert np.isfinite(after).all()
        np.testing.assert_allclose(after, before, rtol=0, atol=1e-9)
        max_difference[domain] = float(np.max(np.abs(after-before)))
    log.append(command(cwd, "plot", "--run-dir", destination))
    (destination/"cli.log").write_text("\n".join(log), encoding="utf-8")
    return {"status": "passed", "case": config.case, "source_run": source.name,
            "checkpoint_sha256": file_hash(destination/"checkpoint.pt"),
            "max_reload_prediction_difference_K": max_difference}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heating", type=Path, required=True)
    parser.add_argument("--cooling", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    cfg = Config({"output": {"root": str(output.parent)}})
    directory = new_run(cfg, output.name, "delivery_checks")
    try:
        with tempfile.TemporaryDirectory(prefix="ring_rib_cwd_") as cwd:
            report = {case: check_run(path.resolve(), directory/case, cwd)
                      for case, path in (("heating", args.heating), ("cooling", args.cooling))}
        write_json(directory/"verification.json", report)
        finish_run(cfg, directory, notes="Moved run directories and all inference CLI commands passed")
        print(json.dumps(report, indent=2))
    except Exception as error:
        finish_run(cfg, directory, error)
        raise


if __name__ == "__main__":
    main()
