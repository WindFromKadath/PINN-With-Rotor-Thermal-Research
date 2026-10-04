"""Repeatable migration audit, CPU tests and GPU model-group smoke check."""

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from ring_rib_pinn.artifacts import environment, file_hash, write_json, new_run, finish_run
from ring_rib_pinn.config import Config

ROOT = Path(__file__).resolve().parents[1]


def migration(source=None):
    with (ROOT/"local_reference_versions/source_manifest.csv").open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    checked, derived = 0, 0
    for row in rows:
        for relative in filter(None, row["copied_to"].split(";")):
            target = ROOT/relative
            if file_hash(target) != row["sha256"]:
                # Processed CSVs may be regenerated; raw copies remain the immutable evidence.
                if relative.startswith("data/processed/") and target.suffix == ".csv":
                    derived += 1
                else:
                    raise ValueError(f"Imported hash mismatch: {relative}")
            else:
                checked += 1
        if source and file_hash(source/row["source_path"]) != row["sha256"]:
            raise ValueError(f"Original source changed: {row['source_path']}")
    return {"source_files": len(rows), "copied_targets_hash_match": checked,
            "regenerated_processed_csvs": derived, "original_source_checked": source is not None}


def gpu(directory):
    import torch
    from ring_rib_pinn.checkpoints import save_checkpoint, load_checkpoint
    from ring_rib_pinn.coupling import CouplingState
    from ring_rib_pinn.legacy import LegacyBundle
    from ring_rib_pinn.materials import Material
    from ring_rib_pinn.physics import Physics
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA validation requires a CUDA PyTorch extra and available GPU")
    results = {"device": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    for name in ("heating_coupled", "cooling_hardconstraint"):
        bundle = LegacyBundle(ROOT/"local_reference_versions"/name, "cuda")
        p = Physics(bundle.dde, bundle.config, bundle.geometry,
                    Material(bundle.config, bundle.device, bundle.dtype), bundle.thickness,
                    bundle.networks, CouplingState(bundle.geometry))
        loss = 0
        before = {}
        for domain in ("upper", "lower", "rib"):
            x = torch.tensor([[.65, 1., .003], [.75, 2., .005]], device="cuda", requires_grad=True)
            y = bundle.networks[domain](x)
            before[domain] = y.detach()
            residual = p.rib(x, y)[0] if domain == "rib" else p.ring(x, y, domain)[0]
            loss = loss+residual.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        for net in bundle.networks.values():
            gradients = [v.grad for v in net.parameters() if v.grad is not None]
            assert gradients and all(torch.isfinite(v).all() for v in gradients)
        path = directory/f"{name}.pt"
        save_checkpoint(path, bundle.config, bundle.geometry, bundle.networks, bundle.thickness)
        _, _, nets, _ = load_checkpoint(path, "cuda")
        x = torch.tensor([[.65, 1., .003], [.75, 2., .005]], device="cuda")
        with torch.no_grad():
            for domain, net in nets.items():
                torch.testing.assert_close(net(x), before[domain], rtol=0, atol=0)
        results[name] = {"pde_loss": loss.item(), "forward_backward_reload": "passed"}
        bundle.dde.grad.clear()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    directory = args.output.resolve()
    cfg = Config({"output": {"root": str(directory.parent)}})
    directory = new_run(cfg, directory.name, "gpu_smoke" if args.gpu else "implementation_checks")
    try:
        report = {"environment": environment(), "migration": migration(args.source)}
        if args.gpu:
            report["gpu"] = gpu(directory)
        else:
            completed = subprocess.run([sys.executable, "-m", "pytest", "-q", "--junitxml", str(directory/"tests.xml")],
                                       cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            (directory/"tests.log").write_text(completed.stdout, encoding="utf-8")
            report["cpu_tests_exit_code"] = completed.returncode
            if completed.returncode:
                write_json(directory/"verification.json", report)
                raise RuntimeError(completed.stdout)
            for name in ("heating_coupled", "cooling_hardconstraint"):
                path = ROOT/"local_reference_versions"/name/"profile.json"
                profile = json.loads(path.read_text(encoding="utf-8"))
                profile.update(status="implementation_compatibility_verified", validation={
                    "run_id": directory.name, "prediction_tolerance_K": 1e-3,
                    "residual_rtol": 1e-4, "residual_atol": 1e-5,
                    "scope": "fixed-point implementation parity; not physical-model validation"})
                write_json(path, profile)
        write_json(directory/"verification.json", report)
        finish_run(cfg, directory, notes="Computational checks; no physical hypothesis conclusion")
    except Exception as error:
        finish_run(cfg, directory, error)
        raise
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
