"""One CLI for explicit runs, portable checkpoints and historical packages."""
import argparse
from pathlib import Path
import json
import sys
import contextlib
import datetime
import numpy as np
import pandas as pd
from ..config import load_config
from ..training.run import Run, write_json, sha256
from ..models.checkpoint import load
from ..evaluation.predict import predict
from ..evaluation.metrics import metrics
from ..plotting.figures import plot_predictions

ROOT = Path(__file__).resolve().parents[3]


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
    def write(self, text):
        self.stream.write(text)
        self.file.write(text)
    def flush(self):
        self.stream.flush()
        self.file.flush()


def _run_id():
    return datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S_%f")


def replay(run_dir, label=None, device="cpu", make_plot=False):
    """Use saved inputs and model binding without reading original config paths."""
    run_dir = Path(run_dir).resolve()
    files = [run_dir/f"{label}.pt"] if label else sorted(run_dir.glob("*.pt"))
    result = {}
    for ck in files:
        input_file = run_dir/f"{ck.stem}_inputs.csv"
        if not input_file.exists():
            if label:
                raise FileNotFoundError(f"Missing saved evaluation inputs: {input_file.name}")
            continue
        model, p, meta = load(ck, device)
        config_file = run_dir/"config.json"
        if config_file.exists():
            saved_config = json.loads(config_file.read_text(encoding="utf-8"))
            if saved_config.get("physics") and saved_config["physics"] != p.to_dict():
                raise ValueError("Saved run configuration and model physical profile mismatch")
        initial_metrics = run_dir/f"{ck.stem}_metrics.json"
        if initial_metrics.exists():
            evidence = json.loads(initial_metrics.read_text(encoding="utf-8"))
            if evidence["checkpoint_sha256"] != sha256(ck) or evidence["point_set_sha256"] != sha256(input_file):
                raise ValueError("Run checkpoint/point-set binding mismatch")
        frame = pd.read_csv(input_file)
        output = predict(model, frame, p, meta["spec"], meta["gauge"])
        cols = ["T"] if meta["spec"]["kind"] in ("thermal", "steady") else ["u_r", "u_z"]
        truth = frame[cols].to_numpy()
        result[ck.stem] = metrics(output, truth, meta["spec"]["kind"])
        if make_plot:
            plot_predictions(frame, output, truth, meta["spec"]["kind"], run_dir/"figures"/ck.stem)
    if not result:
        raise ValueError("No saved models with evaluation input snapshots in the specified run")
    return result


def parser():
    p = argparse.ArgumentParser(description="3D thermal / axisymmetric thermoelastic rotor reference workflows")
    sub = p.add_subparsers(dest="command", required=True)
    for cmd in ("preprocess", "sample", "train-thermal", "train-mechanical", "train-coupled", "train-mlp", "train-iwg", "train-xpinn", "demo"):
        q = sub.add_parser(cmd)
        q.add_argument("--config", type=Path, default=None)
        q.add_argument("--run-id", default=None)
        q.add_argument("--device", default=None)
    for cmd in ("evaluate", "plot"):
        q = sub.add_parser(cmd)
        q.add_argument("--run-dir", type=Path, required=True)
        q.add_argument("--model", default=None)
        q.add_argument("--device", default="cpu")
    q = sub.add_parser("predict")
    q.add_argument("--run-dir", type=Path, required=True)
    q.add_argument("--model", default="thermal")
    q.add_argument("--input", type=Path, required=True, help="Canonical SI r,z,t,theta CSV")
    q.add_argument("--output", type=Path, required=True)
    q.add_argument("--device", default="cpu")
    q = sub.add_parser("legacy-evaluate")
    q.add_argument("--bundle", required=True)
    q.add_argument("--run-id", default=None)
    q.add_argument("--device", default="cpu")
    q = sub.add_parser("compare")
    q.add_argument("--run-dirs", nargs="+", type=Path, required=True)
    q.add_argument("--output", type=Path, required=True)
    return p


def execute(args):
    if args.command in ("evaluate", "plot"):
        print(json.dumps(replay(args.run_dir, args.model, args.device, args.command == "plot"), indent=2))
        return
    if args.command == "predict":
        if args.output.exists():
            raise FileExistsError("Prediction output exists")
        frame = pd.read_csv(args.input)
        if not {"r", "z", "t", "theta"} <= set(frame) or frame.empty:
            raise ValueError("Prediction input requires nonempty canonical SI columns r,z,t,theta")
        if not np.isfinite(frame[["r", "z", "t", "theta"]].to_numpy()).all() or (frame.r <= 0).any() or (frame.t < 0).any():
            raise ValueError("Invalid prediction coordinates")
        model, p, meta = load(args.run_dir/f"{args.model}.pt", args.device)
        output = predict(model, frame, p, meta["spec"], meta["gauge"])
        for j, col in enumerate(["T_prediction_K"] if output.shape[1] == 1 else ["u_r_prediction_m", "u_z_prediction_m"]):
            frame[col] = output[:, j].astype(np.float64)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.output, index=False, float_format="%.17g")
        return
    if args.command == "compare":
        if args.output.exists():
            raise FileExistsError("Comparison output exists")
        data = {f"{i}:{d.name}": replay(d) for i, d in enumerate(args.run_dirs)}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, dict(runs=data, scope="Compare only matching point sets, profiles, budgets and reference frames"))
        return
    if args.command == "legacy-evaluate":
        folder = ROOT/"local_reference_versions/packages"/args.bundle
        if not folder.is_dir() or folder.resolve().parent != (ROOT/"local_reference_versions/packages").resolve():
            raise ValueError("Unknown historical model bundle")
        config = dict(runtime=dict(device=args.device, seed=42), paths={}, bundle=args.bundle)
        from ..training.workflows import finish
        with Run(ROOT/"artifacts", args.run_id or _run_id(), config, args.command) as run:
            reports = {}
            for ck in sorted(folder.glob("*.pt")):
                model, p, meta = load(ck, args.device)
                example = folder/f"{ck.stem}_examples.csv"
                reports[ck.stem] = finish(run, ck.stem, model, meta["spec"], p, pd.read_csv(example), meta["gauge"], meta["source"])
            if not reports:
                raise ValueError("Historical package contains no converted models")
            write_json(run.path/"metrics.json", reports)
        print(str(run.path))
        return
    defaults = {"preprocess": "preprocess", "sample": "demos/coupled", "demo": "demos/coupled"}
    key = defaults.get(args.command, "demos/"+args.command.removeprefix("train-"))
    if key == "demos/xpinn":
        key = "demos/xpinn_steady"
    cfg = load_config(args.config or ROOT/f"configs/{key}.toml")
    if args.device:
        cfg["runtime"]["device"] = args.device
    with Run(cfg["paths"]["artifacts"], args.run_id or _run_id(), cfg, args.command) as run:
        with (run.path/"console.log").open("w", encoding="utf-8") as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                if args.command == "preprocess":
                    from ..data.comsol import preprocess
                    count = preprocess(cfg["paths"]["input"], cfg["paths"]["output"], **cfg["units"])
                    write_json(run.path/"metrics.json", dict(rows=count, output_sha256=sha256(cfg["paths"]["output"])))
                elif args.command == "sample":
                    from ..training.workflows import setup, sample_geometry
                    p, rng = setup(cfg)
                    dom, ic, bc = sample_geometry(cfg, p, rng)
                    dom.to_csv(run.path/"domain.csv", index=False)
                    ic.to_csv(run.path/"initial.csv", index=False)
                    for name, (frame, normals, _) in bc.items():
                        for j, col in enumerate(("nx", "ny", "nz")):
                            frame[col] = normals[:, j]
                        frame.to_csv(run.path/f"boundary_{name}.csv", index=False)
                    write_json(run.path/"metrics.json", dict(domain=len(dom), initial=len(ic)))
                else:
                    from ..training.workflows import train
                    train(cfg, run, cfg["mode"] if args.command == "demo" else args.command.removeprefix("train-"))
    print(str(run.path))


def main():
    args = parser().parse_args()
    try:
        execute(args)
    except (ValueError, FileNotFoundError, FileExistsError, FloatingPointError, RuntimeError) as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
