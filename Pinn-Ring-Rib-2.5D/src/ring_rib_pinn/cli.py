"""Small explicit command surface; all substantive work is importable library code."""

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ring-rib-pinn")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train-thickness", "train", "demo", "legacy-evaluate"):
        command = subparsers.add_parser(name)
        command.add_argument("--config", type=Path, required=True)
        command.add_argument("--device", choices=["cpu", "cuda", "auto"])
        if name != "preprocess":
            command.add_argument("--run-id")
        if name == "legacy-evaluate":
            command.add_argument("--bundle", type=Path, required=True)
    for name in ("evaluate", "plot", "predict"):
        command = subparsers.add_parser(name)
        command.add_argument("--run-dir", type=Path, required=True)
        if name != "plot":
            command.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
        if name == "predict":
            command.add_argument("--domain", choices=["upper", "lower", "rib"], required=True)
            command.add_argument("--input", type=Path, required=True)
            command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command in ("preprocess", "train-thickness", "train", "demo", "legacy-evaluate"):
            from .config import load_config
            config = load_config(args.config)
            if args.device:
                config.t["device"] = args.device
            if args.command == "preprocess":
                from .data import preprocess
                result = preprocess(config)
            elif args.command == "legacy-evaluate":
                from .legacy import legacy_evaluate
                directory, metrics = legacy_evaluate(config, args.bundle, args.run_id)
                result = {"run": str(directory), "metrics": metrics}
            else:
                from .training import train
                if args.command == "demo" and (config.t["lbfgs"] or config.t["outer_iterations"] != 2
                    or config.t["independent_steps"] != 40 or config.t["rib_steps"] != 40
                    or config.t["coupling_steps"] != 40 or config.t["thickness_steps"] != 100):
                    raise ValueError("demo requires the supplied fixed short-budget configuration")
                result = {"run": str(train(config, args.run_id, args.command == "train-thickness"))}
        elif args.command == "evaluate":
            from .checkpoints import evaluate_run
            result = evaluate_run(args.run_dir, args.device)
        elif args.command == "plot":
            from .plotting import plot_run
            result = {"plot": str(plot_run(args.run_dir))}
        else:
            from .checkpoints import load_checkpoint
            from .data import read_table
            from .evaluation import predict
            config, geometry, networks, _ = load_checkpoint(args.run_dir / "checkpoint.pt", args.device)
            frame = read_table(args.input)
            frame["Pred_T"] = predict(networks[args.domain], frame, config, geometry, args.domain)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            frame.to_csv(args.output, index=False)
            result = {"output": str(args.output), "rows": len(frame)}
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        parser.exit(1, f"{type(error).__name__}: {error}\n")
