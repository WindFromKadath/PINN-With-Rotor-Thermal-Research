"""Single command surface with actionable failures and explicit run ids."""

import argparse
import json
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rotor-pinn")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("preprocess", "train-thickness", "train", "demo"):
        command = subcommands.add_parser(name)
        command.add_argument("--config", required=True)
        if name == "preprocess":
            command.add_argument("--overwrite", action="store_true")
        else:
            command.add_argument("--run-id")
            command.add_argument("--device", choices=["auto", "cpu", "cuda"])
    for name in ("evaluate", "plot"):
        command = subcommands.add_parser(name)
        command.add_argument("--run-dir", required=True)
        if name == "evaluate":
            command.add_argument("--split", choices=["train", "validation", "test", "all"], default="test")
            command.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args(argv)
    try:
        if args.command == "evaluate":
            from .evaluation import evaluate_run
            result = evaluate_run(args.run_dir, args.split, args.device)
            print(json.dumps(result, indent=2))
        elif args.command == "plot":
            from .plotting import plot_run
            print(plot_run(args.run_dir))
        else:
            from .config import load_config
            config = load_config(args.config)
            if args.command == "preprocess":
                from .data import preprocess
                print(json.dumps(preprocess(config, args.overwrite)))
            else:
                if args.device:
                    config.training.device = args.device
                from .training import train_pipeline
                print(train_pipeline(config, args.run_id, demo=args.command == "demo",
                                     thickness_only=args.command == "train-thickness"))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError, FloatingPointError) as error:
        print(f"rotor-pinn: {type(error).__name__}: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
