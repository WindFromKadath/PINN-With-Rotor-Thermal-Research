"""Unified dispatcher; individual modules remain callable with python -m."""
import importlib
import sys

COMMANDS = {
    "preprocess": "preprocess_annulus", "train-pino-1d": "run_pino_1d",
    "train-pino-annulus": "run_pino_annulus", "train-pidon": "run_pidon_annulus",
    "train-pi-deeponet": "run_pi_deeponet", "train-deeponet": "run_deeponet_2d",
    "train-pinn": "run_pinn_annulus", "compare": "compare_pino_1d",
    "legacy-evaluate": "legacy_evaluate", "evaluate": "evaluate_run",
}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print("heatops <command> [options]\n" + "\n".join(COMMANDS))
        return 0
    command = argv.pop(0)
    if command not in COMMANDS:
        raise SystemExit(f"未知命令：{command}")
    return importlib.import_module("heatops.cli." + COMMANDS[command]).main(argv)
