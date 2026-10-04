"""Predict with one locally selected, source-bound original version."""

import argparse
from pathlib import Path

from ring_rib_pinn.data import read_table
from ring_rib_pinn.reference import SourceReference


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--domain", choices=["upper", "lower", "rib"], required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    args = parser.parse_args()
    bundle = SourceReference(args.bundle, args.device)
    predicted = bundle.predict(read_table(args.input), args.domain)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    predicted.to_csv(args.output, index=False)
    print(f"{len(predicted)} predictions saved: {args.output}")


if __name__ == "__main__":
    main()
