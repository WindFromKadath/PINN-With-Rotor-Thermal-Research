"""Predict from a preserved local bundle without the historical project or data."""

import argparse
import json
from pathlib import Path

import pandas as pd

from rotor_pinn.legacy import LegacyBundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--region", choices=("upper", "lower", "rib"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--thickness", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_bundle(args.bundle, args.device), ensure_ascii=False, indent=2))
        return
    if args.region is None or args.input is None or args.output is None:
        parser.error("Prediction requires --region, --input and --output")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite output: {args.output}")
    bundle = LegacyBundle(args.bundle, args.device)
    frame = pd.read_csv(args.input)
    frame["predicted_T"] = bundle.predict(frame, args.region)
    if args.thickness:
        frame["predicted_thickness_m"] = bundle.predict_thickness(frame, args.region)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False, float_format="%.17g")
    print(f"Saved {len(frame)} finite predictions to {args.output}")


if __name__ == "__main__":
    main()
