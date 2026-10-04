"""Record a fresh portable-use check for every selected historical bundle."""

import argparse
import json
from pathlib import Path
import shutil

import pandas as pd

from rotor_pinn.artifacts import file_hash, new_run, write_json
from rotor_pinn.config import Config
from rotor_pinn.legacy import LegacyBundle, verify_bundle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, default=Path("local_reference_versions"))
    parser.add_argument("--run-id")
    args = parser.parse_args()
    collection = args.collection.resolve()
    index = json.loads((collection / "index.json").read_text(encoding="utf-8"))
    cfg = Config(base=Path(__file__).resolve().parents[1])
    cfg.output.artifacts = "artifacts"
    cfg.training.device = "cpu"
    cfg.training.dtype = "float32"
    with new_run(cfg, args.run_id, "preserved legacy code/models usable without old source directory; inference, thickness, derivatives and repeated loading",
                 "generated local smoke inputs; no historical metric replication target") as (run, manifest):
        manifest["verification_tool_sha256"] = file_hash(__file__)
        manifest["input_sha256"]["collection_index"] = file_hash(collection / "index.json")
        results = []
        for row in index["selected"]:
            original = collection / row["bundle"]
            if file_hash(original / "profile.json") != row["profile_sha256"]:
                raise ValueError(f"Collection profile hash mismatch: {row['version_id']}")
            # Copy only the portable bundle; make every old source path unavailable.
            isolated = run / "portable_checks" / row["version_id"]
            shutil.copytree(original, isolated)
            profile = json.loads((isolated / "profile.json").read_text(encoding="utf-8"))
            profile["original_source_directory"] = str(run / "unavailable_original_source")
            profile["original_weights"] = {"unavailable": True}
            for info in profile["thickness"].values():
                info["original_path"] = str(run / "unavailable_original_weight")
            write_json(isolated / "profile.json", profile)
            check = verify_bundle(isolated)
            model = LegacyBundle(isolated)
            for region in ("upper", "lower", "rib"):
                frame = pd.read_csv(isolated / "examples" / f"{region}.csv")
                frame["predicted_T"] = model.predict(frame, region)
                if region != "rib":
                    frame["predicted_thickness_m"] = model.predict_thickness(frame, region)
                frame.to_csv(run / f"{row['version_id']}_{region}_predictions.csv", index=False, float_format="%.17g")
            check["historical_source_paths_unavailable"] = True
            results.append(check)
        write_json(run / "metrics.json", {"portable_versions": results, "all_usable": True})
        manifest["has_metrics"] = True
        print(json.dumps(results, indent=2))
        print(run)


if __name__ == "__main__":
    main()
