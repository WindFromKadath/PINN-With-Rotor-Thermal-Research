from pathlib import Path

import numpy as np

from ring_rib_pinn.checkpoints import evaluate_run
from ring_rib_pinn.config import Config
from ring_rib_pinn.data import read_table
from ring_rib_pinn.legacy import legacy_evaluate

ROOT = Path(__file__).resolve().parents[1]


def test_legacy_replay_is_portable(bundle, tmp_path, monkeypatch):
    import ring_rib_pinn.artifacts as artifacts
    monkeypatch.setattr(artifacts, "register_run", lambda *args, **kwargs: None)
    values = bundle.config.serializable()
    values["output"]["root"] = str(tmp_path/"runs")
    values["data"]["rib_json"] = str(bundle.directory/"rib_models.json")
    originals = []
    for domain in ("upper", "lower", "rib"):
        path = tmp_path/f"{domain}.csv"
        read_table(ROOT/"data/processed"/bundle.config.case/f"{domain}_domain.csv").head(16).to_csv(path, index=False)
        values["data"][f"{domain}_csv"] = str(path)
        originals.append(path)
    config = Config(values, tmp_path)
    directory, metrics = legacy_evaluate(config, bundle.directory, "legacy-test")
    for path in originals:
        path.rename(path.with_suffix(".unavailable"))
    assert (directory/"checkpoint.pt").is_file()
    assert (directory/"code.zip").is_file()
    replay = evaluate_run(directory)
    for domain in ("upper", "lower", "rib"):
        np.testing.assert_allclose(replay["domain_mean"][domain]["rmse_K"],
                                   metrics["domain_mean"][domain]["rmse_K"], rtol=0, atol=1e-10)
