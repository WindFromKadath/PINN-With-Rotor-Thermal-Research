from pathlib import Path
import json

import numpy as np
import pytest

from ring_rib_pinn.data import read_table
from ring_rib_pinn.reference import SourceReference

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["heating_v65", "heating_with_data", "cooling_hardconstraint", "cooling_with_data"])
def test_selected_original_version_loads_and_predicts(name):
    directory = ROOT/"local_reference_versions/selected"/name
    bundle = SourceReference(directory)
    profile = json.loads((directory/"profile.json").read_text(encoding="utf-8"))
    for domain in ("upper", "lower", "rib"):
        frame = read_table(ROOT/"data/processed"/bundle.config.case/f"{domain}_domain.csv").sample(n=32, random_state=42)
        prediction = bundle.predict(frame, domain)
        assert len(prediction) == len(frame)
        assert np.isfinite(prediction.Pred_T).all()
        for head in profile["output_heads"][domain][1:]:
            assert np.isfinite(prediction[head]).all()
            assert prediction[head].between(.5, 1.5).all()
