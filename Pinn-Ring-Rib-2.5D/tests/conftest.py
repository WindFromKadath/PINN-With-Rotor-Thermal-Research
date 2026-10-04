from pathlib import Path

import pytest

from ring_rib_pinn.legacy import LegacyBundle

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session", params=["heating_coupled", "cooling_hardconstraint"])
def bundle(request):
    return LegacyBundle(ROOT / "local_reference_versions" / request.param)
