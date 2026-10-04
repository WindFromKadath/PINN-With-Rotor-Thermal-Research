import pytest

from rotor_pinn.config import Config


@pytest.fixture
def config(tmp_path):
    cfg = Config(base=tmp_path)
    cfg.training.device = "cpu"
    cfg.training.dtype = "float64"
    cfg.contact.radial_nodes = 24
    cfg.contact.angular_nodes = 64
    cfg.contact.arc_nodes = 8
    cfg.contact.time_nodes = 3
    return cfg.validate()
