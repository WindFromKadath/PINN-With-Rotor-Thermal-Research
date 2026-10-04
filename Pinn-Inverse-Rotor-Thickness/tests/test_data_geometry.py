import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad

from rotor_pinn.config import from_dict, load_config
from rotor_pinn.data import adapt_comsol, normalized_inputs, preprocess, spatial_split, validate_frame
from rotor_pinn.geometry import RotorGeometry
from rotor_pinn.synthetic import demo_frames


def test_arc_length_and_inverse(config):
    geometry = RotorGeometry(config.geometry)
    g = config.geometry
    reference, _ = quad(lambda r: np.sqrt(1 + (g.rib_slope * r)**2), g.rib_start, g.rib_end)
    assert geometry.length == pytest.approx(reference, rel=1e-12)
    radius = np.linspace(g.rib_start, g.rib_end, 41)
    np.testing.assert_allclose(geometry.s_to_r(geometry.r_to_s(radius)), radius, atol=1e-10)
    with pytest.raises(ValueError):
        geometry.r_to_s(np.array([g.rib_start - 0.01]))
    with pytest.raises(ValueError):
        geometry.s_to_r(np.array([geometry.length + 0.01]))


def test_angle_period_and_discrete_assignment(config):
    geometry = RotorGeometry(config.geometry)
    ids = np.arange(config.geometry.rib_count)
    radius = np.full(len(ids), 0.15)
    angle = geometry.theta(radius, ids)
    np.testing.assert_array_equal(geometry.assign_rib_id(radius, angle + 4 * np.pi), ids)
    frame = demo_frames(config)["upper"].iloc[:10].copy()
    frame["theta"] -= 4 * np.pi
    result = validate_frame(frame, config, "upper")
    assert ((result.theta >= 0) & (result.theta < 2 * np.pi)).all()


def test_group_split_disjoint_order_independent(config):
    frame = validate_frame(demo_frames(config)["upper"], config, "upper")
    result = spatial_split(frame, config.training)
    assert result.groupby(["r", "theta"])["split"].nunique().max() == 1
    groups = {name: set(map(tuple, part[["r", "theta"]].to_numpy()))
              for name, part in result.groupby("split")}
    assert groups["train"].isdisjoint(groups["validation"])
    assert groups["train"].isdisjoint(groups["test"])
    assert groups["validation"].isdisjoint(groups["test"])
    shuffled = spatial_split(frame.sample(frac=1, random_state=6), config.training)
    left = result.sort_values(["r", "theta", "t"])["split"].to_numpy()
    right = shuffled.sort_values(["r", "theta", "t"])["split"].to_numpy()
    np.testing.assert_array_equal(left, right)
    with pytest.raises(ValueError, match="three"):
        spatial_split(frame.iloc[:1], config.training)


@pytest.mark.parametrize("kind", ["missing", "nonfinite", "thickness", "rib_id", "kelvin"])
def test_invalid_data_rejected(config, kind):
    region = "rib" if kind == "rib_id" else "upper"
    frame = demo_frames(config)[region].iloc[:3].copy()
    if kind == "missing":
        frame = frame.drop(columns="t")
    elif kind == "nonfinite":
        frame.loc[0, "True_T"] = np.nan
    elif kind == "thickness":
        frame.loc[0, "thickness"] = 0
    elif kind == "rib_id":
        frame["rib_id"] = frame.rib_id.astype(float)
        frame.loc[0, "rib_id"] = 1.5
    else:
        frame.loc[0, "True_T"] = -1
    with pytest.raises(ValueError):
        validate_frame(frame, config, region)


def test_comsol_wide_units_and_legacy_long(config, tmp_path):
    config.preprocess.length_unit = "mm"
    config.preprocess.thickness_unit = "mm"
    config.preprocess.temperature_unit = "C"
    config.preprocess.columns = {"x": "X", "y": "Y", "thickness": "H"}
    path = tmp_path / "wide.csv"
    original = pd.DataFrame({"% X": [150, 180], "Y": [0, 0], "H": [9, 10],
                             "T (C) @t=0": [20, 21], "T (C) @t=5": [30, 31]})
    original.to_csv(path, index=False)
    snapshot = path.read_bytes()
    result = adapt_comsol(path, config, "upper")
    assert len(result) == 4
    np.testing.assert_allclose(result.r, [0.15, 0.18, 0.15, 0.18])
    np.testing.assert_allclose(result.thickness, [0.009, 0.010, 0.009, 0.010])
    np.testing.assert_allclose(result.True_T, [293.15, 294.15, 303.15, 304.15])
    assert path.read_bytes() == snapshot
    config.preprocess.columns = {}
    config.preprocess.length_unit = config.preprocess.thickness_unit = "m"
    config.preprocess.temperature_unit = "K"
    long = tmp_path / "long.csv"
    result.to_csv(long, index=False)
    pd.testing.assert_frame_equal(adapt_comsol(long, config, "upper"), result)


def test_preprocess_preserves_sources(config, tmp_path):
    for region, frame in demo_frames(config).items():
        source = tmp_path / f"{region}-raw.csv"
        frame.to_csv(source, index=False)
        setattr(config.preprocess, f"{region}_input", source.name)
        setattr(config.data, f"{region}_csv", f"processed/{region}.csv")
    counts = preprocess(config)
    assert all(value > 0 for value in counts.values())
    with pytest.raises(FileExistsError):
        preprocess(config)
    config.data.upper_csv = config.preprocess.upper_input
    with pytest.raises(ValueError, match="source"):
        preprocess(config, overwrite=True)


def test_configuration_paths_and_unknown_settings(tmp_path):
    path = tmp_path / "configs" / "small.toml"
    path.parent.mkdir()
    path.write_text('[data]\nupper_csv="../data/upper.csv"\n', encoding="utf-8")
    config = load_config(path)
    assert config.path(config.data.upper_csv) == tmp_path / "data" / "upper.csv"
    with pytest.raises(ValueError, match="Unknown"):
        from_dict({"training": {"warmup_stpes": 12}}, tmp_path)
    with pytest.raises(ValueError, match="integers"):
        from_dict({"contact": {"arc_nodes": 3.5}}, tmp_path)


def test_rib_input_order(config):
    geometry = RotorGeometry(config.geometry)
    frame = demo_frames(config)["rib"].iloc[-3:]
    inputs = normalized_inputs(frame, "rib", geometry, 1200)
    np.testing.assert_array_equal(inputs[:, 1], frame.rib_id)
    np.testing.assert_allclose(inputs[:, 2], frame.t / 1200)
    np.testing.assert_allclose(inputs[:, 0], geometry.r_to_s(frame.r) / config.geometry.outer_radius)
