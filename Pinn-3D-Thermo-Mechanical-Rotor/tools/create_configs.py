"""Generate checked reference profiles and editable TOML runtime configurations."""
from pathlib import Path
import json
from thermo_rotor.config import Profile


def literal(v):
    if isinstance(v, str):
        return json.dumps(v, ensure_ascii=False)
    if isinstance(v, bool):
        return str(v).lower()
    if isinstance(v, list):
        return "["+", ".join(literal(x) for x in v)+"]"
    return str(v)


def write(path, data):
    rows = []
    def emit(section, items):
        if section:
            rows.append("["+section+"]")
        for k, v in items.items():
            if not isinstance(v, dict):
                rows.append(f"{k} = {literal(v)}")
        rows.append("")
        for k, v in items.items():
            if isinstance(v, dict):
                emit(f"{section}.{k}" if section else k, v)
    emit("", data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(rows), encoding="utf-8")


def main():
    root = Path(__file__).resolve().parents[1]
    base = Profile().to_dict()
    profiles = {"little_transient": base, "fxx_transient": {**base, "name": "fxx_transient"},
                "formal_thermal": {**base, "name": "formal_thermal", "dT": 100., "tau": 100., "q": 50000., "h": 50.},
                "xpinn_steady": {**base, "name": "xpinn_steady", "dT": 100., "tau": 100., "q": 50000., "h": 50.},
                "iwg": {**base, "name": "iwg", "dT": 106.85, "lam": .29/(1.29*.42), "mu": 1/2.58, "beta": 1/.42}}
    for name, p in profiles.items():
        write(root/f"configs/profiles/{name}.toml", dict(physics=p))
    import copy
    for mode in ("coupled", "thermal", "mechanical", "mlp", "iwg", "xpinn", "formal_thermal"):
        p = "little_transient" if mode not in ("iwg", "xpinn", "formal_thermal") else dict(iwg="iwg", xpinn="xpinn_steady", formal_thermal="formal_thermal")[mode]
        cfg = dict(mode="thermal" if mode == "formal_thermal" else mode, profile=f"../profiles/{p}.toml",
                   runtime=dict(seed=42, device="cpu", threads=2, dtype="float32"),
                   paths=dict(artifacts="../../artifacts", stl="../../data/raw/STL/main",
                              thermal_eval="../../data/raw/HeatPINNCheck_long.csv",
                              mechanical_train="../../data/raw/ElsPINNTrain.csv",
                              mechanical_eval="../../data/raw/ElsPINNCheck_long.csv",
                              iwg_constraint="../../data/raw/iwg/constraintCheck.csv",
                              iwg_projection="../../data/raw/iwg/projectionCheck.csv"),
                   data_units={"thermal_eval": dict(length_unit="m", displacement_unit="m", temperature_unit="K", time_unit="s"),
                               "mechanical_train": dict(length_unit="mm", displacement_unit="mm", time_unit="s"),
                               "mechanical_eval": dict(length_unit="mm", displacement_unit="mm", time_unit="s"),
                               "iwg_constraint": dict(length_unit="mm", displacement_unit="mm", time_unit="s"),
                               "iwg_projection": dict(length_unit="mm", displacement_unit="mm", time_unit="s")},
                   sampling=dict(domain=128, boundary=64, initial=64, times=[0.,100.,200.], anchors_per_time=32, evaluation_per_time=128),
                   network=dict(thermal=[64]*5, mechanical=[128]*5, iwg=[64]*4, steady=[64]*3, fourier_dim=64),
                   training=dict(adam_steps=40, thermal_steps=40, mechanical_steps=40, stage_steps=20, lbfgs_steps=0,
                                 lbfgs_max_iter=5, lr=.001, w_pde=1., w_traction=1., w_data=100., label_counts=[32,96]),
                   evaluation=dict(scope="short workflow on bounded COMSOL samples; no accuracy or convergence claim"))
        if mode == "mechanical":
            cfg["paths"]["thermal_checkpoint"] = "../../artifacts/R102/thermal.pt"
        if mode == "formal_thermal":
            cfg["paths"]["stl"] = "../../data/raw/STL/formal"
            cfg["paths"].pop("thermal_eval")
            cfg["sampling"]["times"] = [0.,50.,100.]
            cfg["evaluation"]["scope"] = "diagnostic T0 reference only, no matching formal COMSOL field or accuracy claim"
        if mode == "xpinn":
            cfg["paths"]["stl"] = "../../data/raw/STL/xpinn"
            cfg["evaluation"]["scope"] = "steady three-domain workflow; diagnostic T0 reference, no matching COMSOL field"
        write(root/f"configs/demos/{'xpinn_steady' if mode == 'xpinn' else mode}.toml", cfg)
        full = copy.deepcopy(cfg)
        full["profile"] = f"profiles/{p}.toml"
        full["paths"] = {k: v.removeprefix("../") for k, v in cfg["paths"].items()}
        full["sampling"].update(domain=10000, boundary=2000, initial=2000, times=[float(i) for i in range(0,201,20)], anchors_per_time=100)
        full["training"].update(thermal_steps=5000, mechanical_steps=10000, adam_steps=10000, stage_steps=2000,
                                lbfgs_steps=1000, thermal_lbfgs_steps=500, mechanical_lbfgs_steps=1000, label_counts=[1100,33000])
        if mode in ("xpinn", "formal_thermal"):
            full["sampling"]["times"] = [0.,50.,100.]
        if mode == "xpinn":
            full["training"].update(stage_steps=3000, lbfgs_steps=100)
        if mode == "iwg":
            full["training"]["lbfgs_steps"] = 100
            full["training"]["full_stage_adam_steps"] = 0
        write(root/f"configs/{'xpinn_steady' if mode == 'xpinn' else mode}.toml", full)
    write(root/"configs/preprocess.toml", dict(profile="profiles/little_transient.toml", mode="preprocess",
          paths=dict(artifacts="../artifacts", input="../data/raw/heat_wide_smoke.csv", output="../data/processed/heat_smoke_long.csv"),
          units=dict(length_unit="mm", displacement_unit="mm", temperature_unit="K", time_unit="s")))
    write(root/"configs/preprocess_full.toml", dict(profile="profiles/little_transient.toml", mode="preprocess",
          paths=dict(artifacts="../artifacts", input="../data/raw/HeatPINNCheck.csv", output="../data/processed/heat_full_long.csv"),
          units=dict(length_unit="mm", displacement_unit="mm", temperature_unit="K", time_unit="s")))


if __name__ == "__main__":
    main()
