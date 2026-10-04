"""Package selected historical weights with explicit scales and compatibility evidence."""
from pathlib import Path
import ast
import typing
import json
import numpy as np
import torch
from thermo_rotor.config import Profile
from thermo_rotor.models.checkpoint import legacy_state, save
from thermo_rotor.data.comsol import sample_times
from thermo_rotor.data.transforms import inputs
from thermo_rotor.evaluation.predict import predict, determine_gauge
from thermo_rotor.evaluation.metrics import metrics
from thermo_rotor.training.run import write_json, sha256, Run
from thermo_rotor.physics import thermal, mechanical
from thermo_rotor.coupling.temperature import TemperatureBridge


def definitions(path, names):
    """Execute only reviewed class/function definitions, never archived top-level training."""
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    namespace = dict(torch=torch, nn=torch.nn, np=np, **vars(typing))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def main():
    root = Path(__file__).resolve().parents[1]
    archive = root/"local_reference_versions"
    raw = root/"data/raw"
    source = archive/"source/LittleTinyTest"
    p = Profile()
    iwg = Profile(name="iwg", dT=106.85, lam=.29/(1.29*.42), mu=1/2.58, beta=1/.42)
    heat = sample_times(raw/"HeatPINNCheck_long.csv", 128, [0.,100.,200.], seed=99999, length_unit="m")
    mech = sample_times(raw/"ElsPINNCheck_long.csv", 128, [0.,100.,200.], seed=99999, length_unit="mm")
    specT = dict(kind="thermal", hidden=[64]*5, prefix="layers")
    specM = dict(kind="mechanical", hidden=[128]*5, prefix="net")
    specI = dict(kind="iwg", hidden=[64]*4, fourier_dim=64)
    Tclass = definitions(source/"HeatPINN/network.py", ["PINN_Network"])["PINN_Network"]
    Msource = definitions(source/"MechanicalPINN/train_transient.py", ["TransientMechanicalPINN", "compute_transient_pde_loss", "compute_transient_traction_loss"])
    Iclass = definitions(source/"IWG_Comparison/network.py", ["FourierFeatureLayer", "ThermoelasticNet"])["ThermoelasticNet"]
    report, packages = {}, []
    config = dict(runtime=dict(seed=42, device="cpu"), physics=p.to_dict(), paths={})
    torch.set_num_threads(2)
    with Run(root/"artifacts", "R101", config, "historical-model-curation") as run:
        for bundle, entries, profile in (("little", [("thermal", specT, heat), ("mechanical", specM, mech)], p),
                ("fxx", [("thermal", specT, heat), ("mechanical", specM, mech)], Profile(name="fxx_transient")),
                ("fxx_ntk", [("mechanical", specM, mech)], Profile(name="fxx_transient")),
                ("mlp_1100", [("mechanical", {**specM, "kind": "mlp"}, mech)], p),
                ("mlp_33k", [("mechanical", {**specM, "kind": "mlp"}, mech)], p),
                ("iwg_constraint", [("mechanical", specI, None)], iwg),
                ("iwg_projection", [("mechanical", specI, None)], iwg)):
            folder = archive/"packages"/bundle
            bundle_report = {}
            for label, spec, frame in entries:
                old = folder/f"{label}.pth"
                model, _ = legacy_state(old, spec)
                if spec["kind"] == "iwg":
                    mode = bundle.removeprefix("iwg_")
                    frame = sample_times(raw/f"iwg/{mode}Check.csv", 512, [200.], seed=99999, length_unit="mm")
                    # Source evaluator never sets a constraint point. Retain its raw constraint output.
                    gauge = determine_gauge(model, inputs(frame, profile, "iwg"), "projection" if mode == "projection" else "raw")
                    old_model = Iclass(mode=mode)
                else:
                    gauge = {"mode": "raw"}
                    old_model = Tclass(input_dim=5, hidden_layers=[64]*5) if label == "thermal" else Msource["TransientMechanicalPINN"]()
                old_model.load_state_dict(model.state_dict(), strict=True)
                x = torch.tensor(inputs(frame, profile, spec["kind"]), dtype=torch.float32)
                with torch.no_grad():
                    if label == "thermal":
                        reference = old_model(x).numpy()*profile.dT+profile.T0
                    elif spec["kind"] == "iwg":
                        y = old_model(x[:, 0:1], x[:, 1:2])
                        reference = torch.cat(y, dim=1).numpy()*profile.U
                    else:
                        reference = old_model(x[:, 0:1], x[:, 1:2], x[:, 2:3]).numpy()*profile.U
                new = predict(model, frame, profile, spec, gauge, batch_size=len(frame))
                difference = float(np.abs(reference-new).max())
                tolerance = 1e-3 if label == "thermal" else 1e-9
                if difference > tolerance:
                    raise ValueError(f"Source prediction mismatch: {bundle}/{label}: {difference}")
                save(folder/f"{label}.pt", model, spec, profile, gauge=gauge,
                     source=dict(original_weight_sha256=sha256(old), bundle=bundle, archived_code=True))
                frame.to_csv(folder/f"{label}_examples.csv", index=False, float_format="%.17g")
                truth = frame[["T"]].to_numpy() if label == "thermal" else frame[["u_r", "u_z"]].to_numpy()
                bundle_report[label] = dict(finite=True, source_max_abs_difference=difference,
                            diagnostic_metrics=metrics(new, truth, spec["kind"]), weights_sha256=sha256(old))
                if label == "thermal":
                    test = x[:24].clone().requires_grad_(True)
                    expected = old_model.compute_pde_residual(test.clone(), profile.Fo)
                    actual = thermal.residual(model, test.clone(), profile.Fo)
                    if not torch.allclose(actual, expected, rtol=1e-4, atol=1e-5):
                        raise ValueError("Thermal source PDE mismatch")
                    bundle_report[label]["source_pde_max_difference"] = float((actual-expected).abs().max().detach())
            report[bundle] = bundle_report
            packages.append(dict(bundle_id=bundle, path=f"packages/{bundle}", selection="matched source models; bounded COMSOL diagnostic",
                                 reference_frame="original evaluation raw" if bundle == "iwg_constraint" else "saved global shift" if bundle == "iwg_projection" else "raw",
                                 thermal_bridge="little/thermal.pt" if bundle.startswith("mlp") else "fxx/thermal.pt" if bundle == "fxx_ntk" else "thermal.pt" if bundle in ("little", "fxx") else "linear 400K to 300K"))
        # Same coefficients and input points; compare the full mechanical source loss.
        mm, _ = legacy_state(archive/"packages/little/mechanical.pth", specM)
        tm, _ = legacy_state(archive/"packages/little/thermal.pth", specT)
        bridge = TemperatureBridge(tm, p, p)
        x = torch.tensor(inputs(mech.iloc[:24], p, "mechanical"), dtype=torch.float32)
        reference_bridge = lambda r,z,t: bridge(torch.cat((r,z,t), dim=1))
        expected = Msource["compute_transient_pde_loss"](mm,x[:,0:1],x[:,1:2],x[:,2:3],reference_bridge,p.lam,p.mu,p.beta)
        actual = mechanical.equilibrium(mm,x.clone(),bridge,p).square().sum(dim=1).mean()
        if not torch.allclose(actual,expected,rtol=1e-4,atol=1e-5):
            raise ValueError("Mechanical source PDE mismatch")
        report["mechanical_residual"] = dict(source_loss=float(expected.detach()), rewritten_loss=float(actual.detach()))
        write_json(run.path/"metrics.json", report)
        write_json(archive/"index.json", packages)
        write_json(run.path/"verification.json", dict(status="completed", model_count=9, all_finite=True,
                    source_prediction_checks=True, source_pde_checks=True, packages=packages,
                    limitations=["bounded diagnostic sample; historical training/test separation uncertain",
                    "IWG constraint evaluation preserves original evaluator's unset constraint anchor",
                    "projection gauge saved globally on the fixed diagnostic points", "not a global-best or full-accuracy claim"]))


if __name__ == "__main__":
    main()
