"""Copy raw data and select usable historical packages; originals are read only.

Selection diagnostics are bounded checks on fixed samples, not new full-budget
accuracy reproduction or claims about the global best historical experiment.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import torch
from neuralop.layers.spectral_convolution import SpectralConv
from heatops.models.fno import HeatOperatorFNO, HeatOperatorFNO3D
from heatops.models.deepxde_models import make_deeponet_cartesian, make_fnn, PeriodicFeatureTransform
from heatops.problems import AnnulusParams
from heatops.solvers import solve_annulus_transient_cn
from heatops.training.run_context import file_hash, write_json


def load_raw(path):
    known = [torch._C._nn.gelu, SpectralConv, np.dtype, type(np.dtype("float32")),
             type(np.dtype("float64")), np._core.multiarray.scalar, np._core.multiarray._reconstruct, np.ndarray]
    with torch.serialization.safe_globals(known):
        return torch.load(path, map_location="cpu", weights_only=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="完整外层原目录；只读")
    args = parser.parse_args()
    source = Path(args.source).resolve()
    root = Path(__file__).resolve().parents[1]
    local = root / "local_reference_versions"
    local.mkdir(exist_ok=False)
    cfg = json.loads((root/"configs/legacy_selection.json").read_text(encoding="utf-8"))
    p = AnnulusParams(**cfg["physics"])
    copies = []

    def copy(rel, dest):
        src = source / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src,dest)
        digest = file_hash(src)
        if file_hash(dest) != digest:
            raise RuntimeError("迁移哈希不一致")
        copies.append({"source":str(rel).replace("\\","/"), "destination":str(dest.relative_to(root)).replace("\\","/"), "bytes":src.stat().st_size,"sha256":digest})

    print("copy raw CSVs",flush=True)
    for path in sorted((source/"PythonAgnetCoding/pino_dataset").glob("*.csv")):
        copy(path.relative_to(source),root/"data/raw/comsol_annulus"/path.name)
    for name in ("README.md","AGENTS.md","hypotheses.csv","experiments.csv"):
        copy(Path("neural-operator-exploration")/name,local/"archive_index"/name)

    def package(name,family):
        rd=local/name; rd.mkdir()
        spec={"format_version":1,"family":family,"models":{},"selection_scope":"fixed diagnostic points/samples; historical split unknown; not global-optimum or generalization proof"}
        return rd,spec

    def finish(rd,spec,winners,source_files):
        for label,path,net,network,score in winners:
            copy(path.relative_to(source),rd/"original_weights"/label/path.name)
            dest=rd/f"{label}.pt"
            torch.save({"format_version":1,"family":spec["family"],"model_state_dict":{k:v for k,v in net.state_dict().items() if k!="_metadata"}},dest)
            spec["models"][label]={"checkpoint":dest.name,"sha256":file_hash(dest),"source_checkpoint":str(path.relative_to(source)).replace("\\","/"),"network":network,"selection_score":score}
        for rel in source_files:
            copy(rel,rd/"source"/Path(rel).name)
        write_json(rd/"model_spec.json",spec)
        print(rd.name,"selected",[x.name for _,x,_,_,_ in winners],flush=True)

    rd,spec=package("Fno-1D-Comparison","fno_1d")
    base=source/"PythonAgnetCoding/PINO/DataPINO"
    data=load_raw(base/"Dataset/heat_data.pt")
    copy(Path("PythonAgnetCoding/PINO/DataPINO/Dataset/heat_data.pt"),rd/"data/heat_data.pt")
    truth=data["u_true"][:4].squeeze(-1).numpy()
    nx,nt=truth.shape[1:]
    grid=torch.stack(torch.meshgrid(torch.linspace(0,1,nx),torch.linspace(0,0.5,nt),indexing="ij"),-1)[None].expand(4,-1,-1,-1)
    winners=[]; diagnostics=[]
    for label,folder in (("data","results_fno"),("physics","results_true_pino"),("hybrid","results_pino")):
        choices=[]
        for path in sorted((base/folder).glob("*.pth")):
            net=HeatOperatorFNO(tuple(cfg["fno_1d"]["n_modes"]),cfg["fno_1d"]["hidden_channels"],a=cfg["fno_1d"]["a"])
            state=load_raw(path)
            net.load_state_dict({k:v for k,v in state.items() if k!="_metadata"},strict=True)
            with torch.no_grad(): pred=net(data["u0"][:4],grid).squeeze(-1).numpy()
            score=float(np.linalg.norm(pred-truth)/np.linalg.norm(truth))
            if np.isfinite(score): choices.append((score,path,net))
            diagnostics.append({"model":label,"source":str(path.relative_to(source)).replace("\\","/"),"score":score})
        score,path,net=min(choices,key=lambda x:x[0])
        winners.append((label,path,net,cfg["fno_1d"],score))
    spec.update(transforms={"coordinates":"x in [0,1], t in [0,0.5] seconds","output":"Softplus(u0*exp(-a*t)+t*FNO); historical empirical ansatz"},candidate_scores=diagnostics)
    finish(rd,spec,winners,[Path("PythonAgnetCoding/PINO/DataPINO")/name for name in ("datafno_train.py","datapino_train.py","datagenerator.py","final_compare.py","triple_model_comparison.png")])

    rd,spec=package("Pino-Annulus-Polar","fno_polar")
    base=source/"PythonAgnetCoding/PINO/DataPINO_2D_Annulus"
    ds=load_raw(base/"pino_annulus_4d_polar_64based.pt")
    copy(Path("PythonAgnetCoding/PINO/DataPINO_2D_Annulus/pino_annulus_4d_polar_64based.pt"),rd/"data/polar_legacy.pt")
    path=base/"pino_annulus_polar_optimized.pth"
    net=HeatOperatorFNO3D(tuple(cfg["fno_polar"]["n_modes"]),cfg["fno_polar"]["hidden_channels"])
    net.load_state_dict({k:v for k,v in load_raw(path).items() if k!="_metadata"},strict=True)
    # A small grid confirms execution. Original full grids/data remain in the bundle.
    grid=torch.stack(torch.meshgrid(torch.linspace(0,1,8),torch.linspace(0,1,24),torch.linspace(0,1,24),indexing="ij"),-1)[None]
    with torch.no_grad(): pred=net(torch.ones(1,8,24,24)*0.5,grid)
    if not bool(torch.isfinite(pred).all()): raise FloatingPointError("历史极坐标PINO预测不有限")
    spec.update(transforms={"coordinates":"all three old coordinate channels in [0,1]; not the new dimensionless-radius convention","temperature_range":[float(v) for v in ds["info"]["u_range"]]},selection_note="Only checkpoint with matching polar trainer/input contract selected; Cartesian-trained weights excluded.")
    finish(rd,spec,[("polar",path,net,cfg["fno_polar"],None)],[Path("PythonAgnetCoding/PINO/DataPINO_2D_Annulus")/name for name in ("datapino_2d_polar_train.py","cartesian_2_polar_dataset.py","pino_polar_final_result.png")])

    def reference(qstar):
        pp=AnnulusParams(**{**cfg["physics"],"q_in":p.q_in*qstar})
        out=solve_annulus_transient_cn(pp,nr=32,dt=10.,nt=301,save_every=100)
        times=np.asarray(cfg["comparison_times"])
        fields=np.stack([out["T_history"][np.argmin(np.abs(out["t"]-t))] for t in times])
        return out["r"],times,fields

    def polar_score(net,family,periodic=False):
        truths,predictions=[],[]
        for q in ((1.,3.) if family=="pidon" else (1.,)):
            r,times,truth=reference(q)
            for angle in cfg["comparison_angles"]:
                values=[]
                for t in times:
                    pts=torch.tensor(np.stack([r/p.R_out,np.full_like(r,angle),np.full_like(r,t/p.t_scale)],1),dtype=torch.float32)
                    with torch.no_grad():
                        u=net((torch.full((1,200),q),pts)).numpy().ravel() if family=="pidon" else net(pts).numpy().ravel()
                    values.append(p.T_inf+p.T_scale*u)
                truths.append(truth); predictions.append(values)
        return float(np.linalg.norm(np.asarray(predictions)-truths)/np.linalg.norm(truths))

    for name,family,folder,pattern,source_files in (
        ("Pidon-Annulus","pidon","PythonAgnetCoding/PIDON/Heat_2D_annulus","pidon_heat_polar_optimized_open-*.pt",["heat_2d_annulus_pidon_polar.py"]),
        ("Pinn-Annulus-Transient","pinn_polar","PythonAgnetCoding/PINN/Heat_2D_Annulus_Transient","pinn_heat_polar*.pt",["heat_2d_polar_transient.py","heat_2d_polar_transient_2.py"]),
    ):
        rd,spec=package(name,family); base=source/folder; choices=[]; diagnostics=[]
        for path in sorted((base/"model").glob(pattern)):
            if family=="pinn_polar" and not ("optimized_open" in path.name or "polar_2" in path.name): continue
            periodic="polar_2" in path.name
            network=cfg["pidon"] if family=="pidon" else {"layer_sizes":[4 if periodic else 3,64,64,64,64,1],"periodic_features":periodic}
            net=make_deeponet_cartesian(network["branch_layers"],network["trunk_layers"]) if family=="pidon" else make_fnn(network["layer_sizes"],feature_transform=PeriodicFeatureTransform(1) if periodic else None)
            net.load_state_dict(load_raw(path)["model_state_dict"],strict=True)
            score=polar_score(net,family,periodic)
            diagnostics.append({"source":str(path.relative_to(source)).replace("\\","/"),"score":score})
            if np.isfinite(score): choices.append((score,path,net,network))
        score,path,net,network=min(choices,key=lambda x:x[0])
        spec.update(transforms={"r_min_star":p.R_in_star,"t_final_star":3000/p.t_scale,"T_inf":p.T_inf,"T_scale":p.T_scale},candidate_scores=diagnostics,diagnostic="relative L2 vs C-N on fixed radial profiles (q*=1/3 for PIDON, angles0/pi/2, times0/1000/3000)",legacy_limitations="PIDON historical heat-flux BC misuse retained in source archive; loading does not endorse its training equations" if family=="pidon" else "Short selection diagnostic does not establish full-domain accuracy")
        finish(rd,spec,[("selected",path,net,network,score)],[Path(folder)/f for f in source_files])
    write_json(local/"copied_files.json",copies)
    print("hash complete source inventory",flush=True)
    inventory=[]
    for path in sorted(source.rglob("*")):
        if not path.is_file() or any(part in (".git",".venv","__pycache__",".idea",".ipynb_checkpoints") for part in path.parts): continue
        inventory.append({"source":str(path.relative_to(source)).replace("\\","/"),"bytes":path.stat().st_size,"sha256":file_hash(path)})
    write_json(local/"source_inventory.json",inventory)
    write_json(local/"index.json",{"packages":[p.name for p in local.iterdir() if (p/"model_spec.json").exists()],"copied_file_count":len(copies),"full_source_file_count":len(inventory),"scope":"local archives ignored by Git; no original files changed; selected mature usable versions, bounded best-of-candidates diagnostics"})
    print("Migration complete",len(copies),len(inventory),flush=True)


if __name__=="__main__": main()
