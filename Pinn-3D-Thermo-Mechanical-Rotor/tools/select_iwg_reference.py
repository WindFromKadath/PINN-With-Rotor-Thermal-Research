"""Compare matching IWG candidates; retain the better constraint reference locally."""
from pathlib import Path
import argparse
import csv
import json
import shutil
import torch
import numpy as np
from curate_models import definitions
from thermo_rotor.config import Profile
from thermo_rotor.models.checkpoint import legacy_state, save
from thermo_rotor.data.comsol import sample_times
from thermo_rotor.data.transforms import inputs
from thermo_rotor.evaluation.predict import determine_gauge, predict
from thermo_rotor.evaluation.metrics import metrics
from thermo_rotor.training.run import Run, write_json, sha256


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-root',type=Path,required=True)
    args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    archive=root/'local_reference_versions'
    source=args.source_root/'FXXKINGGOD/PINN_Pls_IWG'
    p=Profile(name='iwg_fxx',dT=106.85,lam=.29/(1.29*.42),mu=1/2.58,beta=1/.42)
    spec=dict(kind='iwg',hidden=[64]*4,fourier_dim=64)
    cfg=dict(runtime=dict(seed=42,device='cpu'),paths={})
    torch.set_num_threads(2)
    with Run(root/'artifacts','R120',cfg,'iwg-candidate-selection') as run:
        frame=sample_times(source/'data/constraintCheck.csv',512,[200.],seed=99999,length_unit='mm')
        chosen=archive/'packages/iwg_fxx_constraint'
        chosen.mkdir(exist_ok=False)
        original=source/'models/best_pinn_constraint.pth'
        shutil.copy2(original,chosen/'mechanical.pth')
        model,_=legacy_state(chosen/'mechanical.pth',spec)
        x=inputs(frame,p,'iwg')
        anchor=[.084/.225,float(frame.z.min()/.225)]
        gauge=determine_gauge(model,x,'constraint',anchor)
        oldclass=definitions(archive/'source/FXXKINGGOD/PINN_Pls_IWG/src/models/stl_ring_model.py',['FourierFeatureLayer','ProjectionNet'])['ProjectionNet']
        old=oldclass()
        old.set_constraint(*anchor)
        old.load_state_dict(model.state_dict(),strict=True)
        xx=torch.tensor(x,dtype=torch.float32)
        with torch.no_grad():
            old_output=torch.cat(old(xx[:,0:1],xx[:,1:2]),dim=1).numpy()*p.U
        output=predict(model,frame,p,spec,gauge)
        difference=float(np.abs(output-old_output).max())
        if difference>1e-9:
            raise ValueError('IWG source prediction mismatch')
        save(chosen/'mechanical.pt',model,spec,p,gauge=gauge,
             source=dict(original_weight_sha256=sha256(original),thermal_bridge='fxx/thermal.pt',thermal_time_s=200.,
                         reason='original explicit constraint evaluator; bounded candidate comparison'))
        frame.to_csv(chosen/'mechanical_examples.csv',index=False,float_format='%.17g')
        data=root/'data/raw/iwg_fxx'
        data.mkdir(exist_ok=True)
        shutil.copy2(source/'data/constraintCheck.csv',data/'constraintCheck.csv')
        new_metrics=metrics(output,frame[['u_r','u_z']].to_numpy(),'iwg')
        little=json.loads((root/'artifacts/R101/metrics.json').read_text())['iwg_constraint']['mechanical']['diagnostic_metrics']
        comparison=dict(recommended='iwg_fxx_constraint',fxx_constraint=new_metrics,little_original_evaluator=little,
                        source_max_difference_m=difference,
                        scope='matching diagnostic protocol with each source evaluator reference frame; no full accuracy claim',
                        reason='Little evaluator omits the trained constraint anchor; FXX explicitly restores its anchor')
        write_json(run.path/'metrics.json',comparison)
        index=json.loads((archive/'index.json').read_text())
        for row in index:
            row['recommended']=row['bundle_id'] in ('little','mlp_33k')
            if row['bundle_id']=='fxx_ntk':
                row['selection']='diagnostic NTK contrast only; not a preferred version'
        index.append(dict(bundle_id='iwg_fxx_constraint',path='packages/iwg_fxx_constraint',recommended=True,
                          selection='preferred constraint candidate by bounded diagnostics',reference_frame='saved original constraint anchor',
                          thermal_bridge='fxx/thermal.pt at physical t=200s'))
        write_json(archive/'index.json',index)
        write_json(archive/'selection.json',comparison)
        manifest=archive/'source_manifest.csv'
        with manifest.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f)
            fields=reader.fieldnames
            rows=list(reader)
        updates={'FXXKINGGOD/PINN_Pls_IWG/models/best_pinn_constraint.pth':chosen/'mechanical.pth',
                 'FXXKINGGOD/PINN_Pls_IWG/data/constraintCheck.csv':data/'constraintCheck.csv'}
        for row in rows:
            if row['source'] in updates:
                path=updates[row['source']]
                if sha256(path)!=row['sha256']:
                    raise ValueError('Additional migration hash mismatch')
                row['copied_to']=path.relative_to(root).as_posix()
        with manifest.open('w',encoding='utf-8',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        write_json(archive/'migration.json',dict(source_files=len(rows),copied_files=sum(bool(r['copied_to']) for r in rows),hash_mismatches=[]))


if __name__=='__main__':
    main()
