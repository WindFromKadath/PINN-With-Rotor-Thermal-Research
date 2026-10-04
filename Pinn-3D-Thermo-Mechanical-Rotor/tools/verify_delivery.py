"""Verify portable CLI/model replay, duplicate protection and seed reproducibility."""
from pathlib import Path
import tempfile
import subprocess
import shutil
import sys
import csv
import json
import numpy as np
import pandas as pd
from thermo_rotor.training.run import Run,write_json,sha256
from thermo_rotor.cli.main import replay


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=dict(runtime=dict(seed=42,device='cpu'),paths={})
    with Run(root/'artifacts','R107',cfg,'delivery-acceptance') as run:
        report={}
        loss_difference={}
        for label in ('thermal','mechanical'):
            a=json.loads((root/f'artifacts/R102/{label}_losses.json').read_text())
            b=json.loads((root/f'artifacts/R108/{label}_losses.json').read_text())
            diff=max(abs(x['total']-y['total']) for x,y in zip(a,b))
            assert len(a)==len(b) and diff==0
            loss_difference[label]=diff
        report['seed_42_reproducibility']=dict(runs=['R102','R108'],loss_max_difference=loss_difference,
                prediction_csvs_identical=all(sha256(root/f'artifacts/R102/{label}_predictions.csv')==sha256(root/f'artifacts/R108/{label}_predictions.csv') for label in ('thermal','mechanical')))
        copied=run.path/'portable_coupled'
        shutil.copytree(root/'artifacts/R102',copied)
        config=json.loads((copied/'config.json').read_text())
        for key in config['paths']:
            config['paths'][key]='unavailable-original-path'
        write_json(copied/'config.json',config)
        report['portable_replay']=replay(copied)
        with tempfile.TemporaryDirectory() as other_cwd:
            commands=[['evaluate','--run-dir',str(copied)],['plot','--run-dir',str(copied)],
                    ['predict','--run-dir',str(copied),'--model','thermal','--input',str(copied/'thermal_inputs.csv'),'--output',str(run.path/'portable_prediction.csv')],
                    ['compare','--run-dirs',str(copied),str(root/'artifacts/R108'),'--output',str(run.path/'comparison.json')],
                    ['sample','--config',str(root/'configs/demos/coupled.toml'),'--run-id','R119']]
            results={}
            for command in commands:
                result=subprocess.run([sys.executable,'-m','thermo_rotor.cli.main',*command],cwd=other_cwd,capture_output=True,text=True,encoding='utf-8')
                (run.path/f"cli_{command[0]}.log").write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
                assert result.returncode==0,(command,result.stderr)
                results[command[0]]='passed'
            marker=sha256(root/'artifacts/R102/status.json')
            duplicate=subprocess.run([sys.executable,'-m','thermo_rotor.cli.main','demo','--config',str(root/'configs/demos/coupled.toml'),'--run-id','R102'],cwd=other_cwd,capture_output=True,text=True)
            assert duplicate.returncode!=0 and sha256(root/'artifacts/R102/status.json')==marker
            results['duplicate_run_id']='rejected_original_unchanged'
            report['entrypoint_checks']=results
        with (root/'local_reference_versions/source_manifest.csv').open(encoding='utf-8',newline='') as f:
            rows=list(csv.DictReader(f))
        mismatches=[r['copied_to'] for r in rows if r['copied_to'] and sha256(root/r['copied_to'])!=r['sha256']]
        assert not mismatches
        report['migration']=dict(indexed_source_files=len(rows),copied_files=sum(bool(r['copied_to']) for r in rows),hash_mismatches=mismatches)
        report['short_workflows']={r:json.loads((root/f'artifacts/{r}/status.json').read_text())['status'] for r in ('R102','R103','R104','R105','R106')}
        assert all(v=='completed' for v in report['short_workflows'].values())
        legacy={}
        for i,bundle in enumerate(json.loads((root/'local_reference_versions/index.json').read_text()),111):
            result=subprocess.run([sys.executable,'-m','thermo_rotor.cli.main','legacy-evaluate','--bundle',bundle['bundle_id'],'--run-id',f'R{i}'],capture_output=True,text=True,encoding='utf-8')
            (run.path/f"legacy_{bundle['bundle_id']}.log").write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
            assert result.returncode==0,result.stderr
            legacy[bundle['bundle_id']]=f'R{i}'
        report['legacy_replays']=legacy
        report['status']='passed'
        write_json(run.path/'verification.json',report)
        print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
