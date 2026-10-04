from pathlib import Path
import json
from thermo_rotor.config import load_config
from thermo_rotor.training.run import Run,write_json
from thermo_rotor.training.workflows import train


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=load_config(root/'configs/demos/mlp.toml')
    cfg['paths']['mechanical_train']='missing-diagnostic-input.csv'
    try:
        with Run(root/'artifacts','R123',cfg,'intentional-missing-data-check') as run:
            train(cfg,run,'mlp')
    except FileNotFoundError as e:
        assert json.loads((run.path/'status.json').read_text())['status']=='failed'
        write_json(run.path/'verification.json',dict(expected_failure=True,clear_error=type(e).__name__,failure_preserved=True))
        return
    raise AssertionError('Expected missing-data failure was not raised')


if __name__=='__main__':
    main()
