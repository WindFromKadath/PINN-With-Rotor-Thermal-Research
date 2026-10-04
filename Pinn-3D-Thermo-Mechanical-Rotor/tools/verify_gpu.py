"""Small CUDA loading/derivative/backward/reload check; no GPU training claim."""
from pathlib import Path
import torch
import json
from thermo_rotor.models.checkpoint import load,save
from thermo_rotor.training.run import Run,write_json
from thermo_rotor.physics.thermal import residual
from thermo_rotor.physics.mechanical import equilibrium
from thermo_rotor.coupling.temperature import TemperatureBridge
from thermo_rotor.data.transforms import inputs
from thermo_rotor.evaluation.predict import predict
import pandas as pd


def main():
    root=Path(__file__).resolve().parents[1]
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no GPU success recorded')
    torch.set_num_threads(2)
    cfg=dict(runtime=dict(seed=42,device='cuda'),paths={})
    with Run(root/'artifacts','R121',cfg,'cuda-implementation-check') as run:
        report={}
        for bundle in json.loads((root/'local_reference_versions/index.json').read_text()):
            folder=root/'local_reference_versions'/bundle['path']
            for ck in folder.glob('*.pt'):
                model,p,meta=load(ck,'cuda')
                f=pd.read_csv(folder/f'{ck.stem}_examples.csv').iloc[:16]
                x=torch.tensor(inputs(f,p,meta['spec']['kind']),dtype=torch.float32,device='cuda').requires_grad_()
                out=model(x)
                grad=torch.autograd.grad(out.sum(),x,create_graph=True)[0]
                assert torch.isfinite(out).all() and torch.isfinite(grad).all()
                model.zero_grad(set_to_none=True)
                loss=out.square().mean()
                if meta['spec']['kind']=='thermal':
                    loss=loss+residual(model,x,p.Fo).square().mean()
                loss.backward()
                assert all(v.grad is None or torch.isfinite(v.grad).all() for v in model.parameters())
                target=run.path/f"{bundle['bundle_id']}_{ck.stem}.pt"
                save(target,model,meta['spec'],p,gauge=meta['gauge'],source=meta['source'])
                again,pp,mm=load(target,'cuda')
                delta=float(abs(predict(model,f,p,meta['spec'],meta['gauge'])-predict(again,f,pp,mm['spec'],mm['gauge'])).max())
                assert delta==0
                report[f"{bundle['bundle_id']}/{ck.stem}"]=dict(finite_forward=True,finite_input_derivative=True,finite_backward=True,reload_difference=delta)
        tm,p,_=load(root/'artifacts/R102/thermal.pt','cuda')
        mm,_,_=load(root/'artifacts/R102/mechanical.pt','cuda')
        bridge=TemperatureBridge(tm,p,p)
        x=torch.tensor([[.5,-.08,.5],[.7,-.05,1.]],device='cuda').requires_grad_()
        loss=equilibrium(mm,x,bridge,p).square().mean()
        loss.backward()
        assert torch.isfinite(loss) and all(v.grad is None for v in tm.parameters())
        report['frozen_temperature_bridge']=dict(finite_loss=float(loss.detach()),thermal_parameter_gradients_none=True)
        write_json(run.path/'verification.json',dict(status='passed',device=torch.cuda.get_device_name(0),torch=torch.__version__,
                    cuda_runtime=torch.version.cuda,models=report,scope='small derivatives/backward/reload; no full GPU training'))


if __name__=='__main__':
    main()
