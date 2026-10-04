"""Regression checks for the defects found during the migration audit."""
import json
import numpy as np
import pytest
import torch
from heatops.physics.finite_difference import first_derivative, second_derivative, residual_1d, polar_derivatives
from heatops.training.run_context import managed_cli, prepare_run, run_directory
from heatops.data.comsol import assemble_cartesian_4d, save_polar_dataset, load_polar_dataset, cartesian_to_polar


def test_field_derivatives_nonuniform_and_backprop():
    x = torch.tensor([0., .1, .3, .6, 1.], dtype=torch.float64)
    y = (x**2).clone().requires_grad_(True)
    assert torch.allclose(first_derivative(y,x,0),2*x,atol=1e-12)
    assert torch.allclose(second_derivative(y,x,0),torch.full_like(x,2.),atol=1e-12)
    second_derivative(y,x,0).square().mean().backward()
    assert torch.isfinite(y.grad).all() and y.grad.abs().sum()>0


def test_fno_field_residual_converges_on_heat_solution():
    errors=[]
    for n in (17,33,65):
        x,t=torch.linspace(0,1,n,dtype=torch.float64),torch.linspace(0,.5,n,dtype=torch.float64)
        grid=torch.stack(torch.meshgrid(x,t,indexing="ij"),-1)[None]
        u=(torch.sin(torch.pi*grid[...,0])*torch.exp(-.4*torch.pi**2*grid[...,1]))[...,None]
        errors.append(float(residual_1d(u,grid,.4)[:,1:-1,1:-1].abs().max()))
    assert errors[2]<errors[1]/3 and errors[1]<errors[0]/3


def test_nonlocal_model_loss_uses_output_field():
    from heatops.physics.losses import compute_loss_terms_1d
    x,t=torch.linspace(0,1,7),torch.linspace(0,.5,7)
    grid=torch.stack(torch.meshgrid(x,t,indexing="ij"),-1)[None]
    def coupled_model(u0,g):
        # Each output depends on every x channel, yet the sampled field is linear in t.
        return (g[...,0].mean()+3*g[...,1])[...,None]
    terms,_=compute_loss_terms_1d(coupled_model,torch.ones(1,7),torch.zeros(1,7,7,1),grid,.4)
    assert float(terms["pde"])==pytest.approx(9.,rel=1e-5)


def test_polar_stencils_and_periodic_endpoint():
    axes=[torch.linspace(0,1,9,dtype=torch.float64),torch.linspace(.4,1,17,dtype=torch.float64),torch.linspace(0,2*torch.pi,65,dtype=torch.float64)]
    grid=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1)[None]
    field=grid[...,1]**2+4*grid[...,0]
    res,ur,uth=polar_derivatives(field,grid,1.)
    assert res[:,1:-1,1:-1].abs().max()<1e-10
    assert torch.allclose(uth[...,0],uth[...,-1])


def test_heat_flux_bc_receives_auxiliary_values():
    import deepxde as dde
    from heatops.physics.operator_boundary import HeatFluxBC
    geom=dde.geometry.Rectangle([.3,0],[1,2*np.pi])
    bc=HeatFluxBC(geom,lambda _,on:on)
    coords=np.array([[.3,0,.1],[.3,1,.2]],dtype=np.float32)
    for q in (1.,3.):
        x=torch.tensor(coords,requires_grad=True)
        u=0*x[:,:1]
        residual=bc.error(coords,x,u,0,2,aux_var=torch.full((2,1),q))
        assert torch.equal(residual,torch.full((2,1),q))
        dde.grad.clear()
    with pytest.raises(ValueError,match="aux_var"):
        bc.error(coords,x,u,0,2)


def test_periodic_bc_checks_independent_endpoint_slopes():
    from heatops.physics.losses import compute_loss_terms_polar
    axes=[torch.linspace(0,1,5),torch.linspace(.4,1,5),torch.linspace(0,2*torch.pi,33)]
    grid=torch.stack(torch.meshgrid(*axes,indexing="ij"),-1)[None]
    def model(q,g):
        theta=g[...,2]
        return theta*(2*torch.pi-theta)
    target=model(None,grid)
    terms,_=compute_loss_terms_polar(model,torch.ones_like(target),target,grid,1.,.1)
    # Endpoint values match, but their one-sided slopes have opposite signs.
    assert float(terms['periodic'])==pytest.approx(16*np.pi**2,rel=1e-4)


def test_csv_actual_time_header_and_unit_guard(tmp_path):
    path=tmp_path/"q.csv"
    path.write_text("% Length unit,mm\n% X,Y,T (degC) @ t=0,T (degC) @ t=99\n0,0,20,30\n0,100,20,30\n100,0,20,30\n100,100,20,30\n",encoding="utf-8")
    out=assemble_cartesian_4d([path],np.array([1e5]))
    assert np.array_equal(out["t"],[0.,99.])
    with pytest.raises(ValueError,match="单位"):
        assemble_cartesian_4d([path],np.array([1e5]),length_unit="m")


def test_mask_aware_interpolation_does_not_introduce_zero_temperature():
    axis=np.linspace(-1,1,11)
    x,y=np.meshgrid(axis,axis,indexing="ij")
    mask=(x*x+y*y>=.2)&(x*x+y*y<=.9)
    u=np.where(mask,20.,0.)
    field=cartesian_to_polar(u,axis,axis,np.linspace(.5,1,8),np.linspace(0,2*np.pi,16),mask=mask)
    assert np.allclose(field,20.)


def test_run_refuses_overwrite_and_records_failure(tmp_path):
    cfg={"experiment":"TEST1","seed":42}
    @managed_cli
    def fail():
        prepare_run(tmp_path/"TEST1",cfg,tmp_path/"test.yaml")
        print("failure retained")
        raise FloatingPointError("nonfinite loss")
    with pytest.raises(FloatingPointError):fail()
    payload=json.loads((tmp_path/"TEST1/run.json").read_text(encoding="utf-8"))
    assert payload["status"]=="failed" and (tmp_path/"TEST1/failure.txt").exists()
    with pytest.raises(FileExistsError):prepare_run(tmp_path/"TEST1",cfg,tmp_path/"test.yaml")
    assert json.loads((tmp_path/"TEST1/run.json").read_text(encoding="utf-8"))["status"]=="failed"
    with pytest.raises(ValueError,match="非法"):
        run_directory(tmp_path/"cfg.yaml","../escape")


def test_polar_dataset_rejects_nonfinite_and_wrong_units(tmp_path):
    path=save_polar_dataset(tmp_path/"p.pt",np.array([1e5]),np.ones((1,3,4,5)),np.arange(3),np.linspace(.1,.2,4),np.linspace(0,2*np.pi,5),metadata={"units":{"u":"K"}})
    with pytest.raises(ValueError,match="单位"):load_polar_dataset(path)


def test_training_rejects_nonfinite_loss_before_saving_success(tmp_path):
    from types import SimpleNamespace
    from heatops.training.torch_loop import TrainConfig, train_operator_model
    dataset=SimpleNamespace(u0=torch.zeros(1,3),u=torch.zeros(1,3,3,1),grid=torch.zeros(1,3,3,2),n_samples=1)
    def bad_loss(model,u0,u,g):
        return torch.tensor(float('nan')),{}
    with pytest.raises(FloatingPointError,match='非有限'):
        train_operator_model(torch.nn.Linear(1,1),dataset,bad_loss,TrainConfig(tmp_path,42,epochs=1,batch_size=1,device='cpu'))
    assert not (tmp_path/'checkpoints/checkpoint_best.pt').exists()


def test_legacy_loader_rejects_tampering_and_wrong_network(tmp_path):
    from heatops.models.legacy import load_legacy_model
    from heatops.training.run_context import file_hash, write_json
    weight=tmp_path/'model.pt'
    weight.write_bytes(b'tampered')
    spec={'format_version':1,'family':'fno_1d','models':{'m':{'checkpoint':'model.pt','sha256':'invalid','network':{'n_modes':[2,2],'hidden_channels':4,'a':.4}}}}
    write_json(tmp_path/'model_spec.json',spec)
    with pytest.raises(ValueError,match='哈希'):load_legacy_model(tmp_path)
    torch.save({'format_version':1,'family':'fno_1d','model_state_dict':{'wrong':torch.ones(1)}},weight)
    spec['models']['m']['sha256']=file_hash(weight)
    write_json(tmp_path/'model_spec.json',spec)
    with pytest.raises(RuntimeError,match='state_dict'):load_legacy_model(tmp_path)
