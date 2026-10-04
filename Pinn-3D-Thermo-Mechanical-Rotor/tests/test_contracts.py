import dataclasses
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import torch
from thermo_rotor.config import Profile, load_config
from thermo_rotor.models import model_from_spec
from thermo_rotor.models.checkpoint import save,load,legacy_state
from thermo_rotor.data.comsol import iter_frames,preprocess,sample_times
from thermo_rotor.evaluation.metrics import metrics
from thermo_rotor.training.run import Run
from thermo_rotor.training.optimizer import optimize
from thermo_rotor.geometry.sampling import mesh,surface,volume


def wide(tmp_path):
    path=tmp_path/'raw.csv'
    path.write_text('% Length unit,mm\n% X,Y,Z,T (K) @ t=0,T (K) @ t=10\n100,0,0,300,310\n120,0,5,301,311\n',encoding='utf-8')
    return path


def test_wide_units_time_and_temperature(tmp_path):
    frames=list(iter_frames(wide(tmp_path),length_unit='mm'))
    assert len(frames)==2
    assert frames[0].r.iloc[0]==.1 and frames[1].t.iloc[0]==10
    assert frames[1]['T'].iloc[0]==310


def test_wide_unit_mismatch(tmp_path):
    with pytest.raises(ValueError,match='metadata'):
        list(iter_frames(wide(tmp_path),length_unit='m'))


@pytest.mark.parametrize('unit',['cm','inch','bogus'])
def test_bad_units(tmp_path,unit):
    with pytest.raises(ValueError,match='unit'):
        list(iter_frames(wide(tmp_path),length_unit=unit))


def test_sample_missing_time_and_missing_file(tmp_path):
    with pytest.raises(ValueError,match='unavailable'):
        sample_times(wide(tmp_path),times=[999],length_unit='mm')
    with pytest.raises(FileNotFoundError):
        list(iter_frames(tmp_path/'missing.csv',length_unit='m'))


def test_time_axis_rejected(tmp_path):
    p=wide(tmp_path)
    p.write_text(p.read_text().replace('t=0,T (K) @ t=10','t=10,T (K) @ t=0'))
    with pytest.raises(ValueError,match='axis'):
        list(iter_frames(p,length_unit='mm'))


def test_preprocess_no_overwrite(tmp_path):
    p=wide(tmp_path)
    output=tmp_path/'long.csv'
    assert preprocess(p,output,length_unit='mm')==4
    with pytest.raises(FileExistsError):
        preprocess(p,output,length_unit='mm')
    with pytest.raises(FileExistsError):
        preprocess(p,p,length_unit='mm')


@pytest.mark.parametrize('values',[dict(r=0.,t=0.,T=300.),dict(r=.1,t=-1.,T=300.),dict(r=.1,t=0.,T=-1.),dict(r=.1,t=0.,T=float('nan'))])
def test_invalid_data(tmp_path,values):
    path=tmp_path/'long.csv'
    pd.DataFrame([dict(z=0.,theta=0.,**values)]).to_csv(path,index=False)
    with pytest.raises(ValueError):
        list(iter_frames(path,length_unit='m'))


def test_displacement_pairs(tmp_path):
    path=tmp_path/'disp.csv'
    path.write_text('% Length unit,mm\n% X,Y,Z,dr (mm) @ t=0,w (mm) @ t=0\n100,0,0,1,2\n')
    f=next(iter_frames(path,length_unit='mm'))
    assert f.u_r.iloc[0]==.001 and f.u_z.iloc[0]==.002
    path.write_text(path.read_text().replace(',w (mm) @ t=0',''))
    with pytest.raises(ValueError,match='paired'):
        list(iter_frames(path,length_unit='mm'))


def test_checkpoint_strict_and_hash(tmp_path):
    spec=dict(kind='thermal',hidden=[8,8],prefix='layers')
    model=model_from_spec(spec).double()
    path=tmp_path/'model.pt'
    save(path,model,spec,Profile())
    again,_,_=load(path)
    x=torch.randn(5,5,dtype=torch.float64)
    assert torch.equal(model(x),again(x))
    with pytest.raises(ValueError,match='profile'):
        load(path,expected_profile=Profile(dT=100.))
    with pytest.raises(ValueError,match='kind'):
        load(path,expected_kind='mechanical')
    with pytest.raises(FileExistsError):
        save(path,model,spec,Profile())
    with path.open('ab') as f:
        f.write(b'broken')
    with pytest.raises(ValueError,match='hash'):
        load(path)


def test_wrong_legacy_network(tmp_path):
    path=tmp_path/'old.pth'
    torch.save({'model_state_dict':model_from_spec(dict(kind='thermal',hidden=[8],prefix='layers')).state_dict()},path)
    with pytest.raises(ValueError,match='mismatch'):
        legacy_state(path,dict(kind='thermal',hidden=[9],prefix='layers'))


def test_failed_run_and_duplicate(tmp_path):
    cfg=dict(runtime=dict(seed=42),paths={})
    with pytest.raises(RuntimeError):
        with Run(tmp_path,'fail',cfg,'test') as run:
            raise RuntimeError('deliberate diagnostic')
    assert json.loads((run.path/'status.json').read_text())['status']=='failed'
    assert (run.path/'failure.txt').exists()
    with pytest.raises(FileExistsError):
        Run(tmp_path,'fail',cfg,'test')
    with pytest.raises(ValueError):
        Run(tmp_path,'../escape',cfg,'test')


def test_nonfinite_loss_and_lbfgs():
    model=torch.nn.Linear(1,1)
    with pytest.raises(FloatingPointError):
        optimize([model],lambda _:dict(bad=model.weight.sum()*float('nan')),1)
    x=torch.ones(3,1)
    history=optimize([model],lambda _:dict(data=(model(x)-2).square().mean()),2,lbfgs_steps=2,lbfgs_max_iter=5)
    assert all(np.isfinite(v['total']) for v in history)
    assert history[-1]['total']<history[0]['total']


def test_geometry_seed_normals_and_volume(tmp_path):
    import trimesh
    m=trimesh.creation.box(extents=[10.,10.,10.])
    p=tmp_path/'box.stl'
    m.export(p)
    loaded=mesh(p)
    a,n=surface(loaded,32,np.random.default_rng(42))
    b,nn=surface(loaded,32,np.random.default_rng(42),flip=True)
    assert np.array_equal(a,b) and np.array_equal(n,-nn)
    assert np.allclose(np.linalg.norm(n,axis=1),1)
    points=volume(loaded,32,np.random.default_rng(42))
    assert loaded.contains(points).all() and abs(points).max()<.005
    with pytest.raises(ValueError):
        mesh(p,'cm')


def test_zero_truth_relative_l2_is_unavailable():
    assert metrics(np.zeros((3,2)),np.zeros((3,2)),'mechanical')['relative_l2'] is None


def test_mm_long_header_unit_mismatch(tmp_path):
    path=tmp_path/'long_mm.csv'
    path.write_text('r_mm,z_mm,t,u_r_mm,u_z_mm\n100,0,0,1,2\n')
    with pytest.raises(ValueError,match='headers'):
        list(iter_frames(path,length_unit='m'))
    with pytest.raises(ValueError,match='headers'):
        list(iter_frames(path,length_unit='mm',displacement_unit='m'))


@pytest.mark.parametrize('settings',[dict(L=0.),dict(dT=-1.),dict(tau=0.),dict(nu=.5),dict(r_min=.3),dict(k=float('nan'))])
def test_invalid_profile(settings):
    with pytest.raises(ValueError):
        Profile(**settings)
