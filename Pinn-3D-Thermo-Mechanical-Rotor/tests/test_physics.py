import numpy as np
import pandas as pd
import pytest
import torch
from thermo_rotor.config import Profile
from thermo_rotor.physics.thermal import angular_derivatives, laplacian, normal_flux, boundary_residual
from thermo_rotor.physics.mechanical import stress, equilibrium, traction, von_mises
from thermo_rotor.models import model_from_spec
from thermo_rotor.coupling.temperature import TemperatureBridge
from thermo_rotor.coupling.interface import interface_loss
from thermo_rotor.data.transforms import inputs, targets, physical
from thermo_rotor.evaluation.predict import determine_gauge, predict


@pytest.fixture
def x():
    theta = torch.linspace(-2, 2, 19, dtype=torch.float64).reshape(-1, 1)
    return torch.cat((torch.full_like(theta, .7), theta*.03, theta*0+.4, theta.sin(), theta.cos()), 1).requires_grad_()


def test_angular_chain_rule(x):
    y = x[:, 3:4]**2-x[:, 4:5]**2
    theta = torch.atan2(x[:, 3:4], x[:, 4:5])
    d, dd = angular_derivatives(y, x)
    assert torch.allclose(d, 2*(2*theta).sin(), atol=1e-12)
    assert torch.allclose(dd, 4*(2*theta).cos(), atol=1e-12)


def test_laplacian_manufactured(x):
    y = x[:, 0:1]**2+x[:, 1:2]**2+x[:, 3:4]
    expected = 6-x[:, 3:4]/x[:, 0:1]**2
    assert torch.allclose(laplacian(y, x), expected, atol=2e-9)


def test_cartesian_normal_flux_and_sign(x):
    model = lambda a: a[:, 0:1]*a[:, 4:5]+2*a[:, 1:2]
    normal = torch.zeros(len(x), 3, dtype=x.dtype)
    normal[:, 0] = 1
    assert torch.allclose(normal_flux(model, x, normal), torch.ones(len(x), 1, dtype=x.dtype), atol=1e-9)
    assert torch.allclose(normal_flux(model, x, -normal), -normal_flux(model, x, normal))
    p = Profile()
    b = boundary_residual(model, x, normal, p, "neumann")
    assert torch.allclose(b, normal_flux(model, x, normal)+p.qstar)


def test_robin_residual(x):
    p = Profile(Tamb=300.)
    model = lambda a: a[:, 0:1]**2
    normal = torch.zeros(len(x), 3, dtype=x.dtype)
    normal[:, 2] = 1
    expected = -p.Bi*(model(x)-(p.Tamb-p.T0)/p.dT)
    assert torch.allclose(boundary_residual(model, x, normal, p, "robin"), expected)


def test_free_thermal_expansion():
    nu = .29
    p = Profile(lam=nu/((1+nu)*(1-2*nu)), mu=1/(2*(1+nu)), beta=1/(1-2*nu))
    x = torch.tensor([[.4, -.1, .3],[.7,.1,.4]], dtype=torch.float64).requires_grad_()
    model = lambda a: .3*a[:, :2]
    temp = lambda a: .3+0*a[:, 0:1]
    s = stress(model, x, temp, p)
    assert torch.allclose(s, torch.zeros_like(s), atol=2e-12)
    assert torch.allclose(equilibrium(model, x, temp, p), torch.zeros(len(x),2,dtype=x.dtype), atol=1e-11)


def test_thermal_gradient_enters_equilibrium():
    p = Profile()
    x = torch.tensor([[.4,-.1,.3],[.7,.1,.4]],dtype=torch.float64).requires_grad_()
    u = lambda a: a[:, :2]*0
    T = lambda a: 2*a[:,0:1]+3*a[:,1:2]
    result = equilibrium(u,x,T,p)
    assert torch.allclose(result,torch.tensor([[-2*p.beta,-3*p.beta]]*2,dtype=x.dtype),atol=1e-10)


def test_traction_and_mises():
    p=Profile()
    x=torch.tensor([[.7,0.,.4]],dtype=torch.float64).requires_grad_()
    s=stress(lambda a: a[:,:2]*0,x,lambda a: a[:,0:1]*0+1,p)
    assert torch.allclose(von_mises(s),torch.zeros(1,1,dtype=x.dtype))
    t=traction(lambda a:a[:,:2]*0,x,torch.tensor([[1.,0.]],dtype=x.dtype),lambda a:a[:,0:1]*0+1,p)
    assert torch.allclose(t,torch.tensor([[-p.beta,0.]],dtype=x.dtype))


def test_frozen_bridge_derivatives_and_scale():
    class Linear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w=torch.nn.Parameter(torch.tensor([2.,3.,4.,0.,0.],dtype=torch.float64))
        def forward(self,a):
            return (a*self.w).sum(1,keepdim=True)
    model=Linear()
    source=Profile(L=.3,dT=100.,tau=50.)
    target=Profile(L=.2,dT=300.,tau=200.)
    bridge=TemperatureBridge(model,source,target)
    x=torch.tensor([[.5,.1,.3]],dtype=torch.float64).requires_grad_()
    g=torch.autograd.grad(bridge(x).sum(),x)[0]
    expected=torch.tensor([[2*target.L/source.L*source.dT/target.dT,
                            3*target.L/source.L*source.dT/target.dT,
                            4*target.tau/source.tau*source.dT/target.dT]],dtype=x.dtype)
    assert torch.allclose(g,expected)
    assert all(not v.requires_grad and v.grad is None for v in model.parameters())


def test_interface_jump_and_detach():
    class Field(torch.nn.Module):
        def __init__(self,k):
            super().__init__()
            self.k=torch.nn.Parameter(torch.tensor(k,dtype=torch.float64))
        def forward(self,a):
            return self.k*a[:,0:1]
    source,target=Field(1.),Field(2.)
    x=torch.tensor([[.5,0.,0.,1.],[.7,.1,0.,1.]],dtype=torch.float64)
    n=torch.tensor([[1.,0.,0.]]*2,dtype=x.dtype)
    terms=interface_loss(source,target,x,n)
    assert terms['temperature']>0 and terms['flux']>0
    sum(terms.values()).backward()
    assert source.k.grad is None and target.k.grad.abs()>0


@pytest.mark.parametrize('kind',['thermal','steady','mechanical','iwg'])
def test_nondimensional_roundtrip(kind):
    p=Profile()
    frame=pd.DataFrame(dict(r=[.1,.2],z=[0.,.01],t=[0.,200.],theta=[-.2,2.],T=[293.15,390.],u_r=[0.,.001],u_z=[.0001,.0002]))
    y=targets(frame,p,kind)
    actual=physical(y,p,kind)
    expected=frame[['T'] if kind in ('thermal','steady') else ['u_r','u_z']].to_numpy()
    assert np.allclose(actual,expected,atol=1e-14)
    xx=inputs(frame,p,kind)
    if kind=='thermal':
        frame.theta+=2*np.pi
        assert np.allclose(xx,inputs(frame,p,kind))


def test_global_projection_independent_of_batch():
    torch.manual_seed(42)
    spec=dict(kind='iwg',hidden=[8,8],fourier_dim=4)
    model=model_from_spec(spec).double()
    p=Profile()
    frame=pd.DataFrame(dict(r=np.linspace(.1,.2,31),z=np.linspace(-.02,0.,31),t=0.,theta=0.))
    x=inputs(frame,p,'iwg')
    gauge=determine_gauge(model,x,'projection',batch_size=7)
    a=predict(model,frame,p,spec,gauge,batch_size=1)
    b=predict(model,frame,p,spec,gauge,batch_size=31)
    assert np.allclose(a,b,atol=1e-15)
    assert abs((a[:,1]*x[:,0]).sum())<1e-14
