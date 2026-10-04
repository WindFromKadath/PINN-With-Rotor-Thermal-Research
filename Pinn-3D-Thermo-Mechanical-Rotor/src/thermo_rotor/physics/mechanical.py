"""Axisymmetric quasi-static thermoelasticity; time changes the thermal load."""
import torch
from .thermal import derivative


def stress(model, x, temperature, p):
    x.requires_grad_(True)
    u = model(x)
    gr = derivative(u[:, 0:1], x)
    gz = derivative(u[:, 1:2], x)
    rr = gr[:, 0:1]
    tt = u[:, 0:1] / (x[:, 0:1]+1e-12)
    zz = gz[:, 1:2]
    rz = 0.5*(gr[:, 1:2]+gz[:, 0:1])
    tr = rr+tt+zz
    T = temperature(x)
    base = p.lam*tr-p.beta*T
    return torch.cat((base+2*p.mu*rr, base+2*p.mu*tt, base+2*p.mu*zz, 2*p.mu*rz), dim=1)


def equilibrium(model, x, temperature, p):
    s = stress(model, x, temperature, p)
    rr = derivative(s[:, 0:1], x)
    zz = derivative(s[:, 2:3], x)
    rz = derivative(s[:, 3:4], x)
    inv = 1/(x[:, 0:1]+1e-12)
    return torch.cat((rr[:, 0:1]+rz[:, 1:2]+(s[:, 0:1]-s[:, 1:2])*inv,
                      rz[:, 0:1]+zz[:, 1:2]+s[:, 3:4]*inv), dim=1)


def traction(model, x, normals_rz, temperature, p):
    s = stress(model, x, temperature, p)
    nr, nz = normals_rz[:, 0:1], normals_rz[:, 1:2]
    return torch.cat((s[:, 0:1]*nr+s[:, 3:4]*nz, s[:, 3:4]*nr+s[:, 2:3]*nz), dim=1)


def von_mises(s):
    rr, tt, zz, rz = s.split(1, dim=1)
    return (0.5*((rr-tt)**2+(tt-zz)**2+(zz-rr)**2)+3*rz**2).clamp_min(0).sqrt()
