"""Historical cylindrical heat operator and signed boundary residuals."""
import torch


def derivative(y, x):
    # Analytic linear/constant test fields legitimately have zero higher derivatives.
    if not y.requires_grad:
        return x * 0
    grad = torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True,
                               retain_graph=True, allow_unused=True)[0]
    return x * 0 if grad is None else grad


def angular_derivatives(y, x):
    si = 3 if x.shape[1] == 5 else 2
    g = derivative(y, x)
    dt = g[:, si:si+1] * x[:, si+1:si+2] - g[:, si+1:si+2] * x[:, si:si+1]
    gd = derivative(dt, x)
    dtt = gd[:, si:si+1] * x[:, si+1:si+2] - gd[:, si+1:si+2] * x[:, si:si+1]
    return dt, dtt


def laplacian(y, x):
    g = derivative(y, x)
    rr = derivative(g[:, 0:1], x)[:, 0:1]
    zz = derivative(g[:, 1:2], x)[:, 1:2]
    _, tt = angular_derivatives(y, x)
    r = x[:, 0:1]
    inv = 1 / (r + 1e-10)
    normal = rr + inv*g[:, 0:1] + inv**2*tt + zz
    return torch.where(r.abs() < 1e-3, 2*rr+zz, normal)


def residual(model, x, Fo):
    x.requires_grad_(True)
    y = model(x)
    lap = laplacian(y, x)
    return derivative(y, x)[:, 2:3] - Fo*lap if x.shape[1] == 5 else lap


def normal_flux(model, x, normals):
    """Normal derivative using Cartesian unit normals, both 4D and 5D input."""
    x.requires_grad_(True)
    y = model(x)
    g = derivative(y, x)
    dt, _ = angular_derivatives(y, x)
    s, c = x[:, -2:-1], x[:, -1:]
    inv = 1 / (x[:, 0:1] + 1e-10)
    gx = g[:, 0:1]*c - dt*s*inv
    gy = g[:, 0:1]*s + dt*c*inv
    return gx*normals[:, 0:1] + gy*normals[:, 1:2] + g[:, 1:2]*normals[:, 2:3]


def boundary_residual(model, x, normals, profile, kind):
    flux = normal_flux(model, x, normals)
    if kind == "neumann":
        return flux + profile.qstar
    if kind == "robin":
        return -flux - profile.Bi*(model(x) - (profile.Tamb-profile.T0)/profile.dT)
    if kind == "adiabatic":
        return flux
    raise ValueError(f"Unknown thermal boundary type: {kind}")
