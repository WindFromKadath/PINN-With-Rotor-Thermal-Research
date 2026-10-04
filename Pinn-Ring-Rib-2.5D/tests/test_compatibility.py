import numpy as np
import torch

from ring_rib_pinn.coupling import CouplingState
from ring_rib_pinn.materials import Material
from ring_rib_pinn.physics import Physics


def sample(bundle, domain):
    rng = np.random.default_rng(42)
    cfg = bundle.config
    if domain == "rib":
        values = np.column_stack([rng.uniform(0.01, 0.65, 16),
                                  rng.integers(0, 13, 16), rng.uniform(0.001, 0.008, 16)])
        values[0, 0], values[-1, 0] = 0, cfg.g["rib_length"]/cfg.g["radius"]
    else:
        values = np.column_stack([rng.uniform(cfg.g[f"{domain}_inner"]/0.225+0.005, 0.99, 16),
                                  rng.uniform(0, 2*np.pi, 16), rng.uniform(0.001, 0.008, 16)])
        radii = [cfg.g[f"{domain}_inner"], .098 if domain == "upper" else .117, .225, .15]
        values[:4, 0] = np.array(radii)/.225
        values[3, 1] = (bundle.geometry.phases[0].item()+cfg.g["rib_slope"]*(.15-.123)) % (2*np.pi)
    return torch.tensor(values, dtype=torch.float32, requires_grad=True)


def physics(bundle):
    return Physics(bundle.dde, bundle.config, bundle.geometry,
                   Material(bundle.config, bundle.device, bundle.dtype), bundle.thickness,
                   bundle.networks, CouplingState(bundle.geometry))


def test_original_predictions(bundle):
    old = bundle.original_networks()
    for domain, net in old.items():
        x = sample(bundle, domain)
        with torch.no_grad():
            difference = (net(x)-bundle.networks[domain](x))*900
        assert difference.abs().max().item() <= 1e-3


def test_coupled_residuals_and_boundaries(bundle):
    ns, p = bundle.oracle(), physics(bundle)
    for domain in ("upper", "lower", "rib"):
        x = sample(bundle, domain)
        y = bundle.networks[domain](x)
        old = ns[f"pde_{domain}_fixed_flux"](x, y) if domain != "rib" else ns["pde_rib"](x, y)
        new = p.ring(x, y, domain, coupled=True) if domain != "rib" else p.rib(x, y)
        torch.testing.assert_close(new[0], old[0], rtol=1e-4, atol=1e-5)
        for outer in (False, True):
            if domain == "rib":
                old = ns["bc_rib_end" if outer else "bc_rib_start"](x, y, None)
                new = p.rib_boundary(x, y, outer)
            else:
                old = ns["bc_outer" if outer else "bc_inner"](x, y, None, bundle.thickness[domain])
                new = p.ring_boundary(x, y, domain, outer)
            torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)
        bundle.dde.grad.clear()


def test_flux_snapshot_and_lookup(bundle):
    ns, p = bundle.oracle(), physics(bundle)
    old = ns["generate_flux_snapshot"](bundle.networks["upper"], bundle.networks["lower"],
                                       bundle.networks["rib"], n_s=16, n_t=8)
    p.coupling.update(p, 16, 8)
    qu, ql, s, t = old
    for name, expected in (("upper", qu), ("lower", ql)):
        torch.testing.assert_close(p.coupling.flux[name], expected, rtol=1e-4, atol=1e-5)
    ns.update(Q_snapshot_upper=qu, Q_snapshot_lower=ql, snapshot_s_grid=s, snapshot_t_grid=t)
    x = sample(bundle, "upper")
    for domain in ("upper", "lower"):
        new = p.coupling.lookup(x[:, :1], x[:, 1:2], x[:, 2:3], domain)
        old = ns["get_q_target"](x[:, :1], x[:, 1:2], x[:, 2:3], domain)
        torch.testing.assert_close(new, old, rtol=1e-4, atol=1e-5)
    assert not CouplingState(bundle.geometry).flux


def test_independent_reference_residuals(bundle):
    from pathlib import Path
    folder = Path(__file__).resolve().parents[1]/"local_reference_versions/source/Re-PINN"
    suffix = "_colding_hardcore" if bundle.config.case == "cooling" else ""
    ring_source = folder / ("pinn_no_rib_data_v6_colding_hardcore.py" if suffix
                            else "pinn_no_rib_data_v6.5.py")
    rib_source = folder / f"pinn_ring2rib_v6{suffix}.py"
    p = physics(bundle)
    ns = bundle.oracle(ring_source)
    for domain in ("upper", "lower"):
        x = sample(bundle, domain)
        y = bundle.networks[domain](x)
        old = ns[f"pde_{domain}"](x, y)
        torch.testing.assert_close(p.ring(x, y, domain)[0], old[0], rtol=1e-4, atol=1e-5)
        bundle.dde.grad.clear()
    ns = bundle.oracle(rib_source)
    x = sample(bundle, "rib")
    y = bundle.networks["rib"](x)
    torch.testing.assert_close(p.rib(x, y, single=True)[0], ns["pde_rib"](x, y)[0],
                               rtol=1e-4, atol=1e-5)
    for outer in (False, True):
        old = ns["rib_bc_end" if outer else "rib_bc_start"](x, y, None)
        torch.testing.assert_close(p.rib_boundary(x, y, outer, single=True), old,
                                   rtol=1e-4, atol=1e-5)
    bundle.dde.grad.clear()
