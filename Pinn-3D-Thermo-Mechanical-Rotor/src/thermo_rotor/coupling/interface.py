from ..physics.thermal import normal_flux


def interface_loss(source, target, x, normals, conductivity=76.2, lambda_T=50., lambda_q=10.):
    # Compute the source flux before detaching. Its parameters receive no interface gradient.
    x.requires_grad_(True)
    T_source = source(x).detach()
    flux_source = normal_flux(source, x, normals).detach()
    T_target = target(x)
    flux_target = normal_flux(target, x, normals)
    return {"temperature": lambda_T*((T_source-T_target)**2).mean(),
            "flux": lambda_q*((conductivity*(flux_source-flux_target))**2).mean()}
