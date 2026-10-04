import torch


class TemperatureBridge:
    """Freeze parameters; convert mechanical scales without breaking input autograd."""
    def __init__(self, model, thermal_profile, mechanical_profile, theta=0.0):
        self.model = model.eval()
        self.source, self.target, self.theta = thermal_profile, mechanical_profile, theta
        for par in model.parameters():
            par.requires_grad_(False)

    def __call__(self, x):
        r = x[:, 0:1]*self.target.L/self.source.L
        z = x[:, 1:2]*self.target.L/self.source.L
        t = x[:, 2:3]*self.target.tau/self.source.tau
        angle = torch.full_like(r, self.theta)
        inp = torch.cat((r, z, t, angle.sin(), angle.cos()), dim=1)
        T = self.model(inp)*self.source.dT+self.source.T0
        return (T-self.target.T0)/self.target.dT


def linear_iwg_temperature(x, p):
    Tmin = (400-p.T0)/p.dT
    Tmax = (300-p.T0)/p.dT
    return Tmin+(Tmax-Tmin)*(x[:, 0:1]*p.L-p.r_min)/(p.r_max-p.r_min)
