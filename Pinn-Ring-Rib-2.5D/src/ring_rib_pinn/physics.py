"""Retained legacy equations. Formula parity is distinct from physical validity."""

import torch


class Physics:
    def __init__(self, dde, config, geometry, material, thickness, networks, coupling):
        self.dde, self.cfg, self.geo, self.mat = dde, config, geometry, material
        self.thickness, self.networks, self.coupling = thickness, networks, coupling

    def derivative(self, value, x, j):
        return self.dde.grad.jacobian(value, x, i=0, j=j)

    @property
    def bi(self):
        return self.cfg.p["convection"] * self.cfg.g["radius"] / self.cfg.p["k_ref"]

    @property
    def qstar(self):
        p, g = self.cfg.p, self.cfg.g
        return p["source"] * g["radius"]**2 / (p["k_ref"] * p["temperature_scale"])

    def radiation(self, kelvin):
        p, g = self.cfg.p, self.cfg.g
        coefficient = p["sigma"] * g["radius"] / (p["k_ref"] * p["temperature_scale"])
        return coefficient * (kelvin**4 - p["ambient"]**4)

    def kelvin(self, y):
        return y * self.cfg.p["temperature_scale"] + self.cfg.p["ambient"]

    def source_distribution(self, radius, domain):
        p, g = self.cfg.p, self.cfg.g
        inner = g[f"{domain}_inner"]
        distribution = (radius < inner + 3*p["source_decay"]).float()
        distribution = distribution * torch.exp(-(radius-inner) / p["source_decay"])
        return distribution if domain == "upper" else distribution * p["gap_factor"]

    def surface_geometry(self, x, domain):
        radius = x[:, :1]
        theta = x[:, 1:2]
        height = self.thickness[domain](radius, theta)
        dh_r = self.derivative(height, x, 0)
        dh_theta = self.derivative(height, x, 1)
        slope_sq = dh_r**2 + (dh_theta / (radius*self.cfg.g["radius"] + 1e-8))**2
        gamma = torch.sqrt(1 + slope_sq)
        metric = 1 + slope_sq / 4 + 1e-8
        lam = 1 / (torch.sqrt(metric) if self.cfg.case == "heating" else metric)
        return height / self.cfg.g["radius"] + 1e-8, gamma, lam

    def ring(self, x, y, domain, coupled=False):
        radius, theta, time = x[:, :1], x[:, 1:2], x[:, 2:3]
        if self.cfg.case == "heating":
            radius = radius.clamp(min=1e-6)
        height, gamma, lam = self.surface_geometry(x, domain)
        kelvin = self.kelvin(y).clamp(250, 2000)
        k, c, _, _ = self.mat.factors(kelvin)
        dy_r, dy_theta = self.derivative(y, x, 0), self.derivative(y, x, 1)
        radial_flux = radius * k * lam * height * dy_r
        if self.cfg.case == "heating":
            radial = lam / radius * self.derivative(radial_flux, x, 0)
            angular_flux = k * height / (lam*lam*radius) * dy_theta
            angular = self.derivative(angular_flux, x, 1) / radius
        else:
            radial = self.derivative(radial_flux, x, 0) / (radius + 1e-8)
            angular_flux = k * lam * height / (radius + 1e-8) * dy_theta
            angular = self.derivative(angular_flux, x, 1) / (radius + 1e-8)
        top, bottom = (gamma, 1.) if domain == "upper" else (1., gamma)
        exposed = 1.
        if coupled:
            mask, _, _ = self.geo.project(radius, theta)
            exposed = 1 - mask  # Retained: Gaussian density can exceed 1; see README.
        convection = self.bi * y * (top + bottom) * exposed
        rad_area = top if domain == "upper" else bottom
        if domain == "lower":
            rad_area = rad_area * (radius*self.cfg.g["radius"] >= 0.1175).float()
        radiation = self.cfg.p["emissivity"] * self.radiation(kelvin) * rad_area * exposed
        source = height * self.qstar * self.source_distribution(radius*self.cfg.g["radius"], domain)
        residual = c * height * self.derivative(y, x, 2) - radial - angular - source
        residual = residual + convection + radiation
        if coupled:
            residual = residual + self.coupling.lookup(radius, theta, time, domain)
        if self.cfg.case == "heating":
            factor = 1 if domain == "upper" else self.cfg.p["gap_factor"]
            residual = residual / (self.qstar * factor / 10)
        return [residual]

    def ring_boundary(self, x, y, domain, outer):
        _, _, lam = self.surface_geometry(x, domain)
        temp = self.kelvin(y)
        k, _, _, _ = self.mat.factors(temp)
        flux = k * lam * self.derivative(y, x, 0)
        if not outer:
            return flux - self.bi * y
        return -flux - (self.bi*y + self.cfg.p["emissivity"]*self.radiation(temp))

    def contact(self, radius_star, theta, upper, lower, rib, single=False):
        # Single-direction scripts clamp physical temperature to material nodes.
        temperatures = [self.kelvin(v) for v in (upper, lower, rib)]
        if single:
            temperatures = [v.clamp(293.15, 773.15) for v in temperatures]
        ku, kl, kr = [self.mat.factors(v)[0] for v in temperatures]
        radius = self.cfg.g["radius"]
        hu = self.thickness["upper"](radius_star, theta) / radius + 1e-12
        hl = self.thickness["lower"](radius_star, theta) / radius + 1e-12
        rib_height = self.cfg.g["rib_height"] / radius
        resistance_rib = rib_height / (2*kr + 1e-12)
        bu = 1 / (hu / (3*ku + 1e-12) + resistance_rib + 1e-6)
        bl = 1 / (hl / (3*kl + 1e-12) + resistance_rib + 1e-6)
        return bu*(upper-rib), bl*(lower-rib)

    def rib(self, x, y, single=False):
        s, rid, time = x[:, :1], x[:, 1:2], x[:, 2:3]
        radius, theta = self.geo.rib_coordinates(s, rid)
        with torch.no_grad():
            ring_x = torch.cat([radius, theta, time], dim=1)
            upper = self.networks["upper"](ring_x)
            lower = self.networks["lower"](ring_x)
        temp = self.kelvin(y)
        if single:
            temp = temp.clamp(293.15, 773.15)
        k, c, _, _ = self.mat.factors(temp)
        q_u, q_l = self.contact(radius, theta, upper, lower, y, single)
        g = self.cfg.g
        side = 2 / (g["rib_width"] / g["radius"]) * self.bi * y
        source = torch.zeros_like(y)
        if single and self.cfg.case == "heating":
            source = (g["rib_width"]*g["rib_height"]/g["radius"]**2 * self.qstar
                      * self.source_distribution(radius*g["radius"], "upper"))
        diffusion = self.dde.grad.hessian(y, x, i=0, j=0)
        residual = c*self.derivative(y, x, 2) - k*diffusion + side - source
        residual = residual - (q_u+q_l)/(g["rib_height"]/g["radius"])
        return [residual / 20.]

    def rib_boundary(self, x, y, outer, single=False):
        temp = self.kelvin(y)
        k = torch.ones_like(y) if single else self.mat.factors(temp)[0]
        flux = k * self.derivative(y, x, 0)
        if not outer:
            return flux - self.bi*y
        if single:
            temp = temp.clamp(293.15, 2000)
        return -flux - (self.bi*y + self.cfg.p["emissivity"]*self.radiation(temp))
