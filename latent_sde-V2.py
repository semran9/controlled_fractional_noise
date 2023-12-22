"""Adapt from
https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
"""
import math
import numpy as np

import torch
import torch.nn as nn
from fractional_neural_sde.fractional_noise import SparseGPNoise
from fractional_neural_sde.utils import _stable_division
from torch import distributions
from torchsde import BaseSDE
from torchdiffeq import odeint

class LatentSDE(BaseSDE):
    def __init__(
        self,
        noise_path: SparseGPNoise,
        theta: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.5,
        noise_type="diagonal",
        sde_type="ito",
    ):
        super().__init__(noise_type, sde_type)
        
        # current problem with latent SDE: how can I add control with the current structure of "getting around" time?
        # try integrating the drift, then passing that through to the hurst function
        # does this work mathematically? possibly
        # but this may not work mathematically

        self.noise_path = noise_path
        # prior SDE - use this for y0 calculation (will need double integration?)
        # SDE only relies on the probability distribution; do we need to add a control part to latent SDE?
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))

        # mean and var of initial point
        log_var = math.log(sigma**2 / (2 * theta))
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[log_var]]))

        # posterior drift
        # *integrate this to then change the h(t, X) function
        self.net = nn.Sequential(
            nn.Linear(3, 10), nn.Sigmoid(), nn.Linear(10, 10), nn.Tanh(), nn.Linear(10, 1)
        )
        
        # nn.init.xavier_uniform_(self.net.parameters()) 
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)
        #self.net[-2].weight.data.fill_(0.0)
        self.net[-3].weight.data.fill_(0.0)
        self.net[-3].bias.data.fill_(0.0)
        #self.net[1].weight.data.fill_(0.0)
        self.net[0].weight.data.fill_(0.0)
        self.net[0].bias.data.fill_(0.0)
        
        # posterior of initial point
        # these are initializing correctly, but smth in precompute_drift is setting them to nan
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.qy0_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=False)
        
        # initial white noise drift path
        self.drift_y0 = torch.zeros(noise_path.num_steps)
        #drift path + dt
        self.drift_yt = torch.zeros(noise_path.num_steps)
    
    @torch.no_grad()
    def drift_pass(self, y0):
        t0, t1, dt = self.noise_path.t0, self.noise_path.t1, self.noise_path.dt
        num_inducings = self.noise_path.num_inducing
        ts = torch.linspace(t0 + dt * 5.5, t1 - dt * 5.5, num_inducings)
        drift_path = y0.clone().detach()
        y_n = drift_path
        for tn in ts:
            dY = self.net(torch.tensor([torch.sin(tn), torch.cos(tn), y0]))
            y_n = y_n + dY
            drift_path = torch.cat([drift_path, y_n]) # This treats this like an ODE euler method solver 
        return drift_path + 1e-5 # (small offset)
        
    def precompute_drift(self, y0):
        # sample initial point
        # get time series to "integrate" over
        with torch.no_grad():
            drift_path = self.drift_pass(y0).reshape(-1,1)
            self.drift_y0 = drift_path[0:(drift_path.shape[0]-1)]
            self.drift_yt = drift_path[1:(drift_path.shape[0])]

    def precompute_white_noise(self, batch_size, ts, y0):
        """Precompute Cholesky for white noise"""
        self.precompute_drift(y0 = y0)
        self.noise_path.precompute(batch_size = batch_size, y0 = self.drift_y0.clone().requires_grad_(), 
                                   yt = self.drift_yt.clone().requires_grad_())

    def f_and_g(self, t, y):
        """Drift and diffusion"""
        # mean, sqrt_var_dt = self.noise_path(t, y)

        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)

        x = torch.cat([torch.sin(t), torch.cos(t), y], dim=-1) #... should we just use positional encoding instead of ODE integration?
        f = self.net(x)
        g = self.sigma.repeat(y.size(0), 1)
        mean, sqrt_var_dt = self.noise_path(t, f, y)

        # new drift and diffusion in Eq. 12
        f = f + g * mean
        g = g * sqrt_var_dt
        return f, g

    def h(self, t, y):
        r"""Drift of prior SDE

        dX_t = h(t, X_t)dt + \sigma(t, X_t)dB_t
        """
        return self.theta * (self.mu - y)

    def f_and_g_aug(self, t, y):
        """Augment drift and diffusion to compute KL while solving SDE

        The output of drift function `f` is added a quantity from Giranov's theorem
        The output of diffusion `g` is added a zero value
        """
        y = y[:, 0:1]

        f, g = self.f_and_g(t, y)
        h = self.h(t, y)

        # g can be very small sometime so that the following is more numerically stable
        u = _stable_division(f - h, g, 1e-3)

        # augmented drift
        f_logqp = 0.5 * (u**2).sum(dim=1, keepdim=True)
        f_aug = torch.cat([f, f_logqp], dim=1)

        # augmented diffusion
        g_logqp = torch.zeros_like(y)
        g_aug = torch.cat([g, g_logqp], dim=1)
        return f_aug, g_aug

    def sample_q(self, ts, batch_size, sdeint_fn, method, dt, bm, eps: None):
        """Sample posterior"""

        if eps is None:
            eps = torch.randn(batch_size, 1).to(self.qy0_mean)

        # sample initial point
        y0 = self.qy0_mean + eps * self.qy0_std
        # make sure precompute inducing before solving SDE
        self.precompute_white_noise(batch_size=batch_size, ts = ts, y0 = y0.data.clone())
        # return the solution of solver with posterior drift and diffusion
        return sdeint_fn(
            sde=self,
            y0=y0,
            ts=ts,
            bm=bm,
            method=method,
            dt=dt,
            names={"drift_and_diffusion": "f_and_g"},
        )

    @property
    def py0_std(self):
        return torch.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(0.5 * self.qy0_logvar)

    def forward(self, ts, batch_size, sdeint_fn, method, dt, bm=None, eps=None):
        """SDE integration

        Args:
            ts: time step at which solution will return
            batch_size: batch size
            sdeint_fn: `torchsde` SDE solver. Normally, we use `euler`
            dt: step size of SDE solver
            bm: Brownian motion
            eps: noise for intial point
        """
        if eps is None:
            eps = torch.randn(batch_size, 1).to(self.qy0_std)

        # sample initial point and compute KL at t=0
        y0 = self.qy0_mean + eps * self.qy0_std
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)
        self.precompute_white_noise(batch_size=batch_size, ts = ts, y0 = y0.data.clone())
        
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)

        # compute cholesky and inducing sample before solving
        # compute prior ode for the white noise
        
        # solve the path from 0 -> T
        aug_ys = sdeint_fn(
            sde=self,
            bm=bm,
            y0=aug_y0,
            ts=ts,
            method=method,
            dt=dt,
            rtol=1e-3,
            atol=1e-3,
            names={"drift_and_diffusion": "f_and_g_aug"},
        )

        # seperate ys and log pq
        ys, logqp_path = aug_ys[:, :, 0:1], aug_ys[-1, :, 1]
        logqp = (logqp0 + logqp_path).mean(dim=0)
        return ys, logqp
