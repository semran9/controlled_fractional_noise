"""Adapt from
https://github.com/google-research/torchsde/blob/master/examples/latent_sde.py
"""
import math
import numpy as np
"""
Try to create FIM and then learn FIM using h(t, X)
FIM - Fractional Ito Motion
This is WIP (very much so)
"""
import torch
import torch.nn as nn
from utils import _stable_division
from torch import distributions
from torchsde import BaseSDE
from torchdiffeq import odeint
import random

class FIM(BaseSDE):
    def __init__(
        self,
        theta: float = 1.0,
        mu: float = 1.0,
        sigma: float = 0.5,
        hurst = 0.5,
        batch_size = None,
        dt = 5 * 1e-3,
        noise_type="diagonal",
        sde_type="ito",
    ):
        super().__init__(noise_type, "ito")
        # prior SDE
        self.register_buffer("theta", torch.tensor([[theta]]))
        self.register_buffer("mu", torch.tensor([[mu]]))
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        # mean and var of initial point
        log_var = math.log(sigma**2 / (2 * theta))
        self.register_buffer("py0_mean", torch.tensor([[mu]]))
        self.register_buffer("py0_logvar", torch.tensor([[log_var]]))
        # dt
        self.dt = dt
        # posterior drift
        self.net = nn.Sequential(
            nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 10), nn.Sigmoid(), nn.Linear(10, 1)
        )
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)
        
        # nn.init.xavier_uniform_(self.net.parameters()) 
        self.hurst_net = nn.Sequential(
            nn.Linear(3, 10), nn.ReLU(), nn.Linear(10, 1), nn.Sigmoid()
        )
        self.hurst_net[-2].weight.data.fill_(0.0)
        self.hurst_net[-2].bias.data.fill_(0.0)
        self.size = batch_size
        # posterior of initial point
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.qy0_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=False)
        #posterior I_H(t)
        self.IH_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.IH_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=False)
        self.IH_var = nn.Parameter(torch.exp(torch.tensor([[log_var]])), requires_grad = True)
        #self.IH_std = self.IH_var ** 0.5
        eps = torch.randn(self.size, 1).to(self.IH_logvar)
        self.sample = nn.Parameter(self.IH_std.repeat(self.size, 1) * eps, requires_grad = False)
        # testing/loss customization ideas
        IH_dist = distributions.Normal(loc=torch.tensor(0).repeat(self.size, 1), scale=self.IH_std)
        self.prob_IH = IH_dist.log_prob(self.sample)
        # update the variance as we move along the trajectory
        self.path_var = torch.exp(torch.tensor([[log_var]]))
        self.path_v0 = torch.exp(torch.tensor([[log_var]]))
        self.I = None
        self.t_0 = 0
        
    @property    
    def IH_std(self):
        return torch.exp(0.5 * self.IH_logvar)
    @property    
    def qy0_std(self):
        return torch.exp(0.5 * self.qy0_logvar) 
    @property    
    def py0_std(self):
        return torch.exp(0.5 * self.py0_logvar)
    @property
    def path_std(self):
        return self.path_var ** 0.5
    @property
    def rand_neg(self):
        return 1 if random.random() < 0.5 else -1
    @property
    def path_dist(self):
        std = self.path_var ** 0.5
        distribution = torch.distributions.Normal(loc = 0, scale = std)
        return distribution
    
    def compute_hurst(self, t):
        t = t.view(-1, 1)
        # time is converted into a positional encoding
        out = self.hurst_net(torch.cat([torch.sin(t), torch.cos(t), t], dim=-1))
        return out
    # @torch.no_grad
    def update_var(self, t, h): # update variance wrt to hurst param
        var_0 = self.path_var
        self.path_v0 = var_0
        dt = self.dt
        var_new = var_0 * (dt ** (2*h))
        self.path_var = var_new
        
    def resample(self, batch_size): # resample I_H for beginning of SDE
        eps = torch.randn(batch_size, 1).to(self.IH_logvar)
        self.sample = nn.Parameter((self.path_var**0.5) * eps, requires_grad = False)
        
    def g_FIM(self, t, y): 
        # hurst param (density of I_H relaies on this)
        # self.resample(y.size(0))
        h = self.hurst_net(torch.cat([torch.sin(t), torch.cos(t), t], dim=-1))
        I_H = self.sample
        out = torch.abs(I_H) ** (1 - 1/(2*h))
        self.update_var(t, h.clone()) # clones bc I'm not sure how the back prop will work
        self.I = out.clone() 
        # from latent sde
        sig = self.sigma.repeat(y.size(0), 1)
        return out * sig

    def f_and_g(self, t, y): 
        """Drift and diffusion"""
        # mean, sqrt_var_dt = self.noise_path(t, y)

        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)

        x = torch.cat([torch.sin(t), torch.cos(t), y], dim=-1)
        f = self.net(x) 
        g = self.g_FIM(t, x)
        f = f
        return f, g

    def h(self, t, y): # testing
        r"""Drift of prior SDE

        dX_t = h(t, X_t)dt + \sigma(t, X_t)dB_t
        """
        std = self.path_std
        h = self.theta * (self.mu - (y - self.path_v0)) # this is the thing that doesn't make sense but was working before
        return h

    def f_and_g_aug(self, t, y):
        """Augment drift and diffusion to compute KL while solving SDE

        The output of drift function `f` is added a quantity from Giranov's theorem
        The output of diffusion `g` is added a zero value
        """
        y = y[:, 0:1]
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        
        f, g = self.f_and_g(t, y)
        h = self.h(t, y)
        
        path_sd = (self.path_var) ** 0.5
        pseudo_prob = path_sd/(self.I*2)
        # g can be very small sometime so that the following is more numerically stable
        # lot of loss testing going on here
        u = _stable_division(f - h, g, 1e-3)
        # u = f-h
        # u = _stable_division(f - h, self.sample, 1e-3)
        
        # u = _stable_division(f - h, g, 1e-3) 
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
        # return the solution of solver with posterior drift and diffusion
        return sdeint_fn(
            sde=self,
            y0=y0,
            ts=ts,
            bm=bm,
            method=method,
            dt=dt,
            names={"drift_and_diffusion": "f_and_g"})

    @property
    def py0_std(self):
        return torch.exp(0.5 * self.py0_logvar)

    @property
    def qy0_std(self):
        return torch.exp(0.5 * self.qy0_logvar)
    
    @property
    def IH_std(self):
        return self.IH_var ** 0.5

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
        self.resample(batch_size)
        self.size = batch_size
        eps_sample = torch.randn(self.size, 1).to(self.IH_logvar)
        self.sample = nn.Parameter(self.IH_std * eps_sample, requires_grad = False)
        if eps is None:
            eps = torch.randn(batch_size, 1).to(self.qy0_std)

        # sample initial point and compute KL at t=0
        
        y0 = self.qy0_mean + eps * self.qy0_std
        
        #KL
        qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)
        
        aug_y0 = torch.cat([y0, torch.zeros(batch_size, 1).to(y0)], dim=1)
        
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
        logqp_path = logqp_path
        logqp = (logqp0 + logqp_path).mean(dim=0)
        return ys, logqp