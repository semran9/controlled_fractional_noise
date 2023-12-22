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
        batch_size = 96,
        dt = 5 * 1e-3,
        noise_type="diagonal",
        sde_type="ito",
    ):
        super().__init__(noise_type, "ito")
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
            nn.Linear(3, 20), nn.ReLU(), nn.Linear(20, 10), nn.Sigmoid(), nn.Linear(10, 1)
        )
        self.net[-1].weight.data.fill_(0.0)
        self.net[-1].bias.data.fill_(0.0)
        
        # nn.init.xavier_uniform_(self.net.parameters()) 
        self.hurst_net = nn.Sequential(
            nn.Linear(3, 10), nn.Tanh(), nn.Linear(10, 1), nn.Sigmoid()
        )
        self.hurst_net[-2].weight.data.fill_(0.0)
        self.hurst_net[-2].bias.data.fill_(0.0)
        
        # posterior of initial point
        self.qy0_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.qy0_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=False)
        #posterior I_H(t)
        self.IH_mean = nn.Parameter(torch.tensor([[mu]]), requires_grad=False)
        self.IH_logvar = nn.Parameter(torch.tensor([[log_var]]), requires_grad=False)
        # self.IH_stdt = self.IH_std
        self.sample = torch.tensor([[mu]])
        self.var_i = None
        
    @torch.no_grad()    
    def compute_hurst(self, t):
        t = t.view(-1, 1)
        # time is converted into a positional encoding
        out = self.hurst_net(torch.cat([torch.sin(t), torch.cos(t), t], dim=-1))
        return out
    
    def rand_neg(self):
        return 1 if random.random() < 0.5 else -1
    # @torch.no_grad()
    def increment_variance(self, increment, h):
        var_1 = torch.exp(self.IH_logvar)
        h_t = h.clone()
        var_inc = (var_1 * increment * h_t) ** (2*h_t-1)
        self.var_i = var_inc
    
    # @torch.no_grad()
    def std(self, t, h):
        std = self.IH_std
        # if h > 1/2:
        #     std = std * (1-h)
        # else if h < 1/2:
        #     std = std * (h)
        # std = std * ((self.dt**(2*h))/self.dt).repeat(std.size(0), 1) #removes t dependence bs multifractional
        return std
    
    # @torch.no_grad()
    def sample_IH(self, t, h, size): # EXPERIMENTAL (treating like gaussian even though that's not true LOL)
        # sample random var
        # "noise"
        eps = torch.randn(size, 1).to(self.IH_std)
        
        # peak/mean of distribution
        peak = self.IH_mean
        
        # stdev of dist
        
        std = self.IH_std
        
        # sub diffusion; batched (NOT WORKING)
        # sub_idx = torch.where(h < 1/2, True, False)
        # sub_h = h[sub_idx]
        # modal_shift = ((1-2*sub_h)/(h**2))*t[sub_idx]**sub_h
        # modal_shift = self.rand_neg()*modal_shift
        # if peak[sub_idx].size(-1) > 0:
        #     peak[sub_idx] = peak[sub_idx][:, None] + modal_shift
        
        # super diffusion; batched NOT WORKING
        # not mathemtically founded (really) but we know that density explodes at origin when h > 1/2
        
        sample = peak + eps*std
        
        # super_idx = torch.where(h < 1/2, True, False)
        # if sample[super_idx].size(-1) > 0:
        #     sample[super_idx] = peak[super_idx] + std * (1-h)
        
        self.sample = sample.clone().detach()
        return sample     
        
    def g_FIM(self, t, y): 
        # hurst param (density of I_H relaies on this
        h = self.hurst_net(torch.cat([torch.sin(t), torch.cos(t), t], dim=-1))
        I_H = self.sample_IH(t, h.clone().detach(), y.size(0))
        sig = self.sigma.repeat(y.size(0), 1)
        out = I_H
        out = torch.abs(out) ** (1 - 1/(2*h))
        self.increment_variance(out, h.clone())
        out = out
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
        std = torch.sqrt(self.var_i)
        h = self.theta * (self.mu - (y - std*self.sigma))
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
        
        # g can be very small sometime so that the following is more numerically stable
        u = _stable_division(f + self.IH_log - h, self.sigma.repeat(y.size(0), 1), 1e-3)

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
        return torch.exp(0.5 * self.IH_logvar)

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
        logqp = (logqp0 + logqp_path).mean(dim=0)
        return ys, logqp

