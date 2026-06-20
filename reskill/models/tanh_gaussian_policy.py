import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TanhGaussianBehaviorPolicy(nn.Module):
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_dim=256,
        log_std_min=-5.0,
        log_std_max=2.0,
        action_low=-1.0,
        action_high=1.0,
        eps=1e-6,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.eps = eps

        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

        action_low = torch.as_tensor(action_low, dtype=torch.float32)
        action_high = torch.as_tensor(action_high, dtype=torch.float32)
        if action_low.ndim == 0:
            action_low = action_low.repeat(action_dim)
        if action_high.ndim == 0:
            action_high = action_high.repeat(action_dim)
        action_scale = (action_high - action_low) / 2.0
        action_bias = (action_high + action_low) / 2.0
        self.register_buffer("action_scale", action_scale)
        self.register_buffer("action_bias", action_bias)

    def forward(self, state):
        h = self.net(state)
        mean = self.mean(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mean, log_std

    def log_prob(self, state, action):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        y = (action - self.action_bias) / self.action_scale
        y = y.clamp(-1.0 + self.eps, 1.0 - self.eps)
        pre_tanh = 0.5 * (torch.log1p(y) - torch.log1p(-y))

        normal_log_prob = -0.5 * (((pre_tanh - mean) / std).pow(2) + 2.0 * log_std + math.log(2.0 * math.pi))
        normal_log_prob = normal_log_prob.sum(dim=-1)

        tanh_correction = torch.log(1.0 - y.pow(2) + self.eps).sum(dim=-1)
        scale_correction = torch.log(self.action_scale + self.eps).sum()
        return normal_log_prob - tanh_correction - scale_correction

    def nll(self, state, action):
        return -self.log_prob(state, action).mean()
