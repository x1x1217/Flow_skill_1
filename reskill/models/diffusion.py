# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from models.helpers import (cosine_beta_schedule,
                            linear_beta_schedule,
                            vp_beta_schedule,
                            extract,
                            Losses)
from utils.utils import Progress, Silent


class Diffusion(nn.Module):
    def __init__(self, state_dim, action_dim, model, max_action,
                 beta_schedule='linear', n_timesteps=100,
                 loss_type='l2', clip_denoised=True, predict_epsilon=True, ddim_steps=20, use_sigma=False):
        super(Diffusion, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.model = model
        self.use_sigma = use_sigma

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(n_timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(n_timesteps)
        elif beta_schedule == 'vp':
            betas = vp_beta_schedule(n_timesteps)

        alphas = 1. - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])

        self.n_timesteps = int(n_timesteps)
        self.ddim_steps = ddim_steps
        self.skip = self.n_timesteps // self.ddim_steps
        self.seq = list(range(0, self.n_timesteps, self.skip))
        if self.seq[-1] != self.n_timesteps-1:
            self.seq.append(self.n_timesteps-1)
        self.seq = list(reversed(self.seq))

        self.clip_denoised = clip_denoised
        self.predict_epsilon = predict_epsilon

        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

        ## log calculation clipped because the posterior variance
        ## is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped',
                             torch.log(torch.clamp(posterior_variance, min=1e-20)))
        self.register_buffer('posterior_mean_coef1',
                             betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        self.register_buffer('posterior_mean_coef2',
                             (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))

        self.loss_fn = Losses[loss_type]()

    # ------------------------------------------ sampling ------------------------------------------#

    def predict_start_from_noise(self, x_t, t, noise):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise
        
    def predict_start_from_noise_grad(self, x_t, t, noise, grad):
        '''
            if self.predict_epsilon, model output is (scaled) noise;
            otherwise, model predicts x0 directly
        '''
        if self.predict_epsilon:
            return (
                    extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                    extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
            )
        else:
            return noise

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped
    
    def q_posterior_ddim(self, x_start, x_t, t_i, t_j, noise, use_sigma=False):  #t_j > t_i
        nta = 0.01
        if use_sigma:
            #sigma = extract(self.sqrt_one_minus_alphas_cumprod, t_i, x_start.shape) / extract(self.sqrt_one_minus_alphas_cumprod, t_j, x_start.shape)
            #sigma *= torch.sqrt(1 - extract(self.alphas, t_j, x_start.shape) / extract(self.alphas, t_i, x_start.shape))
            sigma = nta * extract(self.posterior_variance, t_j, x_start.shape)
        else:
            sigma = torch.zeros([x_start.shape[0]]).to(self.model.device)
        posterior_variance = sigma
        posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        posterior_mean = (
            (extract(self.sqrt_alphas_cumprod, t_i, x_start.shape) / extract(self.sqrt_alphas_cumprod, t_j, x_start.shape)) * x_t - 
            (extract(self.sqrt_one_minus_alphas_cumprod, t_j, x_start.shape) / extract(self.sqrt_alphas_cumprod, t_j, x_start.shape)) * noise +
            (torch.sqrt(1 - extract(self.alphas_cumprod, t_i, x_start.shape) - sigma) * noise)
        )
        #posterior_mean = (
        #        extract(self.sqrt_alphas_cumprod, t_i, x_start.shape) * x_start +
        #        torch.sqrt(1 - extract(self.alphas_cumprod, t_i, x_start.shape) - posterior_log_variance_clipped.exp()) * noise
        #)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, s):
        x_recon = self.predict_start_from_noise(x, t=t, noise=self.model(x, t, s))

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    def p_mean_variance_grad(self, x, t, s, grad):
        x_recon = self.predict_start_from_noise_grad(x, t=t, noise=self.model(x, t, s)-
                                                     extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape) * 3 * grad, grad=grad)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_ddim(self, x, t_i, t_j, s, use_sigma=False):
        noise = self.model(x, t_j, s)
        x_recon = self.predict_start_from_noise(x, t=t_j, noise=noise)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_ddim(x_start=x_recon, x_t=x, t_i=t_i, t_j=t_j, noise=noise, use_sigma=use_sigma)
        return model_mean, posterior_variance, posterior_log_variance
    
    def p_mean_variance_grad_ddim(self, x, t_i, t_j, s, grad, use_sigma=False):
        noise = self.model(x, t_j, s) - extract(self.sqrt_one_minus_alphas_cumprod, t_j, x.shape) * 3 * grad
        x_recon = self.predict_start_from_noise_grad(x, t=t_j, noise=noise, grad=grad)

        if self.clip_denoised:
            x_recon.clamp_(-self.max_action, self.max_action)
        else:
            assert RuntimeError()

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior_ddim(x_start=x_recon, x_t=x, t_i=t_i, t_j=t_j, noise=noise, use_sigma=use_sigma)
        return model_mean, posterior_variance, posterior_log_variance

    # @torch.no_grad()
    def p_sample(self, x, t, s):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, s=s)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    # @torch.no_grad()
    def p_sample_ddim(self, x, t_i, t_j, s, use_sigma=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_ddim(x=x, t_i=t_i, t_j=t_j, s=s, use_sigma=use_sigma)
        if not use_sigma:
            return model_mean
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t_j == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

     # @torch.no_grad()
    def p_sample_grad(self, x, t, s, grad):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_grad(x=x, t=t, s=s, grad=grad)
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

 # @torch.no_grad()
    def p_sample_grad_ddim(self, x, t_i, t_j, s, grad, use_sigma=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance_grad_ddim(x=x, t_i=t_i, t_j=t_j, s=s, grad=grad, use_sigma=use_sigma)
        if not use_sigma:
            return model_mean
        noise = torch.randn_like(x)
        # no noise when t == 0
        nonzero_mask = (1 - (t_j == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # @torch.no_grad()
    def p_sample_loop(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
    # @torch.no_grad()
    def p_sample_loop_ddim(self, state, shape, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in range(len(self.seq)):
            if self.seq[i] == 0:
                timesteps = torch.full((batch_size,), self.seq[i], device=device, dtype=torch.long)
                x = self.p_sample(x, timesteps, state)
            else:
                timesteps_i = torch.full((batch_size,), self.seq[i+1], device=device, dtype=torch.long)
                timesteps_j = torch.full((batch_size,), self.seq[i], device=device, dtype=torch.long)
                x = self.p_sample_ddim(x, timesteps_i, timesteps_j, state, self.use_sigma)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
        # @torch.no_grad()
    def p_sample_loop_grad(self, state, shape, cls, n_obs, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in reversed(range(0, self.n_timesteps)):
            def cond_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor): 
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = cls(torch.concat([state[:, :n_obs], x_in], dim=1)).reshape(-1, 1)
                    log_probs = logits
                    selected = log_probs[range(len(logits)), 0]
                    return torch.autograd.grad(selected.sum(), x_in)[0].float()   # gradient descend
                    
            grad = cond_fn(x, i, y=1) 
            timesteps = torch.full((batch_size,), i, device=device, dtype=torch.long)
            x = self.p_sample_grad(x, timesteps, state, grad)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x
        
        # @torch.no_grad()
    def p_sample_loop_grad_ddim(self, state, shape, cls, n_obs, verbose=False, return_diffusion=False):
        device = self.betas.device

        batch_size = shape[0]
        x = torch.randn(shape, device=device)

        if return_diffusion: diffusion = [x]

        progress = Progress(self.n_timesteps) if verbose else Silent()
        for i in range(len(self.seq)):
            def cond_fn(x: torch.Tensor, t: torch.Tensor, y: torch.Tensor): 
                with torch.enable_grad():
                    x_in = x.detach().requires_grad_(True)
                    logits = cls(torch.concat([state[:, :n_obs], x_in], dim=1)).reshape(-1, 1)
                    log_probs = logits
                    selected = log_probs[range(len(logits)), 0]
                    return torch.autograd.grad(selected.sum(), x_in)[0].float()   # gradient descend
                    
            grad = cond_fn(x, i, y=1) 
            if self.seq[i] == 0:
                timesteps = torch.full((batch_size,), self.seq[i], device=device, dtype=torch.long)
                x = self.p_sample_grad(x, timesteps, state, grad)
            else:
                timesteps_i = torch.full((batch_size,), self.seq[i+1], device=device, dtype=torch.long)
                timesteps_j = torch.full((batch_size,), self.seq[i], device=device, dtype=torch.long)
                x = self.p_sample_grad_ddim(x, timesteps_i, timesteps_j, state, grad, self.use_sigma)

            progress.update({'t': i})

            if return_diffusion: diffusion.append(x)

        progress.close()

        if return_diffusion:
            return x, torch.stack(diffusion, dim=1)
        else:
            return x

    # @torch.no_grad()
    def sample(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)
    
    def sample_ddim(self, state, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop_ddim(state, shape, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)
    
    # @torch.no_grad()
    def sample_grad(self, state, cls, n_obs, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop_grad(state, shape, cls, n_obs, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)
    
    # @torch.no_grad()
    def sample_grad_ddim(self, state, cls, n_obs, *args, **kwargs):
        batch_size = state.shape[0]
        shape = (batch_size, self.action_dim)
        action = self.p_sample_loop_grad_ddim(state, shape, cls, n_obs, *args, **kwargs)
        return action.clamp_(-self.max_action, self.max_action)
    
    def sample_with_noise(self, state, noise):
        device = self.betas.device
        x = noise.to(device)

        progress = Progress(self.n_timesteps)
        for i in reversed(range(0, self.n_timesteps)):
            timesteps = torch.full((1,), i, device=device, dtype=torch.long)
            x = self.p_sample(x, timesteps, state)

            progress.update({'t': i})

        progress.close()
        return x

    # ------------------------------------------ training ------------------------------------------#

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sample = (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

        return sample

    def p_losses(self, x_start, state, t, weights=1.0):
        noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        x_recon = self.model(x_noisy, t, state)

        assert noise.shape == x_recon.shape

        if self.predict_epsilon:
            loss = self.loss_fn(x_recon, noise, weights)
        else:
            loss = self.loss_fn(x_recon, x_start, weights)

        return loss

    def loss(self, x, state, weights=1.0):
        batch_size = len(x)
        t = torch.randint(0, self.n_timesteps, (batch_size,), device=x.device).long()
        return self.p_losses(x, state, t, weights)

    def forward(self, state, *args, **kwargs):
        return self.sample(state, *args, **kwargs)

