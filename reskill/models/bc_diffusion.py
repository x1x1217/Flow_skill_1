# Copyright 2022 Twitter, Inc and Zhendong Wang.
# SPDX-License-Identifier: Apache-2.0

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from reskill.models.diffusion import Diffusion
from reskill.models.model import MLP


class Diffusion_BC(object):
    def __init__(self,
                 state_dim,
                 action_dim,
                 max_action,
                 device,
                 beta_schedule='linear',
                 n_timesteps=100,
                 lr=2e-4,
                 ddim_steps=20,
                 use_sigma=False,
                 ):

        self.model = MLP(state_dim=state_dim, action_dim=action_dim, device=device)
        self.actor = Diffusion(state_dim=state_dim, action_dim=action_dim, model=self.model, max_action=max_action,
                               beta_schedule=beta_schedule, n_timesteps=n_timesteps, ddim_steps=ddim_steps, use_sigma=use_sigma
                               ).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.ddim_steps = ddim_steps

        self.max_action = max_action
        self.action_dim = action_dim
        self.device = device

    def train(self, state, action, iterations, batch_size=100, log_writer=None):

        metric = {'bc_loss': [], 'ql_loss': [], 'actor_loss': [], 'critic_loss': []}
        for _ in range(iterations):
            # Sample replay buffer / batch

            loss = self.actor.loss(action, state)

            self.actor_optimizer.zero_grad()
            loss.backward()
            self.actor_optimizer.step()

            metric['bc_loss'].append(loss.item())

        return metric

    def sample_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action.cpu().data.numpy()
        #return action.cpu().data.numpy().flatten()
    
    def sample_action_ddim(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample_ddim(state)
        return action.cpu().data.numpy()
        #return action.cpu().data.numpy().flatten()

    def sample_action_torch(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample(state)
        return action
        #return action.cpu().data.numpy().flatten()
    
    def sample_action_torch_ddim(self, state):
        #state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample_ddim(state)
        return action
        #return action.cpu().data.numpy().flatten()
    
    def sample_action_guide_repeat(self, state, cls, n_obs, repeat_time):
        state = state.repeat([repeat_time, 1])
        with torch.no_grad():
            action = self.actor.sample_grad(state, cls, n_obs)
        return action
    
    def sample_action_ddim_guide_repeat(self, state, cls, n_obs, repeat_time):
        state = state.repeat([repeat_time, 1])
        with torch.no_grad():
            action = self.actor.sample_grad_ddim(state, cls, n_obs)
        return action

    def sample_determin_action(self, state, noise):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        with torch.no_grad():
            action = self.actor.sample_with_noise(state, noise)
        return action.cpu().data.numpy()

    def save_model(self, dir, id=None):
        if id is not None:
            torch.save(self.actor.state_dict(), f'{dir}/actor_{id}.pth')
        else:
            torch.save(self.actor.state_dict(), f'{dir}/actor.pth')

    def load_model(self, dir, id=None):
        if id is not None:
            self.actor.load_state_dict(torch.load(f'{dir}/actor_{id}.pth'))
        else:
            self.actor.load_state_dict(torch.load(f'{dir}/actor.pth'))

