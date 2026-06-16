import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from reskill.utils.general_utils import AttrDict


device = torch.device('cuda')


def weights_init_(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight, gain=1)
        nn.init.constant_(module.bias, 0)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class TwinQNetwork(nn.Module):
    def __init__(self, state_dim, latent_dim, hidden_dim):
        super().__init__()
        input_dim = state_dim + latent_dim
        self.q1 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.apply(weights_init_)

    def forward(self, state, latent):
        x = torch.cat([state, latent], dim=1)
        return self.q1(x), self.q2(x)


class ChunkReplayBuffer:
    def __init__(self, state_dim, latent_dim, capacity, seed):
        random.seed(seed)
        self.capacity = capacity
        self.position = 0
        self.size = 0
        self.state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.latent = np.zeros((capacity, latent_dim), dtype=np.float32)
        self.reward = np.zeros((capacity, 1), dtype=np.float32)
        self.next_state = np.zeros((capacity, state_dim), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=np.float32)

    def push(self, state, latent, reward, next_state, done):
        self.state[self.position] = state
        self.latent[self.position] = latent
        self.reward[self.position] = reward
        self.next_state[self.position] = next_state
        self.done[self.position] = done
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, size=batch_size)
        return AttrDict(
            state=torch.as_tensor(self.state[idx], dtype=torch.float32, device=device),
            latent=torch.as_tensor(self.latent[idx], dtype=torch.float32, device=device),
            reward=torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            next_state=torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device),
            done=torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
        )

    def __len__(self):
        return self.size


class LatentChunkCritic:
    def __init__(self, state_dim, latent_dim, hidden_dim, num_ensembles, lr, gamma, tau, seq_len):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_ensembles = num_ensembles
        self.gamma = gamma
        self.chunk_discount = gamma ** seq_len
        self.tau = tau

        self.critics = [
            TwinQNetwork(state_dim, latent_dim, hidden_dim).to(device)
            for _ in range(num_ensembles)
        ]
        self.target_critics = [
            TwinQNetwork(state_dim, latent_dim, hidden_dim).to(device)
            for _ in range(num_ensembles)
        ]
        for target_critic, critic in zip(self.target_critics, self.critics):
            hard_update(target_critic, critic)

        params = []
        for critic in self.critics:
            params.extend(list(critic.parameters()))
        self.optimizer = Adam(params, lr=lr)

    def state_dict(self):
        return {
            "critics": [critic.state_dict() for critic in self.critics],
            "target_critics": [critic.state_dict() for critic in self.target_critics],
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "latent_dim": self.latent_dim,
            "num_ensembles": self.num_ensembles,
            "gamma": self.gamma,
            "chunk_discount": self.chunk_discount,
            "tau": self.tau,
        }

    def q_heads(self, state, latent, use_target=False):
        critics = self.target_critics if use_target else self.critics
        heads = []
        for critic in critics:
            q1, q2 = critic(state, latent)
            heads.extend([q1, q2])
        return torch.cat(heads, dim=1)

    def min_q(self, state, latent, use_target=False):
        return self.q_heads(state, latent, use_target=use_target).min(dim=1, keepdim=True).values

    def q_fn_from_obs_latent(self, obs_latent):
        state = obs_latent[:, :self.state_dim]
        latent = obs_latent[:, self.state_dim:]
        return self.min_q(state, latent, use_target=False)

    def update_with_flow_policy(self, replay_buffer, skill_agent, skill_prior, args):
        if len(replay_buffer) < args.chunk_critic_batch_size:
            return AttrDict(
                q_loss=0.0,
                q1_loss=0.0,
                q2_loss=0.0,
                target_q=0.0,
                current_q=0.0,
                updates=0,
            )

        losses = []
        q1_losses = []
        q2_losses = []
        target_qs = []
        current_qs = []

        for _ in range(args.chunk_critic_updates_per_epoch):
            batch = replay_buffer.sample(args.chunk_critic_batch_size)
            with torch.no_grad():
                next_n_np = skill_agent.ac.act_deterministic(batch.next_state)
                next_n = torch.as_tensor(next_n_np, dtype=torch.float32, device=device)
                next_cond = torch.cat((batch.next_state, next_n), dim=1)
                next_z = skill_prior.sample_z_torch(next_cond).detach()
                target_min = self.min_q(batch.next_state, next_z, use_target=True)
                target_q = batch.reward + (1.0 - batch.done) * self.chunk_discount * target_min

            self.optimizer.zero_grad()
            q_loss = 0.0
            q1_loss_value = 0.0
            q2_loss_value = 0.0
            current_q_value = 0.0
            for critic in self.critics:
                q1, q2 = critic(batch.state, batch.latent)
                q1_loss = F.mse_loss(q1, target_q)
                q2_loss = F.mse_loss(q2, target_q)
                q_loss = q_loss + q1_loss + q2_loss
                q1_loss_value += q1_loss.item()
                q2_loss_value += q2_loss.item()
                current_q_value += torch.min(q1, q2).mean().item()
            q_loss.backward()
            self.optimizer.step()

            for target_critic, critic in zip(self.target_critics, self.critics):
                soft_update(target_critic, critic, self.tau)

            losses.append(q_loss.item())
            q1_losses.append(q1_loss_value / self.num_ensembles)
            q2_losses.append(q2_loss_value / self.num_ensembles)
            target_qs.append(target_q.mean().item())
            current_qs.append(current_q_value / self.num_ensembles)

        return AttrDict(
            q_loss=float(np.mean(losses)),
            q1_loss=float(np.mean(q1_losses)),
            q2_loss=float(np.mean(q2_losses)),
            target_q=float(np.mean(target_qs)),
            current_q=float(np.mean(current_qs)),
            updates=args.chunk_critic_updates_per_epoch,
        )

    def update_with_condition_policy(self, replay_buffer, condition_prior, args):
        if len(replay_buffer) < args.condition_critic_batch_size:
            return AttrDict(
                q_loss=0.0,
                q1_loss=0.0,
                q2_loss=0.0,
                target_q=0.0,
                current_q=0.0,
                updates=0,
            )

        losses = []
        q1_losses = []
        q2_losses = []
        target_qs = []
        current_qs = []

        for _ in range(args.condition_critic_updates_per_epoch):
            batch = replay_buffer.sample(args.condition_critic_batch_size)
            with torch.no_grad():
                next_c = condition_prior.sample_z_torch(batch.next_state).detach()
                target_min = self.min_q(batch.next_state, next_c, use_target=True)
                target_q = batch.reward + (1.0 - batch.done) * self.chunk_discount * target_min

            self.optimizer.zero_grad()
            q_loss = 0.0
            q1_loss_value = 0.0
            q2_loss_value = 0.0
            current_q_value = 0.0
            for critic in self.critics:
                q1, q2 = critic(batch.state, batch.latent)
                q1_loss = F.mse_loss(q1, target_q)
                q2_loss = F.mse_loss(q2, target_q)
                q_loss = q_loss + q1_loss + q2_loss
                q1_loss_value += q1_loss.item()
                q2_loss_value += q2_loss.item()
                current_q_value += torch.min(q1, q2).mean().item()
            q_loss.backward()
            self.optimizer.step()

            for target_critic, critic in zip(self.target_critics, self.critics):
                soft_update(target_critic, critic, self.tau)

            losses.append(q_loss.item())
            q1_losses.append(q1_loss_value / self.num_ensembles)
            q2_losses.append(q2_loss_value / self.num_ensembles)
            target_qs.append(target_q.mean().item())
            current_qs.append(current_q_value / self.num_ensembles)

        return AttrDict(
            q_loss=float(np.mean(losses)),
            q1_loss=float(np.mean(q1_losses)),
            q2_loss=float(np.mean(q2_losses)),
            target_q=float(np.mean(target_qs)),
            current_q=float(np.mean(current_qs)),
            updates=args.condition_critic_updates_per_epoch,
        )
