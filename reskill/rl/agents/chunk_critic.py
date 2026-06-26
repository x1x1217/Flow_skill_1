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
    def __init__(
        self,
        state_dim,
        latent_dim,
        hidden_dim,
        hidden_layers=2,
        use_layer_norm=False,
        activation="relu",
    ):
        super().__init__()
        input_dim = state_dim + latent_dim

        def build_q():
            layers = []
            prev_dim = input_dim
            activation_cls = nn.Tanh if activation == "tanh" else nn.ReLU
            for _ in range(hidden_layers):
                layers.append(nn.Linear(prev_dim, hidden_dim))
                if use_layer_norm:
                    layers.append(nn.LayerNorm(hidden_dim))
                layers.append(activation_cls())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            return nn.Sequential(*layers)

        self.q1 = build_q()
        self.q2 = build_q()
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

    def sample(self, batch_size, positive_ratio=0.0, positive_reward_threshold=0.0):
        positive_ratio = float(np.clip(positive_ratio, 0.0, 1.0))
        if positive_ratio > 0.0:
            num_positive = int(batch_size * positive_ratio)
            valid_idx = np.arange(self.size)
            positive_idx = valid_idx[self.reward[: self.size, 0] > positive_reward_threshold]
            if num_positive > 0 and len(positive_idx) > 0:
                pos_idx = np.random.choice(
                    positive_idx,
                    size=num_positive,
                    replace=len(positive_idx) < num_positive,
                )
                rand_idx = np.random.randint(0, self.size, size=batch_size - num_positive)
                idx = np.concatenate([pos_idx, rand_idx])
                np.random.shuffle(idx)
            else:
                idx = np.random.randint(0, self.size, size=batch_size)
        else:
            idx = np.random.randint(0, self.size, size=batch_size)

        positive_fraction = float(np.mean(self.reward[idx, 0] > positive_reward_threshold))
        return AttrDict(
            state=torch.as_tensor(self.state[idx], dtype=torch.float32, device=device),
            latent=torch.as_tensor(self.latent[idx], dtype=torch.float32, device=device),
            reward=torch.as_tensor(self.reward[idx], dtype=torch.float32, device=device),
            next_state=torch.as_tensor(self.next_state[idx], dtype=torch.float32, device=device),
            done=torch.as_tensor(self.done[idx], dtype=torch.float32, device=device),
            positive_fraction=positive_fraction,
        )

    def __len__(self):
        return self.size


class LatentChunkCritic:
    def __init__(
        self,
        state_dim,
        latent_dim,
        hidden_dim,
        num_ensembles,
        lr,
        gamma,
        tau,
        seq_len,
        hidden_layers=2,
        use_layer_norm=False,
        activation="relu",
    ):
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        self.num_ensembles = num_ensembles
        self.gamma = gamma
        self.chunk_discount = gamma ** seq_len
        self.tau = tau

        self.critics = [
            TwinQNetwork(
                state_dim,
                latent_dim,
                hidden_dim,
                hidden_layers=hidden_layers,
                use_layer_norm=use_layer_norm,
                activation=activation,
            ).to(device)
            for _ in range(num_ensembles)
        ]
        self.target_critics = [
            TwinQNetwork(
                state_dim,
                latent_dim,
                hidden_dim,
                hidden_layers=hidden_layers,
                use_layer_norm=use_layer_norm,
                activation=activation,
            ).to(device)
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

    def update_with_flow_policy(self, replay_buffer, skill_agent, skill_prior, args, condition_prior=None):
        if len(replay_buffer) < args.chunk_critic_batch_size:
            return AttrDict(
                q_loss=0.0,
                q1_loss=0.0,
                q2_loss=0.0,
                target_q=0.0,
                current_q=0.0,
                updates=0,
                positive_fraction=0.0,
            )

        losses = []
        q1_losses = []
        q2_losses = []
        target_qs = []
        current_qs = []
        positive_fractions = []

        for _ in range(args.chunk_critic_updates_per_epoch):
            batch = replay_buffer.sample(
                args.chunk_critic_batch_size,
                positive_ratio=getattr(args, "positive_replay_ratio", 0.0),
                positive_reward_threshold=getattr(args, "positive_reward_threshold", 0.0),
            )
            positive_fractions.append(batch.positive_fraction)
            target_guidance_scale = args.guidance_scale if getattr(args, "chunk_critic_update_steps", 0) > 0 else 0.0
            with torch.no_grad():
                if getattr(args, "use_condition_flow", 0) == 1:
                    if condition_prior is None:
                        raise RuntimeError("condition_prior is required for Q_z target with condition flow.")
                    next_n = condition_prior.sample_z_torch(batch.next_state).detach()
                else:
                    next_n_np = skill_agent.ac.act_deterministic(batch.next_state)
                    next_n = torch.as_tensor(next_n_np, dtype=torch.float32, device=device)
                next_cond = torch.cat((batch.next_state, next_n), dim=1)
            if target_guidance_scale > 0.0:
                with torch.enable_grad():
                    next_z = skill_prior.sample_z_guided_torch(
                        next_cond,
                        q_fn=lambda obs_latent: self.min_q(
                            obs_latent[:, : self.state_dim],
                            obs_latent[:, self.state_dim :],
                            use_target=True,
                        ),
                        n_obs=args.n_obs,
                        guidance_scale=target_guidance_scale,
                        grad_clip=args.guidance_grad_clip,
                        guidance_normalize=getattr(args, "guidance_normalize", False),
                    ).detach()
            else:
                with torch.no_grad():
                    next_z = skill_prior.sample_z_torch(next_cond).detach()
            with torch.no_grad():
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
            args.chunk_critic_update_steps = getattr(args, "chunk_critic_update_steps", 0) + 1

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
            positive_fraction=float(np.mean(positive_fractions)),
        )

    def update_with_condition_policy(self, replay_buffer, condition_prior, args):
        batch_size = getattr(args, "condition_critic_batch_size", args.chunk_critic_batch_size)
        updates_per_epoch = getattr(args, "condition_critic_updates_per_epoch", args.chunk_critic_updates_per_epoch)
        positive_ratio = getattr(args, "condition_positive_replay_ratio", getattr(args, "positive_replay_ratio", 0.0))
        positive_reward_threshold = getattr(
            args,
            "condition_positive_reward_threshold",
            getattr(args, "positive_reward_threshold", 0.0),
        )
        if len(replay_buffer) < batch_size:
            return AttrDict(
                q_loss=0.0,
                q1_loss=0.0,
                q2_loss=0.0,
                target_q=0.0,
                current_q=0.0,
                updates=0,
                positive_fraction=0.0,
            )

        losses = []
        q1_losses = []
        q2_losses = []
        target_qs = []
        current_qs = []
        positive_fractions = []

        for _ in range(updates_per_epoch):
            batch = replay_buffer.sample(
                batch_size,
                positive_ratio=positive_ratio,
                positive_reward_threshold=positive_reward_threshold,
            )
            positive_fractions.append(batch.positive_fraction)
            target_guidance_scale = (
                args.condition_guidance_scale
                if getattr(args, "condition_critic_update_steps", 0) > 0
                else 0.0
            )
            if target_guidance_scale > 0.0:
                with torch.enable_grad():
                    next_latent = condition_prior.sample_z_guided_torch(
                        batch.next_state,
                        q_fn=lambda obs_latent: self.min_q(
                            obs_latent[:, : self.state_dim],
                            obs_latent[:, self.state_dim :],
                            use_target=True,
                        ),
                        n_obs=args.n_obs,
                        guidance_scale=target_guidance_scale,
                        grad_clip=args.condition_guidance_grad_clip,
                        guidance_normalize=getattr(args, "condition_guidance_normalize", False),
                    ).detach()
            else:
                with torch.no_grad():
                    next_latent = condition_prior.sample_z_torch(batch.next_state).detach()

            with torch.no_grad():
                target_min = self.min_q(batch.next_state, next_latent, use_target=True)
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
            args.condition_critic_update_steps = getattr(args, "condition_critic_update_steps", 0) + 1

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
            updates=updates_per_epoch,
            positive_fraction=float(np.mean(positive_fractions)),
        )
