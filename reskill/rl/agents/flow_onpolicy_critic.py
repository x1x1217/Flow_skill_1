import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam

import reskill.rl.agents.ppo_core as core
from reskill.rl.utils.mpi_pytorch import sync_params, mpi_avg_grads
from reskill.rl.utils.mpi_tools import proc_id
from reskill.utils.general_utils import AttrDict


device = torch.device("cuda")


class FlowOnPolicyBuffer:
    """On-policy skill buffer with the same path/return lifecycle as PPOBuffer."""

    def __init__(self, obs_dim, condition_dim, z_dim, size, gamma=0.99):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.condition_buf = np.zeros(core.combined_shape(size, condition_dim), dtype=np.float32)
        self.z_buf = np.zeros(core.combined_shape(size, z_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.gamma = gamma
        self.ptr = 0
        self.path_start_idx = 0
        self.max_size = size

    def store(self, obs, condition, z, reward, value):
        assert self.ptr < self.max_size
        self.obs_buf[self.ptr] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.condition_buf[self.ptr] = np.asarray(condition, dtype=np.float32).reshape(-1)
        self.z_buf[self.ptr] = np.asarray(z, dtype=np.float32).reshape(-1)
        self.rew_buf[self.ptr] = reward
        self.val_buf[self.ptr] = float(np.asarray(value).reshape(-1)[0])
        self.ptr += 1

    def finish_path(self, last_val=0.0):
        path_slice = slice(self.path_start_idx, self.ptr)
        if self.path_start_idx == self.ptr:
            return
        last_val = float(np.asarray(last_val).reshape(-1)[0])
        rewards = np.append(self.rew_buf[path_slice], last_val)
        self.ret_buf[path_slice] = core.discount_cumsum(rewards, self.gamma)[:-1]
        self.path_start_idx = self.ptr

    def get(self):
        if self.ptr != self.max_size:
            raise RuntimeError(
                f"FlowOnPolicyBuffer must be full before get(): ptr={self.ptr}, "
                f"max_size={self.max_size}"
            )
        if self.path_start_idx != self.ptr:
            raise RuntimeError(
                "FlowOnPolicyBuffer has an unfinished trajectory before get(): "
                f"path_start_idx={self.path_start_idx}, ptr={self.ptr}"
            )

        size = self.ptr
        positive_fraction = float(np.mean(self.rew_buf[:size] > 0.0))
        data = dict(
            obs=self.obs_buf[:size],
            condition=self.condition_buf[:size],
            z=self.z_buf[:size],
            ret=self.ret_buf[:size],
            value=self.val_buf[:size],
        )
        self.ptr = 0
        self.path_start_idx = 0

        tensors = {
            key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in data.items()
        }
        return AttrDict(
            **tensors,
            size=size,
            positive_fraction=positive_fraction,
        )

    def __len__(self):
        return self.ptr


class FlowOnPolicyCritic(nn.Module):
    """PPO-style full-batch return critics for condition and skill guidance."""

    def __init__(
        self,
        obs_dim,
        condition_dim,
        z_dim,
        hidden_sizes,
        lr,
        train_v_iters,
        seed,
    ):
        super().__init__()
        if train_v_iters <= 0:
            raise ValueError("train_v_iters must be positive")
        process_seed = seed + 10000 * proc_id()
        torch.manual_seed(process_seed)
        np.random.seed(process_seed)

        self.v = core.MLPCritic(obs_dim, hidden_sizes, nn.Tanh)
        self.qc = core.MLPCritic(obs_dim + condition_dim, hidden_sizes, nn.Tanh)
        self.qz = core.MLPCritic(obs_dim + z_dim, hidden_sizes, nn.Tanh)
        self.to(device)
        sync_params(self)

        self.v_optimizer = Adam(self.v.parameters(), lr=lr)
        self.qc_optimizer = Adam(self.qc.parameters(), lr=lr)
        self.qz_optimizer = Adam(self.qz.parameters(), lr=lr)
        self.train_v_iters = train_v_iters
        self.obs_dim = obs_dim

    def value(self, obs):
        with torch.no_grad():
            return self.v(obs).cpu().numpy()

    def qc_fn_from_obs_latent(self, obs_latent):
        return self.qc(obs_latent).reshape(-1, 1)

    def qz_fn_from_obs_latent(self, obs_latent):
        return self.qz(obs_latent).reshape(-1, 1)

    def compute_loss_v(self, data):
        return ((self.v(data.obs) - data.ret) ** 2).mean()

    def compute_loss_qc(self, data):
        inputs = torch.cat((data.obs, data.condition), dim=1)
        return ((self.qc(inputs) - data.ret) ** 2).mean()

    def compute_loss_qz(self, data):
        inputs = torch.cat((data.obs, data.z), dim=1)
        return ((self.qz(inputs) - data.ret) ** 2).mean()

    def update(self, buffer):
        data = buffer.get()
        v_loss_old = self.compute_loss_v(data).item()
        qc_loss_old = self.compute_loss_qc(data).item()
        qz_loss_old = self.compute_loss_qz(data).item()

        for _ in range(self.train_v_iters):
            self.v_optimizer.zero_grad()
            v_loss = self.compute_loss_v(data)
            v_loss.backward()
            mpi_avg_grads(self.v)
            self.v_optimizer.step()

            self.qc_optimizer.zero_grad()
            qc_loss = self.compute_loss_qc(data)
            qc_loss.backward()
            mpi_avg_grads(self.qc)
            self.qc_optimizer.step()

            self.qz_optimizer.zero_grad()
            qz_loss = self.compute_loss_qz(data)
            qz_loss.backward()
            mpi_avg_grads(self.qz)
            self.qz_optimizer.step()

        with torch.no_grad():
            value_mean = self.v(data.obs).mean().item()
            qc_mean = self.qc(torch.cat((data.obs, data.condition), dim=1)).mean().item()
            qz_mean = self.qz(torch.cat((data.obs, data.z), dim=1)).mean().item()

        return AttrDict(
            v_loss=v_loss_old,
            qc_loss=qc_loss_old,
            qz_loss=qz_loss_old,
            delta_v_loss=v_loss.item() - v_loss_old,
            delta_qc_loss=qc_loss.item() - qc_loss_old,
            delta_qz_loss=qz_loss.item() - qz_loss_old,
            return_mean=data.ret.mean().item(),
            value_mean=value_mean,
            qc_mean=qc_mean,
            qz_mean=qz_mean,
            positive_fraction=data.positive_fraction,
            data_size=data.size,
            updates=self.train_v_iters,
        )
