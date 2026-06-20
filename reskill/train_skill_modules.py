
import torch
import torch.optim as optim
import argparse
from typing import List
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
import pdb
from tqdm import tqdm
import os
import time
import yaml
import numpy as np
import json

from reskill.models.skill_vae import SkillVAE
from reskill.data.skill_dataloader import SkillsDataset
from reskill.models.cvae import CVAE
from reskill.models.normal_mlp import NMLP
from reskill.models.rnvp import stacked_NVP
from reskill.models.bc_diffusion import Diffusion_BC
from reskill.utils.general_utils import AttrDict
from reskill.models.bc_flow import Flow_BC
from reskill.models.tanh_gaussian_policy import TanhGaussianBehaviorPolicy
from reskill.models.flow_prior import compute_flow_loss
from reskill.utils.swanlab_writer import SwanLabWriter


class ModelTrainer():
    def __init__(
        self,
        dataset_name,
        config_file,
        prior_model,
        seed,
        writer,
        use_student=True,
        skill_epochs=None,
        prior_epochs=None,
        prior_updates_per_batch=1,
        prior_use_mu=True,
        val_freq=5,
        save_freq=50,
        action_noise_std=0.0,
        condition_reweight=False,
        behavior_policy_epochs=20,
        behavior_policy_lr=3e-4,
        behavior_policy_hidden_dim=256,
        condition_weight_beta=0.2,
        condition_weight_min=0.2,
        condition_weight_max=3.0,
        condition_raw_log_weight_clip_quantile=0.99,
    ):
        self.dataset_name = dataset_name
        self.prior_model = prior_model
        self.seed = seed
        self.use_student = use_student
        self.skill_epochs = skill_epochs
        self.prior_epochs = prior_epochs
        self.prior_updates_per_batch = prior_updates_per_batch
        self.prior_use_mu = prior_use_mu
        self.val_freq = val_freq
        self.save_freq = save_freq
        self.action_noise_std = action_noise_std
        self.condition_reweight = condition_reweight
        self.behavior_policy_epochs = behavior_policy_epochs
        self.behavior_policy_lr = behavior_policy_lr
        self.behavior_policy_hidden_dim = behavior_policy_hidden_dim
        self.condition_weight_beta = condition_weight_beta
        self.condition_weight_min = condition_weight_min
        self.condition_weight_max = condition_weight_max
        self.condition_raw_log_weight_clip_quantile = condition_raw_log_weight_clip_quantile
        curr_dir = os.path.dirname(__file__)
        prior_dir_name = prior_model
        if prior_model == 'Flow':
            prior_dir_name = f"{prior_model}_student{int(use_student)}"
        self.save_dir = os.path.join(
            curr_dir,
            "results",
            "saved_skill_models",
            dataset_name,
            self.prior_model,
            f"seed_{seed}",
            f"skill_prior_{prior_dir_name}",
        )
        os.makedirs(self.save_dir, exist_ok=True)
        self.vae_save_path = self.save_dir + "/skill_vae.pth"
        self.sp_save_path = self.save_dir + "/skill_prior.pth"
        self.condition_prior_save_path = self.save_dir + "/condition_prior.pth"
        self.best_vae_save_path = self.save_dir + "/best_skill_vae.pth"
        self.best_sp_save_path = self.save_dir + "/best_skill_prior.pth"
        self.best_condition_prior_save_path = self.save_dir + "/best_condition_prior.pth"
        self.behavior_policy_save_path = self.save_dir + "/behavior_policy.pth"
        self.condition_weight_stats_path = self.save_dir + "/condition_weight_stats.json"
        self.best_vae_val_loss = float("inf")
        self.best_prior_val_loss = float("inf")
        self.best_condition_val_loss = float("inf")
        self.best_behavior_val_nll = float("inf")
        self.condition_log_mean_weight = None
        self.condition_raw_log_weight_max = None
        
        # config_path = "configs/skill_mdl/" + config_file
        config_path = os.path.join(curr_dir, "configs", "skill_mdl", config_file)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = writer
        print("Device: ", self.device)


        with open(config_path, 'r') as file:
            conf = yaml.safe_load(file)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])        

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])          
        train_data = SkillsDataset(dataset_name, phase="train", subseq_len=conf.skill_vae.subseq_len, transform=transform)
        val_data = SkillsDataset(dataset_name, phase="val", subseq_len=conf.skill_vae.subseq_len, transform=transform)

        self.train_loader = DataLoader(
            train_data,
            batch_size = conf.skill_vae.batch_size,
            shuffle = True,
            drop_last=True,
            prefetch_factor=30,
            num_workers=conf.loader.num_workers,
            pin_memory=True)

        self.val_loader = DataLoader(
            val_data,
            batch_size = 64,
            shuffle = False,
            drop_last=True,
            prefetch_factor=30,
            num_workers=conf.loader.num_workers,
            pin_memory=True)

        self.skill_vae = SkillVAE(n_actions=conf.skill_vae.n_actions, n_obs=conf.skill_vae.n_obs, n_hidden=conf.skill_vae.n_hidden,
                                  seq_length=conf.skill_vae.subseq_len, n_z=conf.skill_vae.n_z, device=self.device).to(self.device)
        
        self.optimizer = optim.Adam(self.skill_vae.parameters(), lr=conf.skill_vae.lr)


        if self.prior_model == 'RNVP':
            self.sp_nvp = stacked_NVP(d=conf.skill_prior_nvp.d, k=conf.skill_prior_nvp.k, n_hidden=conf.skill_prior_nvp.n_hidden,
                                    state_size=conf.skill_vae.n_obs, n=conf.skill_prior_nvp.n_coupling_layers, device=self.device).to(self.device)
            
            self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.sp_optimizer, 0.999)
        
        elif self.prior_model == 'Flow':
            self.sp_nvp = Flow_BC(
                cond_dim=conf.skill_vae.n_obs+conf.skill_vae.n_actions,
                latent_dim=conf.skill_vae.n_z,
                max_action=10,
                device=self.device,
                use_student=self.use_student
            )
            self.condition_prior = Flow_BC(
                cond_dim=conf.skill_vae.n_obs,
                latent_dim=conf.skill_vae.n_actions,
                max_action=1,
                device=self.device,
                use_student=self.use_student
            )
            self.behavior_policy = TanhGaussianBehaviorPolicy(
                state_dim=conf.skill_vae.n_obs,
                action_dim=conf.skill_vae.n_actions,
                hidden_dim=self.behavior_policy_hidden_dim,
                action_low=-1.0,
                action_high=1.0,
            ).to(self.device)
            self.behavior_optimizer = torch.optim.Adam(
                self.behavior_policy.parameters(),
                lr=self.behavior_policy_lr,
            )
        
        elif self.prior_model == 'Diffusion':
            self.sp_nvp = Diffusion_BC(state_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, action_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
        elif self.prior_model == 'MLP':
            self.sp_nvp = NMLP(cond_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, latent_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
            self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        elif self.prior_model == 'CVAE':
            self.sp_nvp = CVAE(z_dim=conf.skill_vae.n_z, cond_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, latent_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
            self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        self.n_epochs = conf.skill_vae.epochs
        if self.skill_epochs is None:
            self.skill_epochs = self.n_epochs
        if self.prior_epochs is None:
            self.prior_epochs = self.n_epochs


    def fit(self, epoch):
        self.skill_vae.train()
        running_loss = 0.0
        for batch_idx, data in enumerate(self.train_loader):
            log_step = epoch * len(self.train_loader) + batch_idx

            data["actions"] = data["actions"].to(self.device)
            data["obs"] = data["obs"].to(self.device)

            # Train skills model
            self.skill_vae.init_hidden(data["actions"].size(0))
            self.optimizer.zero_grad()
            output = self.skill_vae(data)
            losses = self.skill_vae.loss(data, output)
            loss = losses.total_loss
            running_loss += loss.item()
            loss.backward()
            self.optimizer.step()

            # Train skills prior model
            if self.prior_model == 'RNVP':
                self.sp_optimizer.zero_grad()
                sp_input = AttrDict(skill=output.z.detach(),
                                    state=data["obs"][:,0,:])
                z, log_pz, log_jacob = self.sp_nvp(sp_input)
                sp_loss = (-log_pz - log_jacob).mean()
                sp_loss.backward()
                self.sp_optimizer.step()

                if batch_idx % 500 == 0:
                    self.scheduler.step()
                    self.writer.add_scalar('rnvp_prior/lr', self.scheduler.get_lr()[0], log_step)

                if batch_idx % 100 == 0:
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('rnvp_prior/loss', sp_loss.item(), log_step)

            elif self.prior_model == 'Flow':
                skill = output.z.detach()
                state = data["obs"][:, 0, :]
                action = data["actions"][:, 0, :]
                
                action_ori = action
                state_ori = state
                
                for prior_iter in range(100):
                    action = action_ori
                    condtion = torch.cat([state_ori, action], dim=1)
                    
                    metric = self.sp_nvp.train(condtion, skill, iterations=1)
                    condition_metric = self.condition_prior.train(state_ori, action, iterations=1)
                    sp_loss = np.mean(metric['total_loss'])
            
                if batch_idx % 10 == 0:
                    flow_loss = np.mean(metric['flow_loss'])
                    distill_loss = np.mean(metric['distill_loss'])
                    prior_total_loss = np.mean(metric['total_loss'])
                    condition_flow_loss = np.mean(condition_metric['flow_loss'])
                    condition_distill_loss = np.mean(condition_metric['distill_loss'])
                    condition_total_loss = np.mean(condition_metric['total_loss'])
                    print(
                        f"[epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                        f"vae_total={losses.total_loss.item():.4f} "
                        f"vae_bc={losses.bc_loss.item():.4f} "
                        f"vae_kl={losses.kld_loss.item():.4f} "
                        f"flow={flow_loss:.4f} "
                        f"distill={distill_loss:.4f} "
                        f"prior_total={prior_total_loss:.4f} "
                        f"condition_flow={condition_flow_loss:.4f} "
                        f"condition_total={condition_total_loss:.4f}",
                        flush=True,
                    )
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_total_loss', losses.total_loss.item(), log_step)
                    self.writer.add_scalar('flow_prior/flow_loss', flow_loss, log_step)
                    self.writer.add_scalar('flow_prior/distill_loss', distill_loss, log_step)
                    self.writer.add_scalar('flow_prior/total_loss', prior_total_loss, log_step)
                    self.writer.add_scalar('condition_flow/flow_loss', condition_flow_loss, log_step)
                    self.writer.add_scalar('condition_flow/distill_loss', condition_distill_loss, log_step)
                    self.writer.add_scalar('condition_flow/total_loss', condition_total_loss, log_step)
            
            elif self.prior_model == 'Diffusion':
                skill = output.z.detach()
                state = data["obs"][:, 0, :]
                action = data["actions"][:, 0, :]
                action = action / 2.

                action_ori = action
                state_ori = state
                for prior_iter in range(100):
                    action = action_ori + 0.2 * torch.normal(0, 1, action.shape).to(self.device)
                    state = torch.cat([state_ori, action], dim=1)

                    metric = self.sp_nvp.train(state, skill, iterations=1, batch_size=128)
                    sp_loss = np.mean(metric['bc_loss'])

                if batch_idx % 100 == 0:
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('diffusion_prior/bc_loss', sp_loss.item(), log_step)

            elif self.prior_model == 'CVAE':
                skill = output.z.detach()
                state = data["obs"][:, 0, :]
                action = data["actions"][:, 0, :]
                action = action / 2.

                #skill = skill.repeat([20, 1])
                #state = state.repeat([20, 1])
                #action = action.repeat([20, 1])

                action_ori = action
                state_ori = state

                for i in range(100):
                    action = action_ori + 0.2 * torch.normal(0, 1, action.shape).to(self.device)
                    state = torch.cat([state_ori, action], dim=1)

                    bc_loss, kld_loss = self.sp_nvp(skill, state)
                    self.sp_optimizer.zero_grad()
                    loss = bc_loss+kld_loss
                    loss.backward()
                    self.sp_optimizer.step()
                    sp_loss = bc_loss+kld_loss

                if batch_idx % 100 == 0:
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('cvae_prior/total_loss', sp_loss.item(), log_step)

            elif self.prior_model == 'MLP':
                skill = output.z.detach()
                state = data["obs"][:, 0, :]
                action = data["actions"][:, 0, :]
                action = action / 2.

                #skill = skill.repeat([20, 1])
                #state = state.repeat([20, 1])
                #action = action.repeat([20, 1])

                action_ori = action
                state_ori = state

                for i in range(100):
                    action = action_ori + 0.2 * torch.normal(0, 1, action.shape).to(self.device)
                    state = torch.cat([state_ori, action], dim=1)

                    bc_loss = self.sp_nvp(skill, state)
                    self.sp_optimizer.zero_grad()
                    loss = bc_loss
                    loss.backward()
                    self.sp_optimizer.step()
                
                if batch_idx % 100 == 0:
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('mlp_prior/bc_loss', loss.item(), log_step)
            
        train_loss = running_loss / len(self.train_loader)
        return train_loss


    def validate(self):
        self.skill_vae.eval()
        if self.prior_model == 'Flow':
            self.sp_nvp.teacher.eval()
            self.condition_prior.teacher.eval()
            if self.sp_nvp.use_student:
                self.sp_nvp.student.eval()
                self.condition_prior.student.eval()

        vae_bc_losses = []
        vae_kl_losses = []
        vae_total_losses = []
        prior_flow_losses = []
        condition_flow_losses = []
        prior_sample_mses = []
        condition_sample_mses = []

        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                self.skill_vae.init_hidden(data["actions"].size(0))
                output = self.skill_vae(data)
                losses = self.skill_vae.loss(data, output)

                vae_bc_losses.append(losses.bc_loss.item())
                vae_kl_losses.append(losses.kld_loss.item())
                vae_total_losses.append(losses.total_loss.item())

                if self.prior_model == 'Flow':
                    skill = output.z.detach()
                    state = data["obs"][:, 0, :]
                    action = data["actions"][:, 0, :]
                    condition = torch.cat([state, action], dim=1)
                    prior_loss, _ = compute_flow_loss(self.sp_nvp.teacher, condition, skill)
                    condition_loss, _ = compute_flow_loss(self.condition_prior.teacher, state, action)
                    sampled_skill = self.sp_nvp.sample_z_torch(condition)
                    sampled_condition = self.condition_prior.sample_z_torch(state)
                    prior_sample_mse = torch.mean((sampled_skill - skill) ** 2, dim=1).mean()
                    condition_sample_mse = torch.mean((sampled_condition - action) ** 2, dim=1).mean()
                    prior_flow_losses.append(prior_loss.item())
                    condition_flow_losses.append(condition_loss.item())
                    prior_sample_mses.append(prior_sample_mse.item())
                    condition_sample_mses.append(condition_sample_mse.item())

        metrics = AttrDict(
            vae_bc_loss=float(np.mean(vae_bc_losses)),
            vae_kl_loss=float(np.mean(vae_kl_losses)),
            vae_total_loss=float(np.mean(vae_total_losses)),
        )
        if self.prior_model == 'Flow':
            metrics.prior_flow_loss = float(np.mean(prior_flow_losses))
            metrics.condition_flow_loss = float(np.mean(condition_flow_losses))
            metrics.prior_sample_mse = float(np.mean(prior_sample_mses))
            metrics.condition_sample_mse = float(np.mean(condition_sample_mses))
        return metrics


    def train_legacy(self):
        print("Training...") 
        for epoch in tqdm(range(self.n_epochs)):
            print(f"\n[start epoch {epoch:03d}/{self.n_epochs}]", flush=True)
            
            train_epoch_loss = self.fit(epoch)
            self.writer.add_scalar('train_epoch/loss', train_epoch_loss, epoch)

            if epoch % 5 == 0:
                val_metrics = self.validate()
                msg = (
                    f"[end epoch {epoch:03d}] train_loss={train_epoch_loss:.6f} "
                    f"val_vae_bc={val_metrics.vae_bc_loss:.6f} "
                    f"val_vae_total={val_metrics.vae_total_loss:.6f}"
                )
                if self.prior_model == 'Flow':
                    msg += (
                        f" val_prior_flow={val_metrics.prior_flow_loss:.6f} "
                        f"val_prior_mse={val_metrics.prior_sample_mse:.6f} "
                        f"val_condition_flow={val_metrics.condition_flow_loss:.6f} "
                        f"val_condition_mse={val_metrics.condition_sample_mse:.6f}"
                    )
                print(msg, flush=True)

                self.writer.add_scalar('val_epoch/vae_bc_loss', val_metrics.vae_bc_loss, epoch)
                self.writer.add_scalar('val_epoch/vae_kl_loss', val_metrics.vae_kl_loss, epoch)
                self.writer.add_scalar('val_epoch/vae_total_loss', val_metrics.vae_total_loss, epoch)

                if val_metrics.vae_bc_loss < self.best_vae_val_loss:
                    self.best_vae_val_loss = val_metrics.vae_bc_loss
                    torch.save(self.skill_vae, self.best_vae_save_path)
                    self.writer.add_scalar('val_epoch/best_vae_bc_loss', self.best_vae_val_loss, epoch)
                    print(
                        f"[best skill vae epoch {epoch:03d}] val_vae_bc={self.best_vae_val_loss:.6f}",
                        flush=True,
                    )

                if self.prior_model == 'Flow':
                    self.writer.add_scalar('flow_prior_val/flow_loss', val_metrics.prior_flow_loss, epoch)
                    self.writer.add_scalar('flow_prior_val/sample_mse', val_metrics.prior_sample_mse, epoch)
                    self.writer.add_scalar('condition_flow_val/flow_loss', val_metrics.condition_flow_loss, epoch)
                    self.writer.add_scalar('condition_flow_val/sample_mse', val_metrics.condition_sample_mse, epoch)

                    if val_metrics.prior_flow_loss < self.best_prior_val_loss:
                        self.best_prior_val_loss = val_metrics.prior_flow_loss
                        torch.save(self.sp_nvp, self.best_sp_save_path)
                        self.writer.add_scalar('flow_prior_val/best_flow_loss', self.best_prior_val_loss, epoch)
                        print(
                            f"[best skill prior epoch {epoch:03d}] val_flow={self.best_prior_val_loss:.6f}",
                            flush=True,
                        )

                    if val_metrics.condition_flow_loss < self.best_condition_val_loss:
                        self.best_condition_val_loss = val_metrics.condition_flow_loss
                        torch.save(self.condition_prior, self.best_condition_prior_save_path)
                        self.writer.add_scalar('condition_flow_val/best_flow_loss', self.best_condition_val_loss, epoch)
                        print(
                            f"[best condition prior epoch {epoch:03d}] val_flow={self.best_condition_val_loss:.6f}",
                            flush=True,
                        )
                elif val_metrics.vae_bc_loss < self.best_prior_val_loss:
                    self.best_prior_val_loss = val_metrics.vae_bc_loss
                    torch.save(self.sp_nvp, self.best_sp_save_path)
                    self.writer.add_scalar('val_epoch/best_prior_proxy_loss', self.best_prior_val_loss, epoch)

            if epoch % 50 == 0 or epoch == self.n_epochs - 1:
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                if self.prior_model == 'Flow':
                    torch.save(self.condition_prior, self.condition_prior_save_path)

    def train_flow_skill_epoch(self, epoch):
        self.skill_vae.train()
        losses = []
        bc_losses = []
        kl_losses = []
        for batch_idx, data in enumerate(self.train_loader):
            log_step = epoch * len(self.train_loader) + batch_idx
            data["actions"] = data["actions"].to(self.device)
            data["obs"] = data["obs"].to(self.device)

            self.skill_vae.init_hidden(data["actions"].size(0))
            self.optimizer.zero_grad()
            output = self.skill_vae(data)
            loss_out = self.skill_vae.loss(data, output)
            loss_out.total_loss.backward()
            self.optimizer.step()

            losses.append(loss_out.total_loss.item())
            bc_losses.append(loss_out.bc_loss.item())
            kl_losses.append(loss_out.kld_loss.item())

            if batch_idx % 10 == 0:
                print(
                    f"[skill epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                    f"total={loss_out.total_loss.item():.6f} "
                    f"bc={loss_out.bc_loss.item():.6f} "
                    f"kl={loss_out.kld_loss.item():.6f}",
                    flush=True,
                )
                self.writer.add_scalar("skill_train_batch/total_loss", loss_out.total_loss.item(), log_step)
                self.writer.add_scalar("skill_train_batch/bc_loss", loss_out.bc_loss.item(), log_step)
                self.writer.add_scalar("skill_train_batch/kl_loss", loss_out.kld_loss.item(), log_step)

        return AttrDict(
            total_loss=float(np.mean(losses)),
            bc_loss=float(np.mean(bc_losses)),
            kl_loss=float(np.mean(kl_losses)),
        )

    def validate_flow_skill(self, epoch):
        self.skill_vae.eval()
        vae_bc_losses = []
        vae_kl_losses = []
        vae_total_losses = []
        with torch.no_grad():
            for data in self.val_loader:
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                self.skill_vae.init_hidden(data["actions"].size(0))
                output = self.skill_vae(data)
                losses = self.skill_vae.loss(data, output)
                vae_bc_losses.append(losses.bc_loss.item())
                vae_kl_losses.append(losses.kld_loss.item())
                vae_total_losses.append(losses.total_loss.item())
        metrics = AttrDict(
            vae_bc_loss=float(np.mean(vae_bc_losses)),
            vae_kl_loss=float(np.mean(vae_kl_losses)),
            vae_total_loss=float(np.mean(vae_total_losses)),
        )
        print(
            f"[skill val {epoch:03d}] total={metrics.vae_total_loss:.6f} "
            f"bc={metrics.vae_bc_loss:.6f} kl={metrics.vae_kl_loss:.6f}",
            flush=True,
        )
        self.writer.add_scalar("skill_val_epoch/total_loss", metrics.vae_total_loss, epoch)
        self.writer.add_scalar("skill_val_epoch/bc_loss", metrics.vae_bc_loss, epoch)
        self.writer.add_scalar("skill_val_epoch/kl_loss", metrics.vae_kl_loss, epoch)
        return metrics

    def train_behavior_policy_epoch(self, epoch):
        self.behavior_policy.train()
        nlls = []
        logps = []
        for batch_idx, data in enumerate(self.train_loader):
            log_step = epoch * len(self.train_loader) + batch_idx
            state = data["obs"][:, 0, :].to(self.device)
            action0 = data["actions"][:, 0, :].to(self.device)

            nll = self.behavior_policy.nll(state, action0)
            self.behavior_optimizer.zero_grad()
            nll.backward()
            self.behavior_optimizer.step()

            with torch.no_grad():
                log_prob = self.behavior_policy.log_prob(state, action0)
            nlls.append(nll.item())
            logps.append(log_prob.mean().item())

            if batch_idx % 10 == 0:
                print(
                    f"[behavior epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                    f"nll={nll.item():.6f} logp={log_prob.mean().item():.6f}",
                    flush=True,
                )
                self.writer.add_scalar("behavior_policy_train_batch/nll", nll.item(), log_step)
                self.writer.add_scalar("behavior_policy_train_batch/log_prob", log_prob.mean().item(), log_step)

        metrics = AttrDict(
            nll=float(np.mean(nlls)),
            log_prob=float(np.mean(logps)),
        )
        self.writer.add_scalar("behavior_policy_train_epoch/nll", metrics.nll, epoch)
        self.writer.add_scalar("behavior_policy_train_epoch/log_prob", metrics.log_prob, epoch)
        return metrics

    def validate_behavior_policy(self, epoch):
        self.behavior_policy.eval()
        nlls = []
        logps = []
        with torch.no_grad():
            for data in self.val_loader:
                state = data["obs"][:, 0, :].to(self.device)
                action0 = data["actions"][:, 0, :].to(self.device)
                log_prob = self.behavior_policy.log_prob(state, action0)
                nlls.append((-log_prob).mean().item())
                logps.append(log_prob.mean().item())
        metrics = AttrDict(
            nll=float(np.mean(nlls)),
            log_prob=float(np.mean(logps)),
        )
        print(
            f"[behavior val {epoch:03d}] nll={metrics.nll:.6f} logp={metrics.log_prob:.6f}",
            flush=True,
        )
        self.writer.add_scalar("behavior_policy_val/nll", metrics.nll, epoch)
        self.writer.add_scalar("behavior_policy_val/log_prob", metrics.log_prob, epoch)
        return metrics

    def train_behavior_policy(self):
        if not self.condition_reweight:
            return
        print("Training behavior policy for condition weights...", flush=True)
        for epoch in tqdm(range(self.behavior_policy_epochs)):
            self.train_behavior_policy_epoch(epoch)
            if epoch % self.val_freq == 0 or epoch == self.behavior_policy_epochs - 1:
                val_metrics = self.validate_behavior_policy(epoch)
                if val_metrics.nll < self.best_behavior_val_nll:
                    self.best_behavior_val_nll = val_metrics.nll
                    torch.save(self.behavior_policy.state_dict(), self.behavior_policy_save_path)
                    self.writer.add_scalar("behavior_policy_val/best_nll", self.best_behavior_val_nll, epoch)
                    print(
                        f"[best behavior policy epoch {epoch:03d}] val_nll={self.best_behavior_val_nll:.6f}",
                        flush=True,
                    )

        if os.path.exists(self.behavior_policy_save_path):
            self.behavior_policy.load_state_dict(
                torch.load(self.behavior_policy_save_path, map_location=self.device)
            )
        self.behavior_policy.eval()

    def compute_condition_weight_stats(self, batch_size=8192):
        if not self.condition_reweight:
            return

        dataset = self.train_loader.dataset

        raw_log_weights = []
        log_probs = []
        states = []
        actions = []

        self.behavior_policy.eval()

        def flush_batch():
            if not states:
                return
            state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
            action_tensor = torch.as_tensor(np.asarray(actions), dtype=torch.float32, device=self.device)
            with torch.no_grad():
                log_prob = self.behavior_policy.log_prob(state_tensor, action_tensor)
                raw_log_weight = -self.condition_weight_beta * log_prob
            log_probs.extend(log_prob.detach().cpu().numpy().tolist())
            raw_log_weights.extend(raw_log_weight.detach().cpu().numpy().tolist())
            states.clear()
            actions.clear()

        for seq_idx in range(dataset.start, dataset.end):
            seq = dataset.seqs[seq_idx]
            num_starts = max(0, len(seq.actions) - dataset.subseq_len - 1)
            for start_idx in range(num_starts):
                states.append(np.asarray(seq.obs[start_idx], dtype=np.float32))
                actions.append(np.asarray(seq.actions[start_idx], dtype=np.float32))
                if len(states) >= batch_size:
                    flush_batch()
        flush_batch()

        if len(raw_log_weights) == 0:
            raise ValueError("No valid chunks for condition weight computation.")

        raw_log_weights = np.asarray(raw_log_weights, dtype=np.float64)
        log_probs = np.asarray(log_probs, dtype=np.float64)
        raw_log_weight_max = float(
            np.quantile(raw_log_weights, self.condition_raw_log_weight_clip_quantile)
        )
        raw_log_weights_capped = np.minimum(raw_log_weights, raw_log_weight_max)

        max_raw = float(np.max(raw_log_weights_capped))
        log_mean_weight = max_raw + float(np.log(np.mean(np.exp(raw_log_weights_capped - max_raw))))
        weights_before_clip = np.exp(raw_log_weights_capped - log_mean_weight)
        weights = weights_before_clip
        weights = np.clip(weights, self.condition_weight_min, self.condition_weight_max).astype(np.float32)

        self.condition_log_mean_weight = log_mean_weight
        self.condition_raw_log_weight_max = raw_log_weight_max

        effective_sample_size = float((weights.sum() ** 2) / (np.square(weights).sum() + 1e-8))
        percentiles = [1, 5, 50, 90, 95, 99, 99.5, 99.9, 100]
        raw_percentiles = {
            f"raw_log_weight_p{str(p).replace('.', '_')}": float(np.percentile(raw_log_weights, p))
            for p in percentiles
        }
        capped_raw_percentiles = {
            f"raw_log_weight_capped_p{str(p).replace('.', '_')}": float(np.percentile(raw_log_weights_capped, p))
            for p in percentiles
        }
        weight_before_clip_percentiles = {
            f"weight_before_clip_p{str(p).replace('.', '_')}": float(np.percentile(weights_before_clip, p))
            for p in percentiles
        }
        weight_percentiles = {
            f"weight_p{str(p).replace('.', '_')}": float(np.percentile(weights, p))
            for p in percentiles
        }
        stats = {
            "dataset_name": self.dataset_name,
            "num_weighted_chunks": int(len(weights)),
            "beta": float(self.condition_weight_beta),
            "w_min": float(self.condition_weight_min),
            "w_max": float(self.condition_weight_max),
            "raw_log_weight_clip_quantile": float(self.condition_raw_log_weight_clip_quantile),
            "raw_log_weight_max": float(raw_log_weight_max),
            "log_mean_weight": float(log_mean_weight),
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "weight_before_clip_mean": float(np.mean(weights_before_clip)),
            "weight_before_clip_std": float(np.std(weights_before_clip)),
            "weight_before_clip_min": float(np.min(weights_before_clip)),
            "weight_before_clip_max": float(np.max(weights_before_clip)),
            "raw_log_weight_mean": float(np.mean(raw_log_weights)),
            "raw_log_weight_std": float(np.std(raw_log_weights)),
            "raw_log_weight_capped_mean": float(np.mean(raw_log_weights_capped)),
            "raw_log_weight_capped_std": float(np.std(raw_log_weights_capped)),
            "log_prob_mean": float(np.mean(log_probs)),
            "log_prob_std": float(np.std(log_probs)),
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": float(effective_sample_size / max(1, len(weights))),
        }
        stats.update(raw_percentiles)
        stats.update(capped_raw_percentiles)
        stats.update(weight_before_clip_percentiles)
        stats.update(weight_percentiles)
        with open(self.condition_weight_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(
            "[condition weights] "
            f"chunks={stats['num_weighted_chunks']} "
            f"mean={stats['weight_mean']:.6f} std={stats['weight_std']:.6f} "
            f"min={stats['weight_min']:.6f} max={stats['weight_max']:.6f} "
            f"ess_frac={stats['effective_sample_fraction']:.6f}",
            flush=True,
        )
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f"condition_weights/{key}", value, 0)

    def compute_condition_weights(self, state, action0):
        if not self.condition_reweight:
            return None
        if self.condition_log_mean_weight is None:
            if not os.path.exists(self.condition_weight_stats_path):
                raise RuntimeError("Condition reweighting is enabled but condition weight stats are not computed.")
            with open(self.condition_weight_stats_path, "r") as f:
                stats = json.load(f)
            self.condition_log_mean_weight = float(stats["log_mean_weight"])
            self.condition_raw_log_weight_max = float(stats["raw_log_weight_max"])

        with torch.no_grad():
            log_prob = self.behavior_policy.log_prob(state, action0)
            raw_log_weight = -self.condition_weight_beta * log_prob
            raw_log_weight = torch.clamp(raw_log_weight, max=self.condition_raw_log_weight_max)
            log_weight = raw_log_weight - self.condition_log_mean_weight
            weight = torch.exp(log_weight)
            weight = torch.clamp(
                weight,
                min=self.condition_weight_min,
                max=self.condition_weight_max,
            )
        return weight

    def extract_flow_targets(self, data):
        data["actions"] = data["actions"].to(self.device)
        data["obs"] = data["obs"].to(self.device)
        self.skill_vae.init_hidden(data["actions"].size(0))
        output = self.skill_vae(data)
        target_z = output.q.mu if self.prior_use_mu else output.z
        state = data["obs"][:, 0, :]
        action = data["actions"][:, 0, :]
        return state, action, target_z.detach()

    def train_flow_prior_epoch(self, epoch):
        self.skill_vae.eval()
        losses = []
        skill_flow_losses = []
        skill_distill_losses = []
        condition_flow_losses = []
        condition_distill_losses = []
        condition_weight_means = []
        condition_weight_maxes = []

        for batch_idx, data in enumerate(self.train_loader):
            log_step = epoch * len(self.train_loader) + batch_idx
            with torch.no_grad():
                state, action_ori, target_z = self.extract_flow_targets(data)
                action0 = data["actions"][:, 0, :]
                condition_weight = self.compute_condition_weights(state, action0)

            batch_skill_flow = []
            batch_skill_distill = []
            batch_condition_flow = []
            batch_condition_distill = []

            for _ in range(self.prior_updates_per_batch):
                action = action_ori
                condition = torch.cat([state, action], dim=1)
                metric = self.sp_nvp.train(condition, target_z, iterations=1)
                condition_metric = self.condition_prior.train(
                    state,
                    action,
                    iterations=1,
                    sample_weight=condition_weight,
                )
                batch_skill_flow.append(float(np.mean(metric["flow_loss"])))
                batch_skill_distill.append(float(np.mean(metric["distill_loss"])))
                batch_condition_flow.append(float(np.mean(condition_metric["flow_loss"])))
                batch_condition_distill.append(float(np.mean(condition_metric["distill_loss"])))

            skill_flow_loss = float(np.mean(batch_skill_flow))
            skill_distill_loss = float(np.mean(batch_skill_distill))
            condition_flow_loss = float(np.mean(batch_condition_flow))
            condition_distill_loss = float(np.mean(batch_condition_distill))
            total_loss = skill_flow_loss + condition_flow_loss
            losses.append(total_loss)
            skill_flow_losses.append(skill_flow_loss)
            skill_distill_losses.append(skill_distill_loss)
            condition_flow_losses.append(condition_flow_loss)
            condition_distill_losses.append(condition_distill_loss)
            if condition_weight is not None:
                condition_weight_means.append(float(condition_weight.mean().item()))
                condition_weight_maxes.append(float(condition_weight.max().item()))

            if batch_idx % 10 == 0:
                print(
                    f"[prior epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                    f"skill_flow={skill_flow_loss:.6f} "
                    f"skill_distill={skill_distill_loss:.6f} "
                    f"condition_flow={condition_flow_loss:.6f} "
                    f"condition_distill={condition_distill_loss:.6f} "
                    f"condition_weight_mean={condition_weight.mean().item() if condition_weight is not None else 1.0:.6f}",
                    flush=True,
                )
                self.writer.add_scalar("flow_prior_train_batch/flow_loss", skill_flow_loss, log_step)
                self.writer.add_scalar("flow_prior_train_batch/distill_loss", skill_distill_loss, log_step)
                self.writer.add_scalar("condition_flow_train_batch/flow_loss", condition_flow_loss, log_step)
                self.writer.add_scalar("condition_flow_train_batch/distill_loss", condition_distill_loss, log_step)
                if condition_weight is not None:
                    self.writer.add_scalar("condition_flow_train_batch/weight_mean", condition_weight.mean().item(), log_step)
                    self.writer.add_scalar("condition_flow_train_batch/weight_max", condition_weight.max().item(), log_step)

        metrics = AttrDict(
            total_loss=float(np.mean(losses)),
            prior_flow_loss=float(np.mean(skill_flow_losses)),
            prior_distill_loss=float(np.mean(skill_distill_losses)),
            condition_flow_loss=float(np.mean(condition_flow_losses)),
            condition_distill_loss=float(np.mean(condition_distill_losses)),
        )
        if condition_weight_means:
            metrics.condition_weight_mean = float(np.mean(condition_weight_means))
            metrics.condition_weight_max = float(np.mean(condition_weight_maxes))
        else:
            metrics.condition_weight_mean = 1.0
            metrics.condition_weight_max = 1.0
        return metrics

    def validate_flow_priors(self, epoch):
        metrics = self.validate()
        print(
            f"[prior val {epoch:03d}] "
            f"skill_flow={metrics.prior_flow_loss:.6f} "
            f"skill_mse={metrics.prior_sample_mse:.6f} "
            f"condition_flow={metrics.condition_flow_loss:.6f} "
            f"condition_mse={metrics.condition_sample_mse:.6f}",
            flush=True,
        )
        self.writer.add_scalar("flow_prior_val/flow_loss", metrics.prior_flow_loss, epoch)
        self.writer.add_scalar("flow_prior_val/sample_mse", metrics.prior_sample_mse, epoch)
        self.writer.add_scalar("condition_flow_val/flow_loss", metrics.condition_flow_loss, epoch)
        self.writer.add_scalar("condition_flow_val/sample_mse", metrics.condition_sample_mse, epoch)
        return metrics

    def train_flow_two_stage(self):
        print("Training stage 1: SkillVAE...", flush=True)
        for epoch in tqdm(range(self.skill_epochs)):
            train_metrics = self.train_flow_skill_epoch(epoch)
            self.writer.add_scalar("skill_train_epoch/total_loss", train_metrics.total_loss, epoch)
            self.writer.add_scalar("skill_train_epoch/bc_loss", train_metrics.bc_loss, epoch)
            self.writer.add_scalar("skill_train_epoch/kl_loss", train_metrics.kl_loss, epoch)

            if epoch % self.val_freq == 0:
                val_metrics = self.validate_flow_skill(epoch)
                if val_metrics.vae_bc_loss < self.best_vae_val_loss:
                    self.best_vae_val_loss = val_metrics.vae_bc_loss
                    torch.save(self.skill_vae, self.best_vae_save_path)
                    self.writer.add_scalar("skill_val_epoch/best_bc_loss", self.best_vae_val_loss, epoch)
                    print(
                        f"[best skill vae epoch {epoch:03d}] val_bc={self.best_vae_val_loss:.6f}",
                        flush=True,
                    )

            if epoch % self.save_freq == 0 or epoch == self.skill_epochs - 1:
                torch.save(self.skill_vae, self.vae_save_path)

        if os.path.exists(self.best_vae_save_path):
            self.skill_vae = torch.load(self.best_vae_save_path, map_location=self.device)
        self.skill_vae.to(self.device)
        self.skill_vae.eval()
        for param in self.skill_vae.parameters():
            param.requires_grad_(False)

        if self.prior_model == "Flow" and self.condition_reweight:
            self.train_behavior_policy()
            self.compute_condition_weight_stats()

        print("Training stage 2: skill prior + condition prior...", flush=True)
        for epoch in tqdm(range(self.prior_epochs)):
            train_metrics = self.train_flow_prior_epoch(epoch)
            self.writer.add_scalar("flow_prior_train_epoch/flow_loss", train_metrics.prior_flow_loss, epoch)
            self.writer.add_scalar("flow_prior_train_epoch/distill_loss", train_metrics.prior_distill_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/flow_loss", train_metrics.condition_flow_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/distill_loss", train_metrics.condition_distill_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/weight_mean", train_metrics.condition_weight_mean, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/weight_max", train_metrics.condition_weight_max, epoch)

            if epoch % self.val_freq == 0:
                val_metrics = self.validate_flow_priors(epoch)
                if val_metrics.prior_flow_loss < self.best_prior_val_loss:
                    self.best_prior_val_loss = val_metrics.prior_flow_loss
                    torch.save(self.sp_nvp, self.best_sp_save_path)
                    self.writer.add_scalar("flow_prior_val/best_flow_loss", self.best_prior_val_loss, epoch)
                    print(
                        f"[best skill prior epoch {epoch:03d}] val_flow={self.best_prior_val_loss:.6f}",
                        flush=True,
                    )
                if val_metrics.condition_flow_loss < self.best_condition_val_loss:
                    self.best_condition_val_loss = val_metrics.condition_flow_loss
                    torch.save(self.condition_prior, self.best_condition_prior_save_path)
                    self.writer.add_scalar("condition_flow_val/best_flow_loss", self.best_condition_val_loss, epoch)
                    print(
                        f"[best condition prior epoch {epoch:03d}] val_flow={self.best_condition_val_loss:.6f}",
                        flush=True,
                    )

            if epoch % self.save_freq == 0 or epoch == self.prior_epochs - 1:
                torch.save(self.sp_nvp, self.sp_save_path)
                torch.save(self.condition_prior, self.condition_prior_save_path)

    def train(self):
        if self.prior_model == 'Flow':
            self.train_flow_two_stage()
            return
        self.train_legacy()
                
   
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="block/config.yaml")
    parser.add_argument('--pick', type=int, default=1)
    parser.add_argument('--push', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--prior_model', type=str, default='CVAE')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--use_student', type=int, default=1)
    parser.add_argument('--skill_epochs', type=int, default=None)
    parser.add_argument('--prior_epochs', type=int, default=None)
    parser.add_argument('--prior_updates_per_batch', type=int, default=1)
    parser.add_argument('--prior_use_mu', type=int, default=1)
    parser.add_argument('--val_freq', type=int, default=5)
    parser.add_argument('--save_freq', type=int, default=50)
    parser.add_argument('--action_noise_std', type=float, default=0.0)
    parser.add_argument('--condition_reweight', type=int, default=0)
    parser.add_argument('--behavior_policy_epochs', type=int, default=20)
    parser.add_argument('--behavior_policy_lr', type=float, default=3e-4)
    parser.add_argument('--behavior_policy_hidden_dim', type=int, default=256)
    parser.add_argument('--condition_weight_beta', type=float, default=0.2)
    parser.add_argument('--condition_weight_min', type=float, default=0.2)
    parser.add_argument('--condition_weight_max', type=float, default=3.0)
    parser.add_argument('--condition_raw_log_weight_clip_quantile', type=float, default=0.99)
    parser.add_argument('--swanlab_project', type=str, default="Flow_skill_1")
    parser.add_argument('--swanlab_workspace', type=str, default="x1x1217")
    parser.add_argument('--swanlab_mode', type=str, default=None)
    args=parser.parse_args()
    if args.prior_updates_per_batch < 1:
        raise ValueError("--prior_updates_per_batch must be >= 1")
    if args.val_freq < 1:
        raise ValueError("--val_freq must be >= 1")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be >= 1")
    if args.behavior_policy_epochs < 1:
        raise ValueError("--behavior_policy_epochs must be >= 1")
    if not 0 <= args.condition_weight_beta < 1:
        raise ValueError("--condition_weight_beta must satisfy 0 <= beta < 1")
    if args.condition_weight_min <= 0 or args.condition_weight_max < args.condition_weight_min:
        raise ValueError("--condition_weight_min/max are invalid")
    if not 0 < args.condition_raw_log_weight_clip_quantile <= 1:
        raise ValueError("--condition_raw_log_weight_clip_quantile must be in (0, 1].")
    if args.dataset_name is None:
        args.dataset_name = f'fetch_block_push{args.push}_pick{args.pick}'
    flow_suffix = f"_student{args.use_student}" if args.prior_model == "Flow" else ""
    
    curr_dir = os.path.dirname(__file__)
    log_file = os.path.join(
        curr_dir,
        "swanlog",
        "skill_prior",
        args.dataset_name,
        f"seed_{args.seed}_{args.prior_model}{flow_suffix}",
    )
    
    os.makedirs(log_file, exist_ok=True)
    writer = SwanLabWriter(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=f"skill_prior_seed{args.seed}_{args.prior_model}{flow_suffix}",
        config=vars(args),
        logdir=log_file,
        mode=args.swanlab_mode,
        tags=["skill_prior", args.prior_model],
    )

    trainer = ModelTrainer(
        args.dataset_name,
        args.config_file,
        args.prior_model,
        args.seed,
        writer,
        use_student=bool(args.use_student),
        skill_epochs=args.skill_epochs,
        prior_epochs=args.prior_epochs,
        prior_updates_per_batch=args.prior_updates_per_batch,
        prior_use_mu=bool(args.prior_use_mu),
        val_freq=args.val_freq,
        save_freq=args.save_freq,
        action_noise_std=args.action_noise_std,
        condition_reweight=bool(args.condition_reweight),
        behavior_policy_epochs=args.behavior_policy_epochs,
        behavior_policy_lr=args.behavior_policy_lr,
        behavior_policy_hidden_dim=args.behavior_policy_hidden_dim,
        condition_weight_beta=args.condition_weight_beta,
        condition_weight_min=args.condition_weight_min,
        condition_weight_max=args.condition_weight_max,
        condition_raw_log_weight_clip_quantile=args.condition_raw_log_weight_clip_quantile,
    )
    trainer.train()
    writer.close()
