
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

from reskill.models.skill_vae import SkillVAE
from reskill.data.skill_dataloader import SkillsDataset
from reskill.models.cvae import CVAE
from reskill.models.normal_mlp import NMLP
from reskill.models.rnvp import stacked_NVP
from reskill.models.bc_diffusion import Diffusion_BC
from reskill.utils.general_utils import AttrDict
from reskill.models.bc_flow import Flow_BC
from reskill.utils.swanlab_writer import SwanLabWriter


class ModelTrainer():
    def __init__(self, dataset_name, config_file, prior_model, seed, writer, use_student=True):
        self.dataset_name = dataset_name
        self.prior_model = prior_model
        self.seed = seed
        self.use_student = use_student
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
        val_data   = SkillsDataset(dataset_name, phase="val", subseq_len=conf.skill_vae.subseq_len, transform=transform)

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
        
        elif self.prior_model == 'Diffusion':
            self.sp_nvp = Diffusion_BC(state_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, action_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
        elif self.prior_model == 'MLP':
            self.sp_nvp = NMLP(cond_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, latent_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
            self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        elif self.prior_model == 'CVAE':
            self.sp_nvp = CVAE(z_dim=conf.skill_vae.n_z, cond_dim=conf.skill_vae.n_actions+conf.skill_vae.n_obs, latent_dim=conf.skill_vae.n_z, max_action=10, device=self.device)
            self.sp_optimizer = torch.optim.Adam(self.sp_nvp.parameters(), lr=conf.skill_prior_nvp.sp_lr)
        self.n_epochs = conf.skill_vae.epochs


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
                action = data["actions"][:, 0, :] / 2.
                
                action_ori = action
                state_ori = state
                
                for prior_iter in range(100):
                    action = action_ori + 0.2 * torch.normal(0, 1, action.shape).to(self.device)
                    condtion = torch.cat([state_ori, action], dim=1)
                    
                    metric = self.sp_nvp.train(condtion, skill, iterations=1)
                    sp_loss = np.mean(metric['total_loss'])
            
                if batch_idx % 10 == 0:
                    flow_loss = np.mean(metric['flow_loss'])
                    distill_loss = np.mean(metric['distill_loss'])
                    prior_total_loss = np.mean(metric['total_loss'])
                    print(
                        f"[epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                        f"vae_total={losses.total_loss.item():.4f} "
                        f"vae_bc={losses.bc_loss.item():.4f} "
                        f"vae_kl={losses.kld_loss.item():.4f} "
                        f"flow={flow_loss:.4f} "
                        f"distill={distill_loss:.4f} "
                        f"prior_total={prior_total_loss:.4f}",
                        flush=True,
                    )
                    self.writer.add_scalar('train_batch/vae_bc_loss', losses.bc_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_kl_loss', losses.kld_loss.item(), log_step)
                    self.writer.add_scalar('train_batch/vae_total_loss', losses.total_loss.item(), log_step)
                    self.writer.add_scalar('flow_prior/flow_loss', flow_loss, log_step)
                    self.writer.add_scalar('flow_prior/distill_loss', distill_loss, log_step)
                    self.writer.add_scalar('flow_prior/total_loss', prior_total_loss, log_step)
            
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
        running_loss = 0.0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader):
                data["actions"] = data["actions"].to(self.device)
                data["obs"] = data["obs"].to(self.device)
                self.skill_vae.init_hidden(data["actions"].size(0))
                self.optimizer.zero_grad()
                output = self.skill_vae(data)
                losses = self.skill_vae.loss(data, output)

                loss = losses.bc_loss.item()
                running_loss += loss

        val_loss = running_loss / len(self.val_loader)
        return val_loss


    def train(self):
        print("Training...") 
        for epoch in tqdm(range(self.n_epochs)):
            print(f"\n[start epoch {epoch:03d}/{self.n_epochs}]", flush=True)
            
            train_epoch_loss = self.fit(epoch)
            self.writer.add_scalar('train_epoch/loss', train_epoch_loss, epoch)

            if epoch % 5 == 0:
                val_epoch_loss = self.validate()
                print(
                    f"[end epoch {epoch:03d}] train_loss={train_epoch_loss:.6f} "
                    f"val_loss={val_epoch_loss:.6f}",
                    flush=True,
                )
                self.writer.add_scalar('val_epoch/loss', val_epoch_loss, epoch)

            if epoch % 50 == 0:
                torch.save(self.skill_vae, self.vae_save_path)
                torch.save(self.sp_nvp, self.sp_save_path)
                
   
if __name__ == "__main__":

    parser=argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default="block/config.yaml")
    parser.add_argument('--pick', type=int, default=1)
    parser.add_argument('--push', type=int, default=1)
    #parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    parser.add_argument('--prior_model', type=str, default='CVAE')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--use_student', type=int, default=1)
    parser.add_argument('--swanlab_project', type=str, default="Flow_skill_1")
    parser.add_argument('--swanlab_workspace', type=str, default="x1x1217")
    parser.add_argument('--swanlab_mode', type=str, default=None)
    args=parser.parse_args()
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
    )
    trainer.train()
    writer.close()
