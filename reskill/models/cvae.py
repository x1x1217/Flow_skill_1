import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.logger import logger

from models.diffusion import Diffusion
from models.model import MLP
from utils.general_utils import AttrDict

class CVAE(nn.Module):
    def __init__(self, z_dim, cond_dim, latent_dim, max_action, device, lr=2e-4):
        super(CVAE, self).__init__()
        #self.encode = MLP(state_dim=state_dim+cond_dim, action_dim=latent_dim*2, device=device)
        #self.decode = MLP(state_dim=latent_dim+cond_dim, action_dim=latent_dim, device=device)
        self.encode = nn.Sequential(
                    nn.Linear(z_dim+cond_dim, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, latent_dim*2)).to(device)
        self.decode = nn.Sequential(nn.Linear(latent_dim+cond_dim, 64),
                                     nn.LeakyReLU(),
                                     nn.Linear(64, 32),
                                     nn.Linear(32, latent_dim)).to(device)
        self.latent_dim = latent_dim

        self.bc_criterion = nn.MSELoss(reduction="mean")
        self.max_action = max_action

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        sample = mu + (eps*std)
        return sample
    
    def run_inference(self,x):
        # encoding
        out = self.encode(x)
        return out.view(-1,2,self.latent_dim)
    
    def forward(self, z_now, cond, beta=0.00000001):
        x_cat = torch.cat((z_now, cond), 1) 
        x = self.run_inference(x_cat)
        q = AttrDict(mu=x[:,0,:],
                     log_var=x[:,1,:])

        z = self.reparameterize(q.mu, q.log_var)
   
        # Decoding
        # Closed loop decoding
        decode_inputs = torch.cat((z, cond), 1)
        reconstruction = self.decode(decode_inputs)

        bc_loss = self.bc_criterion(z_now, reconstruction)
        kld_loss = (-0.5 * torch.sum(1 + q.log_var - q.mu.pow(2) - q.log_var.exp()))
        
        return bc_loss, beta*kld_loss
