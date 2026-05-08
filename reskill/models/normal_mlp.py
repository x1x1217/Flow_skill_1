import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class NMLP(nn.Module):
    def __init__(self, cond_dim, latent_dim, max_action, device):
        super(NMLP, self).__init__()
        self.net = nn.Sequential(
                    nn.Linear(cond_dim, 64),
                    nn.LeakyReLU(),
                    nn.Linear(64, 32),
                    nn.LeakyReLU(),
                    nn.Linear(32, latent_dim)).to(device)
        
        self.latent_dim = latent_dim

        self.bc_criterion = nn.MSELoss(reduction="mean")
        self.max_action = max_action

    def forward(self, z_now, cond):
        output = self.net(cond)
        bc_loss = self.bc_criterion(z_now, output)
        return bc_loss