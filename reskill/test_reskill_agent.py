
import torch
import gym
import pdb
import numpy as np
import sys
sys.path.append("..")
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from reskill.utils.general_utils import AttrDict
from reskill.models.rnvp import R_NVP, stacked_NVP
import reskill.rl.envs
import argparse

class TestRLAgent():
    def __init__(self, dataset_name, env_name, prior_model):

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.prior_model = prior_model

        skill_mdl_path   = "./results_1999/saved_skill_models/" + dataset_name + "/skill_prior_" + prior_model + "/skill_vae.pth"
        self.skill_mdl = torch.load(skill_mdl_path, map_location=torch.device(self.device))
        self.skill_mdl.eval()
            

        skill_prior_path = "./results_1999/saved_skill_models/" + dataset_name + "/skill_prior_" + prior_model + "/skill_prior.pth"
        self.skill_prior_mdl = torch.load(skill_prior_path, map_location=torch.device(self.device))
        if self.prior_model == "RNVP":
            for i in self.skill_prior_mdl.bijectors:
                i.device = self.device
            self.skill_prior_mdl.eval()

        skill_agent_path = "./results_1999/saved_rl_models/" + dataset_name + "/" + prior_model + "/ppo_agent_" + str(args.seed) + ".pth"
        self.skill_agent = torch.load(skill_agent_path, map_location=torch.device(self.device))

        residual_skill_agent_path = "./results_1999/saved_rl_models/" + dataset_name + "/" + prior_model + "/ppo_residual_agent_" + str(args.seed) + ".pth"
        self.residual_skill_agent = torch.load(residual_skill_agent_path, map_location=torch.device(self.device))

        self.env = gym.make(env_name)
        self.n_features = self.skill_mdl.n_z
        self.seq_len = self.skill_mdl.seq_len
        self.env_name = env_name


    def get_obs(self, obs):
        if self.env_name == "FetchPyramidStack-v0": 
            out = torch.FloatTensor(np.concatenate((obs["observation"][:-6], obs["desired_goal"][-3:]))).unsqueeze(dim=0).to(self.device)
        else:
            out = torch.FloatTensor(np.concatenate((obs["observation"], obs["desired_goal"]))).unsqueeze(dim=0).to(self.device)
        return out

    def test(self):

        obs = self.env.reset()
        obs = self.get_obs(obs)

        steps = 0
        r = 0

        while(True):

            # Use skill agent
            dist, _ = self.skill_agent(obs)
            n = dist.loc
            if self.prior_model == "RNVP":
                skill = AttrDict(noise=n, state=obs)
                z = self.skill_prior_mdl.inverse(skill).noise.detach()
            elif self.prior_model == "Diffusion":
                state_ = torch.cat((obs, n), dim=1).cuda()
                #print(state_)
                z = self.skill_prior_mdl.sample_action_torch(state_).detach()


            for _ in range(self.seq_len):

                obs_z = torch.cat((obs, z), 1)
                a_dec = self.skill_mdl.decoder(obs_z)
                
                # Get residual action
                o_res = torch.cat((obs,z,a_dec), 1)
                dist, _ = self.residual_skill_agent(o_res)
                a_res = dist.loc

                # Add residual action to decoded action
                a = (a_dec.cpu().detach().numpy() + a_res.cpu().detach().numpy())[0]
                obs, reward, done, debug_info = self.env.step(a)
                r += reward
                obs = self.get_obs(obs)

                self.env.render()
                
                steps += 1

            if steps > self.env._max_episode_steps or debug_info['is_success']:

                obs = self.env.reset()
                obs = self.get_obs(obs)
                steps = 0
                break

        print(r)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    parser.add_argument('--env_name', type=str, default="FetchPyramidStack-v0")
    parser.add_argument('--prior_model', type=str, default="Diffusion")
    parser.add_argument('--seed', type=int, default=21)
    args = parser.parse_args()
    t = TestRLAgent(args.dataset_name, args.env_name, args.prior_model)
    t.test()