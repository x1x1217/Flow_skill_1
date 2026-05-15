
import gym
import torch
import pdb
from tqdm import tqdm
import time
import numpy as np
import os
import sys
sys.path.append("..")
from rl.sac.sac import SAC
from rl.sac.replay_memory import ReplayMemory
from rl.utils.mpi_tools import num_procs, mpi_fork, proc_id
from rl.agents.ppo import PPO
from utils.general_utils import AttrDict
import rl.envs
import math
from utils.swanlab_writer import SwanLabWriter


device = torch.device('cuda')


def get_obs(obs, env_name):
    if env_name == "FetchPyramidStack-v0": 
        out = torch.FloatTensor(np.concatenate((obs["observation"][:-6], obs["desired_goal"][-3:]))).unsqueeze(dim=0).to(device)
    else:
        out = torch.FloatTensor(np.concatenate((obs["observation"], obs["desired_goal"]))).unsqueeze(dim=0).to(device)
    return out

def logistic_fn(step, k=0.001, C=18000):
    return 1/(1 + math.exp(-k * (step-C)))

def train(agent, residual_agent, skill_distill, sac_replay, env, skill_vae, skill_prior, logistic_C, logistic_k, 
          save_path, save_path_residual, writer, prior_model, args):

    env_name = env.spec.id
    obs, ep_ret, ep_len = env.reset(), 0, 0
    o = get_obs(obs, env_name)
    updates = 0

    env_step_cnt = 0
    residual_factor = 0.0

    local_steps_per_epoch = int(agent.steps_per_epoch / num_procs())

    for epoch in tqdm(range(agent.epochs)):
        success = 0
        for t in range(local_steps_per_epoch):
            # Select noise vector using high-level policy
            n, v, logp, mu, std = agent.ac.step(torch.as_tensor(o, dtype=torch.float32))
            if prior_model == 'RNVP':
                sample = AttrDict(noise=n, state=o)
                # Warp noise vector to latent space skill
                z = skill_prior.inverse(sample).noise.detach()
            elif prior_model == 'Diffusion':
                state_ = torch.cat((o, n), dim=1).to(device)
                #print(state_)
                z = skill_prior.sample_action_torch(state_)

            o2, skill_r = o, 0
        
            for _ in range(skill_vae.seq_len):
                if len(sac_replay) > args.batch_size:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = skill_distill.update_parameters(sac_replay, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1
                
                obs_z = torch.cat((o2,z), 1)
                a_dec = skill_vae.decoder(obs_z)

                #o_res = torch.cat((o2,z,a_dec), 1)
                #a_res, v_res, logp_res, _, _ = residual_agent.ac.step(o_res)
                
                #a = (a_dec.cpu().detach().numpy() + (a_res.cpu().detach().numpy() * residual_factor))[0]
                a = a_dec.cpu().detach().numpy()[0]
                
                # Step the env
                obs, r, d, _ = env.step(a)

                env_step_cnt += 1

                skill_r += r #Sum rewards for high level policy
                ep_ret += r
                ep_len += 1

                o2 = get_obs(obs, env_name)
                a_dec = skill_vae.decoder(torch.cat((o2,z), 1))
                o2_res = torch.cat((o2,z,a_dec), 1)

                #residual_agent.buf.store(o_res.cpu().detach(), a_res.cpu().detach(), r, v_res, logp_res)

            # Update residual action weighting factor
            residual_factor = logistic_fn(env_step_cnt, k=logistic_k, C=logistic_C)
            if proc_id() == 0:
                writer.add_scalar('ppo/logistic_fn', residual_factor, env_step_cnt)

            # save and log
            agent.buf.store(o.cpu().detach(), n.cpu().detach(), skill_r, v, logp)
            mask = float(d)
            sac_replay.push(o.cpu().detach().numpy()[0], z.cpu().detach().numpy()[0], skill_r, o2.cpu().detach().numpy()[0], mask)

            o = o2
            t += 1

            timeout = ep_len >= agent.max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1
        
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    _, v, _, _, _ = agent.ac.step(o)
                    #_, v_res, _, _, _ = residual_agent.ac.step(o2_res)
                else:
                    v = 0
                    #v_res = 0
                agent.buf.finish_path(v)
                #residual_agent.buf.finish_path(v_res)
                if terminal:
                    if proc_id() == 0:
                        writer.add_scalar('Episode Reture', ep_ret, env_step_cnt)
                obs, ep_ret, ep_len = env.reset(), 0, 0
                o = get_obs(obs, env_name)

        # Save model
        if (epoch % agent.save_freq == 0) or (epoch == agent.epochs-1):
            torch.save(agent.ac.pi, save_path)
            skill_distill.save_checkpoint(env_name, str(args.seed)+"_"+args.prior_model)
            #torch.save(residual_agent.ac.pi, save_path_residual)

        # Perform PPO update!
        losses = agent.update()
        #residual_losses = residual_agent.update()

        success_traj = 0
        total_r = 0
        r_time = 0

        for roll in range(50):
            obs = env.reset()
            obs = get_obs(obs, env_name)

            steps = 0
            r = 0
            while(True):

                # Use skill agent
                n = agent.ac.act_deterministic(obs)
                n = torch.FloatTensor(n)
                if prior_model == "RNVP":
                    skill = AttrDict(noise=n, state=obs)
                    z = skill_prior.inverse(skill).noise.detach()
                elif prior_model == "Diffusion":
                    n = n.cuda()
                    state_ = torch.cat((obs, n), dim=1).cuda()
                    #print(state_)
                    z = skill_prior.sample_action_torch(state_).detach()
                    #z = skill_distill.select_action(obs, evaluate=True)
                elif prior_model == 'MLP':
                    pass
                elif prior_model == 'CVAE':
                    pass

                o2, skill_r = o, 0
            
                for _ in range(skill_vae.seq_len):

                    obs_z = torch.cat((obs, z), 1)
                    a_dec = skill_vae.decoder(obs_z)
                    
                    # Get residual action
                    #o_res = torch.cat((obs,z,a_dec), 1)
                    #dist, _ = self.residual_skill_agent(o_res)
                    #a_res = dist.loc

                    # Add residual action to decoded action
                    # a = (a_dec.cpu().detach().numpy() + a_res.cpu().detach().numpy())[0]
                    a = a_dec.cpu().detach().numpy()[0]
                    obs, reward, done, debug_info = env.step(a)
                    r += reward
                    obs = get_obs(obs, env_name)

                    #env.render()
                    
                    steps += 1

                if steps > env._max_episode_steps or debug_info['is_success']:
                    if debug_info['is_success']:
                        success_traj += 1
                    obs = env.reset()
                    obs = get_obs(obs, env_name)
                    steps = 0
                    total_r += r
                    if r > 0:
                        r_time += 1
                    break

        if proc_id() == 0:
            writer.add_scalar('pi_loss_', losses.LossPi, env_step_cnt)
            writer.add_scalar('v_loss_', losses.LossV, env_step_cnt)
            writer.add_scalar('kl', losses.KL, env_step_cnt)
            writer.add_scalar('entropy', losses.Entropy, env_step_cnt)
            writer.add_scalar('clip_frac', losses.ClipFrac, env_step_cnt)
            writer.add_scalar('delta_loss_pi', losses.DeltaLossPi, env_step_cnt)
            writer.add_scalar('delta_loss_v', losses.DeltaLossV, env_step_cnt)
            writer.add_scalar('success_traj_epoch', success_traj, epoch)
            writer.add_scalar('success_traj_step', success_traj, env_step_cnt)
            writer.add_scalar('success_rate_epoch', success_traj / 50, epoch)
            writer.add_scalar('success_rate_step', success_traj / 50, env_step_cnt)
            writer.add_scalar('r_epoch', total_r, epoch)
            writer.add_scalar('r_step', total_r, env_step_cnt)
            writer.add_scalar('avg_r_epoch', total_r / 50, epoch)
            writer.add_scalar('avg_r_step', total_r / 50, env_step_cnt)
            writer.add_scalar('avg_rtime_epoch', r_time / 50, epoch)
            writer.add_scalar('avg_rtime_step', r_time / 50, env_step_cnt)


def main():
    import argparse
    import yaml
    parser=argparse.ArgumentParser()
    parser.add_argument('--skill_hid', type=int, default=256)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 10 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                        help='Automaically adjust α (default: False)')
    parser.add_argument('--batch_size', type=int, default=512, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=100000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--config_file', type=str, default="table_cleanup/config.yaml")
    parser.add_argument('--dataset_name', type=str, default="fetch_block_40000")
    parser.add_argument('--prior_model', type=str, default='Diffusion')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--swanlab_project', type=str, default="Flow_skill_1")
    parser.add_argument('--swanlab_workspace', type=str, default="x1x1217")
    parser.add_argument('--swanlab_mode', type=str, default=None)
    args=parser.parse_args()

    config_path = "configs/rl/" + args.config_file
    with open(config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = AttrDict(conf)
    for key in conf:
        conf[key] = AttrDict(conf[key])

    mpi_fork(conf.setup.cpu)  #  run parallel code with mpi

    if proc_id() == 0:
        #wandb.init(project=conf.setup.exp_name)
        #wandb.run.name = conf.setup.env + "_reskill_seed_" + str(conf.setup.seed) + '_' + time.asctime().replace(' ', '_')
        log_file = './swanlog/agent/'+args.dataset_name+'/seed_'+str(args.seed)+'_'+args.prior_model+'/'
        os.makedirs(log_file, exist_ok=True)
        writer = SwanLabWriter(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=f"agent_{args.dataset_name}_seed{args.seed}_{args.prior_model}",
            config=vars(args),
            logdir=log_file,
            mode=args.swanlab_mode,
            tags=["agent", args.prior_model],
        )
    else:
        writer = None

    env = gym.make(conf.setup.env)

    save_dir = "./results_1999/saved_rl_models/" + "stack/" + args.dataset_name + "/" + args.prior_model + "/"
    os.makedirs(save_dir, exist_ok=True)
    save_path = save_dir + "ppo_agent_" + str(args.seed) + ".pth"
    save_path_residual = save_dir + "ppo_residual_agent_" + str(args.seed) + ".pth"

    torch.set_num_threads(torch.get_num_threads())

    # Load skills module
    skill_vae_path = "./results_1999/saved_skill_models/" + args.dataset_name + "/skill_prior_" + args.prior_model + "/skill_vae.pth"
    skill_vae = torch.load(skill_vae_path, map_location=device)
    # Load skill prior module
    skill_prior_path = "./results_1999/saved_skill_models/" + args.dataset_name + "/skill_prior_" + args.prior_model + "/skill_prior.pth"
    skill_prior = torch.load(skill_prior_path, map_location=device)
    if args.prior_model == 'RNVP':
        for i in skill_prior.bijectors:
            i.device = device

    n_features = skill_vae.n_z
    n_actions = skill_vae.n_actions
    n_obs = skill_vae.n_obs
    seq_len = skill_vae.seq_len

    if args.prior_model == 'RNVP':
        skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid]*conf.skill_agent.l),
                    gamma=conf.skill_agent.gamma, 
                    seed=args.seed, 
                    steps_per_epoch=conf.skill_agent.steps_per_epoch, 
                    epochs=conf.setup.epochs,
                    clip_ratio=conf.skill_agent.clip_ratio, 
                    pi_lr=conf.skill_agent.pi_lr,
                    vf_lr=conf.skill_agent.vf_lr, 
                    train_pi_iters=conf.skill_agent.train_pi_iters, 
                    train_v_iters=conf.skill_agent.train_v_iters, 
                    lam=conf.skill_agent.lam, 
                    max_ep_len=conf.setup.max_ep_len,
                    target_kl=conf.skill_agent.target_kl, 
                    obs_dim=n_obs, 
                    act_dim=n_features, 
                    act_limit=2)
    elif args.prior_model == 'Diffusion':
        skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid]*conf.skill_agent.l),
                    gamma=conf.skill_agent.gamma, 
                    seed=args.seed, 
                    steps_per_epoch=conf.skill_agent.steps_per_epoch, 
                    epochs=conf.setup.epochs,
                    clip_ratio=conf.skill_agent.clip_ratio, 
                    pi_lr=conf.skill_agent.pi_lr,
                    vf_lr=conf.skill_agent.vf_lr, 
                    train_pi_iters=conf.skill_agent.train_pi_iters, 
                    train_v_iters=conf.skill_agent.train_v_iters, 
                    lam=conf.skill_agent.lam, 
                    max_ep_len=conf.setup.max_ep_len,
                    target_kl=conf.skill_agent.target_kl, 
                    obs_dim=n_obs, 
                    act_dim=n_actions, 
                    act_limit=2)

    action_space = None
    skill_distill_agent = SAC(num_inputs=n_obs,
                              num_actions=n_features,
                              action_space=action_space,
                              args=args
                              )
    sac_memory = ReplayMemory(capacity=args.replay_size, 
                          seed=args.seed
                          )
    
    residual_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.residual_agent.hid]*conf.residual_agent.l),
                        gamma=conf.residual_agent.gamma, 
                        seed=args.seed, 
                        steps_per_epoch=(conf.skill_agent.steps_per_epoch*seq_len), 
                        epochs=conf.setup.epochs,
                        clip_ratio=conf.residual_agent.clip_ratio, 
                        pi_lr=conf.residual_agent.pi_lr,
                        vf_lr=conf.residual_agent.vf_lr, 
                        train_pi_iters=conf.residual_agent.train_pi_iters, 
                        train_v_iters=conf.residual_agent.train_v_iters,
                        lam=conf.residual_agent.lam, 
                        target_kl=conf.residual_agent.target_kl, 
                        obs_dim=n_obs + n_features + env.action_space.shape[0], 
                        act_dim=env.action_space.shape[0], 
                        act_limit=1)

    print("Training RL agent...")
    train(agent=skill_agent,
          residual_agent=residual_agent,  
          skill_distill=skill_distill_agent,
          sac_replay=sac_memory,
          env=env,
          skill_vae=skill_vae,
          skill_prior=skill_prior,
          logistic_C=conf.setup.logistic_C,
          logistic_k=conf.setup.logistic_k,
          save_path=save_path,
          save_path_residual=save_path_residual,
          writer=writer,
          prior_model=args.prior_model,
          args=args)

    if proc_id() == 0 and writer is not None:
        writer.close()
    
    sac_memory.save_buffer(env.spec.id, str(args.seed)+"_"+args.prior_model)


if __name__ == '__main__':
    main()
    
