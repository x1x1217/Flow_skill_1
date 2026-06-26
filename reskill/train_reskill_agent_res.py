
import gym
import torch
import pdb
from tqdm import tqdm
import time
import numpy as np
import os
import sys
from reskill.rl.sac.sac import SAC
from reskill.rl.sac.replay_memory import ReplayMemory
from reskill.rl.utils.mpi_tools import num_procs, mpi_fork, proc_id
from reskill.rl.agents.ppo import PPO
from reskill.rl.agents.chunk_critic import ChunkReplayBuffer, LatentChunkCritic
from reskill.utils.general_utils import AttrDict
import reskill.rl.envs
from reskill.models.bc_diffusion import Diffusion_BC
import math
from reskill.utils.swanlab_writer import SwanLabWriter


device = torch.device('cuda')
RESIDUAL_ACTION_LIMIT = 0.5


def register_legacy_model_modules():
    import reskill.models as reskill_models
    import reskill.models.bc_diffusion as bc_diffusion
    import reskill.models.cvae as cvae
    import reskill.models.diffusion as diffusion
    import reskill.models.helpers as helpers
    import reskill.models.model as model
    import reskill.models.normal_mlp as normal_mlp
    import reskill.models.rnvp as rnvp
    import reskill.models.skill_vae as skill_vae

    sys.modules.setdefault('models', reskill_models)
    sys.modules.setdefault('models.bc_diffusion', bc_diffusion)
    sys.modules.setdefault('models.cvae', cvae)
    sys.modules.setdefault('models.diffusion', diffusion)
    sys.modules.setdefault('models.helpers', helpers)
    sys.modules.setdefault('models.model', model)
    sys.modules.setdefault('models.normal_mlp', normal_mlp)
    sys.modules.setdefault('models.rnvp', rnvp)
    sys.modules.setdefault('models.skill_vae', skill_vae)


def get_obs(obs, env_name):
    if env_name == "FetchPyramidStack-v0": 
        out = torch.FloatTensor(np.concatenate((obs["observation"][:-6], obs["desired_goal"][-3:]))).unsqueeze(dim=0).to(device)
    else:
        out = torch.FloatTensor(np.concatenate((obs["observation"], obs["desired_goal"]))).unsqueeze(dim=0).to(device)
    return out

def logistic_fn(step, k=0.001, C=18000):
    return 1/(1 + math.exp(-k * (step - C)))

def flow_guidance_enabled(args, epoch):
    return (args.prior_model == 'Flow' and args.use_grad == 1 and args.guidance_scale > 0 and epoch >= args.guidance_warmup_epoch)


def condition_guidance_enabled(args, epoch):
    return (
        args.prior_model == 'Flow'
        and args.use_condition_flow == 1
        and args.condition_use_grad == 1
        and args.condition_guidance_scale > 0
        and epoch >= args.condition_guidance_warmup_epoch
    )


def sample_condition(condition_prior, obs, condition_critic, args, epoch, guidance_stats=None):
    if condition_guidance_enabled(args, epoch):
        if condition_critic is None:
            raise RuntimeError("condition_critic is required when condition guidance is enabled.")
        return condition_prior.sample_z_guided_torch(
            obs,
            q_fn=condition_critic.q_fn_from_obs_latent,
            n_obs=args.n_obs,
            guidance_scale=args.condition_guidance_scale,
            grad_clip=args.condition_guidance_grad_clip,
            guidance_normalize=args.condition_guidance_normalize,
            guidance_stats=guidance_stats,
        ).detach(), args.condition_guidance_scale
    return condition_prior.sample_z_torch(obs).detach(), 0.0


def mean_or_zero(values):
    return float(np.mean(values)) if len(values) > 0 else 0.0


def extend_guidance_stats(acc, stats):
    for key, values in stats.items():
        acc.setdefault(key, []).extend(values)


def log_guidance_stats(writer, prefix, stats, step):
    for key, values in stats.items():
        writer.add_scalar(f"{prefix}/{key}", mean_or_zero(values), step)
        if len(values) > 0:
            writer.add_scalar(f"{prefix}/{key}_p90", float(np.percentile(values, 90)), step)
            writer.add_scalar(f"{prefix}/{key}_max", float(np.max(values)), step)


def align_steps_per_epoch(raw_steps_per_epoch, max_ep_len, seq_len):
    episode_skills = int(math.ceil(max_ep_len / seq_len))
    local_raw_steps = int(raw_steps_per_epoch / num_procs())
    local_aligned_steps = (local_raw_steps // episode_skills) * episode_skills
    if local_aligned_steps <= 0:
        raise ValueError(
            "steps_per_epoch is too small for one full episode after MPI split: "
            f"raw_steps_per_epoch={raw_steps_per_epoch}, num_procs={num_procs()}, "
            f"max_ep_len={max_ep_len}, seq_len={seq_len}"
        )
    return local_aligned_steps * num_procs(), episode_skills


def collect_initial_chunk_rollout(env, skill_vae, skill_prior, condition_prior, chunk_replay, args):
    if args.init_rollout_steps <= 0:
        return 0
    if chunk_replay is None:
        return 0

    env_name = env.spec.id
    env_step_cnt = 0
    obs, ep_len = env.reset(), 0
    o = get_obs(obs, env_name)

    while env_step_cnt < args.init_rollout_steps:
        o_start = o
        with torch.no_grad():
            if args.use_condition_flow == 1:
                n = condition_prior.sample_z_torch(o_start).detach()
            else:
                raise RuntimeError("Initial rollout for Q_z guidance requires --use_condition_flow 1.")
            cond = torch.cat((o_start, n), dim=1).to(device)
            z = skill_prior.sample_z_torch(cond).detach()

        next_o = o_start
        chunk_r = 0.0
        done = False

        for step in range(skill_vae.seq_len):
            with torch.no_grad():
                a_dec = skill_vae.decoder(torch.cat((next_o, z), 1))
            obs, r, done, _ = env.step(a_dec.cpu().detach().numpy()[0])
            env_step_cnt += 1
            ep_len += 1
            chunk_r += (args.chunk_critic_gamma ** step) * r
            next_o = get_obs(obs, env_name)

            if done or ep_len >= args.max_ep_len or env_step_cnt >= args.init_rollout_steps:
                break

        terminal = done or ep_len >= args.max_ep_len
        chunk_replay.push(
            o_start.cpu().detach().numpy()[0],
            z.cpu().detach().numpy()[0],
            chunk_r,
            next_o.cpu().detach().numpy()[0],
            float(terminal),
        )

        if terminal:
            obs, ep_len = env.reset(), 0
            o = get_obs(obs, env_name)
        else:
            o = next_o

    return env_step_cnt


def resolve_skill_model_dir(curr_dir, args):
    candidate_dirs = [
        os.path.join(
            curr_dir,
            "results",
            "saved_skill_models",
            args.dataset_name,
            args.prior_model,
            f"seed_{args.seed}",
            f"skill_prior_{args.prior_run_name}",
        ),
        os.path.join(
            curr_dir,
            "results",
            "saved_skill_models",
            args.dataset_name,
            f"seed_{args.seed}",
            f"skill_prior_{args.prior_run_name}",
        ),
        os.path.join(
            curr_dir,
            "results",
            "saved_skill_models",
            args.dataset_name,
            f"seed_{args.seed}",
            f"skill_prior_{args.prior_model}",
        ),
    ]

    required = ["skill_vae", "skill_prior"]
    if args.use_condition_flow == 1:
        required.append("condition_prior")

    for model_dir in candidate_dirs:
        if all(resolve_skill_checkpoint_path(model_dir, name, args.skill_checkpoint) is not None for name in required):
            return model_dir

    checked = "\n".join(candidate_dirs)
    raise FileNotFoundError(
        "Could not find skill module checkpoint directory. Checked:\n"
        f"{checked}\n"
        f"checkpoint={args.skill_checkpoint}, required={required}"
    )


def resolve_skill_checkpoint_path(model_dir, name, checkpoint):
    if checkpoint == "best":
        candidates = [
            os.path.join(model_dir, f"best_{name}.pth"),
            os.path.join(model_dir, f"{name}.pth"),
        ]
    elif checkpoint == "last":
        candidates = [
            os.path.join(model_dir, f"{name}.pth"),
        ]
    else:
        raise ValueError("--skill_checkpoint must be 'best' or 'last'.")

    for path in candidates:
        if os.path.exists(path):
            return path
    return None

def train(agent, latent_q_agent, chunk_critic, chunk_replay, condition_critic, condition_replay, condition_prior,
          residual_agent, env, skill_vae, skill_prior, logistic_C, logistic_k,
          max_residual_factor,
          save_path, save_path_residual, save_path_latent_q, save_path_chunk_critic, save_path_condition_critic,
          writer, prior_model, args):

    env_name = env.spec.id
    init_env_steps = collect_initial_chunk_rollout(
        env,
        skill_vae,
        skill_prior,
        condition_prior,
        chunk_replay,
        args,
    )
    if proc_id() == 0 and args.init_rollout_steps > 0:
        print(
            f"Collected {init_env_steps} initial flow env steps "
            f"({len(chunk_replay) if chunk_replay is not None else 0} chunk transitions).",
            flush=True,
        )
        if writer is not None:
            writer.add_scalar(
                "skill_flow_q_guidance/init_replay_size",
                len(chunk_replay) if chunk_replay is not None else 0,
                init_env_steps,
            )
    obs, ep_ret, ep_len = env.reset(), 0, 0
    o = get_obs(obs, env_name)
    updates = 0

    env_step_cnt = 0
    residual_factor = 0.0
    ep_ret_smooth_window_steps = 10000
    ep_ret_smooth_next_step = ep_ret_smooth_window_steps
    ep_ret_smooth_sum = 0.0
    ep_ret_smooth_count = 0
    ep_ret_recent = []
    ep_ret_recent_window = 20

    local_steps_per_epoch = int(agent.steps_per_epoch / num_procs())

    for epoch in tqdm(range(agent.epochs)):
        rollout_condition_guidance_scales = []
        rollout_skill_guidance_scales = []
        rollout_condition_guidance_stats = {}
        rollout_skill_guidance_stats = {}
        rollout_condition_norm = []
        rollout_z_norm = []
        rollout_decoder_action_norm = []
        rollout_residual_action_norm = []
        rollout_env_action_norm = []
        rollout_chunk_reward = []
        rollout_chunk_discounted_reward = []

        for t in range(local_steps_per_epoch):
            # Select condition vector using either high-level PPO or condition flow policy.
            if args.use_condition_flow == 1:
                condition_stats = {}
                n, condition_guidance_scale = sample_condition(
                    condition_prior,
                    o,
                    condition_critic,
                    args,
                    epoch,
                    guidance_stats=condition_stats,
                )
                extend_guidance_stats(rollout_condition_guidance_stats, condition_stats)
                v_agent, logp_agent = None, None
            else:
                n, v_agent, logp_agent, mu, std = agent.ac.step(torch.as_tensor(o, dtype=torch.float32))
                condition_guidance_scale = 0.0
            rollout_condition_guidance_scales.append(condition_guidance_scale)
            rollout_condition_norm.append(n.norm(dim=1).mean().item())

            skill_guidance_scale = 0.0
            if prior_model == 'RNVP':
                sample = AttrDict(noise=n, state=o)
                # Warp noise vector to latent space skill
                z = skill_prior.inverse(sample).noise.detach()
            elif prior_model == 'Flow':
                cond = torch.cat((o, n), dim=1).to(device)
                if flow_guidance_enabled(args, epoch):
                    skill_guidance_scale = args.guidance_scale
                    skill_stats = {}
                    z = skill_prior.sample_z_guided_torch(
                        cond,
                        q_fn=chunk_critic.q_fn_from_obs_latent,
                        n_obs=args.n_obs,
                        guidance_scale=args.guidance_scale,
                        grad_clip=args.guidance_grad_clip,
                        guidance_normalize=args.guidance_normalize,
                        guidance_stats=skill_stats,
                    ).detach()
                    extend_guidance_stats(rollout_skill_guidance_stats, skill_stats)
                else:
                    z = skill_prior.sample_z_torch(cond).detach()
            elif prior_model == 'Diffusion':
                state_ = torch.cat((o, n), dim=1).to(device)
                #print(state_)
                if args.use_grad == 1:
                    z = skill_prior.sample_action_guide_repeat(state_, latent_q_agent.ac.q, args.n_obs, 1)
                else:
                    z = skill_prior.sample_action_torch(state_)
                v_latent_q, logp_latent_q = latent_q_agent.ac.v_logp(torch.as_tensor(o, dtype=torch.float32), z)
            elif prior_model == 'CVAE':
                state_ = torch.cat((o, n), dim=1).to(device)
                lat = torch.normal(0, 1, (skill_prior.latent_dim, )).unsqueeze(0).cuda()
                input = torch.cat([lat, state_], dim=1)
                z = skill_prior.decode(input).detach()
            elif prior_model == 'MLP':
                state_ = torch.cat((o, n), dim=1).to(device)
                z = skill_prior.net(state_).detach()
            rollout_skill_guidance_scales.append(skill_guidance_scale)
            rollout_z_norm.append(z.norm(dim=1).mean().item())

            o2, skill_r = o, 0
            chunk_r = 0.0
        
            for step in range(skill_vae.seq_len):
                """if len(sac_replay) > args.batch_size:
                    critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha = skill_distill.update_parameters(sac_replay, args.batch_size, updates)
                    writer.add_scalar('loss/critic_1', critic_1_loss, updates)
                    writer.add_scalar('loss/critic_2', critic_2_loss, updates)
                    writer.add_scalar('loss/policy', policy_loss, updates)
                    writer.add_scalar('loss/entropy_loss', ent_loss, updates)
                    writer.add_scalar('entropy_temprature/alpha', alpha, updates)
                    updates += 1"""
                
                obs_z = torch.cat((o2, z), 1)
                a_dec = skill_vae.decoder(obs_z)

                o_res = torch.cat((o2,z, a_dec), 1)
                a_res, _, _, _, _ = residual_agent.ac.step(o_res)
                a_res = torch.clamp(a_res, -RESIDUAL_ACTION_LIMIT, RESIDUAL_ACTION_LIMIT)
                v_res, logp_res = residual_agent.ac.v_logp(o_res, a_res)
                
                a = (a_dec.cpu().detach().numpy() + (a_res.cpu().detach().numpy() * residual_factor))[0]
                #a = a_dec.cpu().detach().numpy()[0]
                rollout_decoder_action_norm.append(a_dec.norm(dim=1).mean().item())
                rollout_residual_action_norm.append(a_res.norm(dim=1).mean().item())
                rollout_env_action_norm.append(float(np.linalg.norm(a)))
                
                # Step the env
                obs, r, d, _ = env.step(a)

                env_step_cnt += 1

                skill_r += r #Sum rewards for high level policy
                chunk_r += (args.chunk_critic_gamma ** step) * r
                ep_ret += r
                ep_len += 1

                o_next = get_obs(obs, env_name)
                #mask = float(d)
                #sac_replay.push(o2.cpu().detach().numpy()[0], a, skill_r, o_next.cpu().detach().numpy()[0], mask)

                o2 = o_next
                
                if step == skill_vae.seq_len - 1:
                    a_dec_next = skill_vae.decoder(torch.cat((o2, z), 1))
                    o2_res = torch.cat((o2, z, a_dec_next), 1)

                residual_agent.buf.store(o_res.cpu().detach(), a_res.cpu().detach(), r, v_res, logp_res)
                
                

            # Update residual action weighting factor
            residual_factor = min(logistic_fn(env_step_cnt, k=logistic_k, C=logistic_C), max_residual_factor)
            if proc_id() == 0:
                writer.add_scalar('ppo/logistic_fn', residual_factor, env_step_cnt)
                writer.add_scalar('residual/factor', residual_factor, env_step_cnt)

            # save and log
            o_start_np = o.cpu().detach().numpy()[0]
            if args.use_condition_flow == 0:
                agent.buf.store(o.cpu().detach(), n.cpu().detach(), skill_r, v_agent, logp_agent)
            if latent_q_agent is not None:
                latent_q_agent.buf.store(o.cpu().detach(), z.cpu().detach(), skill_r, v_latent_q, logp_latent_q)

            o = o2

            timeout = ep_len >= agent.max_ep_len
            terminal = d or timeout
            epoch_ended = t == local_steps_per_epoch-1
            if condition_replay is not None:
                condition_replay.push(
                    o_start_np,
                    n.cpu().detach().numpy()[0],
                    chunk_r,
                    o2.cpu().detach().numpy()[0],
                    float(terminal),
                )
            if chunk_replay is not None:
                chunk_replay.push(
                    o_start_np,
                    z.cpu().detach().numpy()[0],
                    chunk_r,
                    o2.cpu().detach().numpy()[0],
                    float(terminal),
                )
            rollout_chunk_reward.append(skill_r)
            rollout_chunk_discounted_reward.append(chunk_r)
        
            if terminal or epoch_ended:
                if epoch_ended and not(terminal):
                    print('Warning: trajectory cut off by epoch at %d steps.'%ep_len, flush=True)
                # if trajectory didn't reach terminal state, bootstrap value target
                if timeout or epoch_ended:
                    if args.use_condition_flow == 0:
                        _, v_agent, _, _, _ = agent.ac.step(o)
                    if latent_q_agent is not None:
                        _, v_latent_q, _, _, _ = latent_q_agent.ac.step(o)
                    _, v_res, _, _, _ = residual_agent.ac.step(o2_res)
                else:
                    if args.use_condition_flow == 0:
                        v_agent = 0
                    if latent_q_agent is not None:
                        v_latent_q = 0
                    v_res = 0
                if args.use_condition_flow == 0:
                    agent.buf.finish_path(v_agent)
                if latent_q_agent is not None:
                    latent_q_agent.buf.finish_path(v_latent_q)
                residual_agent.buf.finish_path(v_res)
                if terminal:
                    if proc_id() == 0:
                        writer.add_scalar('Episode Return', ep_ret, env_step_cnt)
                        ep_ret_recent.append(ep_ret)
                        if len(ep_ret_recent) > ep_ret_recent_window:
                            ep_ret_recent.pop(0)
                        writer.add_scalar('Episode Return Smoothed Recent', mean_or_zero(ep_ret_recent), env_step_cnt)
                        ep_ret_smooth_sum += ep_ret
                        ep_ret_smooth_count += 1
                        if env_step_cnt >= ep_ret_smooth_next_step and ep_ret_smooth_count > 0:
                            writer.add_scalar('Episode Return Smoothed', ep_ret_smooth_sum / ep_ret_smooth_count, env_step_cnt)
                            ep_ret_smooth_sum = 0.0
                            ep_ret_smooth_count = 0
                            while ep_ret_smooth_next_step <= env_step_cnt:
                                ep_ret_smooth_next_step += ep_ret_smooth_window_steps
                                
                obs, ep_ret, ep_len = env.reset(), 0, 0
                o = get_obs(obs, env_name)

        # Save model
        if (epoch % agent.save_freq == 0) or (epoch == agent.epochs-1):
            if args.use_condition_flow == 0:
                torch.save(agent.ac.pi, save_path)
            if latent_q_agent is not None:
                torch.save(latent_q_agent.ac, save_path_latent_q)
            if chunk_critic is not None:
                torch.save(chunk_critic.state_dict(), save_path_chunk_critic)
            if condition_critic is not None:
                torch.save(condition_critic.state_dict(), save_path_condition_critic)
            #skill_distill.save_checkpoint(env_name, str(args.seed)+"_"+args.prior_model)
            torch.save(residual_agent.ac.pi, save_path_residual)

        # Perform PPO update!
        if args.use_condition_flow == 0:
            losses = agent.update(name='skill')
        else:
            losses = AttrDict(LossPi=0.0, LossV=0.0, KL=0.0, Entropy=0.0, ClipFrac=0.0, PiIters=0, DeltaLossPi=0.0, DeltaLossV=0.0)
        if latent_q_agent is not None:
            latent_q_losses = latent_q_agent.update(name='latent_q')
        residual_losses = residual_agent.update(name='residual')
        if chunk_critic is not None:
            chunk_critic_losses = chunk_critic.update_with_flow_policy(
                chunk_replay,
                agent,
                skill_prior,
                args,
                condition_prior=condition_prior,
            )
        if condition_critic is not None:
            condition_critic_losses = condition_critic.update_with_condition_policy(
                condition_replay,
                condition_prior,
                args,
            )

        success_traj = 0
        total_r = 0
        r_time = 0
        eval_condition_guidance_scales = []
        eval_skill_guidance_scales = []
        eval_condition_guidance_stats = {}
        eval_skill_guidance_stats = {}
        eval_condition_norm = []
        eval_z_norm = []
        eval_decoder_action_norm = []
        eval_residual_action_norm = []
        eval_env_action_norm = []

        for roll in range(50):
            obs = env.reset()
            obs = get_obs(obs, env_name)

            steps = 0
            r = 0
            episode_success = False
            while steps < env._max_episode_steps:

                # Use high-level condition source.
                if args.use_condition_flow == 1:
                    condition_stats = {}
                    n, condition_guidance_scale = sample_condition(
                        condition_prior,
                        obs,
                        condition_critic,
                        args,
                        epoch,
                        guidance_stats=condition_stats,
                    )
                    extend_guidance_stats(eval_condition_guidance_stats, condition_stats)
                else:
                    n = agent.ac.act_deterministic(obs)
                    n = torch.FloatTensor(n)
                    condition_guidance_scale = 0.0
                eval_condition_guidance_scales.append(condition_guidance_scale)
                eval_condition_norm.append(n.norm(dim=1).mean().item())

                skill_guidance_scale = 0.0
                if prior_model == "RNVP":
                    skill = AttrDict(noise=n, state=obs)
                    z = skill_prior.inverse(skill).noise.detach()
                elif prior_model == "Flow":
                    n = n.cuda()
                    cond = torch.cat((obs, n), dim=1).cuda()
                    if flow_guidance_enabled(args, epoch):
                        skill_guidance_scale = args.guidance_scale
                        skill_stats = {}
                        z = skill_prior.sample_z_guided_torch(
                            cond,
                            q_fn=chunk_critic.q_fn_from_obs_latent,
                            n_obs=args.n_obs,
                            guidance_scale=args.guidance_scale,
                            grad_clip=args.guidance_grad_clip,
                            guidance_normalize=args.guidance_normalize,
                            guidance_stats=skill_stats,
                        ).detach()
                        extend_guidance_stats(eval_skill_guidance_stats, skill_stats)
                    else:
                        z = skill_prior.sample_z_torch(cond).detach()
                elif prior_model == "Diffusion":
                    n = n.cuda()
                    state_ = torch.cat((obs, n), dim=1).cuda()
                    if args.use_grad == 1:
                        z = skill_prior.sample_action_guide_repeat(state_, latent_q_agent.ac.q, args.n_obs, 1)
                    else:
                        z = skill_prior.sample_action_torch(state_)
                elif prior_model == 'MLP':
                    n = n.cuda()
                    state_ = torch.cat((obs, n), dim=1).cuda()
                    z = skill_prior.net(state_)
                elif prior_model == 'CVAE':
                    zz = torch.normal(0, 1, (skill_prior.latent_dim,)).unsqueeze(0).cuda()
                    n = n.cuda()
                    state_ = torch.cat((obs, n), dim=1).cuda()
                    input = torch.cat([zz, state_], dim=1).to(device)
                    z = skill_prior.decode(input)
                eval_skill_guidance_scales.append(skill_guidance_scale)
                eval_z_norm.append(z.norm(dim=1).mean().item())

                o2, skill_r = o, 0
            
                for _ in range(skill_vae.seq_len):

                    obs_z = torch.cat((obs, z), 1)
                    a_dec = skill_vae.decoder(obs_z)
                    
                    # Get residual action
                    o_res = torch.cat((obs,z,a_dec), 1)
                    dist = residual_agent.ac.act_deterministic(o_res)
                    a_res = np.clip(dist, -RESIDUAL_ACTION_LIMIT, RESIDUAL_ACTION_LIMIT)

                    # Add residual action to decoded action
                    a = (a_dec.cpu().detach().numpy() + (a_res * residual_factor))[0]
                    #a = a_dec.cpu().detach().numpy()[0]
                    eval_decoder_action_norm.append(a_dec.norm(dim=1).mean().item())
                    eval_residual_action_norm.append(float(np.linalg.norm(a_res)))
                    eval_env_action_norm.append(float(np.linalg.norm(a)))
                    obs, reward, done, debug_info = env.step(a)
                    r += reward
                    episode_success = episode_success or bool(debug_info['is_success'])
                    obs = get_obs(obs, env_name)

                    #env.render()
                    
                    steps += 1
                    if steps >= env._max_episode_steps:
                        break

            if episode_success:
                success_traj += 1
            total_r += r
            if r > 0:
                r_time += 1

        if proc_id() == 0:
            writer.add_scalar('pi_loss_', losses.LossPi, env_step_cnt)
            writer.add_scalar('v_loss_', losses.LossV, env_step_cnt)
            writer.add_scalar('kl', losses.KL, env_step_cnt)
            writer.add_scalar('entropy', losses.Entropy, env_step_cnt)
            writer.add_scalar('clip_frac', losses.ClipFrac, env_step_cnt)
            writer.add_scalar('pi_iters', losses.PiIters, env_step_cnt)
            writer.add_scalar('delta_loss_pi', losses.DeltaLossPi, env_step_cnt)
            writer.add_scalar('delta_loss_v', losses.DeltaLossV, env_step_cnt)
            if latent_q_agent is not None:
                writer.add_scalar('latent_q/pi_loss', latent_q_losses.LossPi, env_step_cnt)
                writer.add_scalar('latent_q/v_loss', latent_q_losses.LossV, env_step_cnt)
                writer.add_scalar('latent_q/q_loss', latent_q_losses.DeltaLossQ, env_step_cnt)
                writer.add_scalar('latent_q/kl', latent_q_losses.KL, env_step_cnt)
                writer.add_scalar('latent_q/entropy', latent_q_losses.Entropy, env_step_cnt)
                writer.add_scalar('latent_q/clip_frac', latent_q_losses.ClipFrac, env_step_cnt)
                writer.add_scalar('latent_q/pi_iters', latent_q_losses.PiIters, env_step_cnt)
            if chunk_critic is not None:
                writer.add_scalar('chunk_critic/q_loss', chunk_critic_losses.q_loss, env_step_cnt)
                writer.add_scalar('chunk_critic/q1_loss', chunk_critic_losses.q1_loss, env_step_cnt)
                writer.add_scalar('chunk_critic/q2_loss', chunk_critic_losses.q2_loss, env_step_cnt)
                writer.add_scalar('chunk_critic/target_q', chunk_critic_losses.target_q, env_step_cnt)
                writer.add_scalar('chunk_critic/current_q', chunk_critic_losses.current_q, env_step_cnt)
                writer.add_scalar('chunk_critic/replay_size', len(chunk_replay), env_step_cnt)
                writer.add_scalar('chunk_critic/positive_fraction', chunk_critic_losses.positive_fraction, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/z_q_loss', chunk_critic_losses.q_loss, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/z_q1_loss', chunk_critic_losses.q1_loss, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/z_q2_loss', chunk_critic_losses.q2_loss, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/z_target_q', chunk_critic_losses.target_q, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/z_current_q', chunk_critic_losses.current_q, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/positive_fraction', chunk_critic_losses.positive_fraction, env_step_cnt)
                writer.add_scalar('skill_flow_q_guidance/replay_size', len(chunk_replay), env_step_cnt)
            if condition_critic is not None:
                writer.add_scalar('condition_critic/q_loss', condition_critic_losses.q_loss, env_step_cnt)
                writer.add_scalar('condition_critic/q1_loss', condition_critic_losses.q1_loss, env_step_cnt)
                writer.add_scalar('condition_critic/q2_loss', condition_critic_losses.q2_loss, env_step_cnt)
                writer.add_scalar('condition_critic/target_q', condition_critic_losses.target_q, env_step_cnt)
                writer.add_scalar('condition_critic/current_q', condition_critic_losses.current_q, env_step_cnt)
                writer.add_scalar('condition_critic/replay_size', len(condition_replay), env_step_cnt)
                writer.add_scalar('condition_critic/positive_fraction', condition_critic_losses.positive_fraction, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/c_q_loss', condition_critic_losses.q_loss, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/c_q1_loss', condition_critic_losses.q1_loss, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/c_q2_loss', condition_critic_losses.q2_loss, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/c_target_q', condition_critic_losses.target_q, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/c_current_q', condition_critic_losses.current_q, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/positive_fraction', condition_critic_losses.positive_fraction, env_step_cnt)
                writer.add_scalar('condition_flow_q_guidance/replay_size', len(condition_replay), env_step_cnt)
            writer.add_scalar('rollout/condition_guidance_scale', mean_or_zero(rollout_condition_guidance_scales), env_step_cnt)
            writer.add_scalar('rollout/skill_guidance_scale', mean_or_zero(rollout_skill_guidance_scales), env_step_cnt)
            writer.add_scalar('rollout/guidance_scale', mean_or_zero(rollout_skill_guidance_scales), env_step_cnt)
            log_guidance_stats(writer, 'rollout/qc_guidance', rollout_condition_guidance_stats, env_step_cnt)
            log_guidance_stats(writer, 'rollout/qz_guidance', rollout_skill_guidance_stats, env_step_cnt)
            writer.add_scalar('rollout/condition_norm', mean_or_zero(rollout_condition_norm), env_step_cnt)
            writer.add_scalar('rollout/noise_norm', mean_or_zero(rollout_condition_norm), env_step_cnt)
            writer.add_scalar('rollout/z_norm', mean_or_zero(rollout_z_norm), env_step_cnt)
            writer.add_scalar('rollout/decoder_action_norm', mean_or_zero(rollout_decoder_action_norm), env_step_cnt)
            writer.add_scalar('rollout/residual_action_norm', mean_or_zero(rollout_residual_action_norm), env_step_cnt)
            writer.add_scalar('rollout/env_action_norm', mean_or_zero(rollout_env_action_norm), env_step_cnt)
            writer.add_scalar('rollout/base_action_norm', mean_or_zero(rollout_decoder_action_norm), env_step_cnt)
            writer.add_scalar('rollout/chunk_reward', mean_or_zero(rollout_chunk_reward), env_step_cnt)
            writer.add_scalar('rollout/chunk_discounted_reward', mean_or_zero(rollout_chunk_discounted_reward), env_step_cnt)
            writer.add_scalar('residual/pi_loss', residual_losses.LossPi, env_step_cnt)
            writer.add_scalar('residual/v_loss', residual_losses.LossV, env_step_cnt)
            writer.add_scalar('residual/q_loss', residual_losses.DeltaLossQ, env_step_cnt)
            writer.add_scalar('residual/kl', residual_losses.KL, env_step_cnt)
            writer.add_scalar('residual/entropy', residual_losses.Entropy, env_step_cnt)
            writer.add_scalar('residual/clip_frac', residual_losses.ClipFrac, env_step_cnt)
            writer.add_scalar('residual/pi_iters', residual_losses.PiIters, env_step_cnt)
            writer.add_scalar('eval_epoch/success_traj', success_traj, epoch)
            writer.add_scalar('eval_epoch/success_rate', success_traj / 50, epoch)
            writer.add_scalar('eval_epoch/reward_sum', total_r, epoch)
            writer.add_scalar('eval_epoch/avg_reward', total_r / 50, epoch)
            writer.add_scalar('eval_epoch/avg_reward_time', r_time / 50, epoch)
            writer.add_scalar('eval_epoch/condition_guidance_scale', mean_or_zero(eval_condition_guidance_scales), epoch)
            writer.add_scalar('eval_epoch/skill_guidance_scale', mean_or_zero(eval_skill_guidance_scales), epoch)
            log_guidance_stats(writer, 'eval_epoch/qc_guidance', eval_condition_guidance_stats, epoch)
            log_guidance_stats(writer, 'eval_epoch/qz_guidance', eval_skill_guidance_stats, epoch)
            writer.add_scalar('eval_epoch/condition_norm', mean_or_zero(eval_condition_norm), epoch)
            writer.add_scalar('eval_epoch/noise_norm', mean_or_zero(eval_condition_norm), epoch)
            writer.add_scalar('eval_epoch/z_norm', mean_or_zero(eval_z_norm), epoch)
            writer.add_scalar('eval_epoch/decoder_action_norm', mean_or_zero(eval_decoder_action_norm), epoch)
            writer.add_scalar('eval_epoch/residual_action_norm', mean_or_zero(eval_residual_action_norm), epoch)
            writer.add_scalar('eval_epoch/env_action_norm', mean_or_zero(eval_env_action_norm), epoch)
            writer.add_scalar('eval_epoch/base_action_norm', mean_or_zero(eval_decoder_action_norm), epoch)

            writer.add_scalar('eval_step/success_traj', success_traj, env_step_cnt)
            writer.add_scalar('eval_step/success_rate', success_traj / 50, env_step_cnt)
            writer.add_scalar('eval_step/reward_sum', total_r, env_step_cnt)
            writer.add_scalar('eval_step/avg_reward', total_r / 50, env_step_cnt)
            writer.add_scalar('eval_step/avg_reward_time', r_time / 50, env_step_cnt)
            writer.add_scalar('eval_step/condition_guidance_scale', mean_or_zero(eval_condition_guidance_scales), env_step_cnt)
            writer.add_scalar('eval_step/skill_guidance_scale', mean_or_zero(eval_skill_guidance_scales), env_step_cnt)
            log_guidance_stats(writer, 'eval_step/qc_guidance', eval_condition_guidance_stats, env_step_cnt)
            log_guidance_stats(writer, 'eval_step/qz_guidance', eval_skill_guidance_stats, env_step_cnt)
            writer.add_scalar('eval_step/condition_norm', mean_or_zero(eval_condition_norm), env_step_cnt)
            writer.add_scalar('eval_step/noise_norm', mean_or_zero(eval_condition_norm), env_step_cnt)
            writer.add_scalar('eval_step/z_norm', mean_or_zero(eval_z_norm), env_step_cnt)
            writer.add_scalar('eval_step/decoder_action_norm', mean_or_zero(eval_decoder_action_norm), env_step_cnt)
            writer.add_scalar('eval_step/residual_action_norm', mean_or_zero(eval_residual_action_norm), env_step_cnt)
            writer.add_scalar('eval_step/env_action_norm', mean_or_zero(eval_env_action_norm), env_step_cnt)
            writer.add_scalar('eval_step/base_action_norm', mean_or_zero(eval_decoder_action_norm), env_step_cnt)


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
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')

    parser.add_argument('--config_file', type=str, default="table_cleanup/config.yaml")
    parser.add_argument('--prior_model', type=str, default='Diffusion')
    parser.add_argument('--seed', type=int, default=21)
    parser.add_argument('--pick', type=int, default=1)
    parser.add_argument('--push', type=int, default=1)
    parser.add_argument('--dataset_name', type=str, default=None)
    parser.add_argument('--use_sigma', type=int, default=1)
    parser.add_argument('--use_grad', type=int, default=1)
    parser.add_argument('--use_student', type=int, default=1)
    parser.add_argument('--guidance_scale', type=float, default=0.0)
    parser.add_argument('--guidance_warmup_epoch', type=int, default=0)
    parser.add_argument('--guidance_grad_clip', type=float, default=0.0)
    parser.add_argument('--guidance_normalize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--condition_use_grad', type=int, default=0)
    parser.add_argument('--condition_guidance_scale', type=float, default=0.0)
    parser.add_argument('--condition_guidance_warmup_epoch', type=int, default=0)
    parser.add_argument('--condition_guidance_grad_clip', type=float, default=0.0)
    parser.add_argument('--condition_guidance_normalize', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--condition_critic_hidden_dim', type=int, default=None)
    parser.add_argument('--condition_critic_hidden_layers', type=int, default=None)
    parser.add_argument('--condition_critic_activation', type=str, default=None, choices=["relu", "tanh"])
    parser.add_argument('--condition_critic_layer_norm', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--condition_critic_lr', type=float, default=None)
    parser.add_argument('--condition_critic_tau', type=float, default=None)
    parser.add_argument('--condition_critic_batch_size', type=int, default=None)
    parser.add_argument('--condition_critic_updates_per_epoch', type=int, default=None)
    parser.add_argument('--condition_critic_replay_size', type=int, default=None)
    parser.add_argument('--condition_critic_ensembles', type=int, default=None)
    parser.add_argument('--condition_positive_replay_ratio', type=float, default=None)
    parser.add_argument('--condition_positive_reward_threshold', type=float, default=None)
    parser.add_argument('--chunk_critic_hidden_dim', type=int, default=256)
    parser.add_argument('--chunk_critic_hidden_layers', type=int, default=2)
    parser.add_argument('--chunk_critic_activation', type=str, default="relu", choices=["relu", "tanh"])
    parser.add_argument('--chunk_critic_layer_norm', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--chunk_critic_lr', type=float, default=3e-4)
    parser.add_argument('--chunk_critic_tau', type=float, default=0.005)
    parser.add_argument('--chunk_critic_gamma', type=float, default=None)
    parser.add_argument('--chunk_critic_batch_size', type=int, default=256)
    parser.add_argument('--chunk_critic_updates_per_epoch', type=int, default=200)
    parser.add_argument('--chunk_critic_replay_size', type=int, default=1000000)
    parser.add_argument('--chunk_critic_ensembles', type=int, default=1)
    parser.add_argument('--init_rollout_steps', type=int, default=0)
    parser.add_argument('--positive_replay_ratio', type=float, default=0.0)
    parser.add_argument('--positive_reward_threshold', type=float, default=0.0)
    parser.add_argument('--max_residual_factor', type=float, default=None)
    parser.add_argument('--use_condition_flow', type=int, default=0)
    parser.add_argument('--skill_checkpoint', type=str, default="best", choices=["best", "last"])
    parser.add_argument('--swanlab_project', type=str, default="Flow_skill_1")
    parser.add_argument('--swanlab_workspace', type=str, default="x1x1217")
    parser.add_argument('--swanlab_mode', type=str, default=None)
    args=parser.parse_args()
    if args.use_condition_flow == 1 and args.prior_model != 'Flow':
        raise ValueError("--use_condition_flow is only implemented for --prior_model Flow.")

    if args.dataset_name is None:
        args.dataset_name = f'fetch_block_push{args.push}_pick{args.pick}'
    args.prior_run_name = args.prior_model
    args.run_variant = f"use_sigma_{args.use_sigma}_grad_{args.use_grad}"
    if args.prior_model == 'Flow':
        args.prior_run_name = f"{args.prior_model}_student{args.use_student}"
        args.run_variant = f"use_student_{args.use_student}_grad_{args.use_grad}"
        if args.use_condition_flow == 1:
            args.run_variant += "_condflow"
        if args.use_grad == 1:
            args.run_variant += (
                f"_gscale_{args.guidance_scale:g}"
                f"_gwarm_{args.guidance_warmup_epoch}"
                f"_gclip_{args.guidance_grad_clip:g}"
                f"_gnorm_{int(args.guidance_normalize)}"
                f"_h{args.chunk_critic_hidden_dim}x{args.chunk_critic_hidden_layers}"
                f"_ln_{int(args.chunk_critic_layer_norm)}"
                f"_act_{args.chunk_critic_activation}"
                f"_init_{args.init_rollout_steps}"
                f"_posratio_{args.positive_replay_ratio:g}"
                f"_chunkq_{args.chunk_critic_ensembles}"
                f"_resmax_{args.max_residual_factor if args.max_residual_factor is not None else 'cfg'}"
            )
        if args.condition_use_grad == 1:
            args.run_variant += (
                f"_cgscale_{args.condition_guidance_scale:g}"
                f"_cgwarm_{args.condition_guidance_warmup_epoch}"
                f"_cgclip_{args.condition_guidance_grad_clip:g}"
                f"_cgnorm_{int(args.condition_guidance_normalize)}"
            )
    curr_dir = os.path.dirname(__file__)
    config_path = os.path.join(curr_dir, "configs", "rl", args.config_file)
    
    with open(config_path, 'r') as file:
        conf = yaml.safe_load(file)
        conf = AttrDict(conf)
    for key in conf:
        conf[key] = AttrDict(conf[key])
    if args.chunk_critic_gamma is None:
        args.chunk_critic_gamma = conf.skill_agent.gamma
    if args.chunk_critic_ensembles < 1:
        raise ValueError("--chunk_critic_ensembles must be >= 1")
    if args.condition_critic_hidden_dim is None:
        args.condition_critic_hidden_dim = args.chunk_critic_hidden_dim
    if args.condition_critic_hidden_layers is None:
        args.condition_critic_hidden_layers = args.chunk_critic_hidden_layers
    if args.condition_critic_activation is None:
        args.condition_critic_activation = args.chunk_critic_activation
    if args.condition_critic_layer_norm is None:
        args.condition_critic_layer_norm = args.chunk_critic_layer_norm
    if args.condition_critic_lr is None:
        args.condition_critic_lr = args.chunk_critic_lr
    if args.condition_critic_tau is None:
        args.condition_critic_tau = args.chunk_critic_tau
    if args.condition_critic_batch_size is None:
        args.condition_critic_batch_size = args.chunk_critic_batch_size
    if args.condition_critic_updates_per_epoch is None:
        args.condition_critic_updates_per_epoch = args.chunk_critic_updates_per_epoch
    if args.condition_critic_replay_size is None:
        args.condition_critic_replay_size = args.chunk_critic_replay_size
    if args.condition_critic_ensembles is None:
        args.condition_critic_ensembles = args.chunk_critic_ensembles
    if args.condition_positive_replay_ratio is None:
        args.condition_positive_replay_ratio = args.positive_replay_ratio
    if args.condition_positive_reward_threshold is None:
        args.condition_positive_reward_threshold = args.positive_reward_threshold
    if args.condition_critic_ensembles < 1:
        raise ValueError("--condition_critic_ensembles must be >= 1")
    if args.condition_use_grad == 1 and args.condition_guidance_scale > 0:
        if args.prior_model != "Flow" or args.use_condition_flow != 1:
            raise ValueError("Condition guidance requires --prior_model Flow and --use_condition_flow 1.")
        if args.use_student == 1:
            raise ValueError("Condition guidance currently requires --use_student 0 for teacher Euler sampling.")
        if args.condition_critic_updates_per_epoch <= 0:
            raise ValueError("Condition guidance requires --condition_critic_updates_per_epoch > 0.")
    if not 0.0 <= args.positive_replay_ratio <= 1.0:
        raise ValueError("--positive_replay_ratio must be in [0, 1].")
    if not 0.0 <= args.condition_positive_replay_ratio <= 1.0:
        raise ValueError("--condition_positive_replay_ratio must be in [0, 1].")
    if args.init_rollout_steps < 0:
        raise ValueError("--init_rollout_steps must be >= 0.")
    args.chunk_critic_update_steps = 0
    args.condition_critic_update_steps = 0

    mpi_fork(conf.setup.cpu)  #  run parallel code with mpi

    if proc_id() == 0:
        #wandb.init(project=conf.setup.exp_name)
        #wandb.run.name = conf.setup.env + "_reskill_seed_" + str(conf.setup.seed) + '_' + time.asctime().replace(' ', '_')
        
        log_file = os.path.join(
            curr_dir,
            "swanlog",
            "agent",
            conf.setup.env,
            args.dataset_name,
            f"seed_{args.seed}",
            args.prior_run_name,
            args.run_variant,
        )

        os.makedirs(log_file, exist_ok=True)
        swanlab_config = vars(args).copy()
        swanlab_config.update(
            {
                "env": conf.setup.env,
                "cpu": conf.setup.cpu,
                "steps_per_epoch": conf.skill_agent.steps_per_epoch,
                "epochs": conf.setup.epochs,
                "max_ep_len": conf.setup.max_ep_len,
            }
        )
        writer = SwanLabWriter(
            project=args.swanlab_project,
            workspace=args.swanlab_workspace,
            experiment_name=(
                f"{conf.setup.env}_{args.prior_model}_grad_{args.use_grad}_seed{args.seed}"
            ),
            config=swanlab_config,
            logdir=log_file,
            mode=args.swanlab_mode,
            tags=["agent", args.prior_run_name, conf.setup.env],
        )
    else:
        writer = None

    env = gym.make(conf.setup.env)

    save_dir = os.path.join(
        curr_dir,
        "results",
        "saved_rl_models",
        conf.setup.env,
        args.dataset_name,
        args.prior_run_name,
        str(args.seed),
        args.run_variant,
    )
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "ppo_agent.pth")
    save_path_latent_q = os.path.join(save_dir, "ppo_latent_q_agent.pth")
    save_path_chunk_critic = os.path.join(save_dir, "chunk_critic.pth")
    save_path_condition_critic = os.path.join(save_dir, "condition_critic.pth")
    save_path_residual = os.path.join(save_dir, "ppo_residual_agent.pth")

    torch.set_num_threads(torch.get_num_threads())

    # Load skills module. Prefer best checkpoints from new training runs, but
    # fall back to last checkpoints and legacy directory layouts.
    skill_model_dir = resolve_skill_model_dir(curr_dir, args)
    skill_vae_path = resolve_skill_checkpoint_path(skill_model_dir, "skill_vae", args.skill_checkpoint)
    skill_prior_path = resolve_skill_checkpoint_path(skill_model_dir, "skill_prior", args.skill_checkpoint)
    condition_prior_path = resolve_skill_checkpoint_path(skill_model_dir, "condition_prior", args.skill_checkpoint)
    if proc_id() == 0:
        print(f"Loading skill modules from: {skill_model_dir}", flush=True)
        print(f"  skill_vae: {skill_vae_path}", flush=True)
        print(f"  skill_prior: {skill_prior_path}", flush=True)
        if args.use_condition_flow == 1:
            print(f"  condition_prior: {condition_prior_path}", flush=True)

    # skill_vae_path = f"./results/saved_skill_models/{args.dataset_name}/seed_{args.seed}/skill_prior_{args.prior_model}/skill_vae.pth"
    # skill_prior_path = f"./results/saved_skill_models/{args.dataset_name}/seed_{args.seed}/skill_prior_{args.prior_model}/skill_prior.pth"

    register_legacy_model_modules()
    skill_vae = torch.load(skill_vae_path, map_location=device)
    skill_prior = torch.load(skill_prior_path, map_location=device)
    condition_prior = None
    if args.use_condition_flow == 1:
        condition_prior = torch.load(condition_prior_path, map_location=device)
    if args.prior_model == 'RNVP':
        for i in skill_prior.bijectors:
            i.device = device
    if args.prior_model == 'Diffusion':
        use_sigma = False
        if args.use_sigma == 1:
            use_sigma = True
        skill_prior2 = Diffusion_BC(state_dim=skill_prior.actor.state_dim, action_dim=skill_prior.actor.action_dim, max_action=skill_prior.actor.max_action, device=device, use_sigma=use_sigma)
        skill_prior2.model = skill_prior.model
        skill_prior2.actor.model = skill_prior.actor.model
        skill_prior = skill_prior2
    if args.prior_model == 'Flow':
        skill_prior.use_student = bool(args.use_student)
        skill_prior.actor = skill_prior.student if skill_prior.use_student else skill_prior.teacher
        if condition_prior is not None:
            condition_prior.use_student = bool(args.use_student)
            condition_prior.actor = condition_prior.student if condition_prior.use_student else condition_prior.teacher

    n_features = skill_vae.n_z
    n_actions = skill_vae.n_actions
    n_obs = skill_vae.n_obs
    seq_len = skill_vae.seq_len
    latent_q_agent = None
    chunk_critic = None
    chunk_replay = None
    condition_critic = None
    condition_replay = None
    args.n_obs = n_obs
    args.max_ep_len = conf.setup.max_ep_len
    raw_steps_per_epoch = conf.skill_agent.steps_per_epoch
    aligned_steps_per_epoch, episode_skills = align_steps_per_epoch(
        raw_steps_per_epoch,
        conf.setup.max_ep_len,
        seq_len,
    )
    conf.skill_agent.steps_per_epoch = aligned_steps_per_epoch
    if proc_id() == 0:
        print(
            "steps_per_epoch alignment: "
            f"raw={raw_steps_per_epoch}, aligned={aligned_steps_per_epoch}, "
            f"num_procs={num_procs()}, seq_len={seq_len}, "
            f"max_ep_len={conf.setup.max_ep_len}, episode_skills={episode_skills}",
            flush=True,
        )

    if args.prior_model == 'RNVP':
        skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid] * conf.skill_agent.l),
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
    elif args.prior_model == 'Flow':
        if args.use_grad == 1 and args.use_student == 1 and args.guidance_scale > 0:
            raise ValueError("Flow guidance currently requires --use_student 0 for teacher Euler sampling.")
        if args.use_grad == 1 and args.guidance_scale > 0 and args.chunk_critic_updates_per_epoch <= 0:
            raise ValueError("Flow guidance requires --chunk_critic_updates_per_epoch > 0.")
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
        if args.use_grad == 1 and args.guidance_scale > 0:
            chunk_critic = LatentChunkCritic(
                state_dim=n_obs,
                latent_dim=n_features,
                hidden_dim=args.chunk_critic_hidden_dim,
                num_ensembles=args.chunk_critic_ensembles,
                lr=args.chunk_critic_lr,
                gamma=args.chunk_critic_gamma,
                tau=args.chunk_critic_tau,
                seq_len=seq_len,
                hidden_layers=args.chunk_critic_hidden_layers,
                use_layer_norm=args.chunk_critic_layer_norm,
                activation=args.chunk_critic_activation,
            )
            chunk_replay = ChunkReplayBuffer(
                state_dim=n_obs,
                latent_dim=n_features,
                capacity=args.chunk_critic_replay_size,
                seed=args.seed,
            )
        if args.condition_use_grad == 1 and args.condition_guidance_scale > 0:
            condition_critic = LatentChunkCritic(
                state_dim=n_obs,
                latent_dim=n_actions,
                hidden_dim=args.condition_critic_hidden_dim,
                num_ensembles=args.condition_critic_ensembles,
                lr=args.condition_critic_lr,
                gamma=args.chunk_critic_gamma,
                tau=args.condition_critic_tau,
                seq_len=seq_len,
                hidden_layers=args.condition_critic_hidden_layers,
                use_layer_norm=args.condition_critic_layer_norm,
                activation=args.condition_critic_activation,
            )
            condition_replay = ChunkReplayBuffer(
                state_dim=n_obs,
                latent_dim=n_actions,
                capacity=args.condition_critic_replay_size,
                seed=args.seed,
            )
    elif args.prior_model == 'Diffusion':
        latent_q_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid] * conf.skill_agent.l),
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
        skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid] * conf.skill_agent.l),
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
    else:
        skill_agent = PPO(ac_kwargs=dict(hidden_sizes=[conf.skill_agent.hid] * conf.skill_agent.l),
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
    #skill_distill_agent = SAC(num_inputs=n_obs,
    #                          num_actions=n_actions,
    #                          action_space=action_space,
    #                          args=args
    #                          )
    #sac_memory = ReplayMemory(capacity=args.replay_size, 
    #                      seed=args.seed
    #                      )
    
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
          latent_q_agent=latent_q_agent,
          chunk_critic=chunk_critic,
          chunk_replay=chunk_replay,
          condition_critic=condition_critic,
          condition_replay=condition_replay,
          condition_prior=condition_prior,
          residual_agent=residual_agent,  
          env=env,
          skill_vae=skill_vae,
          skill_prior=skill_prior,
          logistic_C=conf.setup.logistic_C,
          logistic_k=conf.setup.logistic_k,
          max_residual_factor=(
              args.max_residual_factor
              if args.max_residual_factor is not None
              else getattr(conf.setup, "max_residual_factor", 1.0)
          ),
          save_path=save_path,
          save_path_latent_q=save_path_latent_q,
          save_path_chunk_critic=save_path_chunk_critic,
          save_path_condition_critic=save_path_condition_critic,
          save_path_residual=save_path_residual,
          writer=writer,
          prior_model=args.prior_model,
          args=args)
    
    if proc_id() == 0 and writer is not None:
        writer.close()
    
    #sac_memory.save_buffer(env.spec.id, str(args.seed)+"_"+args.prior_model)


if __name__ == '__main__':
    main()
    
