import argparse
import json
import os

import numpy as np
import torch

from reskill.analysis.sweep_flow_behavior_weight_params import (
    compute_flow_log_probs,
    flow_logprob_given_actions,
)
from reskill.train_skill_modules import ModelTrainer
from reskill.utils.swanlab_writer import SwanLabWriter


class FlowBehaviorWeightModelTrainer(ModelTrainer):
    def __init__(
        self,
        *args,
        flow_behavior_policy_path=None,
        flow_behavior_log_probs_path=None,
        flow_behavior_logprob_batch_size=256,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.flow_behavior_policy_path = flow_behavior_policy_path
        self.flow_behavior_log_probs_path = flow_behavior_log_probs_path
        self.flow_behavior_logprob_batch_size = flow_behavior_logprob_batch_size
        self.flow_behavior_policy = None

    def default_flow_behavior_policy_path(self):
        return os.path.join(self.save_dir, "behavior_flow_policy.pth")

    def default_flow_behavior_log_probs_path(self):
        return os.path.splitext(self.flow_behavior_policy_path)[0] + "_log_probs.npy"

    def train_behavior_policy(self):
        path = self.flow_behavior_policy_path or self.default_flow_behavior_policy_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"Flow behavior policy not found: {path}")
        self.flow_behavior_policy_path = path
        self.flow_behavior_policy = torch.load(path, map_location=self.device)
        self.flow_behavior_policy.device = self.device
        self.flow_behavior_policy.teacher.to(self.device)
        self.flow_behavior_policy.teacher.eval()
        if self.flow_behavior_policy.use_student:
            self.flow_behavior_policy.student.to(self.device)
            self.flow_behavior_policy.student.eval()
        print(f"[flow behavior weights] loaded {path}", flush=True)

    def compute_full_flow_behavior_log_probs(self, batch_size):
        dataset = self.train_loader.dataset
        states = []
        actions = []
        for seq_idx in range(dataset.start, dataset.end):
            seq = dataset.seqs[seq_idx]
            num_starts = max(0, len(seq.actions) - dataset.subseq_len - 1)
            for start_idx in range(num_starts):
                states.append(np.asarray(seq.obs[start_idx], dtype=np.float32))
                actions.append(np.asarray(seq.actions[start_idx], dtype=np.float32))
        if not states:
            raise ValueError("No valid chunks for flow behavior condition weight computation.")

        states = np.asarray(states, dtype=np.float32)
        actions = np.asarray(actions, dtype=np.float32)
        return compute_flow_log_probs(
            self.flow_behavior_policy,
            states,
            actions,
            batch_size,
            int(self.flow_behavior_policy.flow_steps),
            self.device,
        )

    def compute_condition_weight_stats(self, batch_size=8192):
        if not self.condition_reweight:
            return
        if self.flow_behavior_policy is None:
            self.train_behavior_policy()

        log_probs_path = self.flow_behavior_log_probs_path
        if log_probs_path is None:
            log_probs_path = self.default_flow_behavior_log_probs_path()
        self.flow_behavior_log_probs_path = log_probs_path

        if os.path.exists(log_probs_path):
            log_probs = np.load(log_probs_path).astype(np.float64)
            print(f"[flow behavior weights] loaded cached log_probs {log_probs_path}", flush=True)
        else:
            log_probs = self.compute_full_flow_behavior_log_probs(self.flow_behavior_logprob_batch_size)
            os.makedirs(os.path.dirname(log_probs_path), exist_ok=True)
            np.save(log_probs_path, log_probs)
            print(f"[flow behavior weights] saved log_probs {log_probs_path}", flush=True)

        raw_log_weights = -self.condition_weight_beta * log_probs
        raw_log_weight_max = float(
            np.quantile(raw_log_weights, self.condition_raw_log_weight_clip_quantile)
        )
        raw_log_weights_capped = np.minimum(raw_log_weights, raw_log_weight_max)
        log_weight_center = float(np.median(raw_log_weights_capped))
        log_weights = raw_log_weights_capped - log_weight_center
        log_weight_min = float(np.log(self.condition_weight_min))
        log_weight_max = float(np.log(self.condition_weight_max))
        log_weights_clipped = np.clip(log_weights, log_weight_min, log_weight_max)
        weights_clipped = np.exp(log_weights_clipped)
        clipped_weight_mean = float(np.mean(weights_clipped))
        weights = (weights_clipped / (clipped_weight_mean + 1e-8)).astype(np.float32)

        self.condition_log_mean_weight = log_weight_center
        self.condition_log_weight_center = log_weight_center
        self.condition_clipped_weight_mean = clipped_weight_mean
        self.condition_raw_log_weight_max = raw_log_weight_max

        effective_sample_size = float((weights.sum() ** 2) / (np.square(weights).sum() + 1e-8))
        percentiles = [1, 5, 50, 90, 95, 99, 99.5, 99.9, 100]
        stats = {
            "dataset_name": self.dataset_name,
            "weight_source": "flow_behavior_policy",
            "flow_behavior_policy_path": self.flow_behavior_policy_path,
            "flow_behavior_log_probs_path": self.flow_behavior_log_probs_path,
            "num_weighted_chunks": int(len(weights)),
            "beta": float(self.condition_weight_beta),
            "w_min": float(self.condition_weight_min),
            "w_max": float(self.condition_weight_max),
            "raw_log_weight_clip_quantile": float(self.condition_raw_log_weight_clip_quantile),
            "raw_log_weight_max": float(raw_log_weight_max),
            "log_mean_weight": float(log_weight_center),
            "log_weight_center": float(log_weight_center),
            "clipped_weight_mean": float(clipped_weight_mean),
            "weight_mean": float(np.mean(weights)),
            "weight_std": float(np.std(weights)),
            "weight_min": float(np.min(weights)),
            "weight_max": float(np.max(weights)),
            "raw_log_weight_mean": float(np.mean(raw_log_weights)),
            "raw_log_weight_std": float(np.std(raw_log_weights)),
            "log_prob_mean": float(np.mean(log_probs)),
            "log_prob_std": float(np.std(log_probs)),
            "effective_sample_size": effective_sample_size,
            "effective_sample_fraction": float(effective_sample_size / max(1, len(weights))),
        }
        stats.update({
            f"raw_log_weight_p{str(p).replace('.', '_')}": float(np.percentile(raw_log_weights, p))
            for p in percentiles
        })
        stats.update({
            f"weight_p{str(p).replace('.', '_')}": float(np.percentile(weights, p))
            for p in percentiles
        })
        with open(self.condition_weight_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print(
            "[flow behavior condition weights] "
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
        if self.flow_behavior_policy is None:
            self.train_behavior_policy()
        if self.condition_log_weight_center is None:
            if not os.path.exists(self.condition_weight_stats_path):
                raise RuntimeError("Condition reweighting is enabled but condition weight stats are not computed.")
            with open(self.condition_weight_stats_path, "r") as f:
                stats = json.load(f)
            self.condition_log_mean_weight = float(stats["log_mean_weight"])
            self.condition_log_weight_center = float(stats.get("log_weight_center", stats["log_mean_weight"]))
            self.condition_clipped_weight_mean = float(stats["clipped_weight_mean"])
            self.condition_raw_log_weight_max = float(stats["raw_log_weight_max"])

        with torch.enable_grad():
            log_prob = flow_logprob_given_actions(
                self.flow_behavior_policy,
                state,
                action0,
                int(self.flow_behavior_policy.flow_steps),
            )
        raw_log_weight = -self.condition_weight_beta * log_prob
        raw_log_weight = torch.clamp(raw_log_weight, max=self.condition_raw_log_weight_max)
        log_weight = raw_log_weight - self.condition_log_weight_center
        log_weight = torch.clamp(
            log_weight,
            min=float(np.log(self.condition_weight_min)),
            max=float(np.log(self.condition_weight_max)),
        )
        weight = torch.exp(log_weight)
        weight = weight / (self.condition_clipped_weight_mean + 1e-8)
        return weight.detach()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="block/config.yaml")
    parser.add_argument("--pick", type=int, default=1)
    parser.add_argument("--push", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--prior_model", type=str, default="Flow")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--use_student", type=int, default=0)
    parser.add_argument("--skill_epochs", type=int, default=None)
    parser.add_argument("--prior_epochs", type=int, default=None)
    parser.add_argument("--prior_updates_per_batch", type=int, default=1)
    parser.add_argument("--prior_use_mu", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--condition_reweight", type=int, default=1)
    parser.add_argument("--condition_weight_beta", type=float, default=0.2)
    parser.add_argument("--condition_weight_min", type=float, default=1.0)
    parser.add_argument("--condition_weight_max", type=float, default=1000.0)
    parser.add_argument("--condition_raw_log_weight_clip_quantile", type=float, default=1.0)
    parser.add_argument("--flow_behavior_policy_path", type=str, default=None)
    parser.add_argument("--flow_behavior_log_probs_path", type=str, default=None)
    parser.add_argument("--flow_behavior_logprob_batch_size", type=int, default=256)
    parser.add_argument("--swanlab_project", type=str, default="Flow_skill_1")
    parser.add_argument("--swanlab_workspace", type=str, default="x1x1217")
    parser.add_argument("--swanlab_mode", type=str, default=None)
    args = parser.parse_args()

    if args.prior_model != "Flow":
        raise ValueError("--prior_model must be Flow for flow-behavior weighting.")
    if args.prior_updates_per_batch < 1:
        raise ValueError("--prior_updates_per_batch must be >= 1")
    if args.val_freq < 1:
        raise ValueError("--val_freq must be >= 1")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be >= 1")
    if args.condition_weight_beta < 0:
        raise ValueError("--condition_weight_beta must be >= 0")
    if args.condition_weight_min <= 0 or args.condition_weight_max < args.condition_weight_min:
        raise ValueError("--condition_weight_min/max are invalid")
    if not 0 < args.condition_raw_log_weight_clip_quantile <= 1:
        raise ValueError("--condition_raw_log_weight_clip_quantile must be in (0, 1].")
    if args.dataset_name is None:
        args.dataset_name = f"fetch_block_push{args.push}_pick{args.pick}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    curr_dir = os.path.dirname(__file__)
    flow_suffix = f"_student{args.use_student}"
    log_file = os.path.join(
        curr_dir,
        "swanlog",
        "skill_prior",
        args.dataset_name,
        f"seed_{args.seed}_{args.prior_model}{flow_suffix}_flow_behavior_weight",
    )
    os.makedirs(log_file, exist_ok=True)
    writer = SwanLabWriter(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=f"skill_prior_seed{args.seed}_{args.prior_model}{flow_suffix}_flow_behavior_weight",
        config=vars(args),
        logdir=log_file,
        mode=args.swanlab_mode,
        tags=["skill_prior", args.prior_model, "flow_behavior_weight"],
    )

    trainer = FlowBehaviorWeightModelTrainer(
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
        condition_weight_beta=args.condition_weight_beta,
        condition_weight_min=args.condition_weight_min,
        condition_weight_max=args.condition_weight_max,
        condition_raw_log_weight_clip_quantile=args.condition_raw_log_weight_clip_quantile,
        flow_behavior_policy_path=args.flow_behavior_policy_path,
        flow_behavior_log_probs_path=args.flow_behavior_log_probs_path,
        flow_behavior_logprob_batch_size=args.flow_behavior_logprob_batch_size,
    )
    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()
