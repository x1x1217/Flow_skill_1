import argparse
import json
import os

import numpy as np
import torch

from reskill.models.tanh_gaussian_policy import TanhGaussianBehaviorPolicy


def parse_float_list(value):
    return [float(x) for x in value.split(",") if x.strip()]


def load_labeled_sequences(dataset_name):
    path = os.path.join("dataset", dataset_name, "demos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    seqs = np.load(path, allow_pickle=True)
    missing = [i for i, seq in enumerate(seqs) if "mode" not in seq]
    if missing:
        raise ValueError(
            f"Dataset has no labels on {len(missing)} trajectories. "
            "Use a labeled dataset for this sweep."
        )
    return seqs


def collect_chunks(seqs, split_end, subseq_len):
    states = []
    actions = []
    modes = []
    for seq_idx in range(split_end):
        seq = seqs[seq_idx]
        num_starts = max(0, len(seq.actions) - subseq_len - 1)
        for start_idx in range(num_starts):
            states.append(np.asarray(seq.obs[start_idx], dtype=np.float32))
            actions.append(np.asarray(seq.actions[start_idx], dtype=np.float32))
            modes.append(str(seq.mode))

    if not states:
        raise ValueError("No valid chunks found.")

    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(modes),
    )


def compute_log_probs(states, actions, behavior_policy_path, hidden_dim, batch_size, device):
    policy = TanhGaussianBehaviorPolicy(
        state_dim=int(states.shape[1]),
        action_dim=int(actions.shape[1]),
        hidden_dim=hidden_dim,
        action_low=-1.0,
        action_high=1.0,
    ).to(device)
    policy.load_state_dict(torch.load(behavior_policy_path, map_location=device))
    policy.eval()

    log_probs = []
    with torch.no_grad():
        for start in range(0, len(states), batch_size):
            end = start + batch_size
            state_tensor = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
            action_tensor = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
            log_prob = policy.log_prob(state_tensor, action_tensor)
            log_probs.append(log_prob.detach().cpu().numpy())
    return np.concatenate(log_probs).astype(np.float64)


def summarize_mode_counts(modes):
    out = {}
    total = len(modes)
    for mode in sorted(set(modes.tolist())):
        count = int(np.sum(modes == mode))
        out[mode] = {
            "count": count,
            "fraction": float(count / total),
        }
    return out


def weight_summary(weights, mask):
    values = weights[mask]
    if len(values) == 0:
        return None
    return {
        "count": int(len(values)),
        "sum": float(np.sum(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "max": float(np.max(values)),
    }


def array_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "p1": float(np.percentile(values, 1)),
        "p10": float(np.percentile(values, 10)),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p99": float(np.percentile(values, 99)),
        "p99_9": float(np.percentile(values, 99.9)),
        "max": float(np.max(values)),
    }


def sweep(log_probs, modes, betas, w_mins, w_maxs, raw_log_weight_clip_quantile, target_pick_fraction):
    pick_mask = modes == "pick"
    if not np.any(pick_mask):
        raise ValueError("No pick chunks found; cannot compute weighted_pick_fraction.")

    rows = []
    log_weight_percentile = float(raw_log_weight_clip_quantile)
    for beta in betas:
        raw_log_weight = -float(beta) * log_probs
        raw_log_weight_max = float(np.quantile(raw_log_weight, log_weight_percentile))
        raw_log_weight_capped = np.minimum(raw_log_weight, raw_log_weight_max)
        center = float(np.median(raw_log_weight_capped))
        centered = raw_log_weight_capped - center
        weights_before_clip = np.exp(np.clip(centered, -745.0, 709.0))
        weights_before_clip_summary = array_summary(weights_before_clip)
        weights_before_clip_by_mode = {}
        for mode in sorted(set(modes.tolist())):
            weights_before_clip_by_mode[mode] = array_summary(weights_before_clip[modes == mode])

        for w_min in w_mins:
            for w_max in w_maxs:
                if w_min <= 0 or w_max < w_min:
                    continue
                log_weights = np.clip(centered, np.log(w_min), np.log(w_max))
                weights_clipped = np.exp(log_weights)
                clipped_weight_mean = float(np.mean(weights_clipped))
                weights = weights_clipped / (clipped_weight_mean + 1e-8)

                total_weight = float(np.sum(weights))
                pick_weight = float(np.sum(weights[pick_mask]))
                weighted_pick_fraction = pick_weight / total_weight if total_weight > 0 else 0.0
                ess = float((total_weight ** 2) / (np.square(weights).sum() + 1e-8))

                mode_weight_summary = {}
                for mode in sorted(set(modes.tolist())):
                    mode_weight_summary[mode] = weight_summary(weights, modes == mode)

                rows.append(
                    {
                        "beta": float(beta),
                        "w_min": float(w_min),
                        "w_max": float(w_max),
                        "raw_log_weight_clip_quantile": log_weight_percentile,
                        "raw_log_weight_max": raw_log_weight_max,
                        "log_weight_center": center,
                        "weight_before_clip": weights_before_clip_summary,
                        "weight_before_clip_by_mode": weights_before_clip_by_mode,
                        "clipped_weight_mean": clipped_weight_mean,
                        "weighted_pick_fraction": float(weighted_pick_fraction),
                        "target_pick_fraction": float(target_pick_fraction),
                        "abs_error": float(abs(weighted_pick_fraction - target_pick_fraction)),
                        "effective_sample_size": ess,
                        "effective_sample_fraction": float(ess / len(weights)),
                        "weight_mean": float(np.mean(weights)),
                        "weight_std": float(np.std(weights)),
                        "weight_min": float(np.min(weights)),
                        "weight_p50": float(np.percentile(weights, 50)),
                        "weight_p90": float(np.percentile(weights, 90)),
                        "weight_p99": float(np.percentile(weights, 99)),
                        "weight_max": float(np.max(weights)),
                        "by_mode": mode_weight_summary,
                    }
                )

    rows.sort(key=lambda item: (item["abs_error"], -item["effective_sample_fraction"], item["w_max"]))
    return rows


def default_behavior_policy_path(dataset_name, seed, use_student):
    return os.path.join(
        "reskill",
        "results",
        "saved_skill_models",
        dataset_name,
        "Flow",
        f"seed_{seed}",
        f"skill_prior_Flow_student{int(use_student)}",
        "behavior_policy.pth",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--use_student", type=int, default=0)
    parser.add_argument("--behavior_policy_path", type=str, default=None)
    parser.add_argument("--behavior_policy_hidden_dim", type=int, default=256)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--betas", type=str, default="0.05,0.1,0.2,0.3,0.4,0.5,0.7,0.9")
    parser.add_argument("--w_mins", type=str, default="0.02,0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--w_maxs", type=str, default="20,50,100,200,500,1000,2000")
    parser.add_argument("--raw_log_weight_clip_quantile", type=float, default=1.0)
    parser.add_argument("--target_pick_fraction", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--raw_log_probs_path", type=str, default=None)
    parser.add_argument("--save_raw_log_probs", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if not 0 < args.raw_log_weight_clip_quantile <= 1:
        raise ValueError("--raw_log_weight_clip_quantile must be in (0, 1].")

    behavior_policy_path = args.behavior_policy_path or default_behavior_policy_path(
        args.dataset_name,
        args.seed,
        args.use_student,
    )
    if not os.path.exists(behavior_policy_path):
        raise FileNotFoundError(f"Behavior policy not found: {behavior_policy_path}")

    seqs = load_labeled_sequences(args.dataset_name)
    split_end = int(args.train_split * len(seqs))
    states, actions, modes = collect_chunks(seqs, split_end, args.subseq_len)

    raw_log_probs_path = args.raw_log_probs_path
    if raw_log_probs_path is None:
        raw_log_probs_path = os.path.join(
            os.path.dirname(behavior_policy_path),
            "condition_behavior_log_probs.npy",
        )

    if os.path.exists(raw_log_probs_path):
        log_probs = np.load(raw_log_probs_path).astype(np.float64)
        if len(log_probs) != len(modes):
            raise ValueError(
                f"Cached log_probs length {len(log_probs)} does not match chunks {len(modes)}."
            )
    else:
        log_probs = compute_log_probs(
            states,
            actions,
            behavior_policy_path,
            args.behavior_policy_hidden_dim,
            args.batch_size,
            args.device,
        )
        if args.save_raw_log_probs:
            np.save(raw_log_probs_path, log_probs)

    rows = sweep(
        log_probs=log_probs,
        modes=modes,
        betas=parse_float_list(args.betas),
        w_mins=parse_float_list(args.w_mins),
        w_maxs=parse_float_list(args.w_maxs),
        raw_log_weight_clip_quantile=args.raw_log_weight_clip_quantile,
        target_pick_fraction=args.target_pick_fraction,
    )

    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "behavior_policy_path": behavior_policy_path,
        "raw_log_probs_path": raw_log_probs_path,
        "num_train_trajectories": int(split_end),
        "num_chunks": int(len(modes)),
        "mode_distribution": summarize_mode_counts(modes),
        "log_prob_summary": {
            "min": float(np.min(log_probs)),
            "max": float(np.max(log_probs)),
            "mean": float(np.mean(log_probs)),
            "median": float(np.percentile(log_probs, 50)),
            "p1": float(np.percentile(log_probs, 1)),
            "p10": float(np.percentile(log_probs, 10)),
            "p90": float(np.percentile(log_probs, 90)),
            "p99": float(np.percentile(log_probs, 99)),
            "std": float(np.std(log_probs)),
        },
        "top_k": int(args.top_k),
        "top_results": rows[: args.top_k],
        "all_results": rows,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_path:
        with open(args.output_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
