import argparse
import json
import os

import numpy as np
import torch

from reskill.models.tanh_gaussian_policy import TanhGaussianBehaviorPolicy


def load_labeled_sequences(dataset_name):
    path = os.path.join("dataset", dataset_name, "demos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    seqs = np.load(path, allow_pickle=True)
    missing = [i for i, seq in enumerate(seqs) if "mode" not in seq and "mode_id" not in seq]
    if missing:
        raise ValueError(
            f"Dataset has no labels on {len(missing)} trajectories. "
            "Use reskill/data/collect_demos_labeled.py to generate labeled data."
        )
    return seqs


def iter_chunks(seqs, split_end, subseq_len, include_values=False):
    for seq_idx in range(split_end):
        seq = seqs[seq_idx]
        mode = seq.get("mode")
        mode_id = seq.get("mode_id")
        num_starts = max(0, len(seq.actions) - subseq_len - 1)
        for start_idx in range(num_starts):
            chunk = {
                "seq_idx": seq_idx,
                "start_idx": start_idx,
                "mode": mode,
                "mode_id": mode_id,
            }
            if include_values:
                chunk["state"] = np.asarray(seq.obs[start_idx], dtype=np.float32)
                chunk["action"] = np.asarray(seq.actions[start_idx], dtype=np.float32)
            yield chunk


def summarize_counts(chunks):
    counts = {}
    for chunk in chunks:
        mode = chunk["mode"]
        counts[mode] = counts.get(mode, 0) + 1
    total = sum(counts.values())
    return {
        mode: {
            "count": int(count),
            "fraction": float(count / total) if total else 0.0,
        }
        for mode, count in sorted(counts.items())
    }


def compute_weights_from_stats(raw_log_weights, stats):
    raw_log_weight_max = float(stats["raw_log_weight_max"])
    raw_log_weights_capped = np.minimum(raw_log_weights, raw_log_weight_max)
    center = float(stats.get("log_weight_center", stats["log_mean_weight"]))
    w_min = float(stats["w_min"])
    w_max = float(stats["w_max"])
    log_weights = raw_log_weights_capped - center
    log_weights = np.clip(log_weights, np.log(w_min), np.log(w_max))
    weights = np.exp(log_weights)
    if "clipped_weight_mean" in stats:
        weights = weights / (float(stats["clipped_weight_mean"]) + 1e-8)
    return weights.astype(np.float64)


def compute_raw_log_weights(chunks, stats, behavior_policy_path, hidden_dim, batch_size, device):
    first_state = chunks[0]["state"]
    first_action = chunks[0]["action"]
    policy = TanhGaussianBehaviorPolicy(
        state_dim=int(first_state.shape[0]),
        action_dim=int(first_action.shape[0]),
        hidden_dim=hidden_dim,
        action_low=-1.0,
        action_high=1.0,
    ).to(device)
    policy.load_state_dict(torch.load(behavior_policy_path, map_location=device))
    policy.eval()

    raw_log_weights = []
    log_probs = []
    beta = float(stats["beta"])
    with torch.no_grad():
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            states = torch.as_tensor(np.asarray([c["state"] for c in batch]), dtype=torch.float32, device=device)
            actions = torch.as_tensor(np.asarray([c["action"] for c in batch]), dtype=torch.float32, device=device)
            log_prob = policy.log_prob(states, actions)
            raw_log_weight = -beta * log_prob
            log_probs.extend(log_prob.detach().cpu().numpy().tolist())
            raw_log_weights.extend(raw_log_weight.detach().cpu().numpy().tolist())
    return np.asarray(raw_log_weights, dtype=np.float64), np.asarray(log_probs, dtype=np.float64)


def summarize_weighted(chunks, weights):
    sums = {}
    values = {}
    for chunk, weight in zip(chunks, weights):
        mode = chunk["mode"]
        sums[mode] = sums.get(mode, 0.0) + float(weight)
        values.setdefault(mode, []).append(float(weight))

    total = sum(sums.values())
    out = {}
    for mode in sorted(sums):
        arr = np.asarray(values[mode], dtype=np.float64)
        out[mode] = {
            "weighted_sum": float(sums[mode]),
            "weighted_fraction": float(sums[mode] / total) if total else 0.0,
            "weight_mean": float(np.mean(arr)),
            "weight_p50": float(np.percentile(arr, 50)),
            "weight_p90": float(np.percentile(arr, 90)),
            "weight_p99": float(np.percentile(arr, 99)),
            "weight_max": float(np.max(arr)),
        }
    return out


def summarize_batches(chunks, weights, batch_size):
    num_batches = len(chunks) // batch_size
    if num_batches == 0:
        return {}

    modes = sorted({chunk["mode"] for chunk in chunks})
    zero_mode_batches = {mode: 0 for mode in modes}
    mode_count_values = {mode: [] for mode in modes}
    weighted_fraction_values = {mode: [] for mode in modes}

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size
        batch = chunks[start:end]
        batch_weights = weights[start:end]
        total_weight = float(np.sum(batch_weights))

        for mode in modes:
            mask = np.asarray([chunk["mode"] == mode for chunk in batch], dtype=bool)
            count = int(np.sum(mask))
            mode_count_values[mode].append(count)
            if count == 0:
                zero_mode_batches[mode] += 1
            weighted_sum = float(np.sum(batch_weights[mask]))
            weighted_fraction_values[mode].append(weighted_sum / total_weight if total_weight > 0 else 0.0)

    out = {"num_batches": int(num_batches), "batch_size": int(batch_size), "modes": {}}
    for mode in modes:
        counts = np.asarray(mode_count_values[mode], dtype=np.float64)
        weighted_fracs = np.asarray(weighted_fraction_values[mode], dtype=np.float64)
        out["modes"][mode] = {
            "zero_batch_fraction": float(zero_mode_batches[mode] / num_batches),
            "count_mean": float(np.mean(counts)),
            "count_p50": float(np.percentile(counts, 50)),
            "count_p90": float(np.percentile(counts, 90)),
            "weighted_fraction_mean": float(np.mean(weighted_fracs)),
            "weighted_fraction_p50": float(np.percentile(weighted_fracs, 50)),
            "weighted_fraction_p90": float(np.percentile(weighted_fracs, 90)),
        }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--stats_path", type=str, default=None)
    parser.add_argument("--behavior_policy_path", type=str, default=None)
    parser.add_argument("--behavior_policy_hidden_dim", type=int, default=256)
    parser.add_argument("--raw_log_weights_path", type=str, default=None)
    parser.add_argument("--save_raw_log_weights", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    seqs = load_labeled_sequences(args.dataset_name)
    split_end = int(args.train_split * len(seqs))
    chunks = list(iter_chunks(seqs, split_end, args.subseq_len, include_values=True))

    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "num_train_trajectories": int(split_end),
        "num_chunks": int(len(chunks)),
        "raw_chunk_distribution": summarize_counts(chunks),
    }

    stats_path = args.stats_path or os.path.join(
        "reskill",
        "results",
        "saved_skill_models",
        args.dataset_name,
        "Flow",
        f"seed_{args.seed}",
        "skill_prior_Flow_student0",
        "condition_weight_stats.json",
    )
    result["condition_weight_stats_path"] = stats_path

    raw_log_weights_path = args.raw_log_weights_path
    if raw_log_weights_path is None:
        raw_log_weights_path = os.path.join(os.path.dirname(stats_path), "condition_raw_log_weights.npy")
    result["raw_log_weights_path"] = raw_log_weights_path

    behavior_policy_path = args.behavior_policy_path or os.path.join(os.path.dirname(stats_path), "behavior_policy.pth")
    result["behavior_policy_path"] = behavior_policy_path

    if os.path.exists(stats_path) and (os.path.exists(raw_log_weights_path) or os.path.exists(behavior_policy_path)):
        with open(stats_path, "r") as f:
            stats = json.load(f)
        if os.path.exists(raw_log_weights_path):
            raw_log_weights = np.load(raw_log_weights_path)
            log_probs = None
        else:
            raw_log_weights, log_probs = compute_raw_log_weights(
                chunks,
                stats,
                behavior_policy_path,
                args.behavior_policy_hidden_dim,
                args.batch_size,
                args.device,
            )
            if args.save_raw_log_weights:
                np.save(raw_log_weights_path, raw_log_weights)
        if len(raw_log_weights) != len(chunks):
            raise ValueError(
                f"raw_log_weights length {len(raw_log_weights)} does not match labeled chunks {len(chunks)}."
            )
        weights = compute_weights_from_stats(raw_log_weights.astype(np.float64), stats)
        result["weighted_chunk_distribution"] = summarize_weighted(chunks, weights)
        result["batch_distribution"] = summarize_batches(chunks, weights, args.batch_size)
        result["raw_log_weight_summary"] = {
            "mean": float(np.mean(raw_log_weights)),
            "std": float(np.std(raw_log_weights)),
            "p50": float(np.percentile(raw_log_weights, 50)),
            "p90": float(np.percentile(raw_log_weights, 90)),
            "p99": float(np.percentile(raw_log_weights, 99)),
            "max": float(np.max(raw_log_weights)),
        }
        if log_probs is not None:
            result["log_prob_summary"] = {
                "mean": float(np.mean(log_probs)),
                "std": float(np.std(log_probs)),
                "p50": float(np.percentile(log_probs, 50)),
                "p90": float(np.percentile(log_probs, 90)),
                "p99": float(np.percentile(log_probs, 99)),
            }
    else:
        result["weighted_chunk_distribution"] = None
        result["batch_distribution"] = None
        result["warning"] = (
            "Weighted stats require condition_weight_stats.json plus either "
            "condition_raw_log_weights.npy or behavior_policy.pth. Raw label proportions are still reported."
        )

    print(json.dumps(result, indent=2))
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
