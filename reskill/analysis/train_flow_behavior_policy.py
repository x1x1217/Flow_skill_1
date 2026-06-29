import argparse
import json
import os

import numpy as np
import torch

from reskill.analysis.sweep_flow_behavior_weight_params import (
    compute_flow_log_probs,
    default_flow_behavior_path,
    make_flow_behavior,
    summarize,
    train_flow_behavior,
)


def load_behavior_chunks(dataset_name, subseq_len, train_split):
    path = os.path.join("dataset", dataset_name, "demos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    seqs = np.load(path, allow_pickle=True)
    split_end = int(train_split * len(seqs))
    states = []
    actions = []
    for seq in seqs[:split_end]:
        num_starts = max(0, len(seq.actions) - subseq_len - 1)
        for start_idx in range(num_starts):
            states.append(np.asarray(seq.obs[start_idx], dtype=np.float32))
            actions.append(np.asarray(seq.actions[start_idx], dtype=np.float32))

    if not states:
        raise ValueError("No valid chunks found for Flow behavior training.")
    return np.asarray(states, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--use_student", type=int, default=0)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--flow_behavior_path", type=str, default=None)
    parser.add_argument("--flow_log_probs_path", type=str, default=None)
    parser.add_argument("--history_path", type=str, default=None)
    parser.add_argument("--summary_path", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--logprob_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--flow_use_student", type=int, default=0)
    parser.add_argument("--distill_coef", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise ValueError("--epochs must be positive")
    if not 0 < args.train_split <= 1:
        raise ValueError("--train_split must be in (0, 1]")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    states, actions = load_behavior_chunks(args.dataset_name, args.subseq_len, args.train_split)

    behavior_path = args.flow_behavior_path or default_flow_behavior_path(
        args.dataset_name, args.seed, args.use_student
    )
    log_probs_path = args.flow_log_probs_path or os.path.splitext(behavior_path)[0] + "_log_probs.npy"
    history_path = args.history_path or os.path.splitext(behavior_path)[0] + "_train_history.json"
    summary_path = args.summary_path or os.path.splitext(behavior_path)[0] + "_logprob_summary.json"
    for path in (behavior_path, log_probs_path, history_path, summary_path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    flow = make_flow_behavior(states, actions, args)
    history = train_flow_behavior(flow, states, actions, args)
    torch.save(flow, behavior_path)
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    log_probs = compute_flow_log_probs(
        flow,
        states,
        actions,
        args.logprob_batch_size,
        args.flow_steps,
        args.device,
    )
    np.save(log_probs_path, log_probs)
    summary = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "num_chunks": int(len(states)),
        "flow_behavior_path": behavior_path,
        "flow_log_probs_path": log_probs_path,
        "log_prob_summary": summarize(log_probs),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
