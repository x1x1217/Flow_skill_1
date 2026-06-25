import argparse
import json
import math
import os

import numpy as np
import torch


def load_labeled_chunks(dataset_name, subseq_len, train_split):
    path = os.path.join("dataset", dataset_name, "demos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    seqs = np.load(path, allow_pickle=True)
    split_end = int(train_split * len(seqs))
    chunks = []
    for seq_idx in range(split_end):
        seq = seqs[seq_idx]
        if "mode" not in seq:
            raise ValueError(
                f"Trajectory {seq_idx} has no mode label. "
                "Use reskill/data/collect_demos_labeled.py to generate labeled data."
            )
        num_starts = max(0, len(seq.actions) - subseq_len - 1)
        for start_idx in range(num_starts):
            chunks.append(
                (
                    seq.get("mode"),
                    np.asarray(seq.obs[start_idx], dtype=np.float32),
                    np.asarray(seq.actions[start_idx], dtype=np.float32),
                )
            )
    return chunks


def exact_divergence(flow_teacher, cond, z, t):
    z = z.detach().requires_grad_(True)
    v = flow_teacher(cond, z, t)
    div = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
    for dim in range(z.shape[1]):
        grad = torch.autograd.grad(
            v[:, dim].sum(),
            z,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        div = div + grad[:, dim]
    return v.detach(), div.detach()


def standard_normal_logprob(x):
    return -0.5 * (x.pow(2).sum(dim=1) + x.shape[1] * math.log(2.0 * math.pi))


def flow_logprob_given_actions(condition_prior, states, actions, flow_steps):
    teacher = condition_prior.teacher
    teacher.eval()
    dt = 1.0 / float(flow_steps)

    z = actions
    div_integral = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)

    for i in reversed(range(flow_steps)):
        t_value = float(i) / float(flow_steps)
        t = torch.full((z.shape[0], 1), t_value, device=z.device, dtype=z.dtype)
        v, div = exact_divergence(teacher, states, z, t)
        z = z - dt * v
        div_integral = div_integral + dt * div

    base_logprob = standard_normal_logprob(z)
    return base_logprob - div_integral


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return {
        "count": int(len(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p1": float(np.percentile(arr, 1)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
        "std": float(np.std(arr)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--flow_steps", type=int, default=None)
    parser.add_argument("--condition_prior_path", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    condition_prior_path = args.condition_prior_path or os.path.join(
        "reskill",
        "results",
        "saved_skill_models",
        args.dataset_name,
        "Flow",
        f"seed_{args.seed}",
        "skill_prior_Flow_student0",
        "condition_prior.pth",
    )
    if not os.path.exists(condition_prior_path):
        raise FileNotFoundError(f"Condition prior not found: {condition_prior_path}")

    chunks = load_labeled_chunks(args.dataset_name, args.subseq_len, args.train_split)
    condition_prior = torch.load(condition_prior_path, map_location=args.device)
    condition_prior.device = args.device
    flow_steps = int(args.flow_steps or condition_prior.flow_steps)

    by_mode = {}
    for start in range(0, len(chunks), args.batch_size):
        batch = chunks[start:start + args.batch_size]
        modes = [item[0] for item in batch]
        states = torch.as_tensor(np.asarray([item[1] for item in batch]), dtype=torch.float32, device=args.device)
        actions = torch.as_tensor(np.asarray([item[2] for item in batch]), dtype=torch.float32, device=args.device)
        logprob = flow_logprob_given_actions(condition_prior, states, actions, flow_steps)
        for mode, value in zip(modes, logprob.detach().cpu().numpy().tolist()):
            by_mode.setdefault(mode, []).append(float(value))

    result = {
        "dataset_name": args.dataset_name,
        "seed": args.seed,
        "condition_prior_path": condition_prior_path,
        "num_chunks": int(len(chunks)),
        "flow_steps": flow_steps,
        "logprob_by_mode": {mode: summarize(values) for mode, values in sorted(by_mode.items())},
    }

    print(json.dumps(result, indent=2))
    if args.output_path:
        with open(args.output_path, "w") as f:
            json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
