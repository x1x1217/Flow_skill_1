import argparse
import json
import math
import os

import numpy as np
import torch

from reskill.models.bc_flow import Flow_BC


def parse_float_list(value):
    return [float(x) for x in value.split(",") if x.strip()]


def load_labeled_chunks(dataset_name, subseq_len, train_split):
    path = os.path.join("dataset", dataset_name, "demos.npy")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found: {path}")

    seqs = np.load(path, allow_pickle=True)
    split_end = int(train_split * len(seqs))
    states = []
    actions = []
    modes = []
    for seq_idx in range(split_end):
        seq = seqs[seq_idx]
        if "mode" not in seq:
            raise ValueError(
                f"Trajectory {seq_idx} has no mode label. "
                "Use a labeled dataset for flow behavior sweep."
            )
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


def default_save_dir(dataset_name, seed, use_student):
    return os.path.join(
        "reskill",
        "results",
        "saved_skill_models",
        dataset_name,
        "Flow",
        f"seed_{seed}",
        f"skill_prior_Flow_student{int(use_student)}",
    )


def default_flow_behavior_path(dataset_name, seed, use_student):
    return os.path.join(default_save_dir(dataset_name, seed, use_student), "behavior_flow_policy.pth")


def make_flow_behavior(states, actions, args):
    return Flow_BC(
        cond_dim=int(states.shape[1]),
        latent_dim=int(actions.shape[1]),
        max_action=1,
        device=args.device,
        hidden_dim=args.hidden_dim,
        time_dim=args.time_dim,
        flow_steps=args.flow_steps,
        lr=args.lr,
        distill_coef=args.distill_coef,
        use_student=bool(args.flow_use_student),
        grad_clip=args.grad_clip if args.grad_clip > 0 else None,
    )


def train_flow_behavior(flow, states, actions, args):
    rng = np.random.default_rng(args.seed)
    num_samples = states.shape[0]
    steps_per_epoch = args.steps_per_epoch
    if steps_per_epoch is None:
        steps_per_epoch = max(1, math.ceil(num_samples / args.train_batch_size))

    state_tensor = torch.as_tensor(states, dtype=torch.float32, device=args.device)
    action_tensor = torch.as_tensor(actions, dtype=torch.float32, device=args.device)

    history = []
    for epoch in range(args.epochs):
        losses = []
        flow_losses = []
        distill_losses = []
        for step in range(steps_per_epoch):
            idx = rng.integers(0, num_samples, size=args.train_batch_size)
            idx_tensor = torch.as_tensor(idx, dtype=torch.long, device=args.device)
            batch_states = state_tensor.index_select(0, idx_tensor)
            batch_actions = action_tensor.index_select(0, idx_tensor)
            metric = flow.train(batch_states, batch_actions, iterations=1)
            losses.append(float(np.mean(metric["total_loss"])))
            flow_losses.append(float(np.mean(metric["flow_loss"])))
            distill_losses.append(float(np.mean(metric["distill_loss"])))

            if step % args.log_interval == 0:
                print(
                    f"[flow behavior epoch {epoch:03d} step {step:04d}/{steps_per_epoch}] "
                    f"flow={flow_losses[-1]:.6f} "
                    f"distill={distill_losses[-1]:.6f} "
                    f"total={losses[-1]:.6f}",
                    flush=True,
                )

        epoch_stats = {
            "epoch": int(epoch),
            "flow_loss": float(np.mean(flow_losses)),
            "distill_loss": float(np.mean(distill_losses)),
            "total_loss": float(np.mean(losses)),
        }
        history.append(epoch_stats)
        print(
            f"[flow behavior epoch {epoch:03d} end] "
            f"flow={epoch_stats['flow_loss']:.6f} "
            f"distill={epoch_stats['distill_loss']:.6f} "
            f"total={epoch_stats['total_loss']:.6f}",
            flush=True,
        )
    return history


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


def flow_logprob_given_actions(flow, states, actions, flow_steps):
    teacher = flow.teacher
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

    return standard_normal_logprob(z) - div_integral


def compute_flow_log_probs(flow, states, actions, batch_size, flow_steps, device):
    log_probs = []
    for start in range(0, len(states), batch_size):
        end = start + batch_size
        state_tensor = torch.as_tensor(states[start:end], dtype=torch.float32, device=device)
        action_tensor = torch.as_tensor(actions[start:end], dtype=torch.float32, device=device)
        log_prob = flow_logprob_given_actions(flow, state_tensor, action_tensor, flow_steps)
        log_probs.append(log_prob.detach().cpu().numpy())
        if start % max(batch_size * 100, 1) == 0:
            print(f"[flow behavior logprob] {end}/{len(states)}", flush=True)
    return np.concatenate(log_probs).astype(np.float64)


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


def summarize_by_mode(values, modes):
    return {
        mode: summarize(values[modes == mode])
        for mode in sorted(set(modes.tolist()))
    }


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


def mode_counts(modes):
    total = len(modes)
    return {
        mode: {
            "count": int(np.sum(modes == mode)),
            "fraction": float(np.sum(modes == mode) / total),
        }
        for mode in sorted(set(modes.tolist()))
    }


def sweep(log_probs, modes, betas, w_mins, w_maxs, raw_log_weight_clip_quantile, target_pick_fraction):
    pick_mask = modes == "pick"
    if not np.any(pick_mask):
        raise ValueError("No pick chunks found; cannot compute weighted_pick_fraction.")

    rows = []
    for beta in betas:
        raw_log_weight = -float(beta) * log_probs
        raw_log_weight_max = float(np.quantile(raw_log_weight, raw_log_weight_clip_quantile))
        raw_log_weight_capped = np.minimum(raw_log_weight, raw_log_weight_max)
        center = float(np.median(raw_log_weight_capped))
        centered = raw_log_weight_capped - center
        weight_before_clip = np.exp(np.clip(centered, a_min=None, a_max=np.log(np.finfo(np.float64).max)))

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

                by_mode = {}
                before_clip_by_mode = {}
                for mode in sorted(set(modes.tolist())):
                    mask = modes == mode
                    by_mode[mode] = weight_summary(weights, mask)
                    before_clip_by_mode[mode] = weight_summary(weight_before_clip, mask)

                rows.append(
                    {
                        "beta": float(beta),
                        "w_min": float(w_min),
                        "w_max": float(w_max),
                        "raw_log_weight_clip_quantile": float(raw_log_weight_clip_quantile),
                        "raw_log_weight_max": raw_log_weight_max,
                        "log_weight_center": center,
                        "clipped_weight_mean": clipped_weight_mean,
                        "weighted_pick_fraction": float(weighted_pick_fraction),
                        "weighted_push_fraction": float(1.0 - weighted_pick_fraction),
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
                        "by_mode": by_mode,
                        "weight_before_clip_by_mode": before_clip_by_mode,
                    }
                )

    rows.sort(key=lambda item: (item["abs_error"], -item["effective_sample_fraction"], item["w_max"]))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--use_student", type=int, default=0)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--flow_behavior_path", type=str, default=None)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--steps_per_epoch", type=int, default=None)
    parser.add_argument("--train_batch_size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--time_dim", type=int, default=16)
    parser.add_argument("--flow_steps", type=int, default=10)
    parser.add_argument("--flow_use_student", type=int, default=0)
    parser.add_argument("--distill_coef", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=0.0)
    parser.add_argument("--log_interval", type=int, default=100)

    parser.add_argument("--logprob_batch_size", type=int, default=256)
    parser.add_argument("--flow_log_probs_path", type=str, default=None)
    parser.add_argument("--save_flow_log_probs", action="store_true")
    parser.add_argument("--skip_logprob_if_cached", action="store_true")

    parser.add_argument("--betas", type=str, default="0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.12,0.15,0.18,0.2,0.25,0.3")
    parser.add_argument("--w_mins", type=str, default="0.05,0.1,0.2,0.5,1.0")
    parser.add_argument("--w_maxs", type=str, default="500,1000,1500,2000,3000,5000,8000,10000,15000,20000")
    parser.add_argument("--raw_log_weight_clip_quantile", type=float, default=1.0)
    parser.add_argument("--target_pick_fraction", type=float, default=0.5)
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    if not 0 < args.raw_log_weight_clip_quantile <= 1:
        raise ValueError("--raw_log_weight_clip_quantile must be in (0, 1].")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    states, actions, modes = load_labeled_chunks(args.dataset_name, args.subseq_len, args.train_split)
    flow_behavior_path = args.flow_behavior_path or default_flow_behavior_path(
        args.dataset_name,
        args.seed,
        args.use_student,
    )
    os.makedirs(os.path.dirname(flow_behavior_path), exist_ok=True)

    if args.skip_train:
        if not os.path.exists(flow_behavior_path):
            raise FileNotFoundError(f"Flow behavior policy not found: {flow_behavior_path}")
        flow = torch.load(flow_behavior_path, map_location=args.device)
        flow.device = args.device
    else:
        flow = make_flow_behavior(states, actions, args)
        history = train_flow_behavior(flow, states, actions, args)
        torch.save(flow, flow_behavior_path)
        history_path = os.path.splitext(flow_behavior_path)[0] + "_train_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

    flow_steps = int(args.flow_steps or flow.flow_steps)
    flow_log_probs_path = args.flow_log_probs_path
    if flow_log_probs_path is None:
        flow_log_probs_path = os.path.splitext(flow_behavior_path)[0] + "_log_probs.npy"

    if os.path.exists(flow_log_probs_path) and args.skip_logprob_if_cached:
        log_probs = np.load(flow_log_probs_path).astype(np.float64)
        if len(log_probs) != len(modes):
            raise ValueError(f"Cached log_probs length {len(log_probs)} does not match chunks {len(modes)}.")
    else:
        log_probs = compute_flow_log_probs(
            flow,
            states,
            actions,
            args.logprob_batch_size,
            flow_steps,
            args.device,
        )
        if args.save_flow_log_probs:
            np.save(flow_log_probs_path, log_probs)

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
        "flow_behavior_path": flow_behavior_path,
        "flow_log_probs_path": flow_log_probs_path,
        "num_chunks": int(len(modes)),
        "mode_distribution": mode_counts(modes),
        "flow_steps": flow_steps,
        "log_prob_summary": summarize(log_probs),
        "log_prob_by_mode": summarize_by_mode(log_probs, modes),
        "top_k": int(args.top_k),
        "top_results": rows[: args.top_k],
        "all_results": rows,
    }

    text = json.dumps(result, indent=2)
    print(text)
    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        with open(args.output_path, "w") as f:
            f.write(text)


if __name__ == "__main__":
    main()
