import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


LABEL_TO_ID = {"push": 0, "pick": 1}
ID_TO_LABEL = {0: "push", 1: "pick"}


class ModeClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, hidden_layers=2, num_classes=2):
        super().__init__()
        layers = []
        dim = input_dim
        for _ in range(hidden_layers):
            layers.extend([nn.Linear(dim, hidden_dim), nn.ReLU()])
            dim = hidden_dim
        layers.append(nn.Linear(dim, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def repo_root():
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_chunks(dataset_name, subseq_len, train_split, max_chunks_per_mode=None, seed=0):
    path = os.path.join(repo_root(), "dataset", dataset_name, "demos.npy")
    seqs = np.load(path, allow_pickle=True)
    split_end = int(len(seqs) * train_split)
    seqs = seqs[:split_end]

    by_mode = {mode: [] for mode in LABEL_TO_ID}
    for seq in seqs:
        mode = str(seq.get("mode", ""))
        if mode not in LABEL_TO_ID:
            continue
        obs = np.asarray(seq.obs, dtype=np.float32)
        actions = np.asarray(seq.actions, dtype=np.float32)
        max_start = min(len(obs), len(actions)) - subseq_len
        for start in range(max_start):
            x = np.concatenate([obs[start], actions[start]], axis=0)
            by_mode[mode].append(x)

    rng = np.random.default_rng(seed)
    xs, ys = [], []
    counts = {}
    for mode, values in by_mode.items():
        values = np.asarray(values, dtype=np.float32)
        counts[mode] = int(len(values))
        if len(values) == 0:
            continue
        if max_chunks_per_mode is not None and len(values) > max_chunks_per_mode:
            idx = rng.choice(len(values), size=max_chunks_per_mode, replace=False)
            values = values[idx]
        xs.append(values)
        ys.append(np.full(len(values), LABEL_TO_ID[mode], dtype=np.int64))

    if not xs:
        raise RuntimeError(f"No labeled pick/push chunks found in {path}")

    x = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    perm = rng.permutation(len(x))
    return x[perm], y[perm], counts


def split_train_val(x, y, val_fraction):
    n_val = max(1, int(len(x) * val_fraction))
    return x[n_val:], y[n_val:], x[:n_val], y[:n_val]


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    confusion = np.zeros((2, 2), dtype=np.int64)
    losses = []
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        losses.append(F.cross_entropy(logits, yb).item())
        pred = torch.argmax(logits, dim=1)
        correct += int((pred == yb).sum().item())
        total += int(yb.numel())
        for t, p in zip(yb.cpu().numpy(), pred.cpu().numpy()):
            confusion[int(t), int(p)] += 1
    return {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(correct / total) if total else 0.0,
        "confusion": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--train_split", type=float, default=0.99)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--max_chunks_per_mode", type=int, default=200000)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--hidden_layers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    x, y, raw_counts = load_chunks(
        args.dataset_name,
        args.subseq_len,
        args.train_split,
        args.max_chunks_per_mode,
        args.seed,
    )
    x_train, y_train, x_val, y_val = split_train_val(x, y, args.val_fraction)

    mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    std = x_train.std(axis=0, keepdims=True).astype(np.float32)
    std = np.maximum(std, 1e-6)
    x_train = (x_train - mean) / std
    x_val = (x_val - mean) / std

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ModeClassifier(
        input_dim=x_train.shape[1],
        hidden_dim=args.hidden_dim,
        hidden_layers=args.hidden_layers,
    ).to(device)

    class_counts = np.bincount(y_train, minlength=2).astype(np.float32)
    class_weights = class_counts.sum() / np.maximum(class_counts, 1.0)
    class_weights = class_weights / class_weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_val), torch.from_numpy(y_val)),
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    best_state = None
    best_acc = -1.0
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        val = evaluate(model, val_loader, device)
        if val["accuracy"] > best_acc:
            best_acc = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[epoch {epoch:03d}] train_loss={np.mean(losses):.6f} "
            f"val_loss={val['loss']:.6f} val_acc={val['accuracy']:.4f} "
            f"confusion={val['confusion']}",
            flush=True,
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    final_val = evaluate(model, val_loader, device)

    output = args.output
    if output is None:
        output = os.path.join(
            repo_root(),
            "reskill",
            "results",
            "mode_classifiers",
            args.dataset_name,
            "mode_classifier.pth",
        )
    os.makedirs(os.path.dirname(output), exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "input_dim": int(x_train.shape[1]),
        "hidden_dim": int(args.hidden_dim),
        "hidden_layers": int(args.hidden_layers),
        "feature_mean": mean.squeeze(0),
        "feature_std": std.squeeze(0),
        "label_to_id": LABEL_TO_ID,
        "id_to_label": ID_TO_LABEL,
        "dataset_name": args.dataset_name,
        "subseq_len": args.subseq_len,
        "train_split": args.train_split,
        "raw_mode_counts": raw_counts,
        "sampled_mode_counts": {
            mode: int(np.sum(y == label_id))
            for mode, label_id in LABEL_TO_ID.items()
        },
        "val_metrics": final_val,
    }
    torch.save(checkpoint, output)

    metrics_path = os.path.splitext(output)[0] + "_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(
            {
                "output": output,
                "dataset_name": args.dataset_name,
                "raw_mode_counts": raw_counts,
                "sampled_mode_counts": checkpoint["sampled_mode_counts"],
                "val_metrics": final_val,
            },
            f,
            indent=2,
        )
    print(f"Saved classifier: {output}")
    print(f"Saved metrics: {metrics_path}")


if __name__ == "__main__":
    main()
