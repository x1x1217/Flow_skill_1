import argparse
import json
from collections import defaultdict

import numpy as np


def bucket_name(value, edges):
    if value == 0:
        return "return=0"
    prev = None
    for edge in edges:
        if value < edge:
            if prev is None:
                return f"0<return<{edge:g}"
            return f"{prev:g}<=return<{edge:g}"
        prev = edge
    return f"return>={edges[-1]:g}"


def summarize(records):
    if not records:
        return {}
    out = {
        "count": len(records),
        "return_mean": float(np.mean([r["episode_return"] for r in records])),
        "return_max": float(np.max([r["episode_return"] for r in records])),
        "success_rate": float(np.mean([float(r["success"]) for r in records])),
        "push_frac_mean": float(np.mean([r["push_frac"] for r in records])),
        "pick_frac_mean": float(np.mean([r["pick_frac"] for r in records])),
        "avg_push_prob": float(np.mean([r["avg_push_prob"] for r in records])),
        "avg_pick_prob": float(np.mean([r["avg_pick_prob"] for r in records])),
        "chunk_count_mean": float(np.mean([r["chunk_count"] for r in records])),
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_path", required=True)
    parser.add_argument("--edges", type=float, nargs="+", default=[10.0, 20.0, 30.0])
    parser.add_argument("--epoch_min", type=int, default=None)
    parser.add_argument("--epoch_max", type=int, default=None)
    args = parser.parse_args()

    records = []
    with open(args.stats_path) as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            epoch = int(record["epoch"])
            if args.epoch_min is not None and epoch < args.epoch_min:
                continue
            if args.epoch_max is not None and epoch > args.epoch_max:
                continue
            records.append(record)

    buckets = defaultdict(list)
    for record in records:
        buckets[bucket_name(float(record["episode_return"]), args.edges)].append(record)

    result = {
        "stats_path": args.stats_path,
        "epoch_min": args.epoch_min,
        "epoch_max": args.epoch_max,
        "overall": summarize(records),
        "buckets": {name: summarize(values) for name, values in sorted(buckets.items())},
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
