import argparse
import json
import os
import random
import sys

import gym
import numpy as np
from perlin_noise import PerlinNoise
from tqdm import tqdm

sys.path.append("../")
sys.path.append("../rl")

import reskill.rl.envs
from reskill.utils.controllers.hook_controller import get_hook_control
from reskill.utils.controllers.pick_and_place_controller import get_pick_and_place_control
from reskill.utils.controllers.push_controller import get_push_control
from reskill.utils.general_utils import AttrDict


class CollectLabeledDemos:
    """
    Generate Fetch demonstrations and save the controller mode for each trajectory.
    This intentionally lives beside collect_demos.py so existing datasets/training are untouched.
    """

    def __init__(self, dataset_name, num_trajectories=5, subseq_len=10, task="block", push=999, pick=1):
        self.seqs = []
        self.task = task
        self.push = push
        self.pick = pick
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        self.dataset_dir = os.path.join(repo_root, "dataset", dataset_name)
        os.makedirs(self.dataset_dir, exist_ok=True)
        self.save_dir = os.path.join(self.dataset_dir, "demos.npy")
        self.metadata_path = os.path.join(self.dataset_dir, "metadata.json")
        self.num_trajectories = num_trajectories
        self.subseq_len = subseq_len
        self.mode_counts = {"push": 0, "pick": 0, "hook": 0}

        if self.task == "hook":
            self.env = gym.make("FetchHook-v0")
        else:
            self.env = gym.make("FetchPlaceMultiGoal-v0")

    def get_p_noise(self, idx, factor):
        return np.array([
            self.x_noise(idx / factor),
            self.y_noise(idx / factor),
            self.z_noise(idx / factor),
            0,
        ])

    def get_obs(self, obs):
        return np.concatenate([obs["observation"], obs["desired_goal"]])

    def choose_controller(self):
        if self.task != "block":
            return get_hook_control, "hook", 2

        choices = [(get_push_control, "push", 0) for _ in range(self.push)]
        choices += [(get_pick_and_place_control, "pick", 1) for _ in range(self.pick)]
        return random.choice(choices)

    def collect(self):
        print("Collecting labeled demonstrations...")
        attempted_mode_counts = {"push": 0, "pick": 0, "hook": 0}

        for _ in tqdm(range(self.num_trajectories)):
            obs = self.env.reset()
            done = False
            actions = []
            observations = []
            terminals = []

            self.x_noise = PerlinNoise(octaves=3)
            self.y_noise = PerlinNoise(octaves=3)
            self.z_noise = PerlinNoise(octaves=3)

            controller, mode, mode_id = self.choose_controller()
            attempted_mode_counts[mode] += 1
            idx = 0

            while not done:
                observations.append(self.get_obs(obs))

                p_noise = self.get_p_noise(idx, 1000)
                idx += 1

                action, success = controller(obs)
                action += p_noise * 0.5
                actions.append(action)

                obs, _, done, _ = self.env.step(action)
                terminals.append(success)

                if success:
                    break

            if len(actions) <= self.subseq_len + 1:
                continue

            self.mode_counts[mode] += 1
            self.seqs.append(
                AttrDict(
                    obs=observations,
                    actions=actions,
                    mode=mode,
                    mode_id=mode_id,
                    terminals=terminals,
                )
            )

        np.save(self.save_dir, np.array(self.seqs))
        metadata = {
            "dataset_name": os.path.basename(self.dataset_dir),
            "task": self.task,
            "requested_num_trajectories": self.num_trajectories,
            "saved_num_trajectories": len(self.seqs),
            "push": self.push,
            "pick": self.pick,
            "attempted_mode_counts": attempted_mode_counts,
            "saved_mode_counts": self.mode_counts,
            "subseq_len": self.subseq_len,
            "label_schema": {"push": 0, "pick": 1, "hook": 2},
        }
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Dataset generated: {self.save_dir}")
        print(f"Metadata saved: {self.metadata_path}")
        print(f"Saved mode counts: {self.mode_counts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_trajectories", type=int, default=10)
    parser.add_argument("--subseq_len", type=int, default=10)
    parser.add_argument("--task", type=str, default="block", choices=["block", "hook"])
    parser.add_argument("--push", type=int, default=999)
    parser.add_argument("--pick", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default=None)
    args = parser.parse_args()

    dataset_name = args.dataset_name or f"fetch_{args.task}_push{args.push}_pick{args.pick}_labeled"
    print(f"Dataset name: {dataset_name}")

    collector = CollectLabeledDemos(
        dataset_name=dataset_name,
        num_trajectories=args.num_trajectories,
        subseq_len=args.subseq_len,
        task=args.task,
        push=args.push,
        pick=args.pick,
    )
    collector.collect()
