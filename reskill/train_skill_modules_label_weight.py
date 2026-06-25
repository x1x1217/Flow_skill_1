import argparse
import json
import os

import numpy as np
import torch
import yaml
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from reskill.train_skill_modules import ModelTrainer
from reskill.utils.general_utils import AttrDict
from reskill.utils.swanlab_writer import SwanLabWriter


class LabeledSkillsDataset(Dataset):
    SPLIT = AttrDict(train=0.99, val=0.01, test=0.0)

    def __init__(self, dataset_name, phase, subseq_len, transform=None):
        self.phase = phase
        self.subseq_len = subseq_len
        curr_dir = os.path.dirname(__file__)
        fname = os.path.join(curr_dir, "../dataset", dataset_name, "demos.npy")
        self.seqs = np.load(fname, allow_pickle=True)
        self.transform = transform
        self.n_seqs = len(self.seqs)
        print("Dataset size: ", self.n_seqs)

        missing = [i for i, seq in enumerate(self.seqs) if "mode_id" not in seq or "mode" not in seq]
        if missing:
            raise ValueError(
                f"{dataset_name} has {len(missing)} trajectories without mode/mode_id labels. "
                "Use a labeled dataset for label-weight training."
            )

        if self.phase == "train":
            self.start = 0
            self.end = int(self.SPLIT.train * self.n_seqs)
        elif self.phase == "val":
            self.start = int(self.SPLIT.train * self.n_seqs)
            self.end = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
        elif self.phase == "test":
            self.start = int((self.SPLIT.train + self.SPLIT.val) * self.n_seqs)
            self.end = self.n_seqs
        else:
            raise ValueError(f"Unknown phase: {phase}")

    def __getitem__(self, index):
        seq = self._sample_seq()
        start_idx = np.random.randint(0, (len(seq.actions) - self.subseq_len - 1))
        actions = np.array(seq.actions[start_idx:start_idx + self.subseq_len], dtype=np.float32)
        obs = np.array(seq.obs[start_idx:start_idx + self.subseq_len], dtype=np.float32)
        return AttrDict(
            obs=obs,
            actions=actions,
            mode_id=np.int64(seq.mode_id),
            mode=seq.mode,
        )

    def _sample_seq(self):
        return np.random.choice(self.seqs[self.start:self.end])

    def __len__(self):
        return int(self.end - self.start)


class LabelWeightModelTrainer(ModelTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        curr_dir = os.path.dirname(__file__)
        config_file = args[1] if len(args) > 1 else kwargs.get("config_file", "block/config.yaml")
        config_path = os.path.join(curr_dir, "configs", "skill_mdl", config_file)
        with open(config_path, "r") as f:
            conf = yaml.safe_load(f)
            conf = AttrDict(conf)
        for key in conf:
            conf[key] = AttrDict(conf[key])

        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0.5, 0.5)])
        train_data = LabeledSkillsDataset(
            self.dataset_name,
            phase="train",
            subseq_len=conf.skill_vae.subseq_len,
            transform=transform,
        )
        val_data = LabeledSkillsDataset(
            self.dataset_name,
            phase="val",
            subseq_len=conf.skill_vae.subseq_len,
            transform=transform,
        )

        self.train_loader = DataLoader(
            train_data,
            batch_size=conf.skill_vae.batch_size,
            shuffle=True,
            drop_last=True,
            prefetch_factor=30,
            num_workers=conf.loader.num_workers,
            pin_memory=True,
        )
        self.val_loader = DataLoader(
            val_data,
            batch_size=64,
            shuffle=False,
            drop_last=True,
            prefetch_factor=30,
            num_workers=conf.loader.num_workers,
            pin_memory=True,
        )

        self.label_weight_stats_path = os.path.join(self.save_dir, "label_condition_weight_stats.json")
        self.mode_id_to_weight = None
        self.mode_id_to_name = None

    def compute_label_condition_weight_stats(self):
        dataset = self.train_loader.dataset
        counts = {}
        names = {}
        for seq_idx in range(dataset.start, dataset.end):
            seq = dataset.seqs[seq_idx]
            mode_id = int(seq.mode_id)
            counts[mode_id] = counts.get(mode_id, 0) + 1
            names[mode_id] = str(seq.mode)

        if len(counts) != 2:
            raise ValueError(
                f"Expected exactly two labels for N/(2*n_class), got {len(counts)}: {counts}"
            )

        total = sum(counts.values())
        weights = {mode_id: total / (2.0 * count) for mode_id, count in counts.items()}
        weighted_sums = {mode_id: counts[mode_id] * weights[mode_id] for mode_id in counts}
        weighted_total = sum(weighted_sums.values())

        self.mode_id_to_weight = weights
        self.mode_id_to_name = names

        stats = {
            "dataset_name": self.dataset_name,
            "num_train_trajectories": int(total),
            "formula": "w_class = N / (2 * n_class)",
            "counts_by_mode": {
                names[mode_id]: {
                    "mode_id": int(mode_id),
                    "count": int(counts[mode_id]),
                    "raw_fraction": float(counts[mode_id] / total),
                    "weight": float(weights[mode_id]),
                    "weighted_sum": float(weighted_sums[mode_id]),
                    "weighted_fraction": float(weighted_sums[mode_id] / weighted_total),
                }
                for mode_id in sorted(counts)
            },
        }
        with open(self.label_weight_stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        print("[label condition weights]", json.dumps(stats["counts_by_mode"], indent=2), flush=True)
        for mode_name, mode_stats in stats["counts_by_mode"].items():
            self.writer.add_scalar(f"label_condition_weights/{mode_name}_count", mode_stats["count"], 0)
            self.writer.add_scalar(f"label_condition_weights/{mode_name}_weight", mode_stats["weight"], 0)
            self.writer.add_scalar(
                f"label_condition_weights/{mode_name}_weighted_fraction",
                mode_stats["weighted_fraction"],
                0,
            )

    def compute_label_condition_weights(self, data):
        if self.mode_id_to_weight is None:
            if not os.path.exists(self.label_weight_stats_path):
                raise RuntimeError("Label condition weight stats are not computed.")
            with open(self.label_weight_stats_path, "r") as f:
                stats = json.load(f)
            self.mode_id_to_weight = {
                int(mode_stats["mode_id"]): float(mode_stats["weight"])
                for mode_stats in stats["counts_by_mode"].values()
            }
            self.mode_id_to_name = {
                int(mode_stats["mode_id"]): mode_name
                for mode_name, mode_stats in stats["counts_by_mode"].items()
            }

        mode_ids = data["mode_id"].to(self.device).long()
        weights = torch.ones(mode_ids.shape[0], dtype=torch.float32, device=self.device)
        for mode_id, weight in self.mode_id_to_weight.items():
            weights = torch.where(
                mode_ids == int(mode_id),
                torch.full_like(weights, float(weight)),
                weights,
            )
        return weights

    def train_flow_prior_epoch(self, epoch):
        self.skill_vae.eval()
        losses = []
        skill_flow_losses = []
        skill_distill_losses = []
        condition_flow_losses = []
        condition_distill_losses = []
        condition_weight_means = []
        condition_weight_maxes = []

        for batch_idx, data in enumerate(self.train_loader):
            log_step = epoch * len(self.train_loader) + batch_idx
            with torch.no_grad():
                state, action_ori, target_z = self.extract_flow_targets(data)
                condition_weight = self.compute_label_condition_weights(data)

            batch_skill_flow = []
            batch_skill_distill = []
            batch_condition_flow = []
            batch_condition_distill = []

            for _ in range(self.prior_updates_per_batch):
                action = action_ori
                condition = torch.cat([state, action], dim=1)
                metric = self.sp_nvp.train(condition, target_z, iterations=1)
                condition_metric = self.condition_prior.train(
                    state,
                    action,
                    iterations=1,
                    sample_weight=condition_weight,
                )
                batch_skill_flow.append(float(np.mean(metric["flow_loss"])))
                batch_skill_distill.append(float(np.mean(metric["distill_loss"])))
                batch_condition_flow.append(float(np.mean(condition_metric["flow_loss"])))
                batch_condition_distill.append(float(np.mean(condition_metric["distill_loss"])))

            skill_flow_loss = float(np.mean(batch_skill_flow))
            skill_distill_loss = float(np.mean(batch_skill_distill))
            condition_flow_loss = float(np.mean(batch_condition_flow))
            condition_distill_loss = float(np.mean(batch_condition_distill))
            total_loss = skill_flow_loss + condition_flow_loss
            losses.append(total_loss)
            skill_flow_losses.append(skill_flow_loss)
            skill_distill_losses.append(skill_distill_loss)
            condition_flow_losses.append(condition_flow_loss)
            condition_distill_losses.append(condition_distill_loss)
            condition_weight_means.append(float(condition_weight.mean().item()))
            condition_weight_maxes.append(float(condition_weight.max().item()))

            if batch_idx % 10 == 0:
                print(
                    f"[prior epoch {epoch:03d} batch {batch_idx:04d}/{len(self.train_loader)}] "
                    f"skill_flow={skill_flow_loss:.6f} "
                    f"skill_distill={skill_distill_loss:.6f} "
                    f"condition_flow={condition_flow_loss:.6f} "
                    f"condition_distill={condition_distill_loss:.6f} "
                    f"label_weight_mean={condition_weight.mean().item():.6f} "
                    f"label_weight_max={condition_weight.max().item():.6f}",
                    flush=True,
                )
                self.writer.add_scalar("flow_prior_train_batch/flow_loss", skill_flow_loss, log_step)
                self.writer.add_scalar("flow_prior_train_batch/distill_loss", skill_distill_loss, log_step)
                self.writer.add_scalar("condition_flow_train_batch/flow_loss", condition_flow_loss, log_step)
                self.writer.add_scalar("condition_flow_train_batch/distill_loss", condition_distill_loss, log_step)
                self.writer.add_scalar("condition_flow_train_batch/label_weight_mean", condition_weight.mean().item(), log_step)
                self.writer.add_scalar("condition_flow_train_batch/label_weight_max", condition_weight.max().item(), log_step)

        return AttrDict(
            total_loss=float(np.mean(losses)),
            prior_flow_loss=float(np.mean(skill_flow_losses)),
            prior_distill_loss=float(np.mean(skill_distill_losses)),
            condition_flow_loss=float(np.mean(condition_flow_losses)),
            condition_distill_loss=float(np.mean(condition_distill_losses)),
            condition_weight_mean=float(np.mean(condition_weight_means)),
            condition_weight_max=float(np.mean(condition_weight_maxes)),
        )

    def train_flow_two_stage(self):
        print("Training stage 1: SkillVAE...", flush=True)
        for epoch in tqdm(range(self.skill_epochs)):
            train_metrics = self.train_flow_skill_epoch(epoch)
            self.writer.add_scalar("skill_train_epoch/total_loss", train_metrics.total_loss, epoch)
            self.writer.add_scalar("skill_train_epoch/bc_loss", train_metrics.bc_loss, epoch)
            self.writer.add_scalar("skill_train_epoch/kl_loss", train_metrics.kl_loss, epoch)

            if epoch % self.val_freq == 0:
                val_metrics = self.validate_flow_skill(epoch)
                if val_metrics.vae_bc_loss < self.best_vae_val_loss:
                    self.best_vae_val_loss = val_metrics.vae_bc_loss
                    torch.save(self.skill_vae, self.best_vae_save_path)
                    self.writer.add_scalar("skill_val_epoch/best_bc_loss", self.best_vae_val_loss, epoch)
                    print(f"[best skill vae epoch {epoch:03d}] val_bc={self.best_vae_val_loss:.6f}", flush=True)

            if epoch % self.save_freq == 0 or epoch == self.skill_epochs - 1:
                torch.save(self.skill_vae, self.vae_save_path)

        if os.path.exists(self.best_vae_save_path):
            self.skill_vae = torch.load(self.best_vae_save_path, map_location=self.device)
        self.skill_vae.to(self.device)
        self.skill_vae.eval()
        for param in self.skill_vae.parameters():
            param.requires_grad_(False)

        if self.prior_model != "Flow":
            raise ValueError("Label-weight training is only implemented for --prior_model Flow.")

        self.compute_label_condition_weight_stats()

        print("Training stage 2: skill prior + label-weighted condition prior...", flush=True)
        for epoch in tqdm(range(self.prior_epochs)):
            train_metrics = self.train_flow_prior_epoch(epoch)
            self.writer.add_scalar("flow_prior_train_epoch/flow_loss", train_metrics.prior_flow_loss, epoch)
            self.writer.add_scalar("flow_prior_train_epoch/distill_loss", train_metrics.prior_distill_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/flow_loss", train_metrics.condition_flow_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/distill_loss", train_metrics.condition_distill_loss, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/label_weight_mean", train_metrics.condition_weight_mean, epoch)
            self.writer.add_scalar("condition_flow_train_epoch/label_weight_max", train_metrics.condition_weight_max, epoch)

            if epoch % self.val_freq == 0:
                val_metrics = self.validate_flow_priors(epoch)
                if val_metrics.prior_flow_loss < self.best_prior_val_loss:
                    self.best_prior_val_loss = val_metrics.prior_flow_loss
                    torch.save(self.sp_nvp, self.best_sp_save_path)
                    self.writer.add_scalar("flow_prior_val/best_flow_loss", self.best_prior_val_loss, epoch)
                    print(f"[best skill prior epoch {epoch:03d}] val_flow={self.best_prior_val_loss:.6f}", flush=True)
                if val_metrics.condition_flow_loss < self.best_condition_val_loss:
                    self.best_condition_val_loss = val_metrics.condition_flow_loss
                    torch.save(self.condition_prior, self.best_condition_prior_save_path)
                    self.writer.add_scalar("condition_flow_val/best_flow_loss", self.best_condition_val_loss, epoch)
                    print(f"[best condition prior epoch {epoch:03d}] val_flow={self.best_condition_val_loss:.6f}", flush=True)

            if epoch % self.save_freq == 0 or epoch == self.prior_epochs - 1:
                torch.save(self.sp_nvp, self.sp_save_path)
                torch.save(self.condition_prior, self.condition_prior_save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, default="block/config.yaml")
    parser.add_argument("--pick", type=int, default=1)
    parser.add_argument("--push", type=int, default=1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--prior_model", type=str, default="Flow")
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--use_student", type=int, default=1)
    parser.add_argument("--skill_epochs", type=int, default=None)
    parser.add_argument("--prior_epochs", type=int, default=None)
    parser.add_argument("--prior_updates_per_batch", type=int, default=1)
    parser.add_argument("--prior_use_mu", type=int, default=1)
    parser.add_argument("--val_freq", type=int, default=5)
    parser.add_argument("--save_freq", type=int, default=50)
    parser.add_argument("--action_noise_std", type=float, default=0.0)
    parser.add_argument("--swanlab_project", type=str, default="Flow_skill_1")
    parser.add_argument("--swanlab_workspace", type=str, default="x1x1217")
    parser.add_argument("--swanlab_mode", type=str, default=None)
    args = parser.parse_args()

    if args.prior_model != "Flow":
        raise ValueError("--prior_model must be Flow for label-weight training.")
    if args.prior_updates_per_batch < 1:
        raise ValueError("--prior_updates_per_batch must be >= 1")
    if args.val_freq < 1:
        raise ValueError("--val_freq must be >= 1")
    if args.save_freq < 1:
        raise ValueError("--save_freq must be >= 1")
    if args.dataset_name is None:
        args.dataset_name = f"fetch_block_push{args.push}_pick{args.pick}"

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    flow_suffix = f"_student{args.use_student}"
    curr_dir = os.path.dirname(__file__)
    log_file = os.path.join(
        curr_dir,
        "swanlog",
        "skill_prior",
        args.dataset_name,
        f"seed_{args.seed}_{args.prior_model}{flow_suffix}_label_weight",
    )
    os.makedirs(log_file, exist_ok=True)
    writer = SwanLabWriter(
        project=args.swanlab_project,
        workspace=args.swanlab_workspace,
        experiment_name=f"skill_prior_seed{args.seed}_{args.prior_model}{flow_suffix}_label_weight",
        config=vars(args),
        logdir=log_file,
        mode=args.swanlab_mode,
        tags=["skill_prior", args.prior_model, "label_weight"],
    )

    trainer = LabelWeightModelTrainer(
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
        condition_reweight=False,
    )
    trainer.train()
    writer.close()


if __name__ == "__main__":
    main()
