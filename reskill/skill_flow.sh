#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)
cd "$repo_root"

source_dataset=fetch_block_push1_pick500
dataset_name=$source_dataset
legacy_repo_root=${LEGACY_REPO_ROOT:-/home/algroup/x1x/Flow_skill_1}
pick=500
push=1

gpus=(0 1 3)
seeds=(2 3 20)

use_student=0
skill_epochs=400
behavior_epochs=50
prior_epochs=400
prior_updates_per_batch=1
prior_use_mu=1
val_freq=5
save_freq=50
action_noise_std=0.0

condition_reweight=1
condition_weight_beta=0.2
condition_weight_min=0.1
condition_weight_max=300.0
condition_raw_log_weight_clip_quantile=1.0

behavior_train_batch_size=1024
behavior_logprob_batch_size=256
behavior_flow_steps=10
behavior_hidden_dim=256
behavior_time_dim=16
behavior_lr=3e-4

skill_vae_project=Flow_skill_1_offline_skill_vae
weighted_prior_project=Flow_skill_1_offline_flow_behavior_weight

prepare_dataset() {
  if [[ -f "dataset/$dataset_name/demos.npy" ]]; then
    return
  fi

  local source_dir="dataset/$source_dataset"
  if [[ ! -f "$source_dir/demos.npy" ]]; then
    source_dir="$legacy_repo_root/dataset/$source_dataset"
  fi
  if [[ ! -f "$source_dir/demos.npy" ]]; then
    echo "Source dataset not found: $source_dataset" >&2
    exit 2
  fi
  source_dir=$(realpath "$source_dir")
  ln -s "$source_dir" "dataset/$dataset_name"
  echo "[linked] dataset/$dataset_name -> $source_dir"
}

prepare_dataset

archive_existing_models() {
  local model_dir=$1
  if [[ ${FRESH_RUN:-1} == 1 && -d "$model_dir" ]]; then
    local backup_dir="${model_dir}_backup_$(date +%Y%m%d_%H%M%S)"
    mv "$model_dir" "$backup_dir"
    echo "[archived] $model_dir -> $backup_dir"
  fi
  mkdir -p "$model_dir"
}

run_seed() {
  local seed=$1
  local gpu=$2
  local model_dir="reskill/results/saved_skill_models/$dataset_name/Flow/seed_${seed}/skill_prior_Flow_student${use_student}"
  local log_dir="logs/Flow/skill_flow/$dataset_name/seed${seed}"
  local behavior_policy_path="$model_dir/behavior_flow_policy.pth"
  local behavior_log_probs_path="$model_dir/behavior_flow_policy_log_probs.npy"

  mkdir -p "$log_dir"
  archive_existing_models "$model_dir"

  echo "[stage 1/3] seed=$seed gpu=$gpu train SkillVAE"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.train_skill_modules \
    --prior_model Flow \
    --pick "$pick" \
    --push "$push" \
    --dataset_name "$dataset_name" \
    --seed "$seed" \
    --use_student "$use_student" \
    --skill_epochs "$skill_epochs" \
    --prior_epochs 0 \
    --prior_updates_per_batch "$prior_updates_per_batch" \
    --prior_use_mu "$prior_use_mu" \
    --val_freq "$val_freq" \
    --save_freq "$save_freq" \
    --action_noise_std "$action_noise_std" \
    --condition_reweight 0 \
    --swanlab_project "$skill_vae_project" \
    > "$log_dir/01_skill_vae.log" 2>&1

  if [[ ! -f "$model_dir/best_skill_vae.pth" ]]; then
    echo "SkillVAE stage did not produce $model_dir/best_skill_vae.pth" >&2
    return 1
  fi

  echo "[stage 2/3] seed=$seed gpu=$gpu train Flow behavior and compute full log-probs"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.analysis.train_flow_behavior_policy \
    --dataset_name "$dataset_name" \
    --seed "$seed" \
    --use_student "$use_student" \
    --epochs "$behavior_epochs" \
    --train_batch_size "$behavior_train_batch_size" \
    --logprob_batch_size "$behavior_logprob_batch_size" \
    --flow_use_student 0 \
    --flow_steps "$behavior_flow_steps" \
    --hidden_dim "$behavior_hidden_dim" \
    --time_dim "$behavior_time_dim" \
    --lr "$behavior_lr" \
    --distill_coef 1.0 \
    --grad_clip 0.0 \
    --subseq_len 10 \
    --train_split 0.99 \
    --flow_behavior_path "$behavior_policy_path" \
    --flow_log_probs_path "$behavior_log_probs_path" \
    --history_path "$log_dir/02_flow_behavior_train_history.json" \
    --summary_path "$log_dir/02_flow_behavior_logprob_summary.json" \
    > "$log_dir/02_flow_behavior.log" 2>&1

  if [[ ! -f "$behavior_policy_path" || ! -f "$behavior_log_probs_path" ]]; then
    echo "Flow behavior stage did not produce its policy and log-prob cache for seed $seed" >&2
    return 1
  fi

  echo "[stage 3/3] seed=$seed gpu=$gpu train weighted condition prior and skill prior"
  CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.train_skill_modules_flow_behavior_weight \
    --prior_model Flow \
    --pick "$pick" \
    --push "$push" \
    --dataset_name "$dataset_name" \
    --seed "$seed" \
    --use_student "$use_student" \
    --skill_epochs 0 \
    --prior_epochs "$prior_epochs" \
    --prior_updates_per_batch "$prior_updates_per_batch" \
    --prior_use_mu "$prior_use_mu" \
    --val_freq "$val_freq" \
    --save_freq "$save_freq" \
    --action_noise_std "$action_noise_std" \
    --condition_reweight "$condition_reweight" \
    --condition_weight_beta "$condition_weight_beta" \
    --condition_weight_min "$condition_weight_min" \
    --condition_weight_max "$condition_weight_max" \
    --condition_raw_log_weight_clip_quantile "$condition_raw_log_weight_clip_quantile" \
    --flow_behavior_policy_path "$behavior_policy_path" \
    --flow_behavior_log_probs_path "$behavior_log_probs_path" \
    --flow_behavior_logprob_batch_size "$behavior_logprob_batch_size" \
    --swanlab_project "$weighted_prior_project" \
    > "$log_dir/03_weighted_priors.log" 2>&1

  for checkpoint in best_skill_vae.pth best_condition_prior.pth best_skill_prior.pth; do
    if [[ ! -f "$model_dir/$checkpoint" ]]; then
      echo "Offline pipeline missing checkpoint: $model_dir/$checkpoint" >&2
      return 1
    fi
  done
  echo "[complete] seed=$seed offline checkpoints are ready in $model_dir"
}

pids=()
cleanup() {
  if (( ${#pids[@]} > 0 )); then
    kill "${pids[@]}" 2>/dev/null || true
  fi
}
trap cleanup INT TERM

for index in "${!seeds[@]}"; do
  run_seed "${seeds[$index]}" "${gpus[$index]}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done
exit "$status"
