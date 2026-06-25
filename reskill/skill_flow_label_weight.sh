set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow
mkdir -p logs/Flow/skill_flow_label_weight

experiments=(
  "1 300 fetch_block_push300_pick1_labeled fetch_block_push300_pick1_labeled_label_weight 2"
  "500 1 fetch_block_push1_pick500_labeled fetch_block_push1_pick500_labeled_label_weight 3"
)

seeds=(2 3 20)
use_student=0
swanlab_project=Flow_skill_1_offline_label_weight

skill_epochs=0
prior_epochs=400
prior_updates_per_batch=1
prior_use_mu=1
val_freq=5
save_freq=50
action_noise_std=0.0

copy_dataset_if_needed() {
  local src_dataset="$1"
  local dst_dataset="$2"

  if [ ! -d "dataset/${dst_dataset}" ]; then
    cp -a "dataset/${src_dataset}" "dataset/${dst_dataset}"
  fi
}

copy_skill_vae_if_needed() {
  local src_dataset="$1"
  local dst_dataset="$2"
  local seed="$3"
  local src_dir="reskill/results/saved_skill_models/${src_dataset}/Flow/seed_${seed}/skill_prior_Flow_student${use_student}"
  local dst_dir="reskill/results/saved_skill_models/${dst_dataset}/Flow/seed_${seed}/skill_prior_Flow_student${use_student}"

  mkdir -p "$dst_dir"
  for name in best_skill_vae.pth skill_vae.pth; do
    if [ -f "${src_dir}/${name}" ] && [ ! -f "${dst_dir}/${name}" ]; then
      cp "${src_dir}/${name}" "${dst_dir}/${name}"
    fi
  done
}

run_experiment() {
  local pick="$1"
  local push="$2"
  local src_dataset="$3"
  local dataset_name="$4"
  local gpu="$5"

  copy_dataset_if_needed "$src_dataset" "$dataset_name"

  for seed in "${seeds[@]}"; do
    copy_skill_vae_if_needed "$src_dataset" "$dataset_name" "$seed"
    mkdir -p "logs/Flow/skill_flow/${dataset_name}/seed${seed}"

    CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.train_skill_modules_label_weight \
      --prior_model Flow \
      --pick "$pick" \
      --push "$push" \
      --dataset_name "$dataset_name" \
      --seed "$seed" \
      --use_student "$use_student" \
      --skill_epochs "$skill_epochs" \
      --prior_epochs "$prior_epochs" \
      --prior_updates_per_batch "$prior_updates_per_batch" \
      --prior_use_mu "$prior_use_mu" \
      --val_freq "$val_freq" \
      --save_freq "$save_freq" \
      --action_noise_std "$action_noise_std" \
      --swanlab_project "$swanlab_project" \
      > "logs/Flow/skill_flow/${dataset_name}/seed${seed}/pick${pick}_push${push}_condition_flow_label_weight.log" 2>&1 &
  done

  wait
}

for exp in "${experiments[@]}"; do
  read -r pick push src_dataset dataset_name gpu <<< "$exp"
  run_experiment "$pick" "$push" "$src_dataset" "$dataset_name" "$gpu" &
done

wait
