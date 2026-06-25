# cd ..
# python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2

# set -e

# cd ..
# CUDA_VISIBLE_DEVICES=0 python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2 &
# CUDA_VISIBLE_DEVICES=1 python -m reskill.train_skill_modules --prior_model Flow --pick 1 --push 999 --seed 2 &

# wait

set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow
mkdir -p logs/Flow/skill_flow_verify

experiments=(
  "1 300 fetch_block_push300_pick1_labeled_beta05_wmax200 2"
  "500 1 fetch_block_push1_pick500_labeled_beta05_wmax200 3"
)

# seeds=(2 3 20)
seeds=(2 3 20)
use_student=0
swanlab_project=Flow_skill_1_offline_verify

skill_epochs=0
prior_epochs=400
behavior_policy_epochs=50

prior_updates_per_batch=1
prior_use_mu=1
val_freq=5
save_freq=50
action_noise_std=0.0
condition_reweight=1
condition_weight_beta=0.5
condition_weight_min=0.2
condition_weight_max=200.0
condition_raw_log_weight_clip_quantile=1.0

run_experiment() {
  local pick="$1"
  local push="$2"
  local dataset_name="$3"
  local gpu="$4"

  for seed in "${seeds[@]}"; do
    mkdir -p "logs/Flow/skill_flow/${dataset_name}/seed${seed}"

    CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.train_skill_modules \
      --prior_model Flow \
      --pick "$pick" \
      --push "$push" \
      ${dataset_name:+--dataset_name "$dataset_name"} \
      --seed "$seed" \
      --use_student "$use_student" \
      --skill_epochs "$skill_epochs" \
      --prior_epochs "$prior_epochs" \
      --prior_updates_per_batch "$prior_updates_per_batch" \
      --prior_use_mu "$prior_use_mu" \
      --val_freq "$val_freq" \
      --save_freq "$save_freq" \
      --action_noise_std "$action_noise_std" \
      --condition_reweight "$condition_reweight" \
      --behavior_policy_epochs "$behavior_policy_epochs" \
      --condition_weight_beta "$condition_weight_beta" \
      --condition_weight_min "$condition_weight_min" \
      --condition_weight_max "$condition_weight_max" \
      --condition_raw_log_weight_clip_quantile "$condition_raw_log_weight_clip_quantile" \
      --swanlab_project "$swanlab_project" \
      > "logs/Flow/skill_flow/${dataset_name}/seed${seed}/pick${pick}_push${push}_condition_flow_verify_beta${condition_weight_beta}_wmax${condition_weight_max}.log" 2>&1 &
  done

  wait

  for seed in "${seeds[@]}"; do
    python -u -m reskill.analysis.verify_condition_weights \
      --dataset_name "$dataset_name" \
      --seed "$seed" \
      --output_path "logs/Flow/skill_flow_verify/${dataset_name}_seed${seed}_beta${condition_weight_beta}_wmax${condition_weight_max}_weight_verify.json" \
      > "logs/Flow/skill_flow_verify/${dataset_name}_seed${seed}_beta${condition_weight_beta}_wmax${condition_weight_max}_weight_verify.log" 2>&1
  done
}

for exp in "${experiments[@]}"; do
  read -r pick push dataset_name gpu <<< "$exp"
  run_experiment "$pick" "$push" "$dataset_name" "$gpu" &
done

wait
