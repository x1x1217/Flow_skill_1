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

pick=1
push=300
# dataset_name=fetch_block_40000
dataset_name=fetch_block_push${push}_pick${pick}

# seeds=(2 3 20)
seeds=(2 3 20)
use_student=0
swanlab_project=Flow_skill_1_offline

skill_epochs=400
prior_epochs=400
behavior_policy_epochs=50

prior_updates_per_batch=1
prior_use_mu=1
val_freq=5
save_freq=50
action_noise_std=0.0
condition_reweight=1
condition_weight_beta=0.2
condition_weight_min=0.2
condition_weight_max=20.0
condition_raw_log_weight_clip_quantile=1.0

for seed in "${seeds[@]}"; do
  mkdir -p "logs/Flow/skill_flow/seed${seed}"

#   CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_skill_modules \
#     --prior_model Flow  \
#     --pick 999 \
#     --push 1 \
#     --seed "$seed" \
#     --use_student "$use_student" \
#     > "logs/Flow/skill_flow/pick999_push1_seed${seed}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=3 python -u -m reskill.train_skill_modules \
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
    > "logs/Flow/skill_flow/seed${seed}/pick${pick}_push${push}_condition_flow_newClip.log" 2>&1 &

done

wait
