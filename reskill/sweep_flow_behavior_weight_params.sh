set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/flow_behavior_verify9

experiments=(
  "fetch_block_push300_pick1_labeled 2"
  "fetch_block_push1_pick500_labeled 3"
)

seeds=(2 3 20)
use_student=0

skip_train=1
skip_logprob_if_cached=1

epochs=50
train_batch_size=1024
logprob_batch_size=256
flow_use_student=0
flow_steps=10
hidden_dim=256
time_dim=16
lr=3e-4
distill_coef=1.0
grad_clip=0.0

subseq_len=10
train_split=0.99
target_pick_fraction=0.2
raw_log_weight_clip_quantile=1.0

betas="0.2"
w_mins="0.1"
w_maxs="200,300,500,1000"

run_experiment() {
  local dataset_name="$1"
  local gpu="$2"

  for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.analysis.sweep_flow_behavior_weight_params \
      --dataset_name "$dataset_name" \
      --seed "$seed" \
      --use_student "$use_student" \
      --skip_train \
      --skip_logprob_if_cached \
      --epochs "$epochs" \
      --train_batch_size "$train_batch_size" \
      --logprob_batch_size "$logprob_batch_size" \
      --flow_use_student "$flow_use_student" \
      --flow_steps "$flow_steps" \
      --hidden_dim "$hidden_dim" \
      --time_dim "$time_dim" \
      --lr "$lr" \
      --distill_coef "$distill_coef" \
      --grad_clip "$grad_clip" \
      --subseq_len "$subseq_len" \
      --train_split "$train_split" \
      --target_pick_fraction "$target_pick_fraction" \
      --raw_log_weight_clip_quantile "$raw_log_weight_clip_quantile" \
      --betas "$betas" \
      --w_mins "$w_mins" \
      --w_maxs "$w_maxs" \
      --save_flow_log_probs \
      --output_path "logs/Flow/flow_behavior_verify9/${dataset_name}_seed${seed}_flow_behavior_sweep.json" \
      > "logs/Flow/flow_behavior_verify9/${dataset_name}_seed${seed}_flow_behavior_sweep.log" 2>&1
  done
}

for exp in "${experiments[@]}"; do
  read -r dataset_name gpu <<< "$exp"
  run_experiment "$dataset_name" "$gpu" &
done

wait
