set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow_verify3

experiments=(
  "fetch_block_push300_pick1_labeled 2"
  "fetch_block_push1_pick500_labeled 3"
)

seeds=(2 3 20)
use_student=0
batch_size=8192
subseq_len=10
train_split=0.99
target_pick_fraction=0.5
raw_log_weight_clip_quantile=1.0

betas="2"
# w_mins="0.02,0.05,0.1,0.2,0.5,1.0"
# w_maxs="20,50,100,200,500,1000,2000"
w_mins="0.1"
w_maxs="1000"

run_experiment() {
  local dataset_name="$1"
  local gpu="$2"

  for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.analysis.sweep_condition_weight_params \
      --dataset_name "$dataset_name" \
      --seed "$seed" \
      --use_student "$use_student" \
      --batch_size "$batch_size" \
      --subseq_len "$subseq_len" \
      --train_split "$train_split" \
      --target_pick_fraction "$target_pick_fraction" \
      --raw_log_weight_clip_quantile "$raw_log_weight_clip_quantile" \
      --betas "$betas" \
      --w_mins "$w_mins" \
      --w_maxs "$w_maxs" \
      --save_raw_log_probs \
      --output_path "logs/Flow/skill_flow_verify3/${dataset_name}_seed${seed}_condition_weight_sweep.json" \
      > "logs/Flow/skill_flow_verify3/${dataset_name}_seed${seed}_condition_weight_sweep.log" 2>&1
  done
}

for exp in "${experiments[@]}"; do
  read -r dataset_name gpu <<< "$exp"
  run_experiment "$dataset_name" "$gpu" &
done

wait
