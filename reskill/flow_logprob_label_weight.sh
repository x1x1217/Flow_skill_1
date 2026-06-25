set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow_verify

experiments=(
  "fetch_block_push300_pick1_labeled_label_weight 2"
  "fetch_block_push1_pick500_labeled_label_weight 3"
)

seeds=(2 3 20)
batch_size=256

run_experiment() {
  local dataset_name="$1"
  local gpu="$2"

  for seed in "${seeds[@]}"; do
    CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.analysis.verify_flow_logprob \
      --dataset_name "$dataset_name" \
      --seed "$seed" \
      --batch_size "$batch_size" \
      --output_path "logs/Flow/skill_flow_verify/${dataset_name}_seed${seed}_label_weight_flow_logprob.json" \
      > "logs/Flow/skill_flow_verify/${dataset_name}_seed${seed}_label_weight_flow_logprob.log" 2>&1
  done
}

for exp in "${experiments[@]}"; do
  read -r dataset_name gpu <<< "$exp"
  run_experiment "$dataset_name" "$gpu" &
done

wait
