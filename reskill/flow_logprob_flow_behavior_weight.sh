set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow_verify_flowbeh

experiments=(
  "fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax200 1"
  "fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax300 2"
  "fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax1000 3"
)

seed=2
batch_size=256

run_experiment() {
  local dataset_name="$1"
  local gpu="$2"

  CUDA_VISIBLE_DEVICES="$gpu" python -u -m reskill.analysis.verify_flow_logprob \
    --dataset_name "$dataset_name" \
    --seed "$seed" \
    --batch_size "$batch_size" \
    --output_path "logs/Flow/skill_flow_verify_flowbeh/${dataset_name}_seed${seed}_flow_logprob.json" \
    > "logs/Flow/skill_flow_verify_flowbeh/${dataset_name}_seed${seed}_flow_logprob.log" 2>&1
}

for exp in "${experiments[@]}"; do
  read -r dataset_name gpu <<< "$exp"
  run_experiment "$dataset_name" "$gpu" &
done

wait
