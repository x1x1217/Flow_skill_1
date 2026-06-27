#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
repo_root=$(cd -- "$script_dir/.." && pwd)

# Select the configurations to run. All selected configurations run in parallel.
configs=(
    "$script_dir/configs/reskill_flow/rawgrad_qc1_qz1_off-policy.sh"
    "$script_dir/configs/reskill_flow/rawgrad_qc2_qz2_off-policy.sh"
    "$script_dir/configs/reskill_flow/rawgrad_qc5_qz5_off-policy.sh"
    "$script_dir/configs/reskill_flow/rawgrad_qc10_qz10_off-policy.sh"
    "$script_dir/configs/reskill_flow/rawgrad_qc01_qz01_off-policy.sh"
    "$script_dir/configs/reskill_flow/rawgrad_qc05_qz05_off-policy.sh"
)

# Explicit command-line paths temporarily override the selection above.
if (( $# > 0 )); then
    configs=("$@")
fi

launch_config() (
    set -euo pipefail

    config_file=$1
    if [[ ! -f "$config_file" ]]; then
        echo "Config file not found: $config_file" >&2
        exit 2
    fi

    # Keep optional arguments local to this configuration.
    extra_args=()
    source "$config_file"

    : "${gpu:?gpu is required in $config_file}"
    : "${run_name:?run_name is required in $config_file}"
    : "${dataset_name:?dataset_name is required in $config_file}"
    : "${environment_name:?environment_name is required in $config_file}"

    cd "$repo_root"
    local_pids=()

    for seed in "${seeds[@]}"; do
        log_dir="logs/Flow/reskill_flow/${environment_name}/seed${seed}/${log_subdir:-newPara}"
        log_file="${log_dir}/${run_name}.log"
        mkdir -p "$log_dir"

        cmd=(
            python -u -m reskill.train_reskill_agent_res
            --config_file "$environment_name/config_rnvp.yaml"
            --prior_model Flow
            --pick "$pick"
            --push "$push"
            --dataset_name "$dataset_name"
            --seed "$seed"
            --use_grad "$use_grad"
            --guidance_scale "$guidance_scale"
            --guidance_warmup_epoch "$guidance_warmup_epoch"
            --guidance_grad_clip "$guidance_grad_clip"
            --max_residual_factor "$max_residual_factor"
            --use_condition_flow "$use_condition_flow"
            --condition_use_grad "$condition_use_grad"
            --condition_guidance_scale "$condition_guidance_scale"
            --condition_guidance_warmup_epoch "$condition_guidance_warmup_epoch"
            --condition_guidance_grad_clip "$condition_guidance_grad_clip"
            --chunk_critic_ensembles "$chunk_critic_ensembles"
            --chunk_critic_hidden_dim "$chunk_critic_hidden_dim"
            --chunk_critic_hidden_layers "$chunk_critic_hidden_layers"
            --chunk_critic_activation "$chunk_critic_activation"
            --chunk_critic_lr "$chunk_critic_lr"
            --chunk_critic_tau "$chunk_critic_tau"
            --chunk_critic_batch_size "$chunk_critic_batch_size"
            --chunk_critic_updates_per_epoch "$chunk_critic_updates_per_epoch"
            --chunk_critic_replay_size "$chunk_critic_replay_size"
            --condition_critic_ensembles "$condition_critic_ensembles"
            --condition_critic_hidden_dim "$condition_critic_hidden_dim"
            --condition_critic_hidden_layers "$condition_critic_hidden_layers"
            --condition_critic_activation "$condition_critic_activation"
            --condition_critic_lr "$condition_critic_lr"
            --condition_critic_tau "$condition_critic_tau"
            --condition_critic_batch_size "$condition_critic_batch_size"
            --condition_critic_updates_per_epoch "$condition_critic_updates_per_epoch"
            --condition_critic_replay_size "$condition_critic_replay_size"
            --swanlab_project "$swanlab_project"
            "${extra_args[@]}"
        )

        echo "[launch] config=$config_file seed=$seed gpu=$gpu log=$log_file"
        CUDA_VISIBLE_DEVICES="$gpu" "${cmd[@]}" > "$log_file" 2>&1 &
        local_pids+=("$!")
    done

    for pid in "${local_pids[@]}"; do
        wait "$pid"
    done
)

config_pids=()
for config_file in "${configs[@]}"; do
    launch_config "$config_file" &
    config_pids+=("$!")
done

status=0
for pid in "${config_pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
exit "$status"
