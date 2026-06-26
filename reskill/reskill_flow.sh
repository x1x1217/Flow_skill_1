
set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Flow/reskill_flow

seeds=(2)
pick=500
push=1
# dataset_name=fetch_block_40000
dataset_name=fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax300
# dataset_name=fetch_block_push1_pick500_labeled_label_weight

use_student=0
use_grad=1
guidance_scale=0.01
guidance_warmup_epoch=0
guidance_grad_clip=1.0
guidance_normalize=1
init_rollout_steps=1500
positive_replay_ratio=0.5
positive_reward_threshold=0.0

max_residual_factor=1

use_condition_flow=1

condition_use_grad=1
condition_guidance_scale=0.1
condition_guidance_warmup_epoch=0
condition_guidance_grad_clip=1.0
condition_guidance_normalize=1
condition_positive_replay_ratio=0.5
condition_positive_reward_threshold=0.0

chunk_critic_ensembles=1
chunk_critic_hidden_dim=2048
chunk_critic_hidden_layers=3
chunk_critic_activation=tanh
chunk_critic_layer_norm=1
chunk_critic_lr=0.0003
chunk_critic_tau=0.005
chunk_critic_batch_size=256
chunk_critic_updates_per_epoch=200
chunk_critic_replay_size=1000000

condition_critic_ensembles=$chunk_critic_ensembles
condition_critic_hidden_dim=$chunk_critic_hidden_dim
condition_critic_hidden_layers=$chunk_critic_hidden_layers
condition_critic_activation=$chunk_critic_activation
condition_critic_layer_norm=$chunk_critic_layer_norm
condition_critic_lr=$chunk_critic_lr
condition_critic_tau=$chunk_critic_tau
condition_critic_batch_size=$chunk_critic_batch_size
condition_critic_updates_per_epoch=$chunk_critic_updates_per_epoch
condition_critic_replay_size=$chunk_critic_replay_size

environment_name=slippery_push
swanlab_project=Flow_skill_1_${environment_name}

for seed in "${seeds[@]}"; do
    mkdir -p "logs/Flow/reskill_flow/${environment_name}/seed${seed}/"

    CUDA_VISIBLE_DEVICES=2 python -u -m reskill.train_reskill_agent_res \
    --config_file $environment_name/config_rnvp.yaml \
    --prior_model Flow \
    --pick $pick \
    --push $push \
    --dataset_name "$dataset_name" \
    --seed "$seed" \
    --use_student "$use_student" \
    --use_grad "$use_grad" \
    --guidance_scale "$guidance_scale" \
    --guidance_warmup_epoch "$guidance_warmup_epoch" \
    --guidance_grad_clip "$guidance_grad_clip" \
    --guidance_normalize \
    --init_rollout_steps "$init_rollout_steps" \
    --positive_replay_ratio "$positive_replay_ratio" \
    --positive_reward_threshold "$positive_reward_threshold" \
    --max_residual_factor "$max_residual_factor" \
    --use_condition_flow "$use_condition_flow" \
    --condition_use_grad "$condition_use_grad" \
    --condition_guidance_scale "$condition_guidance_scale" \
    --condition_guidance_warmup_epoch "$condition_guidance_warmup_epoch" \
    --condition_guidance_grad_clip "$condition_guidance_grad_clip" \
    --condition_guidance_normalize \
    --condition_positive_replay_ratio "$condition_positive_replay_ratio" \
    --condition_positive_reward_threshold "$condition_positive_reward_threshold" \
    --chunk_critic_ensembles "$chunk_critic_ensembles" \
    --chunk_critic_hidden_dim "$chunk_critic_hidden_dim" \
    --chunk_critic_hidden_layers "$chunk_critic_hidden_layers" \
    --chunk_critic_activation "$chunk_critic_activation" \
    --chunk_critic_layer_norm \
    --chunk_critic_lr "$chunk_critic_lr" \
    --chunk_critic_tau "$chunk_critic_tau" \
    --chunk_critic_batch_size "$chunk_critic_batch_size" \
    --chunk_critic_updates_per_epoch "$chunk_critic_updates_per_epoch" \
    --chunk_critic_replay_size "$chunk_critic_replay_size" \
    --condition_critic_ensembles "$condition_critic_ensembles" \
    --condition_critic_hidden_dim "$condition_critic_hidden_dim" \
    --condition_critic_hidden_layers "$condition_critic_hidden_layers" \
    --condition_critic_activation "$condition_critic_activation" \
    --condition_critic_layer_norm \
    --condition_critic_lr "$condition_critic_lr" \
    --condition_critic_tau "$condition_critic_tau" \
    --condition_critic_batch_size "$condition_critic_batch_size" \
    --condition_critic_updates_per_epoch "$condition_critic_updates_per_epoch" \
    --condition_critic_replay_size "$condition_critic_replay_size" \
    --swanlab_project "$swanlab_project" \
    > "logs/Flow/reskill_flow/${environment_name}/seed${seed}/condflow${use_condition_flow}_${dataset_name}_resmax${max_residual_factor}_qz${guidance_scale}_qc${condition_guidance_scale}_cgwarm${condition_guidance_warmup_epoch}.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
    # --config_file slippery_push/config.yaml \
    # --prior_model Flow \
    # --pick 999 \
    # --push 1 \
    # --seed "$seed" \
    # --use_student "$use_student" \
    # --use_grad "$use_grad" \
    # --guidance_scale "$guidance_scale" \
    # --guidance_warmup_epoch "$guidance_warmup_epoch" \
    # --guidance_grad_clip "$guidance_grad_clip" \
    # > "logs/Flow/reskill_flow/seed${seed}/slippery_push_pick999_push1_seed${seed}_student${use_student}_grad${use_grad}_gscale${guidance_scale}_gwarm${guidance_warmup_epoch}_gclip${guidance_grad_clip}.log" 2>&1 &

done

wait

# python -u -m reskill.train_reskill_agent_res --config_file table_cleanup/config.yaml --prior_model Flow --pick 1 --push 999 --seed 2 --use_sigma 0 --use_grad 0
# tensorboard --logdir reskill/log/agent --host 0.0.0.0 --port 6006
