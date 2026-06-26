set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Flow/reskill_flow/slippery_push/seed2

seed=2
pick=500
push=1
dataset_name=fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax300
classifier_dataset=fetch_block_push1_pick500_labeled

mode_classifier_path="reskill/results/mode_classifiers/${classifier_dataset}/mode_classifier.pth"
mode_stats_path="reskill/results/online_mode_stats/FetchSlipperyPush-v0/${dataset_name}/seed_${seed}/flow_reweight_qz001/eval_episode_mode_stats.jsonl"

use_student=0
use_grad=1
guidance_scale=0.01
guidance_warmup_epoch=0
guidance_grad_clip=1.0
guidance_normalize=1
max_residual_factor=1

use_condition_flow=1

chunk_critic_ensembles=1
chunk_critic_hidden_dim=2048
chunk_critic_hidden_layers=3
chunk_critic_activation=tanh
chunk_critic_lr=0.0003
chunk_critic_tau=0.005
chunk_critic_batch_size=256
chunk_critic_updates_per_epoch=200
chunk_critic_replay_size=1000000

environment_name=slippery_push
swanlab_project=Flow_skill_1_${environment_name}

CUDA_VISIBLE_DEVICES=3 python -u -m reskill.train_reskill_agent_res_mode_stats \
  --config_file $environment_name/config_rnvp.yaml \
  --prior_model Flow \
  --pick "$pick" \
  --push "$push" \
  --dataset_name "$dataset_name" \
  --seed "$seed" \
  --use_student "$use_student" \
  --use_grad "$use_grad" \
  --guidance_scale "$guidance_scale" \
  --guidance_warmup_epoch "$guidance_warmup_epoch" \
  --guidance_grad_clip "$guidance_grad_clip" \
  --guidance_normalize \
  --max_residual_factor "$max_residual_factor" \
  --use_condition_flow "$use_condition_flow" \
  --chunk_critic_ensembles "$chunk_critic_ensembles" \
  --chunk_critic_hidden_dim "$chunk_critic_hidden_dim" \
  --chunk_critic_hidden_layers "$chunk_critic_hidden_layers" \
  --chunk_critic_activation "$chunk_critic_activation" \
  --chunk_critic_lr "$chunk_critic_lr" \
  --chunk_critic_tau "$chunk_critic_tau" \
  --chunk_critic_batch_size "$chunk_critic_batch_size" \
  --chunk_critic_updates_per_epoch "$chunk_critic_updates_per_epoch" \
  --chunk_critic_replay_size "$chunk_critic_replay_size" \
  --mode_classifier_path "$mode_classifier_path" \
  --mode_stats_path "$mode_stats_path" \
  --mode_stats_return_threshold 30 \
  --swanlab_project "$swanlab_project" \
  > "logs/Flow/reskill_flow/${environment_name}/seed${seed}/condflow${use_condition_flow}_${dataset_name}_qz001_mode_stats.log" 2>&1
