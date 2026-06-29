#!/usr/bin/env bash

gpu=0
seeds=(2)

pick=500
push=1
dataset_name=fetch_block_push1_pick500_labeled_flowbeh_beta02_wmin01_wmax300

use_grad=1
guidance_scale=1
guidance_warmup_epoch=1
guidance_grad_clip=10.0

max_residual_factor=1
use_condition_flow=1

condition_use_grad=1
# condition_guidance_scale=1
condition_guidance_scale=$guidance_scale
condition_guidance_warmup_epoch=1
condition_guidance_grad_clip=10.0

environment_name=slippery_push
swanlab_project=Flow_skill_1_${environment_name}
log_subdir=on-policy
run_name=condflow${use_condition_flow}_condgrad${condition_use_grad}_${dataset_name}_qz${guidance_scale}_qc${condition_guidance_scale}_cgwarm${condition_guidance_warmup_epoch}_on-policy_gradClip10.0
