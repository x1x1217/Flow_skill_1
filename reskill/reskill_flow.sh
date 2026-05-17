
set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Flow/reskill_flow

seeds=(2 3 20)
use_student=0
use_grad=1
guidance_scale=0.03
guidance_warmup_epoch=50
guidance_grad_clip=1.0

for seed in "${seeds[@]}"; do
    mkdir -p "logs/Flow/reskill_flow/seed${seed}"

    CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
    --config_file table_cleanup/config.yaml \
    --prior_model Flow \
    --pick 1 \
    --push 999 \
    --seed "$seed" \
    --use_student "$use_student" \
    --use_grad "$use_grad" \
    --guidance_scale "$guidance_scale" \
    --guidance_warmup_epoch "$guidance_warmup_epoch" \
    --guidance_grad_clip "$guidance_grad_clip" \
    > "logs/Flow/reskill_flow/seed${seed}/table_cleanup_pick1_push999_seed${seed}_student${use_student}_grad${use_grad}_gscale${guidance_scale}_gwarm${guidance_warmup_epoch}_gclip${guidance_grad_clip}.log" 2>&1 &

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
