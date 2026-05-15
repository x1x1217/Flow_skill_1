
set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Flow/reskill_flow

seeds=(2 3 20)
use_student=0
use_grad=0

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
    > "logs/Flow/reskill_flow/seed${seed}/table_cleanup_pick1_push999_seed${seed}_${use_student}_${use_grad}.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
    # --config_file slippery_push/config.yaml \
    # --prior_model Flow \
    # --pick 999 \
    # --push 1 \
    # --seed "$seed" \
    # --use_student "$use_student" \
    # --use_grad "$use_grad" \
    # > "logs/Flow/reskill_flow/seed${seed}/slippery_push_pick999_push1_seed${seed}_${use_student}_${use_grad}.log" 2>&1 &

done

wait

# python -u -m reskill.train_reskill_agent_res --config_file table_cleanup/config.yaml --prior_model Flow --pick 1 --push 999 --seed 2 --use_sigma 0 --use_grad 0
# tensorboard --logdir reskill/log/agent --host 0.0.0.0 --port 6006