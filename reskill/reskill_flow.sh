
set -e

cd "$(dirname "$0")/.."
mkdir -p logs/reskill_flow

seeds=(2 3)

for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_reskill_agent_res \
    --config_file table_cleanup/config.yaml \
    --prior_model Flow \
    --pick 1 \
    --push 999 \
    --seed "$seed" \
    --use_sigma 0 \
    --use_grad 0 \
    > "logs/reskill_flow/table_cleanup_pick1_push999_seed${seed}.log" 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
    --config_file slippery_push/config.yaml \
    --prior_model Flow \
    --pick 999 \
    --push 1 \
    --seed "$seed" \
    --use_sigma 0 \
    --use_grad 0 \
    > "logs/reskill_flow/slippery_push_pick999_push1_seed${seed}.log" 2>&1 &

done

wait

# python -u -m reskill.train_reskill_agent_res --config_file table_cleanup/config.yaml --prior_model Flow --pick 1 --push 999 --seed 2 --use_sigma 0 --use_grad 0
# tensorboard --logdir reskill/log/agent --host 0.0.0.0 --port 6006