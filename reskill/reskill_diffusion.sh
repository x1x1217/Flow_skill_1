# cd ~/x1x/Flow_skill_1
# mkdir -p logs/

# CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_reskill_agent_res \
#   --config_file table_cleanup/config.yaml \
#   --prior_model Diffusion \
#   --pick 1 \
#   --push 999 \
#   --seed 2 \
#   --use_sigma 1 \
#   --use_grad 1 \
#   > "logs/Diffusion/reskill_diffusion/seed2/table_cleanup_pick1_push999_1_1.log" 2>&1 &

# CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
#   --config_file table_cleanup/config.yaml \
#   --prior_model Diffusion \
#   --pick 1 \
#   --push 999 \
#   --seed 2 \
#   --use_sigma 1 \
#   --use_grad 0 \
#   > "logs/Diffusion/reskill_diffusion/seed2/table_cleanup_pick1_push999_1_0.log" 2>&1 &

set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Diffusion/reskill_diffusion

seeds=(3 20)

for seed in "${seeds[@]}"; do
    mkdir -p "logs/Diffusion/reskill_diffusion/seed${seed}"

    CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_reskill_agent_res \
        --config_file table_cleanup/config.yaml \
        --prior_model Diffusion \
        --pick 1 \
        --push 999 \
        --seed "$seed" \
        --use_sigma 1 \
        --use_grad 1 \
        > "logs/Diffusion/reskill_diffusion/seed${seed}/table_cleanup_pick1_push999_1_1.log" 2>&1 &

    # CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_reskill_agent_res \
    #     --config_file table_cleanup/config.yaml \
    #     --prior_model Diffusion \
    #     --pick 1 \
    #     --push 999 \
    #     --seed "$seed" \
    #     --use_sigma 1 \
    #     --use_grad 0 \
    #     > "logs/Diffusion/reskill_diffusion/seed${seed}/table_cleanup_pick1_push999_1_0.log" 2>&1 &
done

wait