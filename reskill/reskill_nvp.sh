set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/RNVP/reskill_nvp

# seeds=(2 3 20)
seeds=(2 20)
pick=500
push=1

environment_name=slippery_push
swanlab_project=Flow_skill_1_${environment_name}

for seed in "${seeds[@]}"; do
    mkdir -p "logs/RNVP/reskill_nvp/${environment_name}/seed${seed}/"

    CUDA_VISIBLE_DEVICES=2 python -u -m reskill.train_reskill_agent_res \
    --config_file $environment_name/config_rnvp.yaml \
    --prior_model RNVP \
    --pick $pick \
    --push $push \
    --seed "$seed" \
    --use_sigma 0 \
    --use_grad 0 \
    --swanlab_project "$swanlab_project" \
    > "logs/RNVP/reskill_nvp/${environment_name}/seed${seed}/pick${pick}_push${push}_rnvp.log" 2>&1 &
done

wait
