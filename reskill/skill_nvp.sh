set -e

cd "$(dirname "$0")/.."
mkdir -p logs/RNVP/skill_nvp

seeds=(2 3 20)
pick=1
push=500

prior_model=RNVP
swanlab_project=Flow_skill_1_offline

for seed in "${seeds[@]}"; do
    mkdir -p "logs/RNVP/skill_nvp/pick${pick}_push${push}/seed${seed}"

    CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_skill_modules \
    --prior_model "$prior_model" \
    --pick "$pick" \
    --push "$push" \
    --seed "$seed" \
    --swanlab_project "$swanlab_project" \
    > "logs/RNVP/skill_nvp/pick${pick}_push${push}/seed${seed}/rnvp.log" 2>&1 &
done

wait
