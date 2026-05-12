# cd ..
# python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2

# set -e

# cd ..
# CUDA_VISIBLE_DEVICES=0 python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2 &
# CUDA_VISIBLE_DEVICES=1 python -m reskill.train_skill_modules --prior_model Flow --pick 1 --push 999 --seed 2 &

# wait

set -e

cd "$(dirname "$0")/.."
mkdir -p logs/skill_flow

# seeds=(2 3 20)
seeds=(2 3)

for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_skill_modules \
    --prior_model Flow --pick 999 --push 1 --seed "$seed" \
    > "logs/skill_flow/pick999_push1_seed${seed}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_skill_modules \
    --prior_model Flow --pick 1 --push 999 --seed "$seed" \
    > "logs/skill_flow/pick1_push999_seed${seed}.log" 2>&1 &
done

wait
