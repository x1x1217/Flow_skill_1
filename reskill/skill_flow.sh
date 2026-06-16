# cd ..
# python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2

# set -e

# cd ..
# CUDA_VISIBLE_DEVICES=0 python -m reskill.train_skill_modules --prior_model Flow --pick 999 --push 1 --seed 2 &
# CUDA_VISIBLE_DEVICES=1 python -m reskill.train_skill_modules --prior_model Flow --pick 1 --push 999 --seed 2 &

# wait

set -e

cd "$(dirname "$0")/.."
mkdir -p logs/Flow/skill_flow

pick=1
push=200

# seeds=(2 3 20)
seeds=(2 20)
use_student=0

for seed in "${seeds[@]}"; do
  mkdir -p "logs/Flow/skill_flow/seed${seed}"

#   CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_skill_modules \
#     --prior_model Flow  \
#     --pick 999 \
#     --push 1 \
#     --seed "$seed" \
#     --use_student "$use_student" \
#     > "logs/Flow/skill_flow/pick999_push1_seed${seed}.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_skill_modules \
    --prior_model Flow \
    --pick "$pick" \
    --push "$push" \
    --seed "$seed" \
    --use_student "$use_student" \
    > "logs/Flow/skill_flow/seed${seed}/pick${pick}_push${push}_condition_flow.log" 2>&1 &

done

wait
