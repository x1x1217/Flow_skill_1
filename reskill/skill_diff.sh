# python train_skill_modules.py --pick 1 --push 999 --prior_model 'Diffusion' --seed 2
# python train_skill_modules.py --pick 1 --push 999 --prior_model 'Diffusion' --seed 3
# python train_skill_modules.py --pick 1 --push 999 --prior_model 'Diffusion' --seed 20
# python train_skill_modules.py --pick 1 --push 999 --prior_model 'Diffusion' --seed 21
# python train_skill_modules.py --pick 1 --push 999 --prior_model 'Diffusion' --seed 22
# python train_skill_modules.py --pick 999 --push 1 --prior_model 'Diffusion' --seed 2
# python train_skill_modules.py --pick 999 --push 1 --prior_model 'Diffusion' --seed 3
# python train_skill_modules.py --pick 999 --push 1 --prior_model 'Diffusion' --seed 20
# python train_skill_modules.py --pick 999 --push 1 --prior_model 'Diffusion' --seed 21
# python train_skill_modules.py --pick 999 --push 1 --prior_model 'Diffusion' --seed 22

# python -m reskill.train_skill_modules --pick 1 --push 999 --prior_model 'Diffusion' --seed 2

set -e

cd ~/x1x/Flow_skill_1
mkdir -p logs/Diffusion/skill_diffusion

seeds=(3 20)

for seed in "${seeds[@]}"; do
  mkdir -p "logs/Diffusion/skill_diffusion/seed${seed}"

#   CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_skill_modules \
#     --prior_model Diffusion --pick 999 --push 1 --seed "$seed" \
#     > "logs/Diffusion/skill_diffusion/seed${seed}/pick999_push1.log" 2>&1 &

#   CUDA_VISIBLE_DEVICES=1 python -u -m reskill.train_skill_modules \
#     --prior_model Diffusion --pick 1 --push 999 --seed "$seed" \
#     > "logs/Diffusion/skill_diffusion/seed${seed}/pick1_push999.log" 2>&1 &

  CUDA_VISIBLE_DEVICES=0 python -u -m reskill.train_skill_modules \
    --prior_model Diffusion \
    --pick 1 \
    --push 999 \
    --seed "$seed" \
    > "logs/Diffusion/skill_diffusion/seed${seed}/pick1_push999.log" 2>&1 &
done

wait
