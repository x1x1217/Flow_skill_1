#python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1 --push 1 --seed 1 --use_sigma 0 --use_grad 0 
#python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1 --push 10 --seed 1 --use_sigma 0 --use_grad 0 
#python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1 --push 100 --seed 1 --use_sigma 0 --use_grad 0 
python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1 --push 1000 --seed 1 --use_sigma 0 --use_grad 0 
python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 10 --push 1 --seed 1 --use_sigma 0 --use_grad 0 
python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 100 --push 1 --seed 1 --use_sigma 0 --use_grad 0 
python train_reskill_agent_res.py --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1000 --push 1 --seed 1 --use_sigma 0 --use_grad 0 

python -m reskill.train_reskill_agent_res --config_file table_cleanup/config.yaml --prior_model Diffusion --pick 1 --push 1000 --seed 1 --use_sigma 0 --use_grad 0