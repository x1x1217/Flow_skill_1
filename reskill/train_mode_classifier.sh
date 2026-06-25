set -e

cd ~/x1x/Flow_skill_1

dataset_name=fetch_block_push1_pick500_labeled

CUDA_VISIBLE_DEVICES=3 python -u -m reskill.analysis.train_mode_classifier \
  --dataset_name "$dataset_name" \
  --subseq_len 10 \
  --train_split 0.99 \
  --max_chunks_per_mode 200000 \
  --epochs 20 \
  --batch_size 2048 \
  --hidden_dim 256 \
  --hidden_layers 2 \
  --seed 0
