#!/bin/bash

# 改进的训练脚本 - 专门针对recall@22优化

echo "开始改进的训练 - 使用recall-focused损失函数"

# 基础训练（仅使用排序损失）
echo "=== 实验1: 基础排序损失训练 ==="
python -m src.model.cluster_dist_transformer_loss \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --rank_topk 22 \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --rank_lambda 1.0 \
  --log_interval 200

echo "=== 实验2: 使用recall-focused损失训练 ==="
python -m src.model.cluster_dist_transformer_loss \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --rank_topk 22 \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --rank_lambda 1.0 \
  --use_recall_loss \
  --recall_weight 3.0 \
  --log_interval 200

echo "=== 实验3: 高权重recall-focused训练 ==="
python -m src.model.cluster_dist_transformer_loss \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --rank_topk 22 \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --rank_lambda 1.0 \
  --use_recall_loss \
  --recall_weight 5.0 \
  --log_interval 200

echo "训练完成！请检查生成的图表文件。"
