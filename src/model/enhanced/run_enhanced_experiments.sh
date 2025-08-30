#!/bin/bash

# 设置基础路径
BASE_PATH="/home/xln/PycharmProjects/PredictLeafNode"
DATA_PATH="$BASE_PATH/input/Training_data/gist1M_learn/leafsize20K"

cd $BASE_PATH || exit 1

mkdir -p logs_enhanced_experiments

# ===== 设置实验标识符 =====
EXPERIMENT_ID="G"  # 您可以根据需要修改这个标识符
echo "开始执行实验批次: $EXPERIMENT_ID"
echo "=== Enhanced Cluster Distribution Transformer Experiments ==="


# 基线实验（无增强）
echo "训练 1/5: Baseline (no enhancement)..."
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz $DATA_PATH/train_gist.npz \
  --test_npz $DATA_PATH/test_gist.npz \
  --centroids_path $DATA_PATH/centroids.npy \
  --cluster_info_path $DATA_PATH/cluster_info.npz \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --enhancement_type none \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_enhanced_experiments/baseline.log 2>&1

# 方案1：统计信息增强
echo "训练 2/5: Stats enhancement..."
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz $DATA_PATH/train_gist.npz \
  --test_npz $DATA_PATH/test_gist.npz \
  --centroids_path $DATA_PATH/centroids.npy \
  --cluster_info_path $DATA_PATH/cluster_info.npz \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --enhancement_type stats \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_enhanced_experiments/stats.log 2>&1

# 方案2：多代表向量
echo "训练 3/5: Multi-representative enhancement..."
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz $DATA_PATH/train_gist.npz \
  --test_npz $DATA_PATH/test_gist.npz \
  --centroids_path $DATA_PATH/centroids.npy \
  --cluster_info_path $DATA_PATH/cluster_info.npz \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --enhancement_type multi_rep \
  --num_representatives 3 \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_enhanced_experiments/multi_rep.log 2>&1

# 方案3：距离特征增强
echo "训练 4/5: Distance features enhancement..."
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz $DATA_PATH/train_gist.npz \
  --test_npz $DATA_PATH/test_gist.npz \
  --centroids_path $DATA_PATH/centroids.npy \
  --cluster_info_path $DATA_PATH/cluster_info.npz \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --enhancement_type distance \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_enhanced_experiments/distance.log 2>&1

# 方案4：综合增强
echo "训练 5/5: Comprehensive enhancement..."
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz $DATA_PATH/train_gist.npz \
  --test_npz $DATA_PATH/test_gist.npz \
  --centroids_path $DATA_PATH/centroids.npy \
  --cluster_info_path $DATA_PATH/cluster_info.npz \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --enhancement_type comprehensive \
  --num_representatives 3 \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_enhanced_experiments/comprehensive.log 2>&1

echo "=== All experiments completed! ==="