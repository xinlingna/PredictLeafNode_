# !/bin/bash
# conda activate elpis_torch

# 切换到项目目录
cd /home/xln/PycharmProjects/PredictLeafNode/ || exit 1

# 创建日志目录
mkdir -p logs_mlp_bt256

# ========== 第一次训练 (loss_type=kld_reverse) ==========
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld > logs_mlp_bt256/kld.log 2>&1

# ========== 第二次训练 (loss_type=kld) ==========
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type kld_reverse > logs_mlp_bt256/kld_reverse.log 2>&1

# ========== 第三次训练 (loss_type=recall_focused_loss) ==========
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type recall_focused_loss > logs_mlp_bt256/recall_focused_loss.log 2>&1

# ========== 第四次训练 (loss_type=mse) ==========
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type mse > logs_mlp_bt256/mse.log 2>&1

# ========== 第五次训练 (loss_type=hybrid) ==========
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type mlp \
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type hybrid > logs_mlp_bt256/hybrid.log 2>&1
