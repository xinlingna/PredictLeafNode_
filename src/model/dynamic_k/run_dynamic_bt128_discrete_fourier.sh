!/bin/bash
# conda activate elpis_torch

# 切换到项目目录
cd /home/xln/PycharmProjects/PredictLeafNode/ || exit 1

# 创建日志目录
mkdir -p dynamic_logs_I

# ===== 设置实验标识符 =====
EXPERIMENT_ID="I"  # 您可以根据需要修改这个标识符

echo "开始执行实验批次: $EXPERIMENT_ID"
echo "======================="


# # ========== 第一次训练 (loss_type=recall_focused_loss) ==========
echo "训练 1/5: add"
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type add \
  --training_mode all \
  --loss_type kld > dynamic_logs_I/add.log 2>&1

python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id I \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type add \
  --training_mode all \
  --loss_type kld

# # ========== 第二次训练 (loss_type=recall_focused_loss) ==========
echo "训练 2/5: concat"
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type concat \
  --training_mode all \
  --loss_type kld > dynamic_logs_I/concat.log 2>&1

# # ========== 第三次训练 (loss_type=recall_focused_loss) ==========
echo "训练 3/5: cross_attention"
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type cross_attention \
  --training_mode all \
  --loss_type kld > dynamic_logs_I/cross_attention.log 2>&1

# # ========== 第四次训练 (loss_type=recall_focused_loss) ==========
echo "训练 4/5: gated"
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type gated \
  --training_mode all \
  --loss_type kld > dynamic_logs_I/gated.log 2>&1

# # ========== 第五次训练 (loss_type=recall_focused_loss) ==========
echo "训练 5/5: adaptive"
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --experiment_id $EXPERIMENT_ID \
  --topk 10 \
  --epochs 10 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --nhead 8 \
  --num_layers 6 \
  --d_model 256 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 30 \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type fourier \
  --k_fusion_type adaptive \
  --training_mode all \
  --loss_type kld > dynamic_logs_I/adaptive.log 2>&1


# echo "======================="
# echo "实验批次 $EXPERIMENT_ID 完成!"
# echo "所有模型目录都将以 ${EXPERIMENT_ID}_ 开头"
