#!/bin/bash
# conda activate elpis_torch

# 切换到项目目录
cd /home/xln/PycharmProjects/PredictLeafNode/ || exit 1

# 创建日志目录
mkdir -p logs_bilinear_bt512_pe

# ===== 设置实验标识符 =====
EXPERIMENT_ID="D"  # 您可以根据需要修改这个标识符

echo "开始执行实验批次: $EXPERIMENT_ID"
echo "======================="

# ========== 第一次训练 (loss_type=recall_focused_loss) ==========
echo "训练 1/8: recall_focused_loss"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --loss_type recall_focused_loss > logs_bilinear_bt512_pe/recall_focused_loss.log 2>&1

# ========== 第二次训练 (loss_type=kld) ==========
echo "训练 2/8: kld"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --loss_type kld > logs_bilinear_bt512_pe/kld.log 2>&1

# ========== 第三次训练 (loss_type=kld_reverse) ==========
echo "训练 3/8: kld_reverse"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --loss_type kld_reverse > logs_bilinear_bt512_pe/kld_reverse.log 2>&1

# ========== 第四次训练 (loss_type=mse) ==========
echo "训练 4/8: mse"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --loss_type mse > logs_bilinear_bt512_pe/mse.log 2>&1

# ========== 第五次训练 (loss_type=hybrid) ==========
echo "训练 5/8: hybrid"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --loss_type hybrid > logs_bilinear_bt512_pe/hybrid.log 2>&1

# ========== 第六次训练 (loss_type=listnet) ==========
echo "训练 6/8: listnet"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --listnet_pred_temp 1.5 \
  --listnet_tgt_temp 1.0 \
  --loss_type listnet > logs_bilinear_bt512_pe/listnet.log 2>&1

# ========== 第七次训练 (loss_type=pairwise_hinge) ==========
echo "训练 7/8: pairwise_hinge"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --pair_num_pos 2 \
  --pair_num_neg 15 \
  --pair_margin 0.2 \
  --loss_type pairwise_hinge > logs_bilinear_bt512_pe/pairwise_hinge.log 2>&1

# ========== 第八次训练 (loss_type=listmle) ==========
echo "训练 8/8: listmle"
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --experiment_id $EXPERIMENT_ID \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
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
  --use_type_embed \
  --listmle_topm 20 \
  --loss_type listmle > logs_bilinear_bt512_pe/listmle.log 2>&1

echo "======================="
echo "实验批次 $EXPERIMENT_ID 完成!"
echo "所有模型目录都将以 ${EXPERIMENT_ID}_ 开头"
