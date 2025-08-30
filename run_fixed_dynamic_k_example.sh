#!/bin/bash

# 修复后的动态K值模型训练示例
# 展示正确的训练策略使用方法

echo "🚀 Fixed Dynamic K Model Training Example"
echo "========================================"

# 设置路径变量
BASE_DIR="/home/xln/PycharmProjects/PredictLeafNode"
DATA_DIR="$BASE_DIR/input/Training_data/gist1M_learn/leafsize20K"
TRAIN_NPZ="$DATA_DIR/train_gist.npz"
TEST_NPZ="$DATA_DIR/test_gist.npz"
CENTROIDS_PATH="$DATA_DIR/leaf_centroids.npy"

# 切换到项目目录
cd "$BASE_DIR"

echo "📊 Step 1: Preparing multi-K dataset with CORRECT strategy..."
echo "--------------------------------------------------------"

# 准备多K值数据集
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --prepare_data \
  --original_train_npz "$TRAIN_NPZ" \
  --original_test_npz "$TEST_NPZ" \
  --centroids_path "$CENTROIDS_PATH" \
  --k_values 1 5 10 100

echo ""
echo "🎯 Step 2: Training with FIXED configuration..."
echo "--------------------------------------------"
echo "Key fixes applied:"
echo "  ✅ Using --k_values (not --i)"
echo "  ✅ Using --training_mode all (not random)" 
echo "  ✅ Each query gets all K values as training samples"
echo "  ✅ Training monitoring and validation enabled"
echo ""

# 设置新的数据路径
TRAIN_MULTI_K="$DATA_DIR/train_gist_multi_k.npz"
TEST_MULTI_K="$DATA_DIR/test_gist_multi_k.npz"

# 训练修复后的动态K值模型
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_npz "$TRAIN_MULTI_K" \
  --test_npz "$TEST_MULTI_K" \
  --centroids_path "$CENTROIDS_PATH" \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 5 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 4 \
  --num_layers 2 \
  --dim_ff 256 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 50 \
  --pos_exist \
  --use_type_embed \
  --k_values 1 5 10 100 \
  --k_embed_dim 16 \
  --training_mode all \
  --loss_type kld \
  --experiment_id fixed_dynamic_k

echo ""
echo "📈 What you should see in the output:"
echo "-----------------------------------"
echo "1. Training Configuration:"
echo "   📊 Training Configuration:"
echo "     Original queries: XXXX"
echo "     K values: [1, 5, 10, 100]"
echo "     Training mode: all"
echo "     Dataset size: XXXX"
echo "     Expected size (N×K): XXXX × 4 = XXXX"
echo "     ✅ Each query has 4 training samples (one per K value)"
echo ""
echo "2. K-value Distribution (should be uniform):"
echo "   [Epoch 1] K-value distribution: K=1:XXX(25.0%) | K=5:XXX(25.0%) | K=10:XXX(25.0%) | K=100:XXX(25.0%)"
echo ""
echo "3. Per-K Evaluation Results:"
echo "   [TEST K=1] kld=X.XXXXXX | mae=X.XXXXXX | recall@10=X.XXXX"
echo "   [TEST K=5] kld=X.XXXXXX | mae=X.XXXXXX | recall@10=X.XXXX"
echo "   [TEST K=10] kld=X.XXXXXX | mae=X.XXXXXX | recall@10=X.XXXX"
echo "   [TEST K=100] kld=X.XXXXXX | mae=X.XXXXXX | recall@10=X.XXXX"
echo ""
echo "💡 Key Success Indicators:"
echo "  ✅ Dataset size = Original_queries × Number_of_K_values"
echo "  ✅ K-value distribution shows equal percentages (25% each for 4 K-values)"
echo "  ✅ Each K value gets separate evaluation results"
echo "  ✅ Different K values show different performance metrics"
echo ""
echo "🎉 Training completed with CORRECT dynamic K strategy!"
echo "The model can now predict distributions for K=1,5,10,100 from a single query!"

