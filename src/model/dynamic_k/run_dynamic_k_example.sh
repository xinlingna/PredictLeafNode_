#!/bin/bash

# 动态K值聚类分布预测模型 - 完整运行示例
# 这个脚本展示了如何使用新的动态K值模型进行训练和测试

echo "🚀 Dynamic K Cluster Distribution Transformer - Complete Example"
echo "================================================================"

# 设置路径变量
BASE_DIR="/home/xln/PycharmProjects/PredictLeafNode"
DATA_DIR="$BASE_DIR/input/Training_data/gist1M_learn/leafsize20K"
TRAIN_NPZ="$DATA_DIR/train_gist.npz"
TEST_NPZ="$DATA_DIR/test_gist.npz"
CENTROIDS_PATH="$DATA_DIR/leaf_centroids.npy"

# 检查原始数据文件是否存在
echo "📁 Checking data files..."
if [ ! -f "$TRAIN_NPZ" ]; then
    echo "❌ Training data not found: $TRAIN_NPZ"
    echo "Please make sure you have the original training data."
    exit 1
fi

if [ ! -f "$CENTROIDS_PATH" ]; then
    echo "❌ Centroids file not found: $CENTROIDS_PATH"
    echo "Please make sure you have the centroids file."
    exit 1
fi

echo "✅ Original data files found"
echo "   Training data: $TRAIN_NPZ"
echo "   Test data: $TEST_NPZ"
echo "   Centroids: $CENTROIDS_PATH"

# 切换到项目目录
cd "$BASE_DIR"

# 激活conda环境（如果需要）
# conda activate elpis_torch

echo -e "\n🧪 Step 1: Running basic functionality tests..."
echo "----------------------------------------------"
python test_dynamic_k_model.py

if [ $? -ne 0 ]; then
    echo "❌ Basic tests failed. Please check the model implementation."
    exit 1
fi

echo -e "\n📊 Step 2: Preparing multi-K dataset..."
echo "--------------------------------------"

# 准备多K值数据集
python -m src.model.cluster_dist_transformer_dynamic_k \
  --prepare_data \
  --original_train_npz "$TRAIN_NPZ" \
  --original_test_npz "$TEST_NPZ" \
  --centroids_path "$CENTROIDS_PATH" \
  --k_values 1 5 10 100

if [ $? -ne 0 ]; then
    echo "❌ Data preparation failed."
    exit 1
fi

echo "✅ Multi-K dataset prepared successfully"

# 设置新的数据路径
TRAIN_MULTI_K="$DATA_DIR/train_gist_multi_k.npz"
TEST_MULTI_K="$DATA_DIR/test_gist_multi_k.npz"

echo -e "\n🎯 Step 3: Training dynamic K model (quick demo)..."
echo "------------------------------------------------"

# 快速训练示例（少量epoch用于演示）
python -m src.model.cluster_dist_transformer_dynamic_k \
  --train_npz "$TRAIN_MULTI_K" \
  --test_npz "$TEST_MULTI_K" \
  --centroids_path "$CENTROIDS_PATH" \
  --normalize \
  --val_split 0.05 \
  --topk 10 \
  --epochs 2 \
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
  --training_mode random \
  --loss_type kld \
  --experiment_id demo_quick

if [ $? -ne 0 ]; then
    echo "❌ Quick training failed."
    exit 1
fi

echo "✅ Quick demo training completed"

echo -e "\n🎯 Step 4: Full training example..."
echo "--------------------------------"
echo "For full training, you can use the following command:"
echo ""
echo "python -m src.model.cluster_dist_transformer_dynamic_k \\"
echo "  --train_npz \"$TRAIN_MULTI_K\" \\"
echo "  --test_npz \"$TEST_MULTI_K\" \\"
echo "  --centroids_path \"$CENTROIDS_PATH\" \\"
echo "  --normalize \\"
echo "  --val_split 0.01 \\"
echo "  --topk 10 \\"
echo "  --epochs 20 \\"
echo "  --batch_size 512 \\"
echo "  --lr 1e-3 \\"
echo "  --weight_decay 1e-2 \\"
echo "  --d_model 256 \\"
echo "  --nhead 8 \\"
echo "  --num_layers 4 \\"
echo "  --dim_ff 512 \\"
echo "  --dropout 0.1 \\"
echo "  --score_type bilinear \\"
echo "  --log_interval 10 \\"
echo "  --pos_exist \\"
echo "  --use_type_embed \\"
echo "  --use_gating \\"
echo "  --k_values 1 5 10 100 \\"
echo "  --k_embed_dim 32 \\"
echo "  --training_mode random \\"
echo "  --loss_type kld \\"
echo "  --experiment_id dynamic_k_full"

echo -e "\n📈 Step 5: Model comparison..."
echo "----------------------------"
echo "You can compare the dynamic K model with the original model:"
echo ""
echo "1. Original model (fixed K=100):"
echo "   python -m src.model.cluster_dist_transformer_original ..."
echo ""
echo "2. Dynamic K model (supports K=1,5,10,100):"
echo "   python -m src.model.cluster_dist_transformer_dynamic_k ..."
echo ""
echo "The dynamic K model should show:"
echo "- Better performance on specific K values"
echo "- More flexible inference capabilities"
echo "- Similar overall performance with added versatility"

echo -e "\n🎉 Dynamic K Model Example Completed!"
echo "===================================="
echo ""
echo "✅ What was accomplished:"
echo "   1. ✓ Basic functionality tests passed"
echo "   2. ✓ Multi-K dataset prepared"
echo "   3. ✓ Quick demo training completed"
echo "   4. ✓ Full training command provided"
echo ""
echo "📝 Next steps:"
echo "   1. Run full training with more epochs"
echo "   2. Compare with original baseline model"
echo "   3. Test inference with different K values"
echo "   4. Analyze per-K performance metrics"
echo ""
echo "📊 Check results in:"
echo "   $DATA_DIR/demo_quick_dynamic_k_*/  (quick demo results)"
echo ""
echo "🚀 Ready for production use!"
