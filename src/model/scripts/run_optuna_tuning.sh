#!/bin/bash

# ================= 基础配置 =================
# 激活环境
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null
conda activate elpis_torch

# 定义要运行的 Leaf Size 标识 (对应文件夹名称后缀)
# 根据您的需求：10w, 20w, 30w, 40w, 50w
LEAF_TAGS=("10W" "20W" "30W" "40W" "50W")

# 基础路径
PROJECT_ROOT="/home/xln/PredictLeafNode"
DATA_ROOT_BASE="${PROJECT_ROOT}/input/Training_data/sift10M"
CURRENT_DATE=$(date +%Y%m%d)

echo "========================================================"
echo "🚀 开始执行 Optuna 参数调优 (500 Trials)"
echo "待处理列表: ${LEAF_TAGS[*]}"
echo "========================================================"

for TAG in "${LEAF_TAGS[@]}"; do
    # 动态构建数据文件夹路径
    DATA_DIR="${DATA_ROOT_BASE}/leafsize${TAG}"
    
    # 定义输入文件路径
    TRAIN_NPZ="${DATA_DIR}/train_sift10M.npz"
    TEST_NPZ="${DATA_DIR}/test_sift10M.npz"
    CENTROIDS="${DATA_DIR}/centroids.npy"
    
    # 动态定义 Experiment ID
    EXP_ID="${CURRENT_DATE}_sift10M_leaf${TAG}_OPTUNA"

    echo ""
    echo "########################################################"
    echo "▶️  正在处理: LeafSize ${TAG}"
    echo "   数据路径: ${DATA_DIR}"
    echo "   任务 ID : ${EXP_ID}"
    echo "########################################################"

    # [安全检查] 检查训练数据是否存在
    if [ ! -f "${TRAIN_NPZ}" ]; then
        echo "⚠️  警告: 未找到训练数据 ${TRAIN_NPZ}"
        echo "⏭️  跳过当前任务，进入下一个..."
        continue
    fi

    # 运行 Optuna 调优
    cd ${PROJECT_ROOT} || exit 1
    
    python -m src.model.cluster_dist_transformer_original \
      --train_npz "${TRAIN_NPZ}" \
      --test_npz "${TEST_NPZ}" \
      --centroids_path "${CENTROIDS}" \
      --val_split 0.05 \
      --topk 20 \
      --epochs 2 \
      --pos_exist \
      --use_gating \
      --loss_type kld \
      --experiment_id "${EXP_ID}" \
      --use_optuna \
      --optuna_trials 500 \
      || echo "❌ 任务 ${TAG} 运行中发生错误，已自动跳过，继续后续任务。"

    echo "✅ 任务 ${TAG} 结束。"
done

echo "========================================================"
echo "🎉 所有 Optuna 调优任务执行完毕！"
echo "========================================================"