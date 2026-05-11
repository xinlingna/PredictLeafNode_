import matplotlib
# 必须在导入 pyplot 之前设置后端为 Agg，以便在无头服务器(Linux)上运行
matplotlib.use('Agg') 

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.interpolate import interp1d
import logging

# 忽略字体管理器的报错日志
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)

def load_vectors_from_txt(filename, delimiter=None):
    """
    从txt文件中读取向量。
    每行一个向量，元素用空格或指定分隔符分隔。
    """
    vectors = []
    # 增加错误处理
    if not os.path.exists(filename):
        print(f"[ERROR] File not found: {filename}")
        return np.array([])
        
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # 跳过空行
                parts = line.split(delimiter) if delimiter else line.split()
                vectors.append([float(x) for x in parts])
    return np.array(vectors)

def align_vectors_by_rows(v1, v2, name1="pred", name2="true"):
    """
    将两个向量矩阵按行数对齐，取最小行数
    """
    if len(v1) == 0 or len(v2) == 0:
        print("[ERROR] One of the vector arrays is empty.")
        return v1, v2
        
    n1, n2 = v1.shape[0], v2.shape[0]
    if n1 != n2:
        n = min(n1, n2)
        print(f"[WARN] Row mismatch: {name1}={n1}, {name2}={n2}, truncate to {n}")
        v1 = v1[:n]
        v2 = v2[:n]
    return v1, v2


def plot_avg_prob(vectors):
    """
    计算每个维度的平均概率。
    """
    if len(vectors) == 0:
        return []

    # 先把每个向量归一化成概率分布
    row_sums = np.sum(vectors, axis=1, keepdims=True)
    # 防止除以0
    row_sums[row_sums == 0] = 1e-10
    probs = vectors / row_sums

    # 再计算各维度的平均概率
    avg_probs = np.mean(probs, axis=0)
    avg_probs_8 = [round(x, 8) for x in avg_probs]
    
    return avg_probs_8


def plot_interpolated_distributions(true_dist, pred_dist, num_points=1000, save_path="distribution_result.png"):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    上图展示分布对齐情况，下图展示预测误差（Pred - True）。
    """
    if len(true_dist) == 0 or len(pred_dist) == 0:
        print("[ERROR] Data empty, skipping plot.")
        return

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    # 设置全局字体
    try:
        plt.rcParams["font.family"] = "Arial"
    except:
        plt.rcParams["font.family"] = "sans-serif"
        
    plt.rcParams["font.size"] = 10

    x = np.arange(len(true_dist))

    # 插值（仅用于可视化）
    f_true = interp1d(x, true_dist, kind="cubic")
    f_pred = interp1d(x, pred_dist, kind="cubic")

    x_new = np.linspace(0, len(x) - 1, num_points)

    true_smooth = f_true(x_new)
    pred_smooth = f_pred(x_new)

    # ======================
    # 创建上下双子图
    # ======================
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12, 10), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]}
    )

    # ---------- 上图：分布对比 ----------
    ax1.plot(
        x_new, true_smooth,
        linestyle="--", linewidth=1.2, alpha=0.9,
        label="True distribution"
    )
    ax1.plot(
        x_new, pred_smooth,
        linestyle="-", linewidth=1.2, alpha=0.6,
        label="Predicted distribution"
    )

    ax1.set_ylabel("Probability")
    # 根据数据动态调整 Y 轴范围
    max_val = max(max(true_smooth), max(pred_smooth))
    ax1.set_yticks(np.arange(0.000, max_val * 1.2, 0.005))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- 下图：误差 ----------
    diff = pred_smooth - true_smooth
    ax2.plot(x_new, diff, color="gray", linewidth=1.0)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black")

    fig.suptitle(
        "SIFT10M Distribution Analysis",
        fontsize=10,
        fontweight="bold"
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Pred-True")
    
    # 动态调整 x ticks
    step = 5
    if len(true_dist) > 100: step = 10
    ax2.set_xticks(np.arange(0, len(true_dist) + 1, step))
    
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    
    # 直接使用传入的完整路径保存
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # 关闭图像释放内存
    print(f"[INFO] Plot saved to: {save_path}")

if __name__ == "__main__":
    # === 配置路径 ===
    filename_p = "/home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50W/20260120_sift10M_leaf50W_model_d256_L8_H4_ff512_topk25_bs1024_ep1_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.3/model_d256_L8_H4_ff512_topk25_bs1024_ep1_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.3_pred.txt"
    filename_t = "/home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50W/query_knn_distributions_k100.txt"
    
    print("Loading vectors...")
    vectors_p = load_vectors_from_txt(filename_p)
    vectors_t = load_vectors_from_txt(filename_t)
    
    if len(vectors_p) > 0 and len(vectors_t) > 0:
        # === 行数对齐 ===
        vectors_p, vectors_t = align_vectors_by_rows(
            vectors_p, vectors_t,
            name1="pred", name2="true"
        )
        
        print("Calculating average probabilities...")
        average_distribution_p = plot_avg_prob(vectors_p)
        average_distribution_t = plot_avg_prob(vectors_t)
        
        # === 设置保存路径到 filename_p 所在目录 ===
        output_dir = os.path.dirname(filename_p)
        output_path = os.path.join(output_dir, "sift10m_comparison.png")
        
        print("Plotting distribution comparison...")
        plot_interpolated_distributions(
            average_distribution_t, 
            average_distribution_p, 
            save_path=output_path
        )
    else:
        print("[ERROR] Could not load vectors. Check file paths.")