import matplotlib
# =========================
# 0. Linux 服务器必须配置：使用 Agg 后端（无头模式）
# =========================
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os

# =========================
# 1. 设置全局字体
# =========================
# 注意：如果服务器没有安装 Arial 字体，Matplotlib 会自动回退到默认字体(DejaVu Sans)

matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.size'] = 10

# =========================
# 2. 数据读取
# =========================
def load_matrix(file_path, name="data"):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"文件未找到: {file_path}")
    data = np.loadtxt(file_path)
    # assert data.ndim == 2, f"{name} file format error"
    print(f"[INFO] {name} file: {file_path}")
    print(f"[INFO] {name} shape: {data.shape}")
    return data

def align_rows(counts, probs, name="prob"):
    """
    对齐 counts 和 probs 的行数，取最小行数
    """
    n_counts = counts.shape[0]
    n_probs = probs.shape[0]

    if n_counts != n_probs:
        n = min(n_counts, n_probs)
        print(f"[WARN] Row mismatch: counts={n_counts}, {name}={n_probs}, truncate to {n}")
        counts = counts[:n]
        probs = probs[:n]

    return counts, probs

# =========================
# 3. 累计召回曲线计算
# =========================
def compute_mean_curve(sorted_idx, counts):
    sorted_counts = np.take_along_axis(counts, sorted_idx, axis=1)
    cum_counts = np.cumsum(sorted_counts, axis=1)
    # 防止除以0
    sums = cum_counts[:, -1][:, None]
    sums[sums == 0] = 1 
    cum_ratio = cum_counts / sums
    return np.mean(cum_ratio, axis=0)

# =========================
# 4. 阈值位置
# =========================
def first_reach_threshold(curve, threshold):
    # 找到第一个大于等于 threshold 的索引
    idx_list = np.where(curve >= threshold)[0]
    if len(idx_list) > 0:
        return idx_list[0] + 1  # 1-based
    else:
        return len(curve) # 如果都没达到，返回最大长度

# =========================
# 5. 通用绘图函数
# =========================
def plot_recall_curves(counts, prob_files, threshold=0.99, figsize=(5, 5), save_path=None):
    # --- 计算实际曲线 ---
    sorted_idx_counts = np.argsort(-counts, axis=1)
    mean_cum_counts_actual = compute_mean_curve(sorted_idx_counts, counts)
    
    # --- 计算预测曲线 ---
    mean_cum_counts_list = [mean_cum_counts_actual]
    labels = ["Actual"]

    for label, prob_file in prob_files.items():
        probs = load_matrix(prob_file, name=f"prob ({label})")
        # 对齐数据
        counts_aligned, probs_aligned = align_rows(counts, probs, name=f"prob ({label})")
        
        # 注意：如果 counts 被截断了，actual 曲线计算最好也基于截断后的 counts，
        # 但为了简化，这里假设 counts 足够长或主要关注 prob 的长度。
        # 严谨做法是重新计算 counts_aligned 的 actual curve，但这里沿用你的逻辑。
        
        sorted_idx_pi = np.argsort(-probs_aligned, axis=1)
        mean_cum_counts_pi = compute_mean_curve(sorted_idx_pi, counts_aligned)
        
        mean_cum_counts_list.append(mean_cum_counts_pi)
        labels.append(label)

    # 找到两条曲线达到阈值的最大位置，用作横轴截取
    # list[0] is actual, list[1] is pred
    pos_actual = first_reach_threshold(mean_cum_counts_list[0], threshold)
    pos_pred = first_reach_threshold(mean_cum_counts_list[1], threshold)
    
    xpos_max = max(pos_actual, pos_pred)
    # 可以稍微延伸一点（例如加5），保证图像美观
    xpos_max = min(xpos_max + 5, counts.shape[1])

    x = np.arange(1, xpos_max + 1)

    # 截取曲线到横轴长度
    curve_actual = mean_cum_counts_list[0][:xpos_max]
    curve_pred = mean_cum_counts_list[1][:xpos_max]

    # ================= 绘图 =================
    plt.figure(figsize=figsize)

    # 绘制曲线
    label_actual = f"Actual @{pos_actual}"
    plt.plot(x, curve_actual, marker='o', linestyle="--", markersize=2, linewidth=1, label=label_actual)
    
    label_pred = f"Pred @{pos_pred}"
    plt.plot(x, curve_pred, marker='o', linestyle="-", markersize=2, linewidth=1, label=label_pred)
    
    # 绘制达到阈值的竖直虚线
    plt.axvline(x=pos_actual, linestyle="--", linewidth=1, alpha=0.8, color='black')
    plt.axvline(x=pos_pred, linestyle="--", linewidth=1, alpha=0.8, color='black')

    # 绘制阈值水平虚线
    plt.axhline(y=threshold, linestyle="--", linewidth=1, alpha=0.8, color='gray')

    # 图属性设置
    plt.xlim(0, x[-1])
    plt.xlabel("Selected number of clusters", fontsize=10)
    plt.ylabel("Average recall", fontsize=10)
    
    # 刻度设置
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    
    plt.title("Recall Curve Analysis", fontsize=10, fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='lower right', prop={'size':10}) # 改为 lower right 防止遮挡左上角的起始点

    plt.tight_layout()
    
    # ================= 保存逻辑 =================
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[SUCCESS] 图片已保存至: {save_path}")
    else:
        plt.savefig("result.png", dpi=300, bbox_inches='tight')
        
    plt.close() # 关闭图形，释放内存

# =========================
# 6. 主入口
# =========================
if __name__ == "__main__":

    # 模型预测文件路径
    pred_path = r"/home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50W/20260120_sift10M_leaf50W_model_d256_L8_H4_ff512_topk25_bs1024_ep1_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.3/model_d256_L8_H4_ff512_topk25_bs1024_ep1_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.3_pred.txt"
    
    prob_files = { 
        "Pred": pred_path
    }
    
    count_file = r"/home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50W/query_knn_distributions_k100.txt"
    
    # 加载 Counts
    counts = load_matrix(count_file, name="counts")
    
    # --- 自动确定保存路径 ---
    # 获取 pred_path 所在的目录
    output_dir = os.path.dirname(pred_path)
    # 拼接输出文件名
    output_filename = "recall_curve_comparison.png"
    save_full_path = os.path.join(output_dir, output_filename)
    
    print(f"[INFO] 准备绘图，输出路径: {save_full_path}")

    # 运行绘图并保存
    plot_recall_curves(counts=counts, prob_files=prob_files, threshold=0.99, save_path=save_full_path)