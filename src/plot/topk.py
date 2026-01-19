import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib

# =========================
# 1. 设置全局字体为 10 号 Arial
# =========================
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['font.size'] = 10

# =========================
# 2. 数据读取
# =========================
def load_matrix(file_path, name="data"):
    data = np.loadtxt(file_path)
    assert data.ndim == 2, f"{name} file format error"
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
    cum_ratio = cum_counts / cum_counts[:, -1][:, None]
    return np.mean(cum_ratio, axis=0)

# =========================
# 4. 阈值位置
# =========================
def first_reach_threshold(curve, threshold):
    idx = np.argmax(curve >= threshold)
    return idx + 1  # 1-based

# =========================
# 5. 绘图函数（截取阈值后部分）
# =========================
def plot_recall_curves_sift1M(counts, prob_files, threshold=0.99, figsize=(5, 5)):
    # --- 计算实际曲线 ---
    sorted_idx_counts = np.argsort(-counts, axis=1)
    mean_cum_counts_actual = compute_mean_curve(sorted_idx_counts, counts)
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)

    # --- 计算预测曲线 ---
    mean_cum_counts_list = [mean_cum_counts_actual]
    labels = ["Actual"]

    for label, prob_file in prob_files.items():
        probs = load_matrix(prob_file, name=f"prob ({label})")
        # assert probs.shape == counts.shape, f"{prob_file} 与 counts 形状不匹配"
        counts, probs = align_rows(counts, probs, name=f"prob ({label})")
        
        
        sorted_idx_pi = np.argsort(-probs, axis=1)
        mean_cum_counts_pi = compute_mean_curve(sorted_idx_pi, counts)
        xpos_pi = first_reach_threshold(mean_cum_counts_pi, threshold)

        mean_cum_counts_list.append(mean_cum_counts_pi)
        labels.append(label)

    # 找到两条曲线达到阈值的最大位置，用作横轴截取
    xpos_max = max(first_reach_threshold(mean_cum_counts_actual, threshold),
                   first_reach_threshold(mean_cum_counts_list[1], threshold))
    # 可以稍微延伸一点（例如加2），保证图像美观
    xpos_max = min(xpos_max + 2, counts.shape[1])

    x = np.arange(1, xpos_max + 1)

    # 截取曲线到横轴长度
    mean_cum_counts_actual = mean_cum_counts_actual[:xpos_max]
    mean_cum_counts_pred = mean_cum_counts_list[1][:xpos_max]

    # ================= 绘图 =================
    plt.figure(figsize=figsize)

    plt.plot(x, mean_cum_counts_actual, marker='o',linestyle="--", markersize=2, linewidth=1,
             label=f"Actual @{first_reach_threshold(mean_cum_counts_actual, threshold)}")
    plt.plot(x, mean_cum_counts_pred,marker='o',linestyle="-", markersize=2, linewidth=1,
             label=f"Pred @{first_reach_threshold(mean_cum_counts_pred, threshold)}")
    
    # 绘制达到阈值的竖直虚线
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)
    xpos_pred = first_reach_threshold(mean_cum_counts_pred, threshold)
    
    # 统一用灰色虚线，也可以分别用 'black' 和 'red'
    plt.axvline(x=xpos_actual, linestyle="--", linewidth=1, alpha=0.8, color='black')
    plt.axvline(x=xpos_pred, linestyle="--", linewidth=1, alpha=0.8, color='black')


    # 绘制阈值虚线
    plt.axhline(y=threshold, linestyle="--", linewidth=1, alpha=0.8, color='gray')

    # 图属性设置
    plt.xlim(0, x[-1])
    plt.xlabel("Selected number of clusters", fontsize=10, fontname='Arial')
    plt.ylabel("Average recall", fontsize=10, fontname='Arial')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # plt.title(f"Average recall vs. Number of selected clusters (stop at {int(threshold*100)}%)",
    #           fontsize=10, fontname='Arial')
    plt.title("SIFT10M leafsize20W ALL learn_samples ",
              fontsize=10, fontname='Arial',fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper left', prop={'family':'Arial', 'size':10})

    plt.tight_layout()
    # plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure9a_SIFT1M.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_recall_curves_deep2M(counts, prob_files, threshold=0.99, figsize=(5, 5)):
    # --- 计算实际曲线 ---
    sorted_idx_counts = np.argsort(-counts, axis=1)
    mean_cum_counts_actual = compute_mean_curve(sorted_idx_counts, counts)
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)

    # --- 计算预测曲线 ---
    mean_cum_counts_list = [mean_cum_counts_actual]
    labels = ["Actual"]

    for label, prob_file in prob_files.items():
        probs = load_matrix(prob_file, name=f"prob ({label})")
        assert probs.shape == counts.shape, f"{prob_file} 与 counts 形状不匹配"
        sorted_idx_pi = np.argsort(-probs, axis=1)
        mean_cum_counts_pi = compute_mean_curve(sorted_idx_pi, counts)
        xpos_pi = first_reach_threshold(mean_cum_counts_pi, threshold)

        mean_cum_counts_list.append(mean_cum_counts_pi)
        labels.append(label)

    # 找到两条曲线达到阈值的最大位置，用作横轴截取
    xpos_max = max(first_reach_threshold(mean_cum_counts_actual, threshold),
                   first_reach_threshold(mean_cum_counts_list[1], threshold))
    # 可以稍微延伸一点（例如加2），保证图像美观
    xpos_max = min(xpos_max + 2, counts.shape[1])

    x = np.arange(1, xpos_max + 1)

    # 截取曲线到横轴长度
    mean_cum_counts_actual = mean_cum_counts_actual[:xpos_max]
    mean_cum_counts_pred = mean_cum_counts_list[1][:xpos_max]

    # ================= 绘图 =================
    plt.figure(figsize=figsize)

    plt.plot(x, mean_cum_counts_actual, marker='o',linestyle="--", markersize=2, linewidth=1,
             label=f"Actual @{first_reach_threshold(mean_cum_counts_actual, threshold)}")
    plt.plot(x, mean_cum_counts_pred,marker='o',linestyle="-", markersize=2, linewidth=1,
             label=f"Pred @{first_reach_threshold(mean_cum_counts_pred, threshold)}")
    
    # 绘制达到阈值的竖直虚线
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)
    xpos_pred = first_reach_threshold(mean_cum_counts_pred, threshold)
    
    # 统一用灰色虚线，也可以分别用 'black' 和 'red'
    plt.axvline(x=xpos_actual, linestyle="--", linewidth=1, alpha=0.8, color='black')
    plt.axvline(x=xpos_pred, linestyle="--", linewidth=1, alpha=0.8, color='black')


    # 绘制阈值虚线
    plt.axhline(y=threshold, linestyle="--", linewidth=1, alpha=0.8, color='gray')

    # 图属性设置
    plt.xlim(0, x[-1])
    plt.xlabel("Selected number of clusters", fontsize=10, fontname='Arial')
    plt.ylabel("Average recall", fontsize=10, fontname='Arial')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # plt.title(f"Average recall vs. Number of selected clusters (stop at {int(threshold*100)}%)",
    #           fontsize=10, fontname='Arial')
    plt.title("DEEP2M",
              fontsize=10, fontname='Arial',fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper left', prop={'family':'Arial', 'size':10})

    plt.tight_layout()
    plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure9b_DEEP2M.png", dpi=600, bbox_inches='tight')
    plt.show()
    
def plot_recall_curves_gist1M(counts, prob_files, threshold=0.99, figsize=(5, 5)):
    # --- 计算实际曲线 ---
    sorted_idx_counts = np.argsort(-counts, axis=1)
    mean_cum_counts_actual = compute_mean_curve(sorted_idx_counts, counts)
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)

    # --- 计算预测曲线 ---
    mean_cum_counts_list = [mean_cum_counts_actual]
    labels = ["Actual"]

    for label, prob_file in prob_files.items():
        probs = load_matrix(prob_file, name=f"prob ({label})")
        assert probs.shape == counts.shape, f"{prob_file} 与 counts 形状不匹配"
        sorted_idx_pi = np.argsort(-probs, axis=1)
        mean_cum_counts_pi = compute_mean_curve(sorted_idx_pi, counts)
        xpos_pi = first_reach_threshold(mean_cum_counts_pi, threshold)

        mean_cum_counts_list.append(mean_cum_counts_pi)
        labels.append(label)

    # 找到两条曲线达到阈值的最大位置，用作横轴截取
    xpos_max = max(first_reach_threshold(mean_cum_counts_actual, threshold),
                   first_reach_threshold(mean_cum_counts_list[1], threshold))
    # 可以稍微延伸一点（例如加2），保证图像美观
    xpos_max = min(xpos_max + 2, counts.shape[1])

    x = np.arange(1, xpos_max + 1)

    # 截取曲线到横轴长度
    mean_cum_counts_actual = mean_cum_counts_actual[:xpos_max]
    mean_cum_counts_pred = mean_cum_counts_list[1][:xpos_max]

    # ================= 绘图 =================
    plt.figure(figsize=figsize)

    plt.plot(x, mean_cum_counts_actual, marker='o',linestyle="--", markersize=2, linewidth=1,
             label=f"Actual @{first_reach_threshold(mean_cum_counts_actual, threshold)}")
    plt.plot(x, mean_cum_counts_pred,marker='o',linestyle="-", markersize=2, linewidth=1,
             label=f"Pred @{first_reach_threshold(mean_cum_counts_pred, threshold)}")
    
    # 绘制达到阈值的竖直虚线
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)
    xpos_pred = first_reach_threshold(mean_cum_counts_pred, threshold)
    
    # 统一用灰色虚线，也可以分别用 'black' 和 'red'
    plt.axvline(x=xpos_actual, linestyle="--", linewidth=1, alpha=0.8, color='black')
    plt.axvline(x=xpos_pred, linestyle="--", linewidth=1, alpha=0.8, color='black')


    # 绘制阈值虚线
    plt.axhline(y=threshold, linestyle="--", linewidth=1, alpha=0.8, color='gray')

    # 图属性设置
    plt.xlim(0, x[-1])
    plt.xlabel("Selected number of clusters", fontsize=10, fontname='Arial')
    plt.ylabel("Average recall", fontsize=10, fontname='Arial')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # plt.title(f"Average recall vs. Number of selected clusters (stop at {int(threshold*100)}%)",
    #           fontsize=10, fontname='Arial')
    plt.title("GIST1M",
              fontsize=10, fontname='Arial',fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper left', prop={'family':'Arial', 'size':10})

    plt.tight_layout()
    plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure9c_GIST1M.png", dpi=600, bbox_inches='tight')
    plt.show()

def plot_recall_curves(counts, prob_files, threshold=0.99, figsize=(5, 5)):
    # --- 计算实际曲线 ---
    sorted_idx_counts = np.argsort(-counts, axis=1)
    mean_cum_counts_actual = compute_mean_curve(sorted_idx_counts, counts)
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)

    # --- 计算预测曲线 ---
    mean_cum_counts_list = [mean_cum_counts_actual]
    labels = ["Actual"]

    for label, prob_file in prob_files.items():
        probs = load_matrix(prob_file, name=f"prob ({label})")
        # assert probs.shape == counts.shape, f"{prob_file} 与 counts 形状不匹配"
        counts, probs = align_rows(counts, probs, name=f"prob ({label})")
        
        
        sorted_idx_pi = np.argsort(-probs, axis=1)
        mean_cum_counts_pi = compute_mean_curve(sorted_idx_pi, counts)
        xpos_pi = first_reach_threshold(mean_cum_counts_pi, threshold)

        mean_cum_counts_list.append(mean_cum_counts_pi)
        labels.append(label)

    # 找到两条曲线达到阈值的最大位置，用作横轴截取
    xpos_max = max(first_reach_threshold(mean_cum_counts_actual, threshold),
                   first_reach_threshold(mean_cum_counts_list[1], threshold))
    # 可以稍微延伸一点（例如加2），保证图像美观
    xpos_max = min(xpos_max + 2, counts.shape[1])

    x = np.arange(1, xpos_max + 1)

    # 截取曲线到横轴长度
    mean_cum_counts_actual = mean_cum_counts_actual[:xpos_max]
    mean_cum_counts_pred = mean_cum_counts_list[1][:xpos_max]

    # ================= 绘图 =================
    plt.figure(figsize=figsize)

    plt.plot(x, mean_cum_counts_actual, marker='o',linestyle="--", markersize=2, linewidth=1,
             label=f"Actual @{first_reach_threshold(mean_cum_counts_actual, threshold)}")
    plt.plot(x, mean_cum_counts_pred,marker='o',linestyle="-", markersize=2, linewidth=1,
             label=f"Pred @{first_reach_threshold(mean_cum_counts_pred, threshold)}")
    
    # 绘制达到阈值的竖直虚线
    xpos_actual = first_reach_threshold(mean_cum_counts_actual, threshold)
    xpos_pred = first_reach_threshold(mean_cum_counts_pred, threshold)
    
    # 统一用灰色虚线，也可以分别用 'black' 和 'red'
    plt.axvline(x=xpos_actual, linestyle="--", linewidth=1, alpha=0.8, color='black')
    plt.axvline(x=xpos_pred, linestyle="--", linewidth=1, alpha=0.8, color='black')


    # 绘制阈值虚线
    plt.axhline(y=threshold, linestyle="--", linewidth=1, alpha=0.8, color='gray')

    # 图属性设置
    plt.xlim(0, x[-1])
    plt.xlabel("Selected number of clusters", fontsize=10, fontname='Arial')
    plt.ylabel("Average recall", fontsize=10, fontname='Arial')
    plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(5))
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    # plt.title(f"Average recall vs. Number of selected clusters (stop at {int(threshold*100)}%)",
    #           fontsize=10, fontname='Arial')
    # plt.title("SIFT10M leafsize20W ALL learn_samples ", fontsize=10, fontname='Arial',fontweight="bold")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(loc='upper left', prop={'family':'Arial', 'size':10})

    plt.tight_layout()
    # plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure9a_SIFT1M.png", dpi=600, bbox_inches='tight')
    plt.show()
# =========================
# 6. 主入口示例
# =========================
if __name__ == "__main__":

    # 模型预测文件
    prob_files = { 
        "Pred": r"D:\zmd\deep50M\leafsize20W\model_d128_L4_H8_ff256_topk30_bs256_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.05_pred.txt"
    }
    count_file = r"D:\zmd\deep50M\leafsize20W\query_knn_distributions_k100.txt"
    counts = load_matrix(count_file, name="counts")
    plot_recall_curves(counts=counts, prob_files=prob_files, threshold=0.99)