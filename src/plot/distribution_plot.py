import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from scipy.interpolate import interp1d

def load_vectors_from_txt(filename, delimiter=None):
    """
    从txt文件中读取向量。
    每行一个向量，元素用空格或指定分隔符分隔。
    """
    vectors = []
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
    n1, n2 = v1.shape[0], v2.shape[0]
    if n1 != n2:
        n = min(n1, n2)
        print(f"[WARN] Row mismatch: {name1}={n1}, {name2}={n2}, truncate to {n}")
        v1 = v1[:n]
        v2 = v2[:n]
    return v1, v2


def plot_avg_prob(vectors):
    """
    计算每个维度的平均概率并画直方图。
    概率定义：某个向量的某维度 / 该向量所有维度之和，再对所有向量取平均。
    """
    # 先把每个向量归一化成概率分布
    row_sums = np.sum(vectors, axis=1, keepdims=True)
    probs = vectors / row_sums

    # 再计算各维度的平均概率
    avg_probs = np.mean(probs, axis=0)
    avg_probs = np.mean(probs, axis=0)
    avg_probs_8 = [round(x, 8) for x in avg_probs]
    print(list(avg_probs_8))
    
    df = pd.DataFrame({
    "Cluster_ID": np.arange(1, len(avg_probs_8) + 1),
    "Average_Probability": avg_probs_8
    })

    df.to_excel("avg_probs.xlsx", index=False)
    print(os.getcwd())
    return avg_probs_8



"""     plt.bar(range(1, len(avg_probs)+1), avg_probs)
    plt.xlabel("Cluster ID")
    plt.ylabel("Probability")
    plt.title("Average Actual Probability per Cluster")
    plt.show() """
    

def plot_interpolated_distributions(true_dist, pred_dist, num_points=300):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    用于展示两者在对应维度上的接近程度（不排序）。
    """

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    x = np.arange(len(true_dist))

    # 插值函数（仅用于可视化）
    f_true = interp1d(x, true_dist, kind="cubic")
    f_pred = interp1d(x, pred_dist, kind="cubic")

    x_new = np.linspace(0, len(x) - 1, num_points)

    true_smooth = f_true(x_new)
    pred_smooth = f_pred(x_new)

    plt.figure(figsize=(6, 4))
    plt.xticks(np.arange(0, len(true_dist), 10))
    plt.yticks(np.arange(0, 0.025, 0.001))
    
    plt.rcParams["font.family"] = "Arial"
    plt.rcParams["font.size"] = 10

    plt.plot(x_new, true_smooth, linewidth=1.2, alpha=0.9,label="True distribution", linestyle="--")
    plt.plot(x_new, pred_smooth, linewidth=1.2, alpha=0.6,label="Predicted distribution", linestyle="-")


    plt.xlabel("Cluster ID")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

def plot_interpolated_distributions_two_deep(true_dist, pred_dist, num_points=300):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    上图展示分布对齐情况，下图展示预测误差（Pred - True）。
    """

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    # 设置全局字体（放在最前面更规范）
    plt.rcParams["font.family"] = "Arial"
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
        2, 1, figsize=(6, 5), sharex=True,
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
    ax1.set_yticks(np.arange(0.005, 0.026, 0.005))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- 下图：误差 ----------
    diff = pred_smooth - true_smooth
    ax2.plot(x_new, diff, color="gray", linewidth=1.0)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black")

    fig.suptitle(
        "Deep2M",
        fontsize=10,
        fontweight="bold"
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Pred-True")
    ax2.set_xticks(np.arange(0, 80, 5))
    ax2.set_yticks(np.arange(-0.00085, 0.0016,0.0005))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure8b_DEEP2M.png", dpi=1500, bbox_inches='tight')
    plt.show()
    
def plot_interpolated_distributions_two_sift(true_dist, pred_dist, num_points=300):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    上图展示分布对齐情况，下图展示预测误差（Pred - True）。
    """

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    # 设置全局字体（放在最前面更规范）
    plt.rcParams["font.family"] = "Arial"
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
        2, 1, figsize=(6, 5), sharex=True,
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
    ax1.set_yticks(np.arange(0.000, 0.031, 0.005))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- 下图：误差 ----------
    diff = pred_smooth - true_smooth
    ax2.plot(x_new, diff, color="gray", linewidth=1.0)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black")

#  "Distribution Alignment Between True and Predicted Probabilities",
    fig.suptitle(
        "SIFT1M",
        fontsize=10,
        fontweight="bold"
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Pred-True")
    ax2.set_xticks(np.arange(0, 96, 5))
    ax2.set_yticks(np.arange(-0.0005, 0.0016,0.0005))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure8a_SIFT1M.png", dpi=1500, bbox_inches='tight')
    plt.show()


def plot_interpolated_distributions_two_gist(true_dist, pred_dist, num_points=300):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    上图展示分布对齐情况，下图展示预测误差（Pred - True）。
    """

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    # 设置全局字体（放在最前面更规范）
    plt.rcParams["font.family"] = "Arial"
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
        2, 1, figsize=(6, 5), sharex=True,
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
    ax1.set_yticks(np.arange(0.000, 0.056, 0.005))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- 下图：误差 ----------
    diff = pred_smooth - true_smooth
    ax2.plot(x_new, diff, color="gray", linewidth=1.0)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black")

    fig.suptitle(
        "SIFT10M leafsize=5W aLL learn samples",
        fontsize=10,
        fontweight="bold"
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Pred-True")
    ax2.set_xticks(np.arange(0, 65, 5))
    ax2.set_yticks(np.arange(-0.002, 0.0021,0.001))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure8c_GIST1M.png", dpi=1500, bbox_inches='tight')
    plt.show()
    
    
def plot_interpolated_distributions(true_dist, pred_dist, num_points=1000):
    """
    在相同维度上对真实分布与预测分布进行插值并绘制平滑曲线，
    上图展示分布对齐情况，下图展示预测误差（Pred - True）。
    """

    assert len(true_dist) == len(pred_dist), "Distributions must have the same dimension"

    # 设置全局字体（放在最前面更规范）
    plt.rcParams["font.family"] = "Arial"
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
    ax1.set_yticks(np.arange(0.000, 0.010, 0.005))
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ---------- 下图：误差 ----------
    diff = pred_smooth - true_smooth
    ax2.plot(x_new, diff, color="gray", linewidth=1.0)
    ax2.axhline(0, linestyle="--", linewidth=0.8, color="black")

    fig.suptitle(
        "SIFT10M leafsize=20W aLL learn samples",
        fontsize=10,
        fontweight="bold"
    )
    ax2.set_xlabel("Cluster ID")
    ax2.set_ylabel("Pred-True")
    ax2.set_xticks(np.arange(0, 66, 5))
    ax2.set_yticks(np.arange(-0.002, 0.0021,0.001))
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    # plt.savefig(r"C:\Users\Lenovo\Documents\实验图片\Figure8c_GIST1M.png", dpi=1500, bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    # filename = r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize60K\model_d256_L4_H8_ff512_topk20_bs256_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename = r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize60K\query_knn_distributions_k100.txt"
    
    # filename =  r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize60K\model_d128_L4_H8_ff256_topk40_bs128_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename =  r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize60K\query_knn_distributions_k100.txt"
    
    # filename_p =  r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize40K\model_d256_L4_H8_ff512_topk20_bs256_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"C:\Users\Lenovo\Desktop\zmd\deep2M\leafsize40K\query_knn_distributions_k100.txt"
    # vectors_p = load_vectors_from_txt(filename_p)
    # vectors_t = load_vectors_from_txt(filename_t)
    # average_distribution_p=plot_avg_prob(vectors_p)
    # average_distribution_t=plot_avg_prob(vectors_t)
    # plot_interpolated_distributions_two_deep(average_distribution_t,average_distribution_p)
    
    
    # filename_p =  r"C:\Users\Lenovo\Desktop\zmd\sift1M\leafsize20K\cluster_dist_transformer\model_d256_L4_H8_ff256_bs256_ep150_lr0.001_wd0.01_bilinear_pred.txt"
    # filename_t =  r"C:\Users\Lenovo\Desktop\zmd\sift1M\leafsize20K\cluster_dist_transformer\knn_distributions_query.txt"
    # vectors_p = load_vectors_from_txt(filename_p)
    # vectors_t = load_vectors_from_txt(filename_t)
    # average_distribution_p=plot_avg_prob(vectors_p)
    # average_distribution_t=plot_avg_prob(vectors_t)
    # plot_interpolated_distributions_two_sift(average_distribution_t,average_distribution_p)
    
    
    # filename_p =  r"C:\Users\Lenovo\Desktop\zmd\gist1M\leafsize20k\model_d512_L4_H8_ff512_bs256_ep150_lr0.001_wd0.01_bilinear_pred.txt"
    # filename_t =  r"C:\Users\Lenovo\Desktop\zmd\gist1M\leafsize20k\query_knn_distributions_20k.txt"
    # vectors_p = load_vectors_from_txt(filename_p)
    # vectors_t = load_vectors_from_txt(filename_t)
    # average_distribution_p=plot_avg_prob(vectors_p)
    # average_distribution_t=plot_avg_prob(vectors_t)
    # plot_interpolated_distributions_two_gist(average_distribution_t,average_distribution_p)
    
    # filename_p =  r"D:\zmd\sift10M\leafsize20W\model_d128_L4_H8_ff256_topk30_bs512_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"D:\zmd\sift10M\leafsize20W\query_knn_distributions_k100.txt"
    # vectors_p = load_vectors_from_txt(filename_p)
    # vectors_t = load_vectors_from_txt(filename_t)
    # average_distribution_p=plot_avg_prob(vectors_p)
    # average_distribution_t=plot_avg_prob(vectors_t)
    # plot_interpolated_distributions_two_gist(average_distribution_t,average_distribution_p)
    
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\all_samples\model_d128_L6_H8_ff512_topk20_bs1024_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\samples20W\model_d128_L6_H8_ff512_topk20_bs1024_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\samples10W\model_d128_L6_H8_ff512_topk20_bs128_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\samples10W\query_Samples1K\model_d128_L6_H8_ff256_topk20_bs128_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\samples20W\1KQuerySamples\model_d256_L8_H4_ff512_topk20_bs2560_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_p =  r"D:\zmd\sift10M\leafsize40W\samples20W\5KQuerySamples\model_d256_L8_H4_ff512_topk20_bs2560_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"D:\zmd\sift10M\leafsize40W\query_knn_distributions_k100.txt"


    # filename_p =  r"D:\zmd\sift10M\leafsize5W\allLearn_allQuery\model_d128_L4_H8_ff256_topk100_bs512_ep20_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"D:\zmd\sift10M\leafsize5W\query_knn_distributions_k100.txt"
    
    # filename_p =  r"D:\zmd\sift10M\leafsize20W\5KQuerySampls\model_d256_L4_H8_ff512_topk30_bs1280_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"D:\zmd\sift10M\leafsize20W\query_knn_distributions_k100.txt"      
    
    # filename_p =  r"D:\zmd\deep50M\leafsize100W\model_d128_L4_H8_ff256_topk30_bs256_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_pred.txt"
    # filename_t =  r"D:\zmd\deep50M\leafsize100W\query_knn_distributions_k100.txt"  
    
    filename_p =  r"D:\zmd\deep50M\leafsize20W\model_d128_L4_H8_ff256_topk30_bs256_ep2_lr0.001_wd0.01_bilinear_normalizeFalse_posTrue_gatingTrue_loss_typekld_use_type_embedFalse_dp0.05_pred.txt"
    filename_t =  r"D:\zmd\deep50M\leafsize20W\query_knn_distributions_k100.txt"  
    
    vectors_p = load_vectors_from_txt(filename_p)
    vectors_t = load_vectors_from_txt(filename_t)
    # === 行数对齐（新增）===
    vectors_p, vectors_t = align_vectors_by_rows(
        vectors_p, vectors_t,
        name1="pred", name2="true"
    )
    average_distribution_p=plot_avg_prob(vectors_p)
    average_distribution_t=plot_avg_prob(vectors_t)
    plot_interpolated_distributions(average_distribution_t,average_distribution_p)
    

    