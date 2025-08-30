import argparse
import os
import numpy as np


def load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    """
    从txt文件加载数据矩阵
    """
    # 使用空格分隔加载数据
    arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]  # 如果只有一行，转换为2D
    return arr


def main():
    ap = argparse.ArgumentParser(description="将叶子质心txt文件转换为npy格式")
    ap.add_argument("--centroids_txt", type=str, required=True,  help="叶子质心txt文件路径, 格式: [K, D]")
    ap.add_argument("--out_centroids", type=str, required=True,  help="输出质心npy文件路径")
    ap.add_argument("--dtype", type=str, default="float32",  choices=["float32", "float64"],  help="数据类型")
    args = ap.parse_args()

    # 确定数据类型
    if args.dtype == "float32":
        dtype = np.float32
    else:
        dtype = np.float64

    print(f"正在读取质心文件: {args.centroids_txt}")
    centroids = load_txt_matrix(args.centroids_txt, dtype=dtype)
    
    # 检查形状
    K, D = centroids.shape
    print(f"质心矩阵形状: K={K} (质心数量), D={D} (特征维度)")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.out_centroids), exist_ok=True)
    
    # 保存质心
    np.save(args.out_centroids, centroids.astype(dtype))
    
    print(f"成功保存质心文件: {args.out_centroids}")
    print(f"质心数据统计:")
    print(f"  - 形状: {centroids.shape}")
    print(f"  - 数据类型: {centroids.dtype}")
    print(f"  - 最小值: {centroids.min():.6f}")
    print(f"  - 最大值: {centroids.max():.6f}")
    print(f"  - 平均值: {centroids.mean():.6f}")
    print(f"  - 标准差: {centroids.std():.6f}")


if __name__ == "__main__":
    main()


"""
使用示例:

conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/

# 转换gist1M数据集的质心文件 (leafsize20K)
python -m src.preprocess.convert_centriods_npz \
  --centroids_txt input/Training_data/gist1M_learn/leafsize20K/leaf_centroid.txt \
  --out_centroids input/Training_data/gist1M_learn/leafsize20K/leaf_centroids.npy

# 转换gist1M数据集的质心文件 (leafsize10K)  
python -m src.preprocess.convert_centriods_npz \
  --centroids_txt input/Training_data/gist1M_learn/leafsize10K/leaf_center.txt \
  --out_centroids input/Training_data/gist1M_learn/leafsize10K/centroids.npy

# 转换sift1M数据集的质心文件 (leafsize20K)
python -m src.preprocess.convert_centriods_npz \
  --centroids_txt input/Training_data/sift1M_learn/leafsize20K/leaf_center.txt \
  --out_centroids input/Training_data/sift1M_learn/leafsize20K/centroids.npy

# 如果有leaf_centroid.txt文件，也可以转换
python -m src.preprocess.convert_centriods_npz \
  --centroids_txt input/Training_data/gist1M_learn/leafsize20K/leaf_centroid.txt \
  --out_centroids input/Training_data/gist1M_learn/leafsize20K/centroids_from_centroid.npy

# 使用double precision
python -m src.preprocess.convert_centriods_npz \
  --centroids_txt input/Training_data/gist1M_learn/leafsize20K/leaf_center.txt \
  --out_centroids input/Training_data/gist1M_learn/leafsize20K/centroids_fp64.npy \
  --dtype float64
"""
