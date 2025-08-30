import argparse
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import pickle

def load_vectors_by_cluster(base_path: str, leafsize: str):
    """
    加载每个聚类包含的向量
    假设数据结构：每个聚类的向量保存在单独的文件中
    """
    cluster_vectors = {}
    
    # 这里需要根据你的实际数据结构调整
    # 示例：假设有cluster_0.txt, cluster_1.txt等文件
    cluster_dir = os.path.join(base_path, f"leafsize{leafsize}", "clusters")
    
    if not os.path.exists(cluster_dir):
        print(f"Warning: {cluster_dir} not found. Will create synthetic cluster info.")
        return None
    
    cluster_files = sorted([f for f in os.listdir(cluster_dir) if f.startswith('cluster_') and f.endswith('.txt')])
    
    for i, cluster_file in enumerate(cluster_files):
        cluster_path = os.path.join(cluster_dir, cluster_file)
        try:
            vectors = np.loadtxt(cluster_path, dtype=np.float32)
            if vectors.ndim == 1:
                vectors = vectors[None, :]
            cluster_vectors[i] = vectors
        except:
            print(f"Warning: Could not load {cluster_path}")
            
    return cluster_vectors if cluster_vectors else None

def generate_cluster_info(
    centroids_path: str,
    base_data_path: str,
    leafsize: str,
    output_path: str,
    use_synthetic: bool = False
):
    """
    生成聚类附加信息
    
    Args:
        centroids_path: 质心文件路径
        base_data_path: 数据基础路径
        leafsize: 叶子大小标识符（如"20K"）
        output_path: 输出路径
        use_synthetic: 是否使用合成数据（当无法获取真实聚类向量时）
    """
    print(f"Generating cluster info for leafsize {leafsize}...")
    
    # 加载质心
    centroids = np.load(centroids_path).astype(np.float32)
    K, D = centroids.shape
    print(f"Loaded {K} centroids with dimension {D}")
    
    # 尝试加载每个聚类的向量
    cluster_vectors = load_vectors_by_cluster(base_data_path, leafsize)
    
    if cluster_vectors is None or use_synthetic:
        print("Using synthetic cluster information...")
        cluster_info = generate_synthetic_cluster_info(centroids)
    else:
        print("Computing real cluster statistics...")
        cluster_info = compute_real_cluster_info(cluster_vectors, centroids)
    
    # 生成多代表向量
    if cluster_vectors is not None:
        cluster_info['representative_vectors'] = generate_representative_vectors(
            cluster_vectors, centroids, num_representatives=3
        )
    else:
        cluster_info['representative_vectors'] = generate_synthetic_representatives(
            centroids, num_representatives=3
        )
    
    # 保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **cluster_info)
    
    print(f"Cluster info saved to {output_path}")
    print("Generated features:")
    for key, value in cluster_info.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value.shape}")
        else:
            print(f"  {key}: {type(value)}")

def compute_real_cluster_info(cluster_vectors: dict, centroids: np.ndarray) -> dict:
    """计算真实的聚类统计信息"""
    K, D = centroids.shape
    
    cluster_sizes = []
    cluster_variances = []
    cluster_densities = []
    intra_distances = []
    
    for k in range(K):
        if k in cluster_vectors:
            vectors = cluster_vectors[k]  # [n_k, D]
            centroid = centroids[k]  # [D]
            
            # 聚类大小
            size = len(vectors)
            cluster_sizes.append(size)
            
            # 聚类方差（各维度方差的平均）
            if len(vectors) > 1:
                variance = np.var(vectors, axis=0).mean()
            else:
                variance = 0.1  # 默认值
            cluster_variances.append(variance)
            
            # 聚类内平均距离
            distances_to_centroid = np.linalg.norm(vectors - centroid[None, :], axis=1)
            intra_distance = distances_to_centroid.mean()
            intra_distances.append(intra_distance)
            
            # 聚类密度（基于凸包体积的近似）
            if len(vectors) > D:  # 需要足够的点来估算体积
                # 使用各维度范围的乘积作为体积的粗略估计
                ranges = np.ptp(vectors, axis=0)  # 各维度的范围
                volume = np.prod(ranges + 1e-6)  # 避免零体积
                density = size / volume
            else:
                density = size * 1.0  # 默认密度
            cluster_densities.append(density)
        else:
            # 缺失聚类的默认值
            cluster_sizes.append(1)
            cluster_variances.append(1.0)
            intra_distances.append(1.0)
            cluster_densities.append(1.0)
    
    return {
        'cluster_sizes': np.array(cluster_sizes, dtype=np.float32),
        'cluster_variances': np.array(cluster_variances, dtype=np.float32),
        'cluster_densities': np.array(cluster_densities, dtype=np.float32),
        'intra_distances': np.array(intra_distances, dtype=np.float32),
    }

def generate_synthetic_cluster_info(centroids: np.ndarray) -> dict:
    """生成合成的聚类统计信息"""
    K, D = centroids.shape
    
    # 基于质心之间的距离来生成合理的统计信息
    pairwise_dists = pairwise_distances(centroids)
    
    cluster_sizes = []
    cluster_variances = []
    cluster_densities = []
    intra_distances = []
    
    for k in range(K):
        # 聚类大小：基于与其他质心的距离，距离较远的聚类可能较小
        avg_dist_to_others = pairwise_dists[k].mean()
        size = np.random.uniform(100, 1000) * (1.0 + avg_dist_to_others)
        cluster_sizes.append(size)
        
        # 聚类方差：距离其他质心较近的可能方差较小（更紧密）
        min_dist_to_others = pairwise_dists[k][pairwise_dists[k] > 0].min()
        variance = np.random.uniform(0.5, 2.0) * min_dist_to_others
        cluster_variances.append(variance)
        
        # 内距离：与方差相关
        intra_dist = variance * np.random.uniform(0.8, 1.2)
        intra_distances.append(intra_dist)
        
        # 密度：大小与方差的比率
        density = size / (variance ** D + 1e-6)
        cluster_densities.append(density)
    
    # 标准化到合理范围
    cluster_sizes = np.array(cluster_sizes, dtype=np.float32)
    cluster_variances = np.array(cluster_variances, dtype=np.float32)
    cluster_densities = np.array(cluster_densities, dtype=np.float32)
    intra_distances = np.array(intra_distances, dtype=np.float32)
    
    return {
        'cluster_sizes': cluster_sizes,
        'cluster_variances': cluster_variances, 
        'cluster_densities': cluster_densities,
        'intra_distances': intra_distances,
    }

def generate_representative_vectors(
    cluster_vectors: dict, 
    centroids: np.ndarray, 
    num_representatives: int = 3
) -> np.ndarray:
    """为每个聚类生成多个代表向量"""
    K, D = centroids.shape
    representatives = np.zeros((K, num_representatives, D), dtype=np.float32)
    
    for k in range(K):
        if k in cluster_vectors and len(cluster_vectors[k]) >= num_representatives:
            vectors = cluster_vectors[k]
            centroid = centroids[k]
            
            # 方法1：选择距离质心最近的几个向量
            distances = np.linalg.norm(vectors - centroid[None, :], axis=1)
            closest_indices = np.argsort(distances)[:num_representatives]
            representatives[k] = vectors[closest_indices]
        else:
            # 使用质心周围的随机扰动
            for i in range(num_representatives):
                noise = np.random.normal(0, 0.1, D)
                representatives[k, i] = centroids[k] + noise
    
    return representatives

def generate_synthetic_representatives(
    centroids: np.ndarray, 
    num_representatives: int = 3
) -> np.ndarray:
    """生成合成的代表向量"""
    K, D = centroids.shape
    representatives = np.zeros((K, num_representatives, D), dtype=np.float32)
    
    for k in range(K):
        for i in range(num_representatives):
            # 在质心周围添加不同程度的随机扰动
            scale = 0.05 + 0.1 * i  # 不同的扰动尺度
            noise = np.random.normal(0, scale, D)
            representatives[k, i] = centroids[k] + noise
    
    return representatives

def main():
    parser = argparse.ArgumentParser(description="Generate cluster additional information")
    parser.add_argument("--centroids_path", type=str, required=True,
                       help="Path to centroids file (.npy)")
    parser.add_argument("--base_data_path", type=str, required=True,
                       help="Base path for data (contains leafsizeXX directories)")
    parser.add_argument("--leafsize", type=str, required=True,
                       help="Leaf size identifier (e.g., '20K', '10K')")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for cluster info (.npz)")
    parser.add_argument("--use_synthetic", action="store_true",
                       help="Force use synthetic data even if real data is available")
    
    args = parser.parse_args()
    
    generate_cluster_info(
        args.centroids_path,
        args.base_data_path,
        args.leafsize,
        args.output_path,
        args.use_synthetic
    )

if __name__ == "__main__":
    main()

"""
使用示例：
python -m src.preprocess.generate_cluster_info \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --base_data_path input/Training_data/gist1M_learn \
  --leafsize 20K \
  --output_path input/Training_data/gist1M_learn/leafsize20K/cluster_info.npz \
  --use_synthetic
"""