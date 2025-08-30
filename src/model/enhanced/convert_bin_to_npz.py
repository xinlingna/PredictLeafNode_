#!/usr/bin/env python3
"""
将Hercules生成的二进制文件转换为cluster_info.npz格式
包含4个统计特征：cluster_sizes, cluster_variances, cluster_densities, intra_distances
要求所有生成语句符合cluster_dist_transformer_enhanced.py的逻辑
"""

import os
import argparse
import numpy as np
from pathlib import Path


def load_binary_file(file_path, dtype, expected_size=None):
    """加载二进制文件"""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Binary file not found: {file_path}")
    
    data = np.fromfile(file_path, dtype=dtype)
    if expected_size is not None and len(data) != expected_size:
        raise ValueError(f"Expected {expected_size} elements, got {len(data)} from {file_path}")
    
    print(f"Loaded {file_path}: {data.shape} elements of type {dtype}")
    return data


def convert_bin_to_npz(input_dir, output_path, num_clusters, feature_dim):
    """
    将二进制文件转换为npz格式
    
    Args:
        input_dir: 包含bin文件的目录
        output_path: 输出的npz文件路径
        num_clusters: 聚类数量 (K)
        feature_dim: 特征维度 (D)
    """
    
    input_dir = Path(input_dir)
    
    # 验证参数
    if num_clusters <= 0 or feature_dim <= 0:
        raise ValueError("num_clusters and feature_dim must be positive")
    
    print(f"Converting binary files to npz format...")
    print(f"Parameters: K={num_clusters}, D={feature_dim}")
    
    # 加载各个二进制文件
    try:
        # 1. cluster_sizes: [K] float32
        cluster_sizes = load_binary_file(
            input_dir / "cluster_sizes.bin", 
            dtype=np.float32, 
            expected_size=num_clusters
        )
        
        # 2. cluster_variances: [K] float32  
        cluster_variances = load_binary_file(
            input_dir / "cluster_variances.bin",
            dtype=np.float32,
            expected_size=num_clusters
        )
        
        # 3. cluster_densities: [K] float32
        cluster_densities = load_binary_file(
            input_dir / "cluster_densities.bin", 
            dtype=np.float32,
            expected_size=num_clusters
        )
        
        # 4. intra_distances: [K] float32
        intra_distances = load_binary_file(
            input_dir / "intra_distances.bin",
            dtype=np.float32, 
            expected_size=num_clusters
        )
        
    except Exception as e:
        print(f"Error loading binary files: {e}")
        raise
    
    # 验证数据一致性
    print("\nValidating data consistency...")
    
    # 检查cluster_sizes的合理性
    if np.any(cluster_sizes < 0):
        print("Warning: Found negative cluster sizes")
    
    # 检查cluster_variances的合理性
    if np.any(cluster_variances < 0):
        print("Warning: Found negative cluster variances")
    
    # 打印统计信息
    print(f"\nData statistics:")
    print(f"  cluster_sizes: min={cluster_sizes.min():.2f}, max={cluster_sizes.max():.2f}, mean={cluster_sizes.mean():.2f}")
    print(f"  cluster_variances: min={cluster_variances.min():.6f}, max={cluster_variances.max():.6f}, mean={cluster_variances.mean():.6f}")
    print(f"  cluster_densities: min={cluster_densities.min():.6f}, max={cluster_densities.max():.6f}, mean={cluster_densities.mean():.6f}")
    print(f"  intra_distances: min={intra_distances.min():.6f}, max={intra_distances.max():.6f}, mean={intra_distances.mean():.6f}")
    
    # 构建npz字典，确保符合cluster_dist_transformer_enhanced.py的要求
    cluster_info_dict = {
        'cluster_sizes': cluster_sizes.astype(np.float32),
        'cluster_variances': cluster_variances.astype(np.float32), 
        'cluster_densities': cluster_densities.astype(np.float32),
        'intra_distances': intra_distances.astype(np.float32)
    }
    
    # 保存npz文件
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **cluster_info_dict)
    
    print(f"\nSuccessfully saved cluster_info.npz to: {output_path}")
    
    # 验证保存的文件
    print("\nVerifying saved file...")
    verify_cluster_info_npz(output_path, num_clusters, feature_dim)


def verify_cluster_info_npz(npz_path, expected_K):
    """验证生成的npz文件格式是否正确"""
    
    try:
        data = np.load(npz_path)
        
        required_keys = ['cluster_sizes', 'cluster_variances', 'cluster_densities', 
                        'intra_distances']
        
        print("Checking required keys...")
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
            print(f"  ✓ {key}: {data[key].shape}, dtype: {data[key].dtype}")
        
        # 检查形状
        print("\nChecking shapes...")
        for key in ['cluster_sizes', 'cluster_variances', 'cluster_densities', 'intra_distances']:
            expected_shape = (expected_K,)
            actual_shape = data[key].shape
            if actual_shape != expected_shape:
                raise ValueError(f"{key} shape mismatch: expected {expected_shape}, got {actual_shape}")
            print(f"  ✓ {key}: {actual_shape}")
        

        
        # 检查数据类型
        print("\nChecking data types...")
        for key in required_keys:
            if data[key].dtype != np.float32:
                raise ValueError(f"{key} dtype mismatch: expected float32, got {data[key].dtype}")
            print(f"  ✓ {key}: {data[key].dtype}")
        
        print("\n✅ Cluster info file format is correct!")
        return True
        
    except Exception as e:
        print(f"\n❌ Validation failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert Hercules binary files to cluster_info.npz")
    
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing the binary files")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output path for cluster_info.npz")
    parser.add_argument("--num_clusters", type=int, required=True,
                       help="Number of clusters (K)")
    parser.add_argument("--feature_dim", type=int, required=True,
                       help="Feature dimension (D)")
    
    args = parser.parse_args()
    
    try:
        convert_bin_to_npz(
            input_dir=args.input_dir,
            output_path=args.output_path,
            num_clusters=args.num_clusters,
            feature_dim=args.feature_dim
        )
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Conversion failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())


# 使用示例
"""
python convert_bin_to_npz.py \
    --input_dir input/Training_data/gist1M_learn/leafsize20K/ \
    --output_path input/Training_data/gist1M_learn/leafsize20K/cluster_info.npz \
    --num_clusters 50 \
    --feature_dim 960

# 或者直接在Python中调用
if __name__ == "__test__":
    convert_bin_to_npz(
        input_dir="input/Training_data/gist1M_learn/leafsize20K/",
        output_path="input/Training_data/gist1M_learn/leafsize20K/cluster_info.npz", 
        num_clusters=50,
        feature_dim=960
    )
"""
