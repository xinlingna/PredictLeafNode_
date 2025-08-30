#!/usr/bin/env python3
"""
使用示例：从二进制文件生成cluster_info.npz并测试
"""

import os
import numpy as np
from convert_bin_to_npz import convert_bin_to_npz, verify_cluster_info_npz


def example_usage():
    """示例：如何使用转换脚本"""
    
    # 设置路径和参数
    input_dir = "input/Training_data/gist1M_learn/leafsize20K/"
    output_path = "input/Training_data/gist1M_learn/leafsize20K/cluster_info.npz"
    
    # 根据你的数据设置这些参数
    num_clusters = 50      # 叶子节点数量 (K)
    num_representatives = 3  # 每个聚类的代表向量数量
    feature_dim = 960       # GIST特征维度 (D)
    
    print("=== Hercules Binary to NPZ Conversion Example ===\n")
    
    # 检查输入文件是否存在
    required_files = [
        "cluster_sizes.bin",
        "cluster_variances.bin", 
        "cluster_densities.bin",
        "intra_distances.bin",
        "representatives.bin"
    ]
    
    print("Checking required binary files...")
    missing_files = []
    for file in required_files:
        file_path = os.path.join(input_dir, file)
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"  ✓ {file}: {size} bytes")
        else:
            print(f"  ❌ {file}: NOT FOUND")
            missing_files.append(file)
    
    if missing_files:
        print(f"\nError: Missing files: {missing_files}")
        print("Please run the C++ Hercules code first to generate these binary files.")
        return False
    
    try:
        # 执行转换
        print(f"\nConverting to NPZ format...")
        convert_bin_to_npz(
            input_dir=input_dir,
            output_path=output_path,
            num_clusters=num_clusters,
            num_representatives=num_representatives,
            feature_dim=feature_dim
        )
        
        # 测试加载生成的npz文件
        print(f"\n=== Testing Generated NPZ File ===")
        test_npz_loading(output_path)
        
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False


def test_npz_loading(npz_path):
    """测试npz文件是否能被cluster_dist_transformer_enhanced.py正确加载"""
    
    print(f"Testing NPZ file: {npz_path}")
    
    # 模拟cluster_dist_transformer_enhanced.py中的加载过程
    try:
        data = np.load(npz_path, allow_pickle=True)
        
        if isinstance(data, np.lib.npyio.NpzFile):
            cluster_info = {key: data[key] for key in data.files}
        else:
            cluster_info = data.item() if data.ndim == 0 else data
        
        print("Successfully loaded cluster_info with keys:")
        for key, value in cluster_info.items():
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        
        # 验证代表向量的形状
        if 'representative_vectors' in cluster_info:
            rep_vectors = cluster_info['representative_vectors']
            K, num_rep, D = rep_vectors.shape
            print(f"\nrepresentative_vectors validation:")
            print(f"  Shape: [{K}, {num_rep}, {D}] ✓")
            print(f"  Data type: {rep_vectors.dtype} ✓") 
            
            # 测试reshape操作（模拟multi_rep enhancement的使用）
            rep_flat = rep_vectors.reshape(-1, D)
            print(f"  Flatten test: {rep_vectors.shape} -> {rep_flat.shape} ✓")
        
        print("\n✅ NPZ file passes all loading tests!")
        
    except Exception as e:
        print(f"\n❌ NPZ loading test failed: {e}")
        return False
    
    return True


def generate_sample_bin_files(output_dir, K=10, num_rep=3, D=5):
    """生成示例二进制文件用于测试（如果你没有真实的Hercules输出）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating sample binary files in {output_dir}")
    print(f"Parameters: K={K}, num_representatives={num_rep}, D={D}")
    
    # 生成随机数据
    np.random.seed(42)  # 保证可重复性
    
    # cluster_sizes: [K] float32
    cluster_sizes = np.random.randint(50, 200, size=K).astype(np.float32)
    cluster_sizes.tofile(os.path.join(output_dir, "cluster_sizes.bin"))
    
    # cluster_variances: [K] float32
    cluster_variances = np.random.uniform(0.1, 2.0, size=K).astype(np.float32)
    cluster_variances.tofile(os.path.join(output_dir, "cluster_variances.bin"))
    
    # cluster_densities: [K] float32
    cluster_densities = np.random.uniform(0.5, 5.0, size=K).astype(np.float32)
    cluster_densities.tofile(os.path.join(output_dir, "cluster_densities.bin"))
    
    # intra_distances: [K] float32
    intra_distances = np.random.uniform(1.0, 10.0, size=K).astype(np.float32)
    intra_distances.tofile(os.path.join(output_dir, "intra_distances.bin"))
    
    # representatives: [K*num_rep*D] float32
    representatives = np.random.randn(K * num_rep * D).astype(np.float32)
    representatives.tofile(os.path.join(output_dir, "representatives.bin"))
    
    print("Sample binary files generated successfully!")
    
    # 测试转换
    npz_path = os.path.join(output_dir, "cluster_info.npz")
    convert_bin_to_npz(output_dir, npz_path, K, num_rep, D)
    test_npz_loading(npz_path)


if __name__ == "__main__":
    # 选择运行模式
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        # 测试模式：生成示例文件并转换
        print("=== Test Mode: Generating Sample Files ===")
        generate_sample_bin_files("./test_data/", K=10, num_rep=3, D=5)
    else:
        # 正常模式：转换实际的Hercules输出
        success = example_usage()
        if not success:
            print("\nTip: If you don't have real Hercules binary files yet, run:")
            print("python example_usage.py test")
            sys.exit(1)
