#!/usr/bin/env python3
"""
测试脚本：验证删除representative_vectors后所有功能正常工作
"""

import os
import sys
import numpy as np
import tempfile
from pathlib import Path

# 添加模块路径
sys.path.append(str(Path(__file__).parent))

from convert_bin_to_npz import convert_bin_to_npz, verify_cluster_info_npz


def generate_test_bin_files(output_dir, K=10, D=5):
    """生成测试用的二进制文件（不包含representatives.bin）"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating test binary files in {output_dir}")
    print(f"Parameters: K={K}, D={D}")
    
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
    
    print("Test binary files generated successfully!")
    
    # 验证文件大小
    expected_size = K * 4  # K个float32，每个4字节
    for filename in ["cluster_sizes.bin", "cluster_variances.bin", 
                     "cluster_densities.bin", "intra_distances.bin"]:
        file_path = os.path.join(output_dir, filename)
        actual_size = os.path.getsize(file_path)
        print(f"  {filename}: {actual_size} bytes (expected: {expected_size})")
        assert actual_size == expected_size, f"Size mismatch for {filename}"


def test_npz_conversion():
    """测试二进制文件到NPZ的转换"""
    
    print("\n=== Testing NPZ Conversion ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        K, D = 10, 5
        
        # 生成测试文件
        generate_test_bin_files(temp_dir, K, D)
        
        # 转换为NPZ
        npz_path = os.path.join(temp_dir, "cluster_info.npz")
        convert_bin_to_npz(temp_dir, npz_path, K, D)
        
        # 验证NPZ文件
        verify_result = verify_cluster_info_npz(npz_path, K, D)
        assert verify_result, "NPZ verification failed"
        
        print("✅ NPZ conversion test passed!")


def test_cluster_dataset_loading():
    """测试EnhancedNPZClusterDataset的加载"""
    
    print("\n=== Testing Dataset Loading ===")
    
    try:
        from cluster_dist_transformer_enhanced import EnhancedNPZClusterDataset
        
        with tempfile.TemporaryDirectory() as temp_dir:
            K, D = 5, 8
            N = 20  # 样本数量
            
            # 生成测试NPZ文件和质心
            generate_test_bin_files(temp_dir, K, D)
            
            npz_path = os.path.join(temp_dir, "cluster_info.npz")
            convert_bin_to_npz(temp_dir, npz_path, K, D)
            
            # 生成测试数据
            np.random.seed(42)
            queries = np.random.randn(N, D).astype(np.float32)
            targets = np.random.rand(N, K).astype(np.float32)
            centroids = np.random.randn(K, D).astype(np.float32)
            
            # 保存测试数据
            test_data_path = os.path.join(temp_dir, "test_data.npz")
            np.savez(test_data_path, queries=queries, targets=targets)
            
            # 测试不同的增强类型
            enhancement_types = ["none", "stats", "distance", "multi_rep", "comprehensive"]
            
            for enhancement_type in enhancement_types:
                print(f"  Testing enhancement_type: {enhancement_type}")
                
                try:
                    dataset = EnhancedNPZClusterDataset(
                        npz_path=test_data_path,
                        normalize=False,
                        centroids=centroids,
                        cluster_info_path=npz_path,
                        enhancement_type=enhancement_type,
                        num_representatives=3
                    )
                    
                    # 测试获取样本
                    sample_x, sample_y = dataset[0]
                    print(f"    Sample shape: {sample_x.shape}, target shape: {sample_y.shape}")
                    
                    # 验证数据类型
                    assert sample_x.dtype == np.float32, f"Wrong dtype: {sample_x.dtype}"
                    assert sample_y.dtype == np.float32, f"Wrong dtype: {sample_y.dtype}"
                    
                    print(f"    ✅ {enhancement_type} enhancement test passed!")
                    
                except Exception as e:
                    print(f"    ❌ {enhancement_type} enhancement test failed: {e}")
                    raise
        
        print("✅ Dataset loading test passed!")
        
    except ImportError as e:
        print(f"⚠️  Dataset loading test skipped (import error): {e}")


def test_dimension_calculation():
    """测试输入维度计算"""
    
    print("\n=== Testing Dimension Calculation ===")
    
    try:
        from cluster_dist_transformer_enhanced import calculate_input_dim
        
        D = 100  # 原始维度
        
        test_cases = [
            ("none", D),
            ("stats", D + 4),
            ("distance", D + 5),
            ("multi_rep", D),
            ("comprehensive", D + 9)
        ]
        
        for enhancement_type, expected_dim in test_cases:
            actual_dim = calculate_input_dim(enhancement_type, D)
            print(f"  {enhancement_type}: {actual_dim} (expected: {expected_dim})")
            assert actual_dim == expected_dim, f"Dimension mismatch for {enhancement_type}"
        
        print("✅ Dimension calculation test passed!")
        
    except ImportError as e:
        print(f"⚠️  Dimension calculation test skipped (import error): {e}")


def main():
    """运行所有测试"""
    
    print("🚀 Testing cluster_info generation without representative_vectors\n")
    
    try:
        # 测试NPZ转换
        test_npz_conversion()
        
        # 测试数据集加载
        test_cluster_dataset_loading()
        
        # 测试维度计算
        test_dimension_calculation()
        
        print("\n🎉 All tests passed! The system works correctly without representative_vectors.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
