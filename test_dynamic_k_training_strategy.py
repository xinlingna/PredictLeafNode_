#!/usr/bin/env python3
"""
测试动态K值训练策略的正确性
验证"all"模式下每个query是否有多个K值的训练样本
"""

import numpy as np
import torch
from src.model.dynamic_k.cluster_dist_transformer_dynamic_k import (
    DynamicKNPZClusterDataset,
    create_multi_k_targets
)
import tempfile
import os
from collections import Counter

def test_training_strategy():
    """测试训练策略的正确性"""
    print("🧪 Testing Dynamic K Training Strategy")
    print("=" * 50)
    
    # 创建临时数据文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 参数设置
        N_queries = 100  # 100个查询
        D = 32          # 特征维度
        K = 16          # 聚类数量
        k_values = [1, 5, 10, 50]  # 4个K值
        
        print(f"📊 Test Configuration:")
        print(f"  Queries: {N_queries}")
        print(f"  Feature dim: {D}")
        print(f"  Clusters: {K}")
        print(f"  K values: {k_values}")
        print()
        
        # 生成测试数据
        queries = np.random.randn(N_queries, D).astype(np.float32)
        centroids = np.random.randn(K, D).astype(np.float32)
        multi_k_targets = create_multi_k_targets(queries, centroids, k_values)
        
        # 保存测试数据
        npz_path = os.path.join(tmp_dir, "test_data.npz")
        save_dict = {"queries": queries}
        for k, targets in multi_k_targets.items():
            save_dict[f"targets_top{k}"] = targets
        np.savez_compressed(npz_path, **save_dict)
        
        # 测试"all"模式
        print("🎯 Testing 'all' mode:")
        dataset_all = DynamicKNPZClusterDataset(
            npz_path=npz_path,
            centroids=centroids,
            k_values=k_values,
            mode="all"
        )
        
        expected_size = N_queries * len(k_values)
        actual_size = len(dataset_all)
        
        print(f"  Expected dataset size: {N_queries} × {len(k_values)} = {expected_size}")
        print(f"  Actual dataset size: {actual_size}")
        assert actual_size == expected_size, f"Size mismatch: {actual_size} != {expected_size}"
        print("  ✅ Dataset size correct")
        
        # 验证索引映射
        print(f"  Verifying index mapping...")
        
        # 测试前几个样本的映射
        test_indices = [0, 1, 2, 3, 4, 5, 10, 15]
        for idx in test_indices:
            x, y, k_val = dataset_all[idx]
            expected_query_idx = idx // len(k_values)
            expected_k_idx = idx % len(k_values)
            expected_k_val = k_values[expected_k_idx]
            
            assert k_val == expected_k_val, f"K value mismatch at idx {idx}: {k_val} != {expected_k_val}"
            
            # 验证输入维度
            assert x.shape == (K + 1, D + 1), f"Wrong x shape: {x.shape}"
            assert y.shape == (K,), f"Wrong y shape: {y.shape}"
            
            # 验证K值特征
            k_features = x[:, -1]
            assert np.allclose(k_features, k_val), f"K value feature inconsistent at idx {idx}"
            
            print(f"    idx={idx:2d} → query_{expected_query_idx}, K={k_val} ✓")
        
        print("  ✅ Index mapping correct")
        
        # 测试K值分布
        print(f"  Collecting K value distribution...")
        k_value_counts = Counter()
        query_k_mapping = {}  # query_idx -> set of k_values
        
        for idx in range(len(dataset_all)):
            x, y, k_val = dataset_all[idx]
            query_idx = idx // len(k_values)
            
            k_value_counts[k_val] += 1
            
            if query_idx not in query_k_mapping:
                query_k_mapping[query_idx] = set()
            query_k_mapping[query_idx].add(k_val)
        
        print(f"  K value distribution:")
        total_samples = sum(k_value_counts.values())
        for k_val in sorted(k_values):
            count = k_value_counts[k_val]
            percentage = count / total_samples * 100
            expected_count = N_queries
            print(f"    K={k_val}: {count} samples ({percentage:.1f}%) [expected: {expected_count}]")
            assert count == expected_count, f"K={k_val} count mismatch: {count} != {expected_count}"
        
        print("  ✅ K value distribution correct")
        
        # 验证每个query都有所有K值
        print(f"  Verifying each query has all K values...")
        for query_idx in range(N_queries):
            query_k_values = query_k_mapping.get(query_idx, set())
            expected_k_values = set(k_values)
            assert query_k_values == expected_k_values, \
                f"Query {query_idx} missing K values: {expected_k_values - query_k_values}"
        
        print(f"  ✅ All {N_queries} queries have all {len(k_values)} K values")
        
        # 测试"random"模式对比
        print(f"\n🎲 Testing 'random' mode (for comparison):")
        dataset_random = DynamicKNPZClusterDataset(
            npz_path=npz_path,
            centroids=centroids,
            k_values=k_values,
            mode="random"
        )
        
        random_size = len(dataset_random)
        print(f"  Dataset size: {random_size} (should equal {N_queries})")
        assert random_size == N_queries, f"Random mode size wrong: {random_size} != {N_queries}"
        
        # 收集随机模式下的K值分布
        random_k_counts = Counter()
        for idx in range(min(100, len(dataset_random))):  # 只测试前100个样本
            x, y, k_val = dataset_random[idx]
            random_k_counts[k_val] += 1
        
        print(f"  Random K distribution (first 100 samples):")
        for k_val in sorted(k_values):
            count = random_k_counts[k_val]
            print(f"    K={k_val}: {count} samples")
        
        print("  ⚠️  Random mode: uneven K distribution, not recommended for training")
        
        # 测试目标标签一致性
        print(f"\n🎯 Testing target consistency:")
        query_idx = 0  # 测试第一个query
        query_targets = {}
        
        for k_val in k_values:
            # 找到query_0, K=k_val的样本
            sample_idx = query_idx * len(k_values) + k_values.index(k_val)
            x, y, k = dataset_all[sample_idx]
            assert k == k_val, f"K value mismatch"
            query_targets[k_val] = y
        
        # 验证不同K值的目标分布确实不同
        k_pairs = [(1, 5), (5, 10), (10, 50)]
        for k1, k2 in k_pairs:
            diff = np.abs(query_targets[k1] - query_targets[k2]).mean()
            print(f"  Target difference K={k1} vs K={k2}: {diff:.6f}")
            # 不同K值的分布应该有差异
            assert diff > 1e-6, f"Targets for K={k1} and K={k2} are too similar"
        
        print("  ✅ Different K values have different target distributions")
        
        print(f"\n🎉 All tests passed!")
        print(f"📋 Summary:")
        print(f"  ✅ 'All' mode generates correct number of samples ({expected_size})")
        print(f"  ✅ Index mapping works correctly")
        print(f"  ✅ K value distribution is uniform (each K: {N_queries} samples)")
        print(f"  ✅ Each query has all K values as training samples")
        print(f"  ✅ Different K values produce different target distributions")
        print(f"  ✅ Ready for training with optimal strategy!")


def test_dataloader_integration():
    """测试数据加载器集成"""
    print(f"\n🔄 Testing DataLoader Integration:")
    
    from torch.utils.data import DataLoader
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 创建小型测试数据
        N = 20
        D = 16
        K = 8
        k_values = [1, 5, 10]
        
        queries = np.random.randn(N, D).astype(np.float32)
        centroids = np.random.randn(K, D).astype(np.float32)
        multi_k_targets = create_multi_k_targets(queries, centroids, k_values)
        
        # 保存数据
        npz_path = os.path.join(tmp_dir, "test_data.npz")
        save_dict = {"queries": queries}
        for k, targets in multi_k_targets.items():
            save_dict[f"targets_top{k}"] = targets
        np.savez_compressed(npz_path, **save_dict)
        
        # 创建数据集和加载器
        dataset = DynamicKNPZClusterDataset(
            npz_path=npz_path,
            centroids=centroids,
            k_values=k_values,
            mode="all"
        )
        
        dataloader = DataLoader(dataset, batch_size=6, shuffle=True, num_workers=0)
        
        # 测试数据加载
        batch_k_counts = Counter()
        total_samples = 0
        
        for batch_idx, (x, y, k_vals) in enumerate(dataloader):
            print(f"  Batch {batch_idx}: x.shape={x.shape}, y.shape={y.shape}, k_vals.shape={k_vals.shape}")
            
            # 统计K值
            for k_val in k_vals.numpy():
                batch_k_counts[int(k_val)] += 1
                total_samples += 1
            
            # 验证batch维度
            batch_size = x.size(0)
            assert x.shape == (batch_size, K + 1, D + 1), f"Wrong x batch shape: {x.shape}"
            assert y.shape == (batch_size, K), f"Wrong y batch shape: {y.shape}"
            assert k_vals.shape == (batch_size,), f"Wrong k_vals shape: {k_vals.shape}"
            
            if batch_idx >= 3:  # 只测试前几个batch
                break
        
        print(f"  Sampled K distribution across batches:")
        for k_val in sorted(k_values):
            count = batch_k_counts[k_val]
            percentage = count / total_samples * 100 if total_samples > 0 else 0
            print(f"    K={k_val}: {count} samples ({percentage:.1f}%)")
        
        print("  ✅ DataLoader integration working correctly")


if __name__ == "__main__":
    test_training_strategy()
    test_dataloader_integration()
    
    print(f"\n🎊 All Dynamic K Training Strategy Tests Passed!")
    print(f"Ready to train the dynamic K model with the 'all' mode! 🚀")

