#!/usr/bin/env python3
"""
测试动态K值聚类分布预测模型的基本功能
"""

import numpy as np
import torch
import torch.nn as nn
from src.model.cluster_dist_transformer_dynamic_k import (
    DynamicKClusterDistTransformer,
    DynamicKNPZClusterDataset,
    create_multi_k_targets
)
import tempfile
import os

def test_model_architecture():
    """测试模型架构的基本功能"""
    print("Testing model architecture...")
    
    # 模型参数
    D = 128  # 特征维度
    K = 64   # 聚类数量
    k_values = [1, 5, 10, 100]
    batch_size = 8
    
    # 创建模型
    model = DynamicKClusterDistTransformer(
        input_dim=D + 1,  # D + 1 (包含K值特征)
        d_model=256,
        nhead=8,
        num_layers=2,  # 减少层数以加快测试
        k_values=k_values,
        k_embed_dim=32
    )
    
    # 创建测试输入
    # [batch_size, K+1, D+1] - 最后一维是K值
    x = torch.randn(batch_size, K + 1, D + 1)
    
    # 设置K值（每个样本随机选择一个K值）
    for i in range(batch_size):
        k_val = np.random.choice(k_values)
        x[i, :, -1] = k_val  # 设置K值特征
    
    # 前向传播
    with torch.no_grad():
        logits = model(x)
    
    # 检查输出形状
    expected_shape = (batch_size, K)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    
    # 检查输出是否为有限值
    assert torch.isfinite(logits).all(), "Model output contains non-finite values"
    
    print(f"✓ Model architecture test passed!")
    print(f"  Input shape: {x.shape}")
    print(f"  Output shape: {logits.shape}")
    print(f"  Output range: [{logits.min().item():.3f}, {logits.max().item():.3f}]")


def test_k_value_embedding():
    """测试K值嵌入是否正常工作"""
    print("\nTesting K value embedding...")
    
    D = 64
    K = 32
    k_values = [1, 5, 10, 100]
    
    model = DynamicKClusterDistTransformer(
        input_dim=D + 1,
        d_model=128,
        nhead=4,
        num_layers=1,
        k_values=k_values,
        k_embed_dim=16
    )
    
    # 测试不同K值是否产生不同的输出
    x_base = torch.randn(1, K + 1, D + 1)
    outputs = {}
    
    for k_val in k_values:
        x = x_base.clone()
        x[0, :, -1] = k_val  # 设置K值
        
        with torch.no_grad():
            logits = model(x)
            outputs[k_val] = logits.clone()
    
    # 检查不同K值是否产生不同的输出
    k_pairs = [(1, 5), (5, 10), (10, 100)]
    for k1, k2 in k_pairs:
        diff = torch.abs(outputs[k1] - outputs[k2]).mean().item()
        assert diff > 1e-6, f"Outputs for K={k1} and K={k2} are too similar (diff={diff})"
        print(f"  ✓ K={k1} vs K={k2}: mean difference = {diff:.6f}")
    
    print("✓ K value embedding test passed!")


def test_multi_k_targets_generation():
    """测试多K值标签生成功能"""
    print("\nTesting multi-K targets generation...")
    
    # 创建测试数据
    N = 100  # 查询数量
    D = 64   # 特征维度
    K = 32   # 聚类数量
    k_values = [1, 5, 10, 50]
    
    # 生成随机查询和聚类中心
    queries = np.random.randn(N, D).astype(np.float32)
    centroids = np.random.randn(K, D).astype(np.float32)
    
    # 生成多K值标签
    multi_k_targets = create_multi_k_targets(queries, centroids, k_values)
    
    # 检查标签形状和性质
    for k in k_values:
        targets = multi_k_targets[k]
        
        # 检查形状
        assert targets.shape == (N, K), f"Wrong shape for K={k}: {targets.shape}"
        
        # 检查概率分布性质
        prob_sums = targets.sum(axis=1)
        assert np.allclose(prob_sums, 1.0, atol=1e-5), f"Probabilities don't sum to 1 for K={k}"
        
        # 检查非负性
        assert (targets >= 0).all(), f"Negative probabilities found for K={k}"
        
        print(f"  ✓ K={k}: shape={targets.shape}, sum_check=passed, non_negative=passed")
    
    # 检查K=1的特殊性质（应该是one-hot分布）
    k1_targets = multi_k_targets[1]
    for i in range(min(10, N)):  # 检查前10个样本
        max_vals = k1_targets[i].max()
        num_max = (k1_targets[i] == max_vals).sum()
        assert num_max == 1, f"K=1 should produce one-hot distribution, but sample {i} has {num_max} max values"
    
    print("✓ Multi-K targets generation test passed!")


def test_dataset_functionality():
    """测试数据集功能"""
    print("\nTesting dataset functionality...")
    
    # 创建临时数据文件
    with tempfile.TemporaryDirectory() as tmp_dir:
        # 参数
        N = 50
        D = 32
        K = 16
        k_values = [1, 5, 10]
        
        # 生成测试数据
        queries = np.random.randn(N, D).astype(np.float32)
        centroids = np.random.randn(K, D).astype(np.float32)
        multi_k_targets = create_multi_k_targets(queries, centroids, k_values)
        
        # 保存测试数据
        npz_path = os.path.join(tmp_dir, "test_data.npz")
        save_dict = {"queries": queries}
        for k, targets in multi_k_targets.items():
            save_dict[f"targets_top{k}"] = targets
        np.savez_compressed(npz_path, **save_dict)
        
        # 测试数据集类
        dataset = DynamicKNPZClusterDataset(
            npz_path=npz_path,
            centroids=centroids,
            k_values=k_values,
            mode="random"
        )
        
        # 检查数据集长度
        assert len(dataset) == N, f"Expected dataset length {N}, got {len(dataset)}"
        
        # 检查数据样本
        for i in range(min(5, N)):
            x, y, k_val = dataset[i]
            
            # 检查形状
            assert x.shape == (K + 1, D + 1), f"Wrong x shape: {x.shape}"
            assert y.shape == (K,), f"Wrong y shape: {y.shape}"
            assert k_val in k_values, f"Invalid k_val: {k_val}"
            
            # 检查K值特征
            k_features = x[:, -1]
            assert np.allclose(k_features, k_val), f"K value feature inconsistent"
            
            # 检查概率分布
            assert np.allclose(y.sum(), 1.0, atol=1e-5), f"Target probabilities don't sum to 1"
            assert (y >= 0).all(), f"Negative target probabilities"
        
        print(f"  ✓ Dataset basic functionality: {len(dataset)} samples")
        print(f"  ✓ Sample shapes: x={x.shape}, y={y.shape}")
        print(f"  ✓ K values: {set(dataset[i][2] for i in range(min(10, len(dataset))))}")
    
    print("✓ Dataset functionality test passed!")


def test_forward_backward():
    """测试前向和反向传播"""
    print("\nTesting forward and backward pass...")
    
    # 模型参数
    D = 64
    K = 32
    k_values = [1, 5, 10]
    batch_size = 4
    
    # 创建模型
    model = DynamicKClusterDistTransformer(
        input_dim=D + 1,
        d_model=128,
        nhead=4,
        num_layers=2,
        k_values=k_values,
        k_embed_dim=16
    )
    
    # 创建测试数据
    x = torch.randn(batch_size, K + 1, D + 1, requires_grad=True)
    for i in range(batch_size):
        k_val = np.random.choice(k_values)
        x.data[i, :, -1] = k_val
    
    # 创建目标标签
    targets = torch.softmax(torch.randn(batch_size, K), dim=-1)
    
    # 前向传播
    logits = model(x)
    
    # 计算损失
    loss_fn = nn.KLDivLoss(reduction="batchmean")
    log_probs = torch.log_softmax(logits, dim=-1)
    loss = loss_fn(log_probs, targets)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    param_count = 0
    grad_count = 0
    for name, param in model.named_parameters():
        param_count += 1
        if param.grad is not None:
            grad_count += 1
            assert torch.isfinite(param.grad).all(), f"Non-finite gradients in {name}"
    
    assert grad_count > 0, "No gradients computed"
    
    print(f"  ✓ Forward pass: loss = {loss.item():.6f}")
    print(f"  ✓ Backward pass: {grad_count}/{param_count} parameters have gradients")
    print("✓ Forward and backward pass test passed!")


def run_all_tests():
    """运行所有测试"""
    print("🧪 Testing Dynamic K Cluster Distribution Transformer")
    print("=" * 60)
    
    try:
        test_model_architecture()
        test_k_value_embedding()
        test_multi_k_targets_generation()
        test_dataset_functionality()
        test_forward_backward()
        
        print("\n" + "=" * 60)
        print("🎉 All tests passed! The dynamic K model is working correctly.")
        print("\nYou can now proceed with:")
        print("1. Preparing your multi-K dataset using --prepare_data")
        print("2. Training the model with your actual data")
        print("3. Evaluating the model on different K values")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    run_all_tests()
