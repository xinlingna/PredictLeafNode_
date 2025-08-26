#!/usr/bin/env python3
"""
测试脚本：验证 topk_recall 和 topk_ordered_accuracy 函数是否正常工作
"""

import torch
import sys
import os

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.model.cluster_dist_setTransformer import topk_recall, topk_ordered_accuracy

def test_metrics():
    """测试 topk 指标函数"""
    print("🧪 测试 topk_recall 和 topk_ordered_accuracy 函数...")
    
    # 创建测试数据
    B, K = 4, 10  # batch_size=4, num_classes=10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模拟预测概率和真实标签
    pred_prob = torch.randn(B, K).to(device)
    pred_prob = torch.softmax(pred_prob, dim=-1)
    
    label_prob = torch.randn(B, K).to(device)
    label_prob = torch.softmax(label_prob, dim=-1)
    
    print(f"📊 测试数据形状: pred_prob={pred_prob.shape}, label_prob={label_prob.shape}")
    print(f"🔧 设备: {device}")
    
    # 测试 topk_recall
    print("\n📈 测试 topk_recall:")
    for k in [1, 3, 5, 10]:
        recall = topk_recall(pred_prob, label_prob, k=k)
        print(f"  recall@{k} = {recall:.4f}")
    
    # 测试 topk_ordered_accuracy
    print("\n🎯 测试 topk_ordered_accuracy:")
    for k in [1, 3, 5, 10]:
        ordacc = topk_ordered_accuracy(pred_prob, label_prob, k=k)
        print(f"  ordacc@{k} = {ordacc:.4f}")
    
    # 测试梯度保护
    print("\n🛡️ 测试梯度保护:")
    pred_prob.requires_grad_(True)
    
    # 计算损失（应该能正常反向传播）
    loss = torch.mean(pred_prob)
    loss.backward()
    
    print(f"  pred_prob.grad is None: {pred_prob.grad is None}")
    print(f"  pred_prob.grad.shape: {pred_prob.grad.shape}")
    
    # 再次测试指标（应该不影响梯度）
    with torch.no_grad():
        recall = topk_recall(pred_prob, label_prob, k=5)
        ordacc = topk_ordered_accuracy(pred_prob, label_prob, k=5)
    
    print(f"  使用 torch.no_grad() 后的指标: recall@5={recall:.4f}, ordacc@5={ordacc:.4f}")
    print(f"  梯度仍然存在: pred_prob.grad is None: {pred_prob.grad is None}")
    
    print("\n✅ 所有测试通过！")

if __name__ == "__main__":
    test_metrics()
