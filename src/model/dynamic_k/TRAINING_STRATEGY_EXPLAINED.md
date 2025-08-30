# 动态K值模型训练策略详解

## 核心思想

对于动态K值模型，关键是要让模型学习到**同一个query在不同K值下的不同分布特征**。

### 训练数据组织

#### 单个Query的多个训练样本

对于每个查询向量query，我们需要生成多个训练样本：

```
原始数据: 1000个queries
K值列表: [1, 5, 10, 100]
训练数据: 4000个样本

Query_0 → 4个训练样本:
  - (query_0, centroids, K=1)   → target_top1_0
  - (query_0, centroids, K=5)   → target_top5_0  
  - (query_0, centroids, K=10)  → target_top10_0
  - (query_0, centroids, K=100) → target_top100_0

Query_1 → 4个训练样本:
  - (query_1, centroids, K=1)   → target_top1_1
  - (query_1, centroids, K=5)   → target_top5_1
  - (query_1, centroids, K=10)  → target_top10_1
  - (query_1, centroids, K=100) → target_top100_1

...依此类推
```

#### 数据集索引映射

在"all"模式下，数据集大小为 `N_queries × len(k_values)`：

```python
# 数据集索引映射示例
N_queries = 1000
k_values = [1, 5, 10, 100]
dataset_size = 4000

索引映射:
idx=0  → query_0, K=1
idx=1  → query_0, K=5  
idx=2  → query_0, K=10
idx=3  → query_0, K=100
idx=4  → query_1, K=1
idx=5  → query_1, K=5
...

公式:
query_idx = idx // len(k_values)
k_idx = idx % len(k_values)
k_value = k_values[k_idx]
```

## 训练模式对比

### 🚀 推荐模式: "all"

```python
training_mode = "all"
```

**优势:**
- ✅ 每个query的所有K值都被训练
- ✅ 训练分布均匀，每个K值训练样本数相等
- ✅ 模型能充分学习K值感知特征
- ✅ 训练稳定，结果可重复

**训练效果:**
```
Epoch 1 K-value distribution: K=1:1000(25.0%) | K=5:1000(25.0%) | K=10:1000(25.0%) | K=100:1000(25.0%)
```

### ⚠️ 不推荐模式: "random"

```python
training_mode = "random"  # 不推荐
```

**问题:**
- ❌ 每个query只训练一个随机K值
- ❌ K值分布不均匀，随机性大
- ❌ 模型可能无法充分学习所有K值
- ❌ 训练不稳定，结果难以重现

## 数据准备

### 多K值标签生成

为每个K值生成对应的目标分布：

```python
def create_multi_k_targets(queries, centroids, k_values):
    # 为queries=[N,D], centroids=[K,D] 生成多K值标签
    multi_k_targets = {}
    
    for k in k_values:
        targets_k = np.zeros((N, K))
        for i in range(N):
            # 计算query_i的top-k最近邻在聚类中的分布
            targets_k[i] = compute_topk_distribution(queries[i], centroids, k)
        multi_k_targets[k] = targets_k
    
    return multi_k_targets

# 保存格式:
{
    "queries": [N, D],
    "targets_top1": [N, K],
    "targets_top5": [N, K], 
    "targets_top10": [N, K],
    "targets_top100": [N, K]
}
```

## 模型架构

### K值嵌入机制

```python
# K值编码
k_to_idx = {1: 0, 5: 1, 10: 2, 100: 3}
k_embedding = nn.Embedding(4, k_embed_dim)

# 输入构建
input_features = [query[D], centroids[K,D], k_value_feature]
k_embed = k_embedding(k_value_idx)  # [k_embed_dim]

# 特征融合
combined_features = concat([feature_proj, k_embed_expanded])
```

### 训练输入格式

```python
# 每个训练样本
x: [K+1, D+1]  # 最后一维是K值特征
- x[0, :D] = query_vector
- x[0, D] = k_value
- x[1:, :D] = centroids  
- x[1:, D] = k_value (重复)

y: [K]  # 对应K值的目标分布
k_val: scalar  # K值标量
```

## 训练验证

### K值分布统计

训练过程中会显示每个epoch的K值分布：

```
[Epoch 1] K-value distribution: K=1:1000(25.0%) | K=5:1000(25.0%) | K=10:1000(25.0%) | K=100:1000(25.0%)
[Epoch 2] K-value distribution: K=1:1000(25.0%) | K=5:1000(25.0%) | K=10:1000(25.0%) | K=100:1000(25.0%)
```

**验证标准:**
- ✅ 在"all"模式下，各K值的训练样本数应该相等
- ✅ 每个K值的比例应该是 1/len(k_values)
- ❌ 如果分布不均匀，说明训练策略有问题

### 配置验证

启动训练时会显示配置信息：

```
📊 Training Configuration:
  Original queries: 1000
  K values: [1, 5, 10, 100]
  Training mode: all
  Dataset size: 4000
  Expected size (N×K): 1000 × 4 = 4000
  ✅ Each query has 4 training samples (one per K value)
  Batch size: 512
  Batches per epoch: 8
```

## 推理使用

训练完成后，模型可以动态预测不同K值：

```python
# 推理时指定K值
model.eval()
with torch.no_grad():
    # 预测top-5分布
    x = prepare_input(query, centroids, k_value=5)
    logits = model(x)
    top5_dist = torch.softmax(logits, dim=-1)
    
    # 预测top-100分布
    x = prepare_input(query, centroids, k_value=100)  
    logits = model(x)
    top100_dist = torch.softmax(logits, dim=-1)
```

## 最佳实践

### 1. 数据准备
```bash
# 生成多K值数据集
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --prepare_data \
  --original_train_npz train.npz \
  --centroids_path centroids.npy \
  --k_values 1 5 10 100
```

### 2. 模型训练
```bash
# 使用"all"模式训练
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_npz train_multi_k.npz \
  --centroids_path centroids.npy \
  --k_values 1 5 10 100 \
  --training_mode all \  # 关键！
  --epochs 20 \
  --batch_size 512
```

### 3. 性能验证
- 为每个K值单独评估模型性能
- 比较不同K值下的预测准确度
- 验证K值感知能力

## 总结

通过"all"训练模式，我们确保：
1. **完整性**: 每个query的所有K值都被训练
2. **均匀性**: 各K值的训练样本数相等
3. **有效性**: 模型能充分学习K值感知特征
4. **稳定性**: 训练过程稳定可重现

这种训练策略是实现高质量动态K值预测模型的关键！

