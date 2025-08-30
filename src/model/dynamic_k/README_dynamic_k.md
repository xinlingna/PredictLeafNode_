# 动态K值聚类分布预测模型

这个模型扩展了原始的聚类分布Transformer，支持根据不同的K值预测top-K最近邻在聚簇中的分布。

## 主要特性

1. **动态K值支持**：模型可以根据输入的K值（如1, 5, 10, 100）预测对应的top-K最近邻分布
2. **K值嵌入**：将K值编码为嵌入向量，与查询向量和聚类中心一起处理
3. **多样化训练**：支持随机K值训练或固定K值训练
4. **兼容性**：保持与原始模型相似的架构和训练流程

## 模型架构

### 输入结构
```
输入：[K+1, D+1]
- 第1行：[query_vector[D], k_value]
- 第2~K+1行：[centroid_i[D], k_value]
```

### 模型组件
1. **K值嵌入层**：将K值映射到嵌入空间
2. **特征投影层**：处理D维原始特征
3. **组合投影层**：融合特征和K值嵌入
4. **Transformer编码器**：处理序列信息和注意力
5. **评分机制**：计算查询向量与各聚类中心的相似度

## 使用方法

### 1. 数据准备

首先需要从原始数据准备包含多个K值标签的数据集：

```bash
# 准备多K值数据集
python -m src.model.cluster_dist_transformer_dynamic_k \
  --prepare_data \
  --original_train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --original_test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/leaf_centroids.npy \
  --k_values 1 5 10 100
```

这会生成包含多个K值标签的新数据集：
- `train_gist_multi_k.npz`：包含 `targets_top1`, `targets_top5`, `targets_top10`, `targets_top100`
- `test_gist_multi_k.npz`：同样包含多个K值的标签

### 2. 模型训练

```bash
# 训练动态K值模型
python -m src.model.cluster_dist_transformer_dynamic_k \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist_multi_k.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist_multi_k.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/leaf_centroids.npy \
  --normalize \
  --val_split 0.01 \
  --topk 10 \
  --epochs 20 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_type_embed \
  --use_gating \
  --k_values 1 5 10 100 \
  --k_embed_dim 32 \
  --training_mode random \
  --loss_type kld \
  --experiment_id dynamic_k_v1
```

### 3. 关键参数说明

- `--k_values`：支持的K值列表，如 `1 5 10 100`
- `--k_embed_dim`：K值嵌入的维度，默认32
- `--training_mode`：训练模式
  - `random`：每个样本随机选择一个K值
  - `all`：为每个查询生成所有K值的训练样本
- `--prepare_data`：是否运行数据准备阶段

## 模型优势

### 1. 统一架构
单个模型可以处理多种K值的查询需求，避免了为每个K值训练单独模型的开销。

### 2. K值感知
模型通过K值嵌入学习到不同K值对应的不同分布特征：
- K=1：更集中的分布，偏向最相似的聚类
- K=100：更平滑的分布，考虑更多聚类的贡献

### 3. 灵活推理
推理时可以动态指定K值，满足不同应用场景的需求。

## 训练策略

### 随机K值训练（推荐）
```python
training_mode = "random"
```
- 每个epoch中，每个查询样本随机选择一个K值进行训练
- 平衡了不同K值的训练，避免模型偏向某个特定K值
- 训练效率高，收敛快

### 全K值训练
```python
training_mode = "all"
```
- 每个查询生成所有K值的训练样本
- 数据集大小增加4倍（假设4个K值）
- 训练更充分，但耗时更长

## 评估机制

模型支持为每个K值单独评估：

```
[TEST K=1] kld=0.234567 | mae=0.123456 | mse=0.345678 | ord_acc@10=0.8765 | recall@10=0.9123
[TEST K=5] kld=0.198765 | mae=0.109876 | mse=0.298765 | ord_acc@10=0.8923 | recall@10=0.9287
[TEST K=10] kld=0.176543 | mae=0.098765 | mse=0.276543 | ord_acc@10=0.9045 | recall@10=0.9356
[TEST K=100] kld=0.165432 | mae=0.087654 | mse=0.265432 | ord_acc@10=0.9123 | recall@10=0.9423
```

## 数据格式

### 输入数据格式
```
train_gist_multi_k.npz:
- queries: [N, D] 查询向量
- targets_top1: [N, K] top-1分布标签
- targets_top5: [N, K] top-5分布标签  
- targets_top10: [N, K] top-10分布标签
- targets_top100: [N, K] top-100分布标签
```

### 模型输入格式
```
x: [batch_size, K+1, D+1]
- x[:, 0, :D] = query_vectors
- x[:, 0, D] = k_value
- x[:, 1:, :D] = centroids
- x[:, 1:, D] = k_value (重复)
```

## 实际应用

在实际应用中，你可以这样使用训练好的模型：

```python
# 推理时指定不同的K值
model.eval()
with torch.no_grad():
    # 预测top-5分布
    k_value = torch.tensor([5])
    logits = model(input_data, k_value)
    top5_distribution = torch.softmax(logits, dim=-1)
    
    # 预测top-100分布  
    k_value = torch.tensor([100])
    logits = model(input_data, k_value)
    top100_distribution = torch.softmax(logits, dim=-1)
```

这样，一个模型就能满足不同K值的查询需求，大大提高了模型的实用性和灵活性。
