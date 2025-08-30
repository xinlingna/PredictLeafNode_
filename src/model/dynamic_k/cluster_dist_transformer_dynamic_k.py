import argparse
import math
import os
import random
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from src.model.plot import plot_epoch1_batch_metrics, plot_epoch_metrics

# 导入原始函数
from src.model.cluster_dist_transformer_original import (
    set_seed, load_centroids, topk_recall, topk_ordered_accuracy, PositionalEncoding
)


class DynamicKNPZClusterDataset(Dataset):
    """支持动态K值的聚类数据集"""
    
    def __init__(
        self,
        npz_path: str = None,
        normalize: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None,
        k_values: List[int] = [1, 5, 10, 100],
        mode: str = "random",  # "random", "all", "specific"
        # 新增参数：支持用户提供的标签NPZ文件
        queries_path: Optional[str] = None, # 支持npz文件和npy文件
        centroids_path: Optional[str] = None, # 支持npz文件和npy文件
        labels_path: Optional[str] = None,  # 包含所有K值标签的NPZ文件路径
    ):
        
        # 判断使用哪种数据加载方式
        if queries_path is not None and centroids_path is not None and labels_path is not None:
            # 使用用户提供的文件格式
            self._load_user_data(queries_path, centroids_path, labels_path, k_values)
        elif npz_path is not None:
            # 使用传统NPZ格式
            self._load_npz_data(npz_path, k_values)
        else:
            raise ValueError("Must provide either (npz_path) or (queries_path + centroids_path + labels_path)")
        
        self.k_values = k_values
        self.mode = mode

        # 如果通过参数提供了centroids，优先使用
        if centroids is not None:
            self.centroids = centroids.astype(np.float32)
        
        # 形状检查
        N, D = self.queries.shape
        K_centroids, D_centroids = self.centroids.shape
        assert D == D_centroids, f"D mismatch: queries={D}, centroids={D_centroids}"
        
        self.N, self.D, self.K = N, D, K_centroids
        
        # 数据标准化
        if normalize:
            self._normalize_data(mean, std)
        else:
            self.norm_stats = None
            
        # 根据模式确定数据集大小
        if self.mode == "all":
            self._length = self.N * len(self.k_values)
        else:
            self._length = self.N
    
    def _load_user_data(self, queries_path: str, centroids_path: str, labels_path: str, k_values: List[int]):
        """加载用户提供的数据文件"""
        print(f"Loading user data files:")
        print(f"  Queries: {queries_path}")
        print(f"  Centroids: {centroids_path}")
        print(f"  Labels: {labels_path}")
        
        # 加载查询向量
        try:
            if queries_path.endswith('.npz'):
                queries_data = np.load(queries_path)
                if 'queries' in queries_data:
                    self.queries = queries_data['queries'].astype(np.float32)
                else:
                    # 如果NPZ文件中没有'queries'键，尝试使用第一个数组
                    if len(queries_data.keys()) == 0:
                        raise ValueError(f"Empty NPZ file: {queries_path}")
                    key = list(queries_data.keys())[0]
                    self.queries = queries_data[key].astype(np.float32)
                    print(f"Warning: 'queries' key not found, using '{key}' as queries")
            else:
                self.queries = np.load(queries_path).astype(np.float32)
                
            if self.queries.size == 0:
                raise ValueError(f"Empty queries array loaded from {queries_path}")
                
            # 验证查询向量的形状
            if self.queries.ndim != 2:
                raise ValueError(f"Queries must be 2D array [N, D], got shape {self.queries.shape}")
                
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Failed to load queries file {queries_path}: {str(e)}")
        except (ValueError, KeyError) as e:
            raise ValueError(f"Invalid queries data in {queries_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Unexpected error loading queries from {queries_path}: {str(e)}")
        
        # 加载质心向量
        try:
            if centroids_path.endswith('.npz'): # npz文件
                centroids_data = np.load(centroids_path)
                if 'centroids' in centroids_data:
                    self.centroids = centroids_data['centroids'].astype(np.float32)
                else:
                    # 如果NPZ文件中没有'centroids'键，尝试使用第一个数组
                    if len(centroids_data.keys()) == 0:
                        raise ValueError(f"Empty NPZ file: {centroids_path}")
                    key = list(centroids_data.keys())[0]
                    self.centroids = centroids_data[key].astype(np.float32)
                    print(f"Warning: 'centroids' key not found, using '{key}' as centroids")
            else: # npy文件
                self.centroids = np.load(centroids_path).astype(np.float32)
            
            # 验证质心数据
            if self.centroids.size == 0:
                raise ValueError(f"Empty centroids array loaded from {centroids_path}")
            if self.centroids.ndim != 2:
                raise ValueError(f"Centroids must be 2D array [K, D], got shape {self.centroids.shape}")
                
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Failed to load centroids file {centroids_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading centroids from {centroids_path}: {str(e)}")
        
        # 加载多K值标签（从单个NPZ文件中）
        print(f"  Loading multi-K labels from: {labels_path}")
        try:
            labels_data = np.load(labels_path)
            if len(labels_data.keys()) == 0:
                raise ValueError(f"Empty labels NPZ file: {labels_path}")
        except (FileNotFoundError, IOError) as e:
            raise ValueError(f"Failed to load labels file {labels_path}: {str(e)}")
        except Exception as e:
            raise ValueError(f"Error loading labels from {labels_path}: {str(e)}")
        
        self.multi_k_targets = {}
        for k in k_values:
            # 根据K值查找对应的标签属性
            target_key = f"targets_k{k}"
            
            if target_key not in labels_data:
                # 尝试其他可能的键名
                possible_keys = [f'targets_top{k}', f'labels_k{k}', f'targets_{k}']
                found_key = None
                for key in possible_keys:
                    if key in labels_data:
                        found_key = key
                        break
                
                if found_key is None:
                    raise ValueError(f"Label for K={k} not found in {labels_path}. Expected key: '{target_key}' or similar.")
                else:
                    target_key = found_key
                    print(f"  Using '{target_key}' as labels for K={k}")
            
            targets = labels_data[target_key].astype(np.float32)
            print(f"  Loaded K={k} labels: {target_key}, shape={targets.shape}")
            
            # 形状检查和归一化
            if targets.ndim == 2:
                N_targets, K_targets = targets.shape
                if N_targets != self.queries.shape[0]:
                    raise ValueError(f"Number of targets ({N_targets}) doesn't match number of queries ({self.queries.shape[0]}) for K={k}")
                
                # 检查是否需要归一化（每行和为1的概率分布）
                row_sums = targets.sum(axis=1)
                needs_normalization = not np.allclose(row_sums, 1.0, atol=1e-6)
                
                if needs_normalization:
                    print(f"  Normalizing K={k} labels to probability distributions")
                    # 处理全零行
                    zero_rows = (row_sums == 0)
                    if zero_rows.any():
                        print(f"  Warning: Found {zero_rows.sum()} zero-sum rows for K={k}, using uniform distribution")
                        targets[zero_rows] = 1.0 / K_targets
                        row_sums[zero_rows] = 1.0
                    
                    # 归一化非零行
                    non_zero_rows = (row_sums > 0)
                    targets[non_zero_rows] = targets[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
                
                self.multi_k_targets[k] = targets
                print(f"  Loaded K={k} labels: shape={targets.shape}, sum_range=[{row_sums.min():.6f}, {row_sums.max():.6f}]")
            else:
                raise ValueError(f"Expected 2D array for K={k} labels, got shape {targets.shape}")
    
    def _load_npz_data(self, npz_path: str, k_values: List[int]):
        """加载传统NPZ格式数据"""
        print(f"Loading traditional NPZ data: {npz_path}")
        data = np.load(npz_path)
        
        if "queries" not in data:
            raise ValueError(f"{npz_path} must contain 'queries' [N, D].")
        
        self.queries = data["queries"].astype(np.float32)   # [N, D]
        
        if hasattr(self, 'centroids') and self.centroids is not None:
            # centroids已经通过参数提供
            pass
        else:
            raise ValueError("centroids is required (global [K, D]); provide via --centroids_path or centroids parameter.")
        
        # 检查是否有预计算的多K值标签
        self.multi_k_targets = {}
        for k in k_values:
            target_key = f"targets_top{k}"
            if target_key in data:
                targets = data[target_key].astype(np.float32)
                
                # 归一化检查和处理（与用户提供的多K标签语义一致）
                row_sums = targets.sum(axis=1)
                needs_normalization = not np.allclose(row_sums, 1.0, atol=1e-6)
                
                if needs_normalization:
                    print(f"Normalizing {target_key} to probability distributions")
                    # 处理全零行
                    zero_rows = (row_sums == 0)
                    if zero_rows.any():
                        print(f"Warning: Found {zero_rows.sum()} zero-sum rows for K={k}, using uniform distribution")
                        targets[zero_rows] = 1.0 / targets.shape[1]
                        row_sums[zero_rows] = 1.0
                    
                    # 归一化非零行
                    non_zero_rows = (row_sums > 0)
                    targets[non_zero_rows] = targets[non_zero_rows] / row_sums[non_zero_rows, np.newaxis]
                
                self.multi_k_targets[k] = targets
            else:
                # 如果没有预计算的标签，尝试从默认targets推导
                if "targets" in data:
                    print(f"Warning: {target_key} not found, using default targets as approximation")
                    self.multi_k_targets[k] = data["targets"].astype(np.float32)
                else:
                    raise ValueError(f"Neither {target_key} nor 'targets' found in {npz_path}")
    
    def _normalize_data(self, mean: Optional[np.ndarray], std: Optional[np.ndarray]):
        """数据标准化"""
        if mean is None or std is None:
            all_vectors = [self.queries, self.centroids]
            combined = np.concatenate(all_vectors, axis=0)
            mean = combined.mean(axis=0, keepdims=True)
            std = combined.std(axis=0, keepdims=True) + 1e-6
        
        self.queries = (self.queries - mean) / std
        self.centroids = (self.centroids - mean) / std
        
        self.norm_stats = (mean, std)

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        # 选择 query 与 K
        if self.mode == "all":
            query_idx = idx // len(self.k_values)
            k_idx = idx % len(self.k_values)
            k_value = int(self.k_values[k_idx])
        else:
            query_idx = idx
            if self.mode == "random":
                k_value = int(random.choice(self.k_values))
            else:
                # 如果提供了 "specific" 模式，这里可以按需实现；默认用第一个
                k_value = int(self.k_values[0])
    
        query = self.queries[query_idx]                              # [D]
        target = self.multi_k_targets[k_value][query_idx]            # [M]
        M = self.centroids.shape[0]
        D = self.queries.shape[1]
    
        # -------- 数据组织方式 --------
        # 简化为单一方式：纵向堆叠 query 与 centroids
        # K值通过模型的embedding系统处理，不在数据中编码
        x = np.vstack([query[None, :], self.centroids])        # [M+1, D]
    
        return x.astype(np.float32), target.astype(np.float32), k_value



class DynamicKClusterDistTransformer(nn.Module):
    """支持动态K值的聚类分布Transformer"""
    
    def __init__(
        self,
        input_dim: int,  # D + 1 (包含K值特征)
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
        use_type_embed: bool = True,
        score_type: str = "bilinear",
        pos_exist: bool = False,
        use_gating: bool = False,
        k_values: List[int] = [1, 5, 10, 100],
        k_embed_dim: int = 32,
    ):
        super().__init__()
        
        # 验证输入参数
        if not k_values or len(k_values) == 0:
            raise ValueError("k_values cannot be empty")
        if k_embed_dim <= 0:
            raise ValueError("k_embed_dim must be positive")
        if d_model <= k_embed_dim:
            raise ValueError("d_model must be greater than k_embed_dim")
        
        self.k_values = k_values
        self.k_embed_dim = k_embed_dim
        self.pos_exist = pos_exist
        self.use_gating = use_gating
        self.d_model = d_model
        
        # K值嵌入：将K值映射到嵌入空间
        # 创建K值到索引的映射
        self.k_to_idx = {k: i for i, k in enumerate(k_values)}
        self.k_embedding = nn.Embedding(len(k_values), k_embed_dim)
        
        # 输入投影：处理原始特征 + K值嵌入
        self.feature_proj = nn.Linear(input_dim, d_model - k_embed_dim)  # D -> d_model - k_embed_dim
        self.input_proj = nn.Linear(d_model, d_model)  # 最终投影到d_model
        
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.type_embed = nn.Embedding(2, d_model) if use_type_embed else None
        
        if use_gating:
            self.gate_linear = nn.Linear(d_model, d_model)
            self.gate_activation = nn.Sigmoid()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.score_type = score_type
        if score_type == "bilinear":
            self.query_head = nn.Linear(d_model, d_model, bias=True)
            self.centroid_head = nn.Linear(d_model, d_model, bias=True)
        elif score_type == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            raise ValueError("score_type must be 'bilinear' or 'mlp'")
    
    def _k_values_to_indices(self, k_values: torch.Tensor) -> torch.Tensor:
        """
        高效地将K值批量转换为对应的索引
        
        Args:
            k_values: [B] K值tensor
        Returns:
            k_indices: [B] 对应的索引tensor
        """
        device = k_values.device
        
        # 创建K值到索引的映射tensor (在GPU上)
        if not hasattr(self, '_k_values_tensor') or self._k_values_tensor.device != device:
            self._k_values_tensor = torch.tensor(self.k_values, device=device, dtype=k_values.dtype)
            self._k_indices_tensor = torch.arange(len(self.k_values), device=device, dtype=torch.long)
        
        # 验证输入K值的有效性
        if torch.any(k_values < 0):
            raise ValueError(f"K values must be non-negative, got: {k_values}")
        
        # 批量化处理：找到每个K值对应的索引
        # 使用广播计算距离矩阵
        distances = torch.abs(k_values.unsqueeze(1) - self._k_values_tensor.unsqueeze(0))  # [B, num_k_values]
        closest_indices = torch.argmin(distances, dim=1)  # [B]
        
        # 验证是否找到了精确匹配
        matched_k_values = self._k_values_tensor[closest_indices]
        if not torch.allclose(k_values.float(), matched_k_values.float(), atol=1e-6):
            mismatched = k_values[~torch.isclose(k_values.float(), matched_k_values.float(), atol=1e-6)]
            if len(mismatched) > 0:
                print(f"Warning: Some K values not found in supported list: {mismatched.unique().tolist()}")
                print(f"Supported K values: {self.k_values}")
        
        return closest_indices
    
    def forward(self, x: torch.Tensor, k_values: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, M+1, D] 输入特征,其中M是聚类数量
            k_values: [B] K值,必须提供
        Returns:
            logits: [B, M] 预测的聚类分布logits,M是聚类数量
        """
        bsz, seqlen, feat_dim = x.size()
        
        # K值现在必须从外部提供
        if k_values is None:
            raise ValueError("k_values must be provided")
        
        # 将K值转换为对应的索引 - 优化版本
        k_indices = self._k_values_to_indices(k_values)
        
        # 获取K值嵌入
        k_embed = self.k_embedding(k_indices)  # [B, k_embed_dim]
        
        # 投影原始特征
        feat_proj = self.feature_proj(x)  # [B, M+1, d_model - k_embed_dim]
        
        # 为每个序列位置添加K值嵌入
        k_embed_expanded = k_embed.unsqueeze(1).expand(-1, seqlen, -1)  # [B, M+1, k_embed_dim]
        
        # 拼接特征和K值嵌入 - 这确保K值信息融入到每个token
        combined = torch.cat([feat_proj, k_embed_expanded], dim=-1)  # [B, M+1, d_model]
        
        # 最终投影
        t = self.input_proj(combined)  # [B, M+1, d_model]
        
        # 类型嵌入
        if self.type_embed is not None:
            type_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=x.device)
            if seqlen > 1:
                type_ids[:, 1:] = 1
            t = t + self.type_embed(type_ids)
        
        # 位置编码
        if self.pos_exist:
            t = self.pos_enc(t)
        
        # Transformer编码
        h = self.encoder(t)  # [B, M+1, d_model]
        
        # 门控机制
        if self.use_gating:
            gate = self.gate_activation(self.gate_linear(h))
            h = h * gate
        
        # 提取query和centroid表示
        q = h[:, 0, :]  # [B, d_model] - query表示
        c = h[:, 1:, :]  # [B, M, d_model] - centroid表示，M是聚类数量
        
        # 计算相似度分数
        if self.score_type == "bilinear":
            qh = self.query_head(q)
            ch = self.centroid_head(c)
            logits = torch.einsum("bd,bkd->bk", qh, ch) / math.sqrt(self.d_model)
        else:
            q_expand = q.unsqueeze(1).expand(-1, c.size(1), -1)
            logits = self.scorer(torch.cat([q_expand, c], dim=-1)).squeeze(-1)
        
        return logits


def build_dynamic_k_loaders(
    train_npz: str = None,
    val_npz: Optional[str] = None,
    centroids_path: str = None,
    batch_size: int = 256,
    num_workers: int = 4,
    normalize: bool = False,
    val_split: float = 0.1,
    seed: int = 42,
    k_values: List[int] = [1, 5, 10, 100],
    mode: str = "random",
    # 新增参数：支持用户提供的标签文件
    train_queries_path: Optional[str] = None, # 训练数据
    train_labels_path: Optional[str] = None,  # 训练标签
    val_queries_path: Optional[str] = None,  # 验证数据
    val_labels_path: Optional[str] = None,  # 验证标签
):
    """构建支持动态K值的数据加载器
    
    支持两种数据格式：
    1. 传统NPZ格式：使用 train_npz, val_npz, centroids_path
    2. 用户数据格式：使用 train_queries_path, train_labels_path, centroids_path
    
    Args:
        # 传统格式参数
        train_npz: 训练NPZ文件路径
        val_npz: 验证NPZ文件路径
        centroids_path: 质心文件路径
        
        # 用户数据格式参数
        train_queries_path: 训练查询向量文件路径
        train_labels_path: 训练标签NPZ文件路径(包含 targets_k1, targets_k5 等)
        val_queries_path: 验证查询向量文件路径
        val_labels_path: 验证标签NPZ文件路径 (包含 targets_k1, targets_k5 等）
        
        # 其他参数
        batch_size, num_workers, normalize, val_split, seed: 训练参数
        k_values: 支持的K值列表
        mode: 训练模式 ("all" 推荐, "random")
    """
    # 加载质心 npz文件或npy文件
    centroids = load_centroids(centroids_path) if centroids_path else None
    
    # 确定使用哪种数据格式
    use_user_format = (train_queries_path is not None and 
                           train_labels_path is not None and 
                           centroids_path is not None)
    
    if use_user_format:
        print("Using user data format for data loading")
        # 使用用户数据格式
        full_train = DynamicKNPZClusterDataset(
            normalize=normalize,
            centroids=centroids,
            k_values=k_values,
            mode=mode,
            queries_path=train_queries_path,
            centroids_path=centroids_path,
            labels_path=train_labels_path
        )
        
        # 处理验证集
        if val_queries_path is not None and val_labels_path is not None:
            # 使用提供的验证集文件
            mean, std = (None, None)
            if full_train.norm_stats is not None:
                mean, std = full_train.norm_stats
            
            val_ds = DynamicKNPZClusterDataset(
                normalize=normalize,
                mean=mean,
                std=std,
                centroids=centroids,
                k_values=k_values,
                mode=mode,
                queries_path=val_queries_path,
                centroids_path=centroids_path,
                labels_path=val_labels_path
            )
        elif val_split > 0:
            # 从训练集分割验证集
            n_total = len(full_train)
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            g = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
        else:
            val_ds = None
            train_ds = full_train
        
        # 返回原始完整数据集用于获取元信息
        sample_dataset = full_train
    
    else:
        print("Using traditional NPZ format for data loading")
        # 使用传统NPZ格式
        if train_npz is None:
            raise ValueError("Must provide train_npz for traditional format")
        
        full_train = DynamicKNPZClusterDataset(
            npz_path=train_npz,
            normalize=normalize, 
            centroids=centroids,
            k_values=k_values,
            mode=mode
        )
        
        if val_npz is None and val_split > 0:
            n_total = len(full_train)
            n_val = int(n_total * val_split)
            n_train = n_total - n_val
            g = torch.Generator().manual_seed(seed)
            train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
        elif val_npz is None:
            train_ds, val_ds = full_train, None
        else:
            train_ds = full_train
            mean, std = (None, None)
            if full_train.norm_stats is not None:
                mean, std = full_train.norm_stats
            val_ds = DynamicKNPZClusterDataset(
                npz_path=val_npz,
                normalize=normalize, 
                mean=mean, 
                std=std, 
                centroids=centroids,
                k_values=k_values,
                mode=mode
            )
        
        # 返回原始完整数据集用于获取元信息
        sample_dataset = full_train
    
    # 创建数据加载器
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, 
                                                      shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, sample_dataset


@torch.no_grad()
def evaluate_dynamic_k_batch(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    k_vals: torch.Tensor,
    device: torch.device, 
    topk: int = 10, 
    loss_type: str = 'kld',
    **loss_kwargs
) -> Tuple[float, float, float, float, float]:
    """评估支持动态K值的单个batch"""
    was_training = model.training
    model.eval()
    
    x = x.to(device)
    y = y.to(device)
    k_vals = k_vals.to(device)
    
    logits = model(x, k_vals)  # 传递K值信息
    log_probs = torch.log_softmax(logits, dim=-1)
    preds = torch.softmax(logits, dim=-1)
    
    # 计算损失（与原始版本保持一致）
    if loss_type == "kld":
        kldiv = nn.KLDivLoss(reduction="batchmean")
        loss_val = kldiv(log_probs, y)
    elif loss_type == "kld_reverse":
        y_log = torch.log(y + 1e-8)
        loss_val = (preds * (log_probs - y_log)).sum(dim=-1).mean()
    elif loss_type == "mse":
        mse_loss_fn = nn.MSELoss()
        loss_val = mse_loss_fn(preds, y)
    # 添加其他损失函数...
    else:
        kldiv = nn.KLDivLoss(reduction="batchmean")
        loss_val = kldiv(log_probs, y)
    
    mae = torch.mean(torch.abs(preds - y)) # 平均绝对误差
    mse = torch.mean((preds - y) ** 2)  # 均方差
    recall = topk_recall(preds, y, k=topk)
    ordered_acc = topk_ordered_accuracy(preds, y, k=topk)
    
    if was_training:
        model.train()
    
    return loss_val.item(), mae.item(), mse.item(), ordered_acc, recall


@torch.no_grad()
def evaluate_dynamic_k(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: int = 10,
    loss_type: str = "kld",
    **loss_kwargs
) -> Tuple[float, float, float, float, float]:
    """评估支持动态K值的完整数据集"""
    was_training = model.training
    model.eval()
    
    mae_meter, mse_meter, loss_meter = 0.0, 0.0, 0.0
    total = 0
    recall_meter = 0.0
    ordered_acc_meter = 0.0
    
    kldiv = nn.KLDivLoss(reduction="batchmean")
    mse_loss_fn = nn.MSELoss()
    
    for batch_data in loader:
        if len(batch_data) == 3:
            x, y, k_vals = batch_data
        else:
            x, y = batch_data
            k_vals = None
        
        x = x.to(device)
        y = y.to(device)
        if k_vals is not None:
            k_vals = k_vals.to(device)
        
        logits = model(x, k_vals)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.softmax(logits, dim=-1)
        
        # 计算损失
        if loss_type == "kld":
            loss_val = kldiv(log_probs, y)
        elif loss_type == "kld_reverse":
            y_log = torch.log(y + 1e-8)
            loss_val = (preds * (log_probs - y_log)).sum(dim=-1).mean()
        elif loss_type == "mse":
            loss_val = mse_loss_fn(preds, y)
        else:
            loss_val = kldiv(log_probs, y)
        
        mae = torch.mean(torch.abs(preds - y))
        mse = torch.mean((preds - y) ** 2)
        recall = topk_recall(preds, y, k=topk)
        ordered_acc = topk_ordered_accuracy(preds, y, k=topk)
        
        bs = x.size(0)
        loss_meter += loss_val.item() * bs
        mae_meter += mae.item() * bs
        mse_meter += mse.item() * bs
        recall_meter += recall * bs
        ordered_acc_meter += ordered_acc * bs
        total += bs
    
    if was_training:
        model.train()
    
    return loss_meter / total, mae_meter / total, mse_meter / total, ordered_acc_meter / total, recall_meter / total


def train_dynamic_k(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-2,
    grad_clip: float = 1.0,
    amp: bool = True,
    save_dir: Optional[str] = None,
    save_name: str = "best.pt",
    eval_topk: int = 10,
    log_interval: int = 100,
    loss_type: str = "kld",
    **loss_kwargs
):
    """训练支持动态K值的模型"""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp and torch.cuda.is_available())
    
    if loss_type == "kld":
        lossFunc = nn.KLDivLoss(reduction="batchmean")
    elif loss_type == "mse":
        lossFunc = nn.MSELoss()
    else: # 未指定则使用kld
        lossFunc = nn.KLDivLoss(reduction="batchmean")
    
    best_val = float("inf")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    epoch_history = []
    epoch1_batch_metrics = []
    
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_ordacc = 0.0
        running_recall = 0.0
        n = 0
        
        # 统计本epoch中不同K值的训练样本数量
        k_value_counts = {}
        
        if ep == 1 and val_loader is not None:
            batch_count = 0
        
        # 初始化变量以避免作用域问题
        k_vals = None
        
        for batch_data in train_loader:
            if len(batch_data) == 3:
                x, y, k_vals = batch_data  # x:[B,M+1,D] y：[B, M] k_vals:[B]
            else:
                x, y = batch_data
                k_vals = None
            
            x = x.to(device)
            y = y.to(device)
            if k_vals is not None:
                k_vals = k_vals.to(device)
                
                # 统计K值分布
                for k_val in k_vals.cpu().numpy():
                    k_val = int(k_val)
                    k_value_counts[k_val] = k_value_counts.get(k_val, 0) + 1
            
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x, k_vals)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                if loss_type == "kld_reverse":
                    y_log = torch.log(y + 1e-8)
                    loss = (probs * (log_probs - y_log)).sum(dim=-1).mean()
                elif loss_type == "kld":
                    loss = lossFunc(log_probs, y)
                elif loss_type == "mse":
                    loss = lossFunc(probs, y)
                else:  # 不指定则使用kld
                    loss = lossFunc(log_probs, y)
                
                train_ordacc = topk_ordered_accuracy(probs, y, k=eval_topk)
                train_recall = topk_recall(probs, y, k=eval_topk)
            
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            
            running += loss.item() * x.size(0)
            running_ordacc += train_ordacc * x.size(0)
            running_recall += train_recall * x.size(0)
            n += x.size(0)
            
            # Epoch 1 batch metrics
            if ep == 1 and val_loader is not None:
                batch_count += 1
                if batch_count % 10 == 0:
                    val_batch_metrics = []
                    for val_batch_data in val_loader:
                        if len(val_batch_data) == 3:
                            val_x, val_y, val_k_vals = val_batch_data
                        else:
                            val_x, val_y = val_batch_data
                            val_k_vals = None
                        
                        val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate_dynamic_k_batch(
                            model, val_x, val_y, val_k_vals, device, topk=eval_topk, loss_type=loss_type
                        )
                        val_batch_metrics.append((val_kld, val_mae, val_mse, val_ordacc, val_recall))
                    
                    # 计算平均指标
                    avg_val_kld = sum(m[0] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_mae = sum(m[1] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_mse = sum(m[2] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_ordacc = sum(m[3] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_recall = sum(m[4] for m in val_batch_metrics) / len(val_batch_metrics)
                    
                    epoch1_batch_metrics.append({
                        'batch': batch_count,
                        'val_kld': avg_val_kld,
                        'val_mae': avg_val_mae,
                        'val_mse': avg_val_mse,
                        'val_ordacc': avg_val_ordacc,
                        'val_recall': avg_val_recall,
                        'train_kld': loss.item(),
                        'train_ordacc': train_ordacc,
                        'train_recall': train_recall,
                    })
                    print(f"Epoch {ep:03d} | Batch {batch_count} | val_kld={avg_val_kld:.6f} | val_ordacc@{eval_topk}={avg_val_ordacc:.6f} | val_recall@{eval_topk}={avg_val_recall:.6f}")
            
            # Periodic batch logging
            if log_interval > 0:
                step_idx = n // x.size(0)
                if step_idx % log_interval == 0:
                    lr_now = opt.param_groups[0]['lr']
                    # 安全地获取K值信息
                    if k_vals is not None:
                        k_info = f" | k_vals={k_vals[:5].tolist()}" # 只输出前5个k_vals值
                    else:
                        k_info = " | k_vals=None"
                    print(f"Epoch {ep:03d} | Step {step_idx} | lr={lr_now:.6e} | loss={loss.item():.6f}{k_info}")
        
        sched.step()
        train_loss = running / n
        train_ordacc = running_ordacc / n
        train_recall = running_recall / n
        
        # 打印K值分布统计
        if k_value_counts:
            total_samples = sum(k_value_counts.values())
            k_dist_str = " | ".join([f"K={k}:{count}({count/total_samples*100:.1f}%)" 
                                   for k, count in sorted(k_value_counts.items())])
            print(f"[Epoch {ep}] K-value distribution: {k_dist_str}")
        
        if val_loader is None:
            epoch_history.append({
                "train_loss": train_loss,
                "train_ordacc": train_ordacc,
                "train_recall": train_recall
            })
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f}")
        else:
            val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate_dynamic_k(
                model, val_loader, device, topk=eval_topk, loss_type=loss_type
            )
            epoch_history.append({
                "train_loss": train_loss,
                "train_ordacc": train_ordacc,
                "train_recall": train_recall,
                "val_kld": val_kld,
                "val_ordacc": val_ordacc,
                "val_recall": val_recall
            })
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f} | val_kld={val_kld:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | ord_acc@{eval_topk}={val_ordacc:.4f} | recall@{eval_topk}={val_recall:.4f}")
            
            if val_kld < best_val and save_dir:
                best_val = val_kld
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "val_kld": val_kld},
                    os.path.join(save_dir, save_name),
                )
    
    # 绘制训练曲线
    if len(epoch1_batch_metrics) > 0 and save_dir:
        plot_epoch1_batch_metrics(epoch1_batch_metrics, save_dir, save_name, eval_topk)
    
    if val_loader is not None and save_dir:
        plot_epoch_metrics(save_dir, save_name, epoch_history)


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Dynamic K Transformer for predicting NN cluster distribution")
    
    # 数据参数 - 传统NPZ格式
    p.add_argument("--train_npz", type=str, default=None, help="Training npz (contains queries and multi-k targets)")
    p.add_argument("--val_npz", type=str, default=None, help="Validation npz (optional)")
    p.add_argument("--test_npz", type=str, default=None, help="Test npz (optional)")
    p.add_argument("--centroids_path", type=str, default=None, help="Global shared centroids (.npy or .npz), shape [K, D]")
    
    # 数据参数 - 用户数据格式（用户提供的格式）
    p.add_argument("--train_queries_path", type=str, default=None, help="Training queries file (.npy or .npz)")
    p.add_argument("--train_labels_path", type=str, default=None, help="Training labels NPZ file (contains targets_k1, targets_k5, etc.)")
    p.add_argument("--val_queries_path", type=str, default=None, help="Validation queries file (.npy or .npz)")
    p.add_argument("--val_labels_path", type=str, default=None, help="Validation labels NPZ file (contains targets_k1, targets_k5, etc.)")
    
    # 通用参数
    p.add_argument("--val_split", type=float, default=0.1, help="Split ratio from training set when no val_npz is provided")
    p.add_argument("--normalize", action="store_true", help="Apply feature-wise standardization")
    
    # K值相关参数
    p.add_argument("--k_values", nargs='+', type=int, default=[1, 5, 10, 100], 
                   help="List of K values to support")
    p.add_argument("--k_embed_dim", type=int, default=32, help="K value embedding dimension")
    p.add_argument("--training_mode", type=str, default="all", choices=["random", "all"],
                   help="Training mode: all (all K values per query - recommended) or random (random K per sample)")
    
    # 训练参数
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (max norm)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_dir", type=str, default="./results", help="Directory to save results")
    
    # 模型参数
    p.add_argument("--d_model", type=int, default=256, help="Transformer hidden size")
    p.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=4, help="Number of Transformer encoder layers")
    p.add_argument("--dim_ff", type=int, default=512, help="Feedforward hidden size")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    p.add_argument("--max_len", type=int, default=4096, help="Maximum supported sequence length")
    p.add_argument("--score_type", type=str, default="bilinear", choices=["bilinear", "mlp"], help="Scoring function type")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--topk", type=int, default=10, help="Top-K for recall metric")
    p.add_argument("--log_interval", type=int, default=100, help="Steps between batch logs (0 to disable)")
    
    # 模型结构参数
    p.add_argument("--pos_exist", action="store_true", help="Whether to use positional encoding")
    p.add_argument("--use_type_embed", action="store_true", help="Whether to use type embedding")
    p.add_argument("--use_gating", action="store_true", help="Whether to use gating mechanism")
    
    # 损失函数参数
    p.add_argument("--loss_type", type=str, choices=["kld", "kld_reverse", "mse"], default="kld")
    
    # 实验标识符
    p.add_argument("--experiment_id", type=str, default=None, help="Experiment identifier prefix for result directories")
    
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.train_npz is None and args.train_queries_path is None:
        raise ValueError("You must provide either --train_npz or --train_queries_path")
    if args.centroids_path is None:
        raise ValueError("You must provide --centroids_path")
    
    # 检查用户数据格式
    
    if args.train_queries_path is not None:
        # 使用用户数据格式
        if args.train_labels_path is None:
            raise ValueError("Must provide --train_labels_path when using --train_queries_path")
        
        print(f"Using user data format:")
        print(f"  Train queries: {args.train_queries_path}")
        print(f"  Train labels: {args.train_labels_path}")
        if args.val_queries_path:
            print(f"  Val queries: {args.val_queries_path}")
            print(f"  Val labels: {args.val_labels_path}")
    
    # 加载质心以确定维度
    C = load_centroids(args.centroids_path)
    K, D = C.shape
    print(f"Detected K={K}, D={D}")
    print(f"Supporting K values: {args.k_values}")
    
    # 输入维度：D (不再包含K值特征，K值通过embedding处理)
    input_dim = D
    
    # 构建数据加载器
    train_loader, val_loader, sample_dataset = build_dynamic_k_loaders(
        train_npz=args.train_npz,
        val_npz=args.val_npz,
        centroids_path=args.centroids_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        k_values=args.k_values, # 
        mode=args.training_mode,
        # 用户数据参数
        train_queries_path=args.train_queries_path,
        train_labels_path=args.train_labels_path,
        val_queries_path=args.val_queries_path,
        val_labels_path=args.val_labels_path,
    )
    
    # 打印训练配置信息
    # 处理不同的数据集类型
    if hasattr(sample_dataset, 'N'):
        original_queries = sample_dataset.N
    elif hasattr(sample_dataset, 'dataset'):
        # 如果是random_split的结果，需要访问原始数据集
        if hasattr(sample_dataset.dataset, 'N'):
            original_queries = sample_dataset.dataset.N
        else:
            original_queries = len(sample_dataset.dataset) // len(args.k_values) if args.training_mode == "all" else len(sample_dataset.dataset)
    else:
        original_queries = len(sample_dataset) // len(args.k_values) if args.training_mode == "all" else len(sample_dataset)
        
    dataset_size = len(sample_dataset)
    print(f"\n Training Configuration:")
    print(f"  Original queries: {original_queries}")
    print(f"  K values: {args.k_values}")
    print(f"  Training mode: {args.training_mode}")
    print(f"  Dataset size: {dataset_size}")
    if args.training_mode == "all":
        expected_size = original_queries * len(args.k_values)
        print(f"  Expected size (N*K): {original_queries} * {len(args.k_values)} = {expected_size}")
        assert dataset_size == expected_size, f"Dataset size mismatch: {dataset_size} != {expected_size}"
        print(f"  Each query has {len(args.k_values)} training samples (one per K value)")
    else:
        print(f" Random mode: each query trains only one random K value per epoch")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    print()
    
    # 创建模型
    model = DynamicKClusterDistTransformer(
        input_dim=input_dim,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        score_type=args.score_type,
        pos_exist=args.pos_exist,
        use_type_embed=args.use_type_embed,
        use_gating=args.use_gating,
        k_values=args.k_values,
        k_embed_dim=args.k_embed_dim,
    ).to(device)
    
    # 设置保存路径
    ckpt_dir = os.path.dirname(args.centroids_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    k_values_str = "_".join(map(str, args.k_values))
    ckpt_name = (
        f"dynamic_k_{k_values_str}_d{args.d_model}_L{args.num_layers}_"
        f"bs{args.batch_size}_ep{args.epochs}_{args.score_type}_{args.loss_type}.pt"
    )
    subdir_name = os.path.splitext(ckpt_name)[0]
    if args.experiment_id:
        subdir_name = f"{args.experiment_id}_{subdir_name}"
        print(f"Using experiment ID: {args.experiment_id}")
    
    result_dir = os.path.join(ckpt_dir, subdir_name)
    os.makedirs(result_dir, exist_ok=True)
    
    # 训练
    if args.epochs > 0:
        train_dynamic_k(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip,
            amp=not args.no_amp,
            save_dir=result_dir,
            save_name=ckpt_name,
            eval_topk=args.topk,
            log_interval=args.log_interval,
            loss_type=args.loss_type,
        )
    
    # 测试评估
    if args.test_npz:
        best_path = os.path.join(result_dir, ckpt_name)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device, weights_only=False)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint from {best_path}")
        
        # 构建测试数据加载器
        mean, std = (None, None)
        # 安全地获取归一化统计信息
        if hasattr(sample_dataset, 'norm_stats') and sample_dataset.norm_stats is not None:
            mean, std = sample_dataset.norm_stats
        elif hasattr(sample_dataset, 'dataset') and hasattr(sample_dataset.dataset, 'norm_stats') and sample_dataset.dataset.norm_stats is not None:
            mean, std = sample_dataset.dataset.norm_stats
            
        test_ds = DynamicKNPZClusterDataset(
            args.test_npz, 
            normalize=args.normalize, 
            mean=mean, 
            std=std, 
            centroids=C,
            k_values=args.k_values,
            mode="all"  # 测试时使用所有K值
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
        
        # 为每个K值分别评估 - 优化版本
        print("\nEvaluating for each K value:")
        
        # 一次性收集所有测试样本并按K值分组
        k_specific_samples = {k: [] for k in args.k_values}
        
        for i in range(len(test_ds)):
            try:
                x, y, k = test_ds[i]
                if k in k_specific_samples:
                    k_specific_samples[k].append((x, y, k))
            except Exception as e:
                print(f"Warning: Skipping test sample {i} due to error: {e}")
                continue
        
        # 为每个K值评估
        for k_val in args.k_values:
            if k_specific_samples[k_val]:
                # 创建临时加载器
                k_loader = DataLoader(k_specific_samples[k_val], batch_size=args.batch_size, 
                                    shuffle=False, num_workers=0)
                
                try:
                    kld, mae, mse, ordacc, recall = evaluate_dynamic_k(
                        model, k_loader, device, topk=args.topk, loss_type=args.loss_type
                    )
                    print(f"[TEST K={k_val}] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | ord_acc@{args.topk}={ordacc:.4f} | recall@{args.topk}={recall:.4f}")
                except Exception as e:
                    print(f"Error evaluating K={k_val}: {e}")
            else:
                print(f"[TEST K={k_val}] No test samples found")
        
        # 全体测试
        kld, mae, mse, ordacc, recall = evaluate_dynamic_k(
            model, test_loader, device, topk=args.topk, loss_type=args.loss_type
        )
        print(f"[TEST ALL] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | ord_acc@{args.topk}={ordacc:.4f} | recall@{args.topk}={recall:.4f}")
        
        # 保存预测结果
        print("Saving model predictions to file...")
        model.eval()
        out_pred = os.path.join(result_dir, os.path.basename(ckpt_name).rsplit('.', 1)[0] + "_pred.txt")
        with torch.no_grad(), open(out_pred, "w") as f:
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    x, _, k_vals = batch_data
                else:
                    x, _ = batch_data
                    k_vals = None
                
                x = x.to(device)
                if k_vals is not None:
                    k_vals = k_vals.to(device)
                
                probs = torch.softmax(model(x, k_vals), dim=-1).cpu()
                for i, row in enumerate(probs):
                    k_val = k_vals[i].item() if k_vals is not None else "unknown"
                    f.write(f"K={k_val}: " + " ".join(f"{v.item():.6f}" for v in row) + "\n")
        print(f"Saved predictions -> {out_pred}")
    
    print(f"Training completed! Results saved to {result_dir}")


if __name__ == "__main__":
    main()

"""
使用示例：

# 方式1: 传统NPZ格式 - 训练动态K值模型（如果已经有多K标签的NPZ文件）
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
  --training_mode all \
  --loss_type kld \
  --experiment_id dynamic_k_v1

# 方式2: 用户数据格式（推荐，用户提供的数据格式）
# 数据文件结构：
#   - queries.npy: [N, D] 查询向量
#   - centroids.npy: [K, D] 质心向量
#   - labels.npz: 包含 targets_k1, targets_k5, targets_k10, targets_k100 等属性的NPZ文件
#     每个属性为 [N, K] 形状,表示对应K值下每个query的最近邻在聚簇中的分布

python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --val_split 0.1 \
  --normalize \
  --topk 10 \
  --epochs 20 \
  --batch_size 128 \
  --lr 5e-4 \
  --weight_decay 5e-3 \
  --d_model 512 \
  --nhead 8 \
  --num_layers 6 \
  --dim_ff 1024 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_type_embed \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --training_mode all \
  --loss_type kld \
  --experiment_id DynamicK_v1
# 说明：
# 1. 每个标签文件应包含 [N, K] 形状的数组,其中N是查询数量,K是聚类数量
# 2. 每行标签与查询向量按顺序一一对应
# 3. 标签表示"每个query的top-K最近邻在所有聚簇中的数量分布"
# 4. 如果标签不是概率分布(行和不为1),系统会自动归一化
# 5. 如果某行全零，会自动设置为均匀分布
"""
