import argparse
import math
import os
import random
import time
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
from torch.utils.data import Subset


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
        mode: str = "random",  # "random", "all", "specific", "subsample"
        subsample_m: int = 1,  # 新增: subsample模式下每个query采样的K值数量
        # 新增参数：支持用户提供的标签NPZ文件
        queries_path: Optional[str] = None, # 支持npz文件和npy文件
        centroids_path: Optional[str] = None, # 支持npz文件和npy文件
        labels_path: Optional[str] = None,  # 包含所有K值标签的NPZ文件路径
    ):
        
        # 如果通过参数提供了centroids，先设置以便_load_npz_data使用
        if centroids is not None:
            self.centroids = centroids.astype(np.float32)
        else:
            self.centroids = None
        
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
        self.subsample_m = subsample_m  # 新增
        
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
            
        # 为 subsample 模式预计算每个 query 的 K 值采样
        if self.mode == "subsample":
            self._precompute_subsample_k_values()
            
        # 根据模式确定数据集大小
        if self.mode == "all":
            self._length = self.N * len(self.k_values)
        elif self.mode == "subsample":
            self._length = self.N * self.subsample_m  # 新增: N * m
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
                    # 测试模式：没有标签数据时创建虚拟标签（全为0）
                    print(f"Warning: No labels found for K={k}. Creating dummy targets for testing.")
                    n_samples = self.queries.shape[0]
                    n_centroids = self.centroids.shape[0] if self.centroids is not None else 100  # 默认假设100个聚类
                    dummy_targets = np.zeros((n_samples, n_centroids), dtype=np.float32)
                    # 创建均匀分布作为虚拟标签
                    dummy_targets[:, :] = 1.0 / n_centroids
                    self.multi_k_targets[k] = dummy_targets
    
    def _precompute_subsample_k_values(self):
        """为 subsample 模式预计算每个 query 的 K 值采样，确保无重复且均衡"""
        if self.subsample_m >= len(self.k_values):
            # 如果 m >= K值总数，等价于 "all" 模式
            print(f"Warning: subsample_m ({self.subsample_m}) >= num_k_values ({len(self.k_values)}), using all K values")
            self.subsample_k_mapping = {}
            for query_idx in range(self.N):
                for m_idx in range(len(self.k_values)):
                    sample_idx = query_idx * len(self.k_values) + m_idx
                    self.subsample_k_mapping[sample_idx] = self.k_values[m_idx]
            self.subsample_m = len(self.k_values)  # 调整 m 值
            self._length = self.N * self.subsample_m  # 重新计算长度
        else:
            # 正常 subsample：为每个 query 无重复地采样 m 个 K 值
            self.subsample_k_mapping = {}
            np.random.seed(42)  # 固定种子确保可重现
            
            for query_idx in range(self.N):
                # 为每个 query 随机采样 m 个不重复的 K 值
                sampled_k_values = np.random.choice(self.k_values, size=self.subsample_m, replace=False)
                
                # 为该 query 的 m 个样本分配 K 值
                for m_idx in range(self.subsample_m):
                    sample_idx = query_idx * self.subsample_m + m_idx
                    self.subsample_k_mapping[sample_idx] = int(sampled_k_values[m_idx])
            
            print(f"Precomputed subsample K values: {self.subsample_m} per query, total samples: {len(self.subsample_k_mapping)}")
            
            # 统计 K 值分布，确保均衡
            k_counts = {}
            for k_val in self.subsample_k_mapping.values():
                k_counts[k_val] = k_counts.get(k_val, 0) + 1
            print(f"K value distribution in subsample: {k_counts}")
    
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
        elif self.mode == "subsample":  # 改进: 使用预计算的 K 值映射
            query_idx = idx // self.subsample_m
            k_value = self.subsample_k_mapping[idx]  # 直接从预计算的映射中获取
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
        k_embed_type: str = "discrete",  # "discrete", "positional", "learned_continuous", "fourier", "attention"
        k_fusion_type: str = "concat",  # "concat", "add", "cross_attention", "gated", "adaptive"
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
        self.k_embed_type = k_embed_type
        self.k_fusion_type = k_fusion_type
        self.pos_exist = pos_exist
        self.use_gating = use_gating
        self.d_model = d_model
        
        # K值嵌入：根据类型选择不同的嵌入策略
        self._init_k_embedding()
        
        # 根据融合方式设置输入投影
        self._init_feature_projection(input_dim)
        
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
    
    def _init_k_embedding(self):
        """初始化不同类型的K值嵌入"""
        if self.k_embed_type == "discrete":
            # 方法1: 离散嵌入 (原有方法)
            self.k_to_idx = {k: i for i, k in enumerate(self.k_values)}
            self.k_embedding = nn.Embedding(len(self.k_values), self.k_embed_dim)
            
        elif self.k_embed_type == "positional":
            # 方法2: 位置编码式嵌入
            self.k_to_idx = {k: i for i, k in enumerate(self.k_values)}
            # 使用sinusoidal位置编码
            self.register_buffer('k_pos_encoding', self._get_positional_encoding())
            
        elif self.k_embed_type == "learned_continuous":
            # 方法3: 学习的连续嵌入
            self.k_proj = nn.Sequential(
                nn.Linear(1, self.k_embed_dim // 2),
                nn.ReLU(),
                nn.Linear(self.k_embed_dim // 2, self.k_embed_dim),
            )
            
        elif self.k_embed_type == "fourier":
            # 方法4: 傅里叶特征嵌入
            self.fourier_dim = self.k_embed_dim // 2
            # 固定的傅里叶基频率
            freqs = torch.exp(torch.linspace(0, torch.log(torch.tensor(10000.0)), self.fourier_dim))
            self.register_buffer('fourier_freqs', freqs)
            
        elif self.k_embed_type == "attention":
            # 方法5: 注意力机制嵌入
            self.k_to_idx = {k: i for i, k in enumerate(self.k_values)}
            self.k_embedding = nn.Embedding(len(self.k_values), self.k_embed_dim)
            self.k_attention = nn.MultiheadAttention(self.k_embed_dim, num_heads=4, batch_first=True)
            # 可学习的查询向量
            self.k_query = nn.Parameter(torch.randn(1, 1, self.k_embed_dim))
            
        else:
            raise ValueError(f"Unknown k_embed_type: {self.k_embed_type}")
    
    def _get_positional_encoding(self):
        """生成sinusoidal位置编码"""
        max_len = len(self.k_values)
        pe = torch.zeros(max_len, self.k_embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, self.k_embed_dim, 2).float() * 
                           (-math.log(10000.0) / self.k_embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe
    
    def _get_k_embedding(self, k_values: torch.Tensor) -> torch.Tensor:
        """根据嵌入类型获取K值嵌入"""
        if self.k_embed_type == "discrete":
            # 离散嵌入
            k_indices = self._k_values_to_indices(k_values)
            return self.k_embedding(k_indices) # [B, k_embed_dim]
            
        elif self.k_embed_type == "positional":
            # 位置编码嵌入
            k_indices = self._k_values_to_indices(k_values)
            return self.k_pos_encoding[k_indices]
            
        elif self.k_embed_type == "learned_continuous":
            # 连续学习嵌入 - 将K值归一化到[0,1]
            k_normalized = (k_values.float() - min(self.k_values)) / (max(self.k_values) - min(self.k_values))
            k_input = k_normalized.unsqueeze(-1)  # [B, 1]
            return self.k_proj(k_input)
            
        elif self.k_embed_type == "fourier":
            # 傅里叶特征嵌入
            k_normalized = k_values.float() / max(self.k_values)  # 归一化到[0,1]
            # 计算sin和cos特征
            angles = k_normalized.unsqueeze(-1) * self.fourier_freqs  # [B, fourier_dim]
            sin_features = torch.sin(angles)
            cos_features = torch.cos(angles)
            return torch.cat([sin_features, cos_features], dim=-1)  # [B, k_embed_dim]
            
        elif self.k_embed_type == "attention":
            # 注意力机制嵌入
            k_indices = self._k_values_to_indices(k_values)
            k_embed = self.k_embedding(k_indices)  # [B, k_embed_dim]
            
            # 使用注意力机制增强K值表示
            batch_size = k_embed.size(0)
            k_query_expanded = self.k_query.expand(batch_size, -1, -1)  # [B, 1, k_embed_dim]
            k_embed_expanded = k_embed.unsqueeze(1)  # [B, 1, k_embed_dim]
            
            enhanced_k, _ = self.k_attention(k_query_expanded, k_embed_expanded, k_embed_expanded)
            return enhanced_k.squeeze(1)  # [B, k_embed_dim]
    
    def _init_feature_projection(self, input_dim: int):
        """根据融合类型初始化特征投影层"""
        if self.k_fusion_type == "concat":
            # 拼接方式：需要为K值嵌入预留空间
            self.feature_proj = nn.Linear(input_dim, self.d_model - self.k_embed_dim)
            self.input_proj = nn.Linear(self.d_model, self.d_model)
            
        elif self.k_fusion_type == "add":
            # 加法方式：特征和K值嵌入维度必须相同
            self.feature_proj = nn.Linear(input_dim, self.d_model)
            self.k_proj = nn.Linear(self.k_embed_dim, self.d_model)  # 将K值嵌入投影到d_model
            
        elif self.k_fusion_type == "cross_attention":
            # 交叉注意力方式
            self.feature_proj = nn.Linear(input_dim, self.d_model)
            self.k_proj = nn.Linear(self.k_embed_dim, self.d_model)
            self.cross_attention = nn.MultiheadAttention(self.d_model, num_heads=8, batch_first=True)
            
        elif self.k_fusion_type == "gated":
            # 门控融合方式
            self.feature_proj = nn.Linear(input_dim, self.d_model)
            self.k_proj = nn.Linear(self.k_embed_dim, self.d_model)
            self.gate_proj = nn.Linear(self.k_embed_dim + self.d_model, self.d_model)
            
        elif self.k_fusion_type == "adaptive":
            # 自适应融合方式
            self.feature_proj = nn.Linear(input_dim, self.d_model)
            self.k_proj = nn.Linear(self.k_embed_dim, self.d_model)
            self.adaptive_weights = nn.Sequential(
                nn.Linear(self.k_embed_dim, self.d_model // 4),
                nn.ReLU(),
                nn.Linear(self.d_model // 4, self.d_model),
                nn.Sigmoid()
            )
            
        else:
            raise ValueError(f"Unknown k_fusion_type: {self.k_fusion_type}")
    
    def _fuse_k_features(self, feat_proj: torch.Tensor, k_embed: torch.Tensor, seqlen: int) -> torch.Tensor:
        """根据融合类型融合特征和K值嵌入"""
        if self.k_fusion_type == "concat":
            # 方法1: 拼接 (原有方式)
            k_embed_expanded = k_embed.unsqueeze(1).repeat(1, seqlen, 1) # [B, M+1, k_embed_dim]    
            combined = torch.cat([feat_proj, k_embed_expanded], dim=-1) # [B, M+1, d_model]
            return self.input_proj(combined)
            
        elif self.k_fusion_type == "add":
            # 方法2: 加法融合
            k_proj = self.k_proj(k_embed)  # [B, d_model]
            k_expanded = k_proj.unsqueeze(1).expand(-1, seqlen, -1)  # [B, M+1, d_model]
            return feat_proj + k_expanded
            
        elif self.k_fusion_type == "cross_attention":
            # 方法3: 交叉注意力融合
            k_proj = self.k_proj(k_embed)  # [B, d_model]
            k_key_value = k_proj.unsqueeze(1)  # [B, 1, d_model]
            
            # 使用特征作为query，K值作为key和value
            enhanced_feat, _ = self.cross_attention(feat_proj, k_key_value, k_key_value) # [B, M+1, d_model]
            return enhanced_feat # [B, M+1, d_model]
            
        elif self.k_fusion_type == "gated": 
            # 方法4: 门控融合
            k_proj = self.k_proj(k_embed)  # [B, d_model]
            k_expanded = k_proj.unsqueeze(1).expand(-1, seqlen, -1)  # [B, M+1, d_model]
            
            # 计算门控权重
            gate_input = torch.cat([
                feat_proj, 
                k_expanded
            ], dim=-1)  # [B, M+1, d_model + d_model]
            
            # 对每个位置计算门控
            gate_input_flat = gate_input.view(-1, gate_input.size(-1))  # [B*M+1, d_model*2]
            gate_weights = self.gate_proj(gate_input_flat)  # [B*M+1, d_model]
            gate_weights = torch.sigmoid(gate_weights)
            gate_weights = gate_weights.view(feat_proj.size())  # [B, M+1, d_model]
            
            return feat_proj * gate_weights + k_expanded * (1 - gate_weights) # [B, M+1, d_model]
            
        elif self.k_fusion_type == "adaptive":
            # 方法5: 自适应融合
            k_proj = self.k_proj(k_embed)  # [B, d_model]
            adaptive_weights = self.adaptive_weights(k_embed)  # [B, d_model]
            
            k_weighted = k_proj * adaptive_weights  # [B, d_model]
            k_expanded = k_weighted.unsqueeze(1).expand(-1, seqlen, -1)  # [B, M+1, d_model]
            
            return feat_proj + k_expanded
            
        else:
            raise ValueError(f"Unknown k_fusion_type: {self.k_fusion_type}")
    
    def _k_values_to_indices(self, k_values: torch.Tensor) -> torch.Tensor:
        """
        高效地将K值批量转换为对应的索引
        
        Args:
            k_values: [B] K值tensor
        Returns:
            k_indices: [B] 对应的索引tensor
        """
        device = k_values.device
        
        # 创建K值到索引的映射字典 (更高效的查找)
        if not hasattr(self, '_k_to_idx_tensor'):
            # 创建一个足够大的查找表，用于O(1)查找
            max_k = max(self.k_values) + 1
            self._k_lookup_table = torch.full((max_k,), -1, device=device, dtype=torch.long)
            for idx, k in enumerate(self.k_values):
                self._k_lookup_table[k] = idx
            self._max_k = max_k
        
        # 确保查找表在正确的设备上
        if self._k_lookup_table.device != device:
            self._k_lookup_table = self._k_lookup_table.to(device)
        
        # 快速批量查找 - O(1)复杂度
        # 首先验证K值范围
        # valid_mask = (k_values >= 0) & (k_values < self._max_k)
        # if not valid_mask.all():
        #     invalid_k = k_values[~valid_mask]
        #     raise ValueError(f"Invalid K values: {invalid_k.tolist()}")
        
        indices = self._k_lookup_table[k_values]
        
        # # 检查是否有未找到的K值（索引为-1）
        # invalid_mask = indices == -1
        # if invalid_mask.any():
        #     invalid_k = k_values[invalid_mask]
        #     raise ValueError(f"K values not found in supported list: {invalid_k.unique().tolist()}, supported: {self.k_values}")
        
        return indices
    
    def forward(self, x: torch.Tensor, k_values: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: [B, M+1, D] 输入特征,其中M是聚类数量
            k_values: [B] K值,必须提供
        Returns:
            logits: [B, M] 预测的聚类分布logits,M是聚类数量
        """
        bsz, seqlen, feat_dim = x.size() # seqlen=M+1
        
        # K值现在必须从外部提供
        if k_values is None:
            raise ValueError("k_values must be provided")
        
        # 获取K值嵌入 - 使用统一接口
        k_embed = self._get_k_embedding(k_values)  # [B, k_embed_dim]
        
        # 投影原始特征
        feat_proj = self.feature_proj(x)  # [B, M+1, feature_dim]
        
        # 融合特征和K值嵌入 - 使用可选择的融合策略
        t = self._fuse_k_features(feat_proj, k_embed, seqlen)  # [B, M+1, d_model]
        
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


def stratified_split_by_k(dataset, val_split: float, k_values: List[int], seed: int = 42):
    """
    按K值进行分层抽样划分数据集，确保每个K值在训练/验证集中比例均衡
    
    Args:
        dataset: DynamicKNPZClusterDataset in 'all' mode
        val_split: 验证集比例
        k_values: K值列表
        seed: 随机种子
    
    Returns:
        train_ds, val_ds: 训练集和验证集
    """
    print(f"Using stratified split by K values to ensure balanced distribution")
    
    # 获取数据集信息
    n_queries = dataset.N  # 原始query数量
    n_k_values = len(k_values)
    
    # 计算每个K值应该分配的验证样本数
    samples_per_k = n_queries  # 每个K值有n_queries个样本
    val_samples_per_k = int(samples_per_k * val_split)  # 每个K值对应的验证样本数
    train_samples_per_k = samples_per_k - val_samples_per_k  # 每个K值对应的训练样本数
    
    # 设置随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    train_indices = []
    val_indices = []
    
    # 对每个K值分别进行划分
    for k_idx, k_value in enumerate(k_values):
        # 计算当前K值对应的样本索引范围
        # 在'all'模式下，索引规律：query_i的k_value_j对应的全局索引 = i * len(k_values) + j
        k_start_indices = [i * n_k_values + k_idx for i in range(n_queries)]
        
        # 随机打乱当前K值的样本索引
        k_indices_shuffled = k_start_indices.copy()
        random.shuffle(k_indices_shuffled)
        
        # 划分训练集和验证集索引
        k_val_indices = k_indices_shuffled[:val_samples_per_k]
        k_train_indices = k_indices_shuffled[val_samples_per_k:]
        
        train_indices.extend(k_train_indices)
        val_indices.extend(k_val_indices)
        
        print(f"  K={k_value}: train={len(k_train_indices)}, val={len(k_val_indices)}")
    
    # 打乱最终的训练集和验证集索引（保持内部平衡的同时增加随机性）
    random.shuffle(train_indices)
    random.shuffle(val_indices)
    
    # 创建Subset数据集
    train_ds = Subset(dataset, train_indices)
    val_ds = Subset(dataset, val_indices)
    
    print(f"Stratified split completed: train_size={len(train_indices)}, val_size={len(val_indices)}")
    print(f"Total samples per K-value: train={train_samples_per_k}, val={val_samples_per_k}")
    
    return train_ds, val_ds


def get_k_batch_iterator(train_loader: DataLoader, k_values: List[int]):
    """
    创建按K值批次训练的迭代器
    
    Args:
        train_loader: 训练数据加载器
        k_values: K值列表，按此顺序进行训练
    
    Returns:
        按K值顺序组织的批次迭代器
    """
    # 收集所有训练数据并按K值分组
    k_batches = {k: [] for k in k_values}
    
    print("Collecting and grouping training data by K values...")
    for batch_data in train_loader:
        if len(batch_data) == 3:
            x, y, k_vals = batch_data
            
            # 按K值将当前批次的样本分组
            for k in k_values:
                # 找到当前批次中属于K值k的样本
                k_mask = (k_vals == k)
                if k_mask.sum() > 0:
                    k_x = x[k_mask]
                    k_y = y[k_mask] 
                    k_k_vals = k_vals[k_mask]
                    k_batches[k].append((k_x, k_y, k_k_vals))
    
    # 按K值顺序生成批次
    for k in k_values:
        if k_batches[k]:
            print(f"  Training K={k}: {len(k_batches[k])} batches")
            for batch_data in k_batches[k]:
                yield batch_data


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
    subsample_m: int = 1,  # 新增: 传递subsample_m
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
        subsample_m: 新增: 传递subsample_m
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
            subsample_m=subsample_m,  # 新增
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
                subsample_m=subsample_m,  # 新增
                queries_path=val_queries_path,
                centroids_path=centroids_path,
                labels_path=val_labels_path
            )
        elif val_split > 0:
            # 从训练集分割验证集 - 使用分层抽样确保K值分布均衡
            if mode == "all":
                # 分层抽样：确保每个K值在训练/验证集中比例均衡
                train_ds, val_ds = stratified_split_by_k(full_train, val_split, k_values, seed)
            else:
                # 随机模式下使用原来的划分方式
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
            mode=mode,
            subsample_m=subsample_m  # 新增
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
                mode=mode,
                subsample_m=subsample_m  # 新增
            )
        
        # 返回原始完整数据集用于获取元信息
        sample_dataset = full_train
    
    # 创建数据加载器 - 优化版本
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True, 
                            prefetch_factor=2, persistent_workers=True if num_workers > 0 else False)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, 
                                                      shuffle=False, num_workers=num_workers, pin_memory=True,
                                                      prefetch_factor=2, persistent_workers=True if num_workers > 0 else False)
    
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
    k_batch_training: bool = False,  # 新增：是否按K值批次进行训练
    k_values: Optional[List[int]] = None,  # K值列表，用于k_batch_training
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
        
        # 选择训练方式：按K值批次训练 vs 随机训练
        if k_batch_training and k_values is not None:
            # 按K值批次训练：每个epoch内先训练所有K=1样本，再训练K=5样本等
            print(f"[Epoch {ep}] Using K-batch training mode (K values in order: {k_values})")
            train_iterator = get_k_batch_iterator(train_loader, k_values)
        else:
            # 传统随机训练
            train_iterator = train_loader
        
        for batch_data in train_iterator:
            if len(batch_data) == 3:
                x, y, k_vals = batch_data  # x:[B,M+1,D] y：[B, M] k_vals:[B]
            else:
                x, y = batch_data
                k_vals = None
            
            x = x.to(device)
            y = y.to(device)
            if k_vals is not None:
                k_vals = k_vals.to(device)
                
                # 统计K值分布 - 批处理版本
                unique_k, counts = torch.unique(k_vals.cpu(), return_counts=True)
                for k_val, count in zip(unique_k.numpy(), counts.numpy()):
                    k_val = int(k_val)
                    k_value_counts[k_val] = k_value_counts.get(k_val, 0) + int(count)
            
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
            
            # Epoch 1 batch metrics - 优化版本
            if ep == 1 and val_loader is not None:
                batch_count += 1
                if batch_count % 10 == 0:
                    # 使用现有的批量评估函数，更高效
                    val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate_dynamic_k(
                        model, val_loader, device, topk=eval_topk, loss_type=loss_type
                    )
                    
                    epoch1_batch_metrics.append({
                        'batch': batch_count,
                        'val_kld': val_kld,
                        'val_mae': val_mae,
                        'val_mse': val_mse,
                        'val_ordacc': val_ordacc,
                        'val_recall': val_recall,
                        'train_kld': loss.item(),
                        'train_ordacc': train_ordacc,
                        'train_recall': train_recall,
                    })
                    # 显示当前batch的K值分布
                    if k_vals is not None:
                        current_k_counts = {}
                        unique_k, counts = torch.unique(k_vals.cpu(), return_counts=True)
                        for k_val, count in zip(unique_k.numpy(), counts.numpy()):
                            current_k_counts[int(k_val)] = int(count)
                        k_dist_str = ", ".join([f"K{k}={count}" for k, count in sorted(current_k_counts.items())])
                        print(f"Epoch {ep:03d} | Batch {batch_count} | val_kld={val_kld:.6f} | val_ordacc@{eval_topk}={val_ordacc:.6f} | val_recall@{eval_topk}={val_recall:.6f}")
                        print(f"Epoch {ep:03d} | Batch {batch_count} | K-values: {k_dist_str}")
                    else:
                        print(f"Epoch {ep:03d} | Batch {batch_count} | val_kld={val_kld:.6f} | val_ordacc@{eval_topk}={val_ordacc:.6f} | val_recall@{eval_topk}={val_recall:.6f}")
            
        
        sched.step()
        train_loss = running / n
        train_ordacc = running_ordacc / n
        train_recall = running_recall / n
        
        # 输出K值分布统计
        if k_value_counts:
            k_dist_str = ", ".join([f"K{k}={count}" for k, count in sorted(k_value_counts.items())])
            print(f"[Epoch {ep}] train_loss={train_loss:.6f} | train_ordacc={train_ordacc:.6f} | train_recall={train_recall:.6f}")
            print(f"[Epoch {ep}] K-value distribution: {k_dist_str}")
        else:
            print(f"[Epoch {ep}] train_loss={train_loss:.6f} | train_ordacc={train_ordacc:.6f} | train_recall={train_recall:.6f}")
        
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
    p.add_argument("--k_embed_type", type=str, default="discrete", 
                   choices=["discrete", "positional", "learned_continuous", "fourier", "attention"],
                   help="K value embedding type")
    p.add_argument("--k_fusion_type", type=str, default="concat",
                   choices=["concat", "add", "cross_attention", "gated", "adaptive"],
                   help="K value fusion type")
    p.add_argument("--training_mode", type=str, default="all", choices=["random", "all", "subsample"],
                   help="Training mode: all (all K values per query), random (random K per sample), or subsample (m K values per query)")
    p.add_argument("--k_batch_training", action="store_true",
                   help="Enable K-batch training: train all K=1 samples first, then K=5, etc. (within each epoch)")
    p.add_argument("--subsample_m", type=int, default=1, 
                   help="Number of K values to sample per query in 'subsample' mode (1 = random mode equivalent)")
    
    # 训练参数
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (max norm)")
    p.add_argument("--num_workers", type=int, default=8, help="DataLoader workers")
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
    p.add_argument("--no_compile", action="store_true", help="Disable torch.compile optimization")
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
    # ========== 程序总时间记录开始 ==========
    program_start_time = time.time()
    
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
    
    # 构建数据加载器：每条数据组成（query+centriods, label, k_values）
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
        subsample_m=args.subsample_m,  # 新增
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
    print(f"\nTraining Configuration:")
    print(f"  Original queries: {original_queries}")
    print(f"  K values: {args.k_values}")
    print(f"  Training mode: {args.training_mode}")
    if args.training_mode == "subsample":
        print(f"  Subsample m: {args.subsample_m} (each query uses {args.subsample_m} K values)")
    print(f"  Dataset size: {dataset_size}")
    if args.training_mode == "all":
        expected_size = original_queries * len(args.k_values)
        print(f"  Expected size (N*K): {original_queries} * {len(args.k_values)} = {expected_size}")
        assert dataset_size == expected_size, f"Dataset size mismatch: {dataset_size} != {expected_size}"
        print(f"  Each query has {len(args.k_values)} training samples (one per K value)")
    elif args.training_mode == "subsample":
        expected_size = original_queries * args.subsample_m
        print(f"  Expected size (N*m): {original_queries} * {args.subsample_m} = {expected_size}")
        print(f"  Each query has {args.subsample_m} randomly sampled K values")
    else:
        print(f"  Random mode: each query trains only one random K value per epoch")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Batches per epoch: {len(train_loader)}")
    
    # 如果是subsample模式，显示K值采样统计
    if args.training_mode == "subsample" and hasattr(sample_dataset, 'subsample_k_mapping'):
        k_counts = {}
        for k_val in sample_dataset.subsample_k_mapping.values():
            k_counts[k_val] = k_counts.get(k_val, 0) + 1
        print(f"Subsample K-value distribution:")
        total_samples = sum(k_counts.values())
        for k_val in sorted(k_counts.keys()):
            count = k_counts[k_val]
            percentage = (count / total_samples) * 100 if total_samples > 0 else 0
            print(f"     K={k_val}: {count} samples ({percentage:.1f}%)")
        print(f"     Total: {total_samples} samples")
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
        k_embed_type=args.k_embed_type,
        k_fusion_type=args.k_fusion_type,
    ).to(device)
    
    # 如果使用PyTorch 2.0+，启用编译优化（除非被禁用）
    if not args.no_compile:
        try:
            if hasattr(torch, 'compile') and torch.cuda.is_available():
                print("Enabling torch.compile optimization...")
                model = torch.compile(model, mode="reduce-overhead")
        except Exception as e:
            print(f"torch.compile not available or failed: {e}")
    else:
        print("torch.compile optimization disabled by --no_compile flag")
    
    # 设置保存路径
    ckpt_dir = os.path.dirname(args.centroids_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    k_values_str = "_".join(map(str, args.k_values))
    ckpt_name = (
        f"dynamic_k_{k_values_str}_d{args.d_model}_L{args.num_layers}_"
        f"bs{args.batch_size}_ep{args.epochs}_{args.score_type}_{args.loss_type}_{args.training_mode}_{args.subsample_m}.pt"
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
            k_batch_training=args.k_batch_training,
            k_values=args.k_values,
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
            npz_path = args.test_npz, 
            normalize=args.normalize, 
            mean=mean, 
            std=std, 
            centroids=C,
            k_values=args.k_values,
            mode="all"  # 测试时使用所有K值
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
        
        # 为每个K值分别评估 - 高效批处理版本
        print("\n" + "="*60)
        print("Starting test evaluation for each K value:")
        print("="*60)
        
        # 使用DataLoader直接处理，避免预先收集所有样本
        from collections import defaultdict
        k_metrics = defaultdict(lambda: {'kld': [], 'mae': [], 'mse': [], 'ordacc': [], 'recall': [], 'count': 0})
        
        # 统计测试数据中的K值分布
        test_k_distribution = defaultdict(int)
        
        # 批量处理测试数据，同时收集每个K值的指标
        model.eval()
        with torch.no_grad():
            for batch_data in test_loader:
                if len(batch_data) == 3:
                    x, y, k_vals = batch_data
                else:
                    x, y = batch_data
                    continue  # 跳过没有K值的batch
                
                x = x.to(device)
                y = y.to(device)
                k_vals = k_vals.to(device)
                
                # 获得批量预测
                logits = model(x, k_vals)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 按K值分组计算指标
                unique_k = torch.unique(k_vals)
                for k_val in unique_k:
                    mask = (k_vals == k_val)
                    if mask.sum() == 0:
                        continue
                    
                    k_int = int(k_val.item())
                    if k_int not in args.k_values:
                        continue
                    
                    # 统计K值分布
                    test_k_distribution[k_int] += mask.sum().item()
                    
                    # 提取当前K值的样本
                    k_probs = probs[mask]
                    k_y = y[mask]
                    k_log_probs = log_probs[mask]
                    
                    # 计算指标
                    kldiv = nn.KLDivLoss(reduction="batchmean")
                    batch_kld = kldiv(k_log_probs, k_y).item()
                    batch_mae = torch.mean(torch.abs(k_probs - k_y)).item()
                    batch_mse = torch.mean((k_probs - k_y) ** 2).item()
                    batch_ordacc = topk_ordered_accuracy(k_probs, k_y, k=args.topk)
                    batch_recall = topk_recall(k_probs, k_y, k=args.topk)
                    
                    # 累积指标
                    batch_size = k_probs.size(0)
                    k_metrics[k_int]['kld'].append(batch_kld * batch_size)
                    k_metrics[k_int]['mae'].append(batch_mae * batch_size)
                    k_metrics[k_int]['mse'].append(batch_mse * batch_size)
                    k_metrics[k_int]['ordacc'].append(batch_ordacc * batch_size)
                    k_metrics[k_int]['recall'].append(batch_recall * batch_size)
                    k_metrics[k_int]['count'] += batch_size
        
        # 首先打印测试数据的K值分布统计
        print("\nTest data K-value distribution:")
        total_test_samples = sum(test_k_distribution.values())
        for k_val in sorted(test_k_distribution.keys()):
            count = test_k_distribution[k_val]
            percentage = (count / total_test_samples) * 100 if total_test_samples > 0 else 0
            print(f"   K={k_val}: {count} samples ({percentage:.1f}%)")
        print(f"   Total: {total_test_samples} samples")
        
        # 计算并打印每个K值的平均指标
        print("\nTest results for each K value:")
        for k_val in sorted(args.k_values):
            if k_val in k_metrics and k_metrics[k_val]['count'] > 0:
                metrics = k_metrics[k_val]
                total_count = metrics['count']
                
                avg_kld = sum(metrics['kld']) / total_count
                avg_mae = sum(metrics['mae']) / total_count
                avg_mse = sum(metrics['mse']) / total_count
                avg_ordacc = sum(metrics['ordacc']) / total_count
                avg_recall = sum(metrics['recall']) / total_count
                
                print(f"[TEST K={k_val}] kld={avg_kld:.6f} | mae={avg_mae:.6f} | mse={avg_mse:.6f} | ord_acc@{args.topk}={avg_ordacc:.4f} | recall@{args.topk}={avg_recall:.4f} | samples={total_count}")
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

    # ========== 程序总时间记录结束 ==========
    program_end_time = time.time()
    total_program_time_seconds = program_end_time - program_start_time
    
    # 转换为时分秒格式
    hours = int(total_program_time_seconds // 3600)
    minutes = int((total_program_time_seconds % 3600) // 60)
    seconds = int(total_program_time_seconds % 60)
    
    print(f"Total program execution time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_program_time_seconds:.2f} seconds)")


if __name__ == "__main__":
    main()

"""
使用示例：

# 用户数据格式（推荐，用户提供的数据格式）
# 数据文件结构：
#   - queries.npy: [N, D] 查询向量
#   - centroids.npy: [K, D] 质心向量
#   - labels.npz: 包含 targets_k1, targets_k5, targets_k10, targets_k100 等属性的NPZ文件
#     每个属性为 [N, K] 形状,表示对应K值下每个query的最近邻在聚簇中的分布
'''



'''
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/siftsmall/siftsmall5H/siftsmall_learn.npz \
  --centroids_path input/Training_data/siftsmall/siftsmall5H/leaf_center.npz \
  --train_labels_path input/Training_data/siftsmall/siftsmall5H/learn/train_labels.npz \
  --test_npz input/Training_data/siftsmall/siftsmall5H/siftsmall_query.npz \
  --val_split 0.1 \
  --normalize \
  --topk 10 \
  --epochs 3 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 6 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 10 \
  --pos_exist \
  --use_type_embed \
  --use_gating \
  --k_values 1 10 20 50 100 \
  --k_embed_dim 32 \
  --k_embed_type discrete \
  --k_fusion_type add \
  --training_mode all \
  --loss_type kld \
  --experiment_id Batch_DynamicK
'''


'''
# gist1M_learn
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.dynamic_k.cluster_dist_transformer_dynamic_k \
  --train_queries_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --train_labels_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_labels.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz \
  --val_split 0.1 \
  --normalize \
  --topk 30 \
  --epochs 2 \
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
  --k_embed_dim 16 \
  --k_embed_type discrete \
  --k_fusion_type add \
  --loss_type kld \
  --training_mode subsample \
  --subsample_m 1 \
  --experiment_id Subsample
'''


'''
# 说明：
# 1. 每个标签文件应包含 [N, K] 形状的数组,其中N是查询数量,K是聚类数量
# 2. 每行标签与查询向量按顺序一一对应
# 3. 标签表示"每个query的top-K最近邻在所有聚簇中的数量分布"
# 4. 如果标签不是概率分布(行和不为1),系统会自动归一化
# 5. 如果某行全零，会自动设置为均匀分布
# 
# 关于数据划分和训练策略的改进：
# 1. **分层抽样数据划分**: 自动确保每个K值在训练/验证集中比例均衡
#    - 在 training_mode="all" 时自动启用
#    - 每个K值都有相同比例的样本进入验证集
#    - 避免验证指标因K值分布不均而产生偏差
#
# 2. **按K值批次训练**: 使用 --k_batch_training 参数启用
#    - 每个epoch内按K值顺序训练：K=1 -> K=5 -> K=10 -> K=50 -> K=100
#    - 有助于模型学习不同K值的渐进特征
#    - 可能提高收敛速度和最终性能
#
# 3. **建议的最佳实践**:
#    - 使用 --training_mode all (确保所有K值都被训练)
#    - 结合 --k_batch_training (按K值顺序训练)
#    - 监控每个K值的验证指标确保均衡学习
"""
