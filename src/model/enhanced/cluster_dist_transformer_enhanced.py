import argparse
import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.model.plot import plot_epoch1_batch_metrics, plot_epoch_metrics
from src.model.allLoss import *

# 导入原始函数
from src.model.cluster_dist_transformer_original import (
    set_seed, load_centroids, make_synth_dataset, save_npz,
    topk_recall, topk_ordered_accuracy, PositionalEncoding,
    evaluate_batch, evaluate, train
)


class EnhancedNPZClusterDataset(Dataset):
    """增强版聚类数据集"""
    
    def __init__(
        self,
        npz_path: str,
        normalize: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None,
        cluster_info_path: Optional[str] = None,
        enhancement_type: str = "none",
        num_representatives: int = 3,
    ):
        data = np.load(npz_path) #（query,target）
        
        if "queries" not in data:
            raise ValueError(f"{npz_path} must contain 'queries' [N, D].")
        tgt = data.get("targets", data.get("labels", None))
        if tgt is None:
            raise ValueError(f"{npz_path} must contain 'targets' [N, K] (or 'labels').")

        self.queries = data["queries"].astype(np.float32)   # [N, D]
        self.targets = tgt.astype(np.float32)               # [N, K]

        if centroids is None:
            raise ValueError("centroids is required (global [K, D]); provide via --centroids_path.")
        self.centroids = centroids.astype(np.float32)       # [K, D]
        
        # 增强参数
        self.enhancement_type = enhancement_type
        self.num_representatives = num_representatives  # 代表向量的个数
        self.cluster_info_path = cluster_info_path

        # 形状检查
        N, D = self.queries.shape
        Nk, K = self.targets.shape
        Kc, Dc = self.centroids.shape
        assert N == Nk, f"N mismatch: queries={N}, targets={Nk}"
        assert K == Kc, f"K mismatch: targets={K}, centroids={Kc}"
        assert D == Dc, f"D mismatch: queries={D}, centroids={Dc}"
        
        self.N, self.D, self.K = N, D, K

        # 加载聚类附加信息
        self.cluster_info = self._load_cluster_info()
        
        # 数据标准化 (QUERY, CENTROIDS)
        if normalize:
            self._normalize_data(mean, std)
        else:
            self.norm_stats = None
            
        # 准备增强特征
        self._prepare_enhanced_features()

    def _load_cluster_info(self):
        """加载聚类附加信息"""
        if self.cluster_info_path is None or not os.path.exists(self.cluster_info_path):
            print(f"Warning: cluster_info_path not provided or not found. Using default values.")
            return self._generate_default_cluster_info()
        
        try:
            data = np.load(self.cluster_info_path, allow_pickle=True)
            if isinstance(data, np.lib.npyio.NpzFile):
                return {key: data[key] for key in data.files}
            else:
                return data.item() if data.ndim == 0 else data
        except Exception as e:
            print(f"Error loading cluster info: {e}. Using default values.")
            return self._generate_default_cluster_info()
    
    def _generate_default_cluster_info(self):
        """生成默认聚类信息"""
        K, D = self.K, self.D
        return {
            'cluster_sizes': np.ones(K, dtype=np.float32),
            'cluster_variances': np.ones(K, dtype=np.float32),
            'cluster_densities': np.ones(K, dtype=np.float32),
            'intra_distances': np.ones(K, dtype=np.float32)
        }

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
    
    def _prepare_enhanced_features(self):
        """准备增强特征"""
        if self.enhancement_type == "stats":
            self._prepare_stats_features()
        elif self.enhancement_type == "multi_rep":
            self._prepare_multi_rep_features()
        elif self.enhancement_type == "distance":
            self.distance_mode = True
        elif self.enhancement_type == "comprehensive":
            self._prepare_stats_features()
            self._prepare_multi_rep_features()
            self.distance_mode = True
        else:
            self.distance_mode = False
    
    def _prepare_stats_features(self):
        """准备统计特征"""
        stats_list = []
        
        # 聚类大小（归一化）
        sizes = self.cluster_info['cluster_sizes'].astype(np.float32)
        sizes = sizes / (sizes.sum() + 1e-6)
        stats_list.append(sizes.reshape(-1, 1))
        
        # 其他统计特征（标准化）
        for key in ['cluster_densities', 'cluster_variances', 'intra_distances']:
            if key in self.cluster_info:
                values = self.cluster_info[key].astype(np.float32)
                values = (values - values.mean()) / (values.std() + 1e-6)
                stats_list.append(values.reshape(-1, 1))
        
        self.cluster_stats = np.concatenate(stats_list, axis=1)  # [K, num_stats=4]
    
    def _prepare_multi_rep_features(self):
        """准备多代表向量特征（使用质心的噪声变体）"""
        # 由于没有representative_vectors，使用质心的噪声变体来模拟多代表向量
        np.random.seed(42)  # 保证可重复性
        self.representative_vectors = np.zeros((self.K, self.num_representatives, self.D), dtype=np.float32)
        
        for k in range(self.K):
            for rep in range(self.num_representatives):
                if rep == 0:
                    # 第一个代表向量就是质心本身
                    self.representative_vectors[k, rep] = self.centroids[k]
                else:
                    # 其他代表向量是质心的轻微噪声变体
                    noise_scale = 0.1 * np.std(self.centroids[k])
                    noise = np.random.normal(0, noise_scale, self.D).astype(np.float32)
                    self.representative_vectors[k, rep] = self.centroids[k] + noise
    
    def _compute_distance_features(self, query: np.ndarray) -> np.ndarray:
        """计算距离特征"""
        # 欧几里得距离
        euclidean_dists = np.linalg.norm(self.centroids - query[None, :], axis=1) # [K]
        
        # 余弦相似度
        query_norm = np.linalg.norm(query) # [1]
        centroids_norm = np.linalg.norm(self.centroids, axis=1) # [K]   
        cosine_sim = np.dot(self.centroids, query) / (centroids_norm * query_norm + 1e-8) # [K] 
        
        # L1距离
        l1_dists = np.sum(np.abs(self.centroids - query[None, :]), axis=1) # [K]
        
        # 排序特征
        euclidean_ranks = np.argsort(np.argsort(euclidean_dists)).astype(np.float32)
        cosine_ranks = np.argsort(np.argsort(-cosine_sim)).astype(np.float32)
        
        # 拼接并标准化
        features = np.column_stack([
            euclidean_dists, cosine_sim, l1_dists, euclidean_ranks, cosine_ranks
        ])
        features = (features - features.mean(axis=0)) / (features.std(axis=0) + 1e-6)
        
        return features.astype(np.float32)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        query = self.queries[idx]  # [D]
        target = self.targets[idx]  # [K]
        
        if self.enhancement_type == "none":
            # 原始方法
            x = np.vstack([query[None, :], self.centroids])  # [K+1, D]
            
        elif self.enhancement_type == "stats":
            # 统计信息增强
            enhanced_centroids = np.concatenate([self.centroids, self.cluster_stats], axis=1) # [K, D+4]
            query_stats = np.zeros(self.cluster_stats.shape[1], dtype=np.float32) # [4]
            enhanced_query = np.concatenate([query, query_stats]) # [D+4]
            x = np.vstack([enhanced_query[None, :], enhanced_centroids])    # [K+1, D+4]
            
        elif self.enhancement_type == "multi_rep":
            # 多代表向量
            rep_vectors = self.representative_vectors.reshape(-1, self.D)  # [K*num_rep, D]
            x = np.vstack([query[None, :], rep_vectors])  # [1+K*num_rep, D]
            
        elif self.enhancement_type == "distance":
            # 距离特征增强
            distance_features = self._compute_distance_features(query) # [K，5]
            enhanced_centroids = np.concatenate([self.centroids, distance_features], axis=1)
            query_dist_features = np.zeros(distance_features.shape[1], dtype=np.float32)
            enhanced_query = np.concatenate([query, query_dist_features])
            x = np.vstack([enhanced_query[None, :], enhanced_centroids])
            
        elif self.enhancement_type == "comprehensive":
            # 综合增强
            distance_features = self._compute_distance_features(query)
            enhanced_centroids = np.concatenate([
                self.centroids, self.cluster_stats, distance_features
            ], axis=1)
            query_extra_features = np.zeros(
                self.cluster_stats.shape[1] + distance_features.shape[1], 
                dtype=np.float32
            )
            enhanced_query = np.concatenate([query, query_extra_features])
            x = np.vstack([enhanced_query[None, :], enhanced_centroids])
        
        return x.astype(np.float32), target.astype(np.float32)


class EnhancedClusterDistTransformer(nn.Module):
    """增强版聚类分布Transformer"""
    
    def __init__(
        self,
        input_dim: int,
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
        enhancement_type: str = "none",
        num_representatives: int = 3,
    ):
        super().__init__()
        self.enhancement_type = enhancement_type
        self.num_representatives = num_representatives
        self.pos_exist = pos_exist
        self.use_gating = use_gating
        self.d_model = d_model
        
        self.input_proj = nn.Linear(input_dim, d_model)
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
            self.query_head = nn.Linear(d_model, d_model, bias=False)
            self.centroid_head = nn.Linear(d_model, d_model, bias=False)
        elif score_type == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
        else:
            raise ValueError("score_type must be 'bilinear' or 'mlp'")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = x.size()
        t = self.input_proj(x)  # [B, L, d_model]
        
        if self.type_embed is not None:
            type_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=x.device)
            if seqlen > 1:
                type_ids[:, 1:] = 1
            t = t + self.type_embed(type_ids)
        
        if self.pos_exist:
            t = self.pos_enc(t)
        
        h = self.encoder(t)  # [B, L, d_model]
        
        if self.use_gating:
            gate = self.gate_activation(self.gate_linear(h))
            h = h * gate
        
        q = h[:, 0, :]  # [B, d_model]
        
        if self.enhancement_type == "multi_rep":
            # 处理多代表向量
            cluster_reps = h[:, 1:, :].view(bsz, -1, self.num_representatives, self.d_model)
            c = cluster_reps.mean(dim=2)  # [B, K, d_model]
        else:
            c = h[:, 1:, :]  # [B, K, d_model]
        
        if self.score_type == "bilinear":
            qh = self.query_head(q)
            ch = self.centroid_head(c)
            logits = torch.einsum("bd,bkd->bk", qh, ch) / math.sqrt(self.d_model)
        else:
            q_expand = q.unsqueeze(1).expand(-1, c.size(1), -1)
            logits = self.scorer(torch.cat([q_expand, c], dim=-1)).squeeze(-1)
        
        return logits


def build_enhanced_loaders(
    train_npz: str,
    val_npz: Optional[str],
    centroids_path: str,
    cluster_info_path: Optional[str],
    enhancement_type: str,
    batch_size: int,
    num_workers: int,
    normalize: bool,
    val_split: float,
    seed: int,
    num_representatives: int = 3,
):
    """构建增强版数据加载器"""
    centroids = load_centroids(centroids_path)
    
    full_train = EnhancedNPZClusterDataset(
        train_npz, 
        normalize=normalize, 
        centroids=centroids,
        cluster_info_path=cluster_info_path,
        enhancement_type=enhancement_type,
        num_representatives=num_representatives
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
        val_ds = EnhancedNPZClusterDataset(
            val_npz, 
            normalize=normalize, 
            mean=mean, 
            std=std, 
            centroids=centroids,
            cluster_info_path=cluster_info_path,
            enhancement_type=enhancement_type,
            num_representatives=num_representatives
        )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                            num_workers=num_workers, pin_memory=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, 
                                                      shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, full_train


def calculate_input_dim(enhancement_type: str, original_dim: int) -> int:
    """根据增强类型计算输入维度"""
    if enhancement_type == "none":
        return original_dim
    elif enhancement_type == "stats":
        return original_dim + 4  # +4 statistical features
    elif enhancement_type == "distance":
        return original_dim + 5  # +5 distance features
    elif enhancement_type == "comprehensive":
        return original_dim + 9  # +4 stats + 5 distance
    elif enhancement_type == "multi_rep":
        return original_dim  # dimension unchanged, sequence length changes to use synthetic representatives
    else:
        raise ValueError(f"Unknown enhancement_type: {enhancement_type}")


def parse_args():
    """解析命令行参数"""
    p = argparse.ArgumentParser(description="Enhanced Transformer for predicting NN cluster distribution")
    
    # 基础参数（从原始文件复制）
    p.add_argument("--train_npz", type=str, default=None, help="Training npz (contains queries and targets)")
    p.add_argument("--val_npz", type=str, default=None, help="Validation npz (optional)")
    p.add_argument("--test_npz", type=str, default=None, help="Test npz (optional)")
    p.add_argument("--centroids_path", type=str, default=None, help="Global shared centroids (.npy or .npz), shape [K, D]")
    p.add_argument("--val_split", type=float, default=0.1, help="Split ratio from training set when no val_npz is provided")
    p.add_argument("--normalize", action="store_true", help="Apply feature-wise standardization")
    
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
    p.add_argument("--loss_type", type=str,
                   choices=["kld","kld_reverse","mse","hybrid","recall_focused_loss",
                            "listnet","listmle","pairwise_hinge"], default="kld")
    p.add_argument("--listmle_topm", type=int, default=None, help="ListMLE 仅用前 m 个目标")
    p.add_argument("--listnet_pred_temp", type=float, default=1.0)
    p.add_argument("--listnet_tgt_temp", type=float, default=None)
    p.add_argument("--pair_num_pos", type=int, default=1)
    p.add_argument("--pair_num_neg", type=int, default=20)
    p.add_argument("--pair_margin", type=float, default=0.1)

    # ===== 新增：实验标识符 =====
    p.add_argument("--experiment_id", type=str, default=None, 
                   help="Experiment identifier prefix for result directories (e.g., 'A', 'exp1', 'batch1')")
    
    # ===== 增强功能参数 =====
    p.add_argument("--enhancement_type", type=str, default="none", 
                   choices=["none", "stats", "multi_rep", "distance", "comprehensive"],
                   help="Enhancement type for cluster representation")
    p.add_argument("--cluster_info_path", type=str, default=None, 
                   help="Path to cluster additional information file (.npz)")
    p.add_argument("--num_representatives", type=int, default=3,
                   help="Number of synthetic representative vectors per cluster (for multi_rep mode)")
    
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if args.train_npz is None:
        raise ValueError("You must provide --train_npz")
    if args.centroids_path is None:
        raise ValueError("You must provide --centroids_path")
    
    # 加载质心以确定维度
    C = load_centroids(args.centroids_path)
    K, D = C.shape
    print(f"Detected K={K}, D={D}")
    
    # 根据增强类型计算输入维度
    input_dim = calculate_input_dim(args.enhancement_type, D)
    print(f"Enhancement type: {args.enhancement_type}")
    print(f"Input dimension: {input_dim} (original: {D})")
    
    # 构建数据加载器
    train_loader, val_loader, sample_dataset = build_enhanced_loaders(
        args.train_npz,
        args.val_npz,
        args.centroids_path,
        args.cluster_info_path,
        args.enhancement_type,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        num_representatives=args.num_representatives,
    )
    
    # 创建模型
    model = EnhancedClusterDistTransformer(
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
        enhancement_type=args.enhancement_type,
        num_representatives=args.num_representatives,
    ).to(device)
    
    # 设置保存路径
    ckpt_dir = os.path.dirname(args.centroids_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    ckpt_name = (
        f"enhanced_{args.enhancement_type}_d{args.d_model}_L{args.num_layers}_"
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
        train(
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
            listmle_topm=args.listmle_topm,
            listnet_pred_temp=args.listnet_pred_temp,
            listnet_tgt_temp=args.listnet_tgt_temp,
            pair_num_pos=args.pair_num_pos,
            pair_num_neg=args.pair_num_neg,
            pair_margin=args.pair_margin
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
        if sample_dataset.norm_stats is not None:
            mean, std = sample_dataset.norm_stats
            
        test_ds = EnhancedNPZClusterDataset(
            args.test_npz, 
            normalize=args.normalize, 
            mean=mean, 
            std=std, 
            centroids=C,
            cluster_info_path=args.cluster_info_path,
            enhancement_type=args.enhancement_type,
            num_representatives=args.num_representatives
        )
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
        
        kld, mae, mse, ordacc, recall = evaluate(
            model, test_loader, device, topk=args.topk, loss_type=args.loss_type,
            listmle_topm=args.listmle_topm,
            listnet_pred_temp=args.listnet_pred_temp,
            listnet_tgt_temp=args.listnet_tgt_temp,
            pair_num_pos=args.pair_num_pos,
            pair_num_neg=args.pair_num_neg,
            pair_margin=args.pair_margin
        )
        # 将模型预测的概率保存到文件中
        print("Saving model predictions to file...")
        model.eval()
        out_pred = os.path.join(result_dir, os.path.basename(ckpt_name).rsplit('.', 1)[0] + "_pred.txt")
        with torch.no_grad(), open(out_pred, "w") as f:
            for x, _ in test_loader:
                x = x.to(device)
                probs = torch.softmax(model(x), dim=-1).cpu()
                for row in probs:
                    f.write(" ".join(f"{v.item():.6f}" for v in row) + "\n")
        print(f"Saved predictions -> {out_pred}")
        print(f"[TEST] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | ord_acc@{args.topk}={ordacc:.4f} | recall@{args.topk}={recall:.4f}")
    
    print(f"Training completed! Results saved to {result_dir}")


if __name__ == "__main__":
    main()

'''
python -m src.model.enhanced.cluster_dist_transformer_enhanced \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/leaf_centroids.npy \
  --cluster_info_path input/Training_data/gist1M_learn/leafsize20K/cluster_info.npz \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
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
  --enhancement_type multi_rep \
  --num_representatives 3 \
  --loss_type pairwise_hinge \
  --pair_num_pos 1 \
  --pair_num_neg 20 \
  --pair_margin 0.2 \
'''