import os
import math
import random
import argparse
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 假设这些模块存在，若报错请确保路径正确
try:
    from src.model.plot import plot_epoch1_batch_metrics, plot_epoch_metrics
    from src.model.allLoss import *
except ImportError:
    # 简单的回退方案逻辑
    def plot_epoch1_batch_metrics(*args, **kwargs): pass
    def plot_epoch_metrics(*args, **kwargs): pass

# ===================== 优化后的指标计算 =====================
def topk_recall(pred_prob: torch.Tensor, label_prob: torch.Tensor, k: int = 10) -> float:
    """向量化实现的 Top-K Recall"""
    B, K_dim = pred_prob.size()
    k = min(k, K_dim)
    _, topk_pred = torch.topk(pred_prob, k, dim=1)
    _, topk_true = torch.topk(label_prob, k, dim=1)
    
    # 使用 scatter 构造掩码实现快速交集计算
    mask = torch.zeros_like(label_prob, dtype=torch.bool)
    mask.scatter_(1, topk_true, True)
    hits = mask.gather(1, topk_pred).float().sum(dim=1)
    return (hits / float(k)).mean().item()

def topk_ordered_accuracy(pred_prob: torch.Tensor, label_prob: torch.Tensor, k: int = 10) -> float:
    """向量化实现的顺序准确率"""
    B, K_dim = pred_prob.size()
    k = min(k, K_dim)
    _, pred_idx = torch.topk(pred_prob, k, dim=1)
    _, true_idx = torch.topk(label_prob, k, dim=1)
    correct = (pred_idx == true_idx).float().mean()
    return correct.item()

# ===================== 优化后的 Dataset =====================
class NPZClusterDataset(Dataset):
    def __init__(self, npz_path: str, normalize: bool = False, mean=None, std=None):
        data = np.load(npz_path)
        self.queries = data["queries"].astype(np.float32)
        tgt = data.get("targets", data.get("labels", None))
        if tgt is None: raise ValueError("Target data missing")
        self.targets = tgt.astype(np.float32)

        if normalize:
            if mean is None or std is None:
                mean, std = self.queries.mean(0), self.queries.std(0) + 1e-6
            self.queries = (self.queries - mean) / std
            self.norm_stats = (mean, std)
        else:
            self.norm_stats = None

    def __len__(self): return self.queries.shape[0]

    def __getitem__(self, idx):
        # 仅返回 Query 和 Target，不再在 CPU 上拼接 Centroids
        return self.queries[idx], self.targets[idx]

# ===================== 优化后的模型 (Cross-Attention) =====================
class ClusterDistTransformer(nn.Module):
    def __init__(self, input_dim: int, centroids: np.ndarray, d_model: int = 256, 
                 nhead: int = 8, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # 将质心注册为 buffer，自动移动到 GPU 且不计入梯度更新
        self.register_buffer("centroids", torch.from_numpy(centroids).float())
        K, D = self.centroids.shape

        self.input_proj = nn.Linear(input_dim, d_model)
        self.centroid_proj = nn.Linear(input_dim, d_model)
        
        # 使用 Cross-Attention 替代全量 Self-Attention
        # Query 序列长度为 1，Centroids 序列长度为 K
        self.transformer_layers = nn.ModuleList([
            nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(nn.Linear(d_model, dim_feedforward), nn.GELU(), nn.Linear(dim_feedforward, d_model))
            for _ in range(num_layers)
        ])

        self.query_head = nn.Linear(d_model, d_model)
        self.centroid_head = nn.Linear(d_model, d_model)

    def forward(self, q_raw: torch.Tensor) -> torch.Tensor:
        # q_raw: [B, D]
        B = q_raw.size(0)
        q = self.input_proj(q_raw).unsqueeze(1) # [B, 1, d_model]
        c = self.centroid_proj(self.centroids).unsqueeze(0).expand(B, -1, -1) # [B, K, d_model]

        for layer, norm, ffn in zip(self.transformer_layers, self.norms, self.ffns):
            # Cross-Attention: q attends to c
            attn_out, _ = layer(q, c, c)
            q = norm(q + attn_out)
            q = q + ffn(q)

        # Scoring
        q_final = self.query_head(q.squeeze(1)) # [B, d_model]
        c_final = self.centroid_head(self.centroid_proj(self.centroids)) # [K, d_model]
        
        logits = torch.matmul(q_final, c_final.t()) / math.sqrt(self.d_model)
        return logits

# ===================== 训练与主逻辑优化 =====================
def evaluate(model, loader, device, topk, loss_type="kld"):
    model.eval()
    total_loss, total_recall, total_ord, count = 0, 0, 0, 0
    criterion = nn.KLDivLoss(reduction="batchmean") if loss_type == "kld" else nn.MSELoss()
    
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            if loss_type == "kld":
                loss = criterion(torch.log_softmax(logits, dim=-1), y)
            else:
                loss = criterion(torch.softmax(logits, dim=-1), y)
            
            total_loss += loss.item() * x.size(0)
            total_recall += topk_recall(torch.softmax(logits, -1), y, topk) * x.size(0)
            total_ord += topk_ordered_accuracy(torch.softmax(logits, -1), y, topk) * x.size(0)
            count += x.size(0)
    return total_loss/count, total_ord/count, total_recall/count

def train(model, train_loader, val_loader, device, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=not args.no_amp)
    criterion = nn.KLDivLoss(reduction="batchmean") if args.loss_type == "kld" else nn.MSELoss()

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            
            with torch.amp.autocast(device_type="cuda", enabled=not args.no_amp):
                logits = model(x)
                if args.loss_type == "kld":
                    loss = criterion(torch.log_softmax(logits, dim=-1), y)
                else:
                    loss = criterion(torch.softmax(logits, dim=-1), y)

            scaler.scale(loss).backward()
            if args.grad_clip:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            if i % args.log_interval == 0:
                print(f"Epoch {ep} | Batch {i} | Loss: {loss.item():.4f}")

        sched.step()
        if val_loader:
            v_loss, v_ord, v_rec = evaluate(model, val_loader, device, args.topk, args.loss_type)
            print(f">> Val Epoch {ep} | Loss: {v_loss:.4f} | Recall@{args.topk}: {v_rec:.4f}")
            if v_loss < best_val:
                best_val = v_loss
                torch.save(model.state_dict(), os.path.join(args.save_dir, "best_model.pt"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_npz", type=str, required=True)
    parser.add_argument("--centroids_path", type=str, required=True)
    parser.add_argument("--test_npz", type=str)
    parser.add_argument("--val_split", type=float, default=0.05)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--topk", type=int, default=20)
    parser.add_argument("--loss_type", type=str, default="kld")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--experiment_id", type=str, default="exp")
    args = parser.parse_args()

    # 加载质心
    centroids = np.load(args.centroids_path).astype(np.float32)
    K, D = centroids.shape
    args.save_dir = f"./results/{args.experiment_id}"
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 构建数据
    full_ds = NPZClusterDataset(args.train_npz, normalize=True)
    train_size = int((1 - args.val_split) * len(full_ds))
    train_ds, val_ds = random_split(full_ds, [train_size, len(full_ds) - train_size])
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = ClusterDistTransformer(D, centroids, d_model=args.d_model).to(device)
    
    train(model, train_loader, val_loader, device, args)

if __name__ == "__main__":
    main()
    
""" 
python -m src.model.cluster_dist_transformer_original_copy \
  --train_npz   /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize10W_learn5M/train_deep50M.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize10W_learn5M/test_deep50M.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize10W_learn5M/centroids.npy \
  --val_split 0.05 \
  --topk 200 \
  --epochs 1 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 6 \
  --dim_ff 256 \
  --dropout 0.05 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20260129_deep50M_leaf10W 
  """