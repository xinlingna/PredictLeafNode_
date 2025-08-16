import math
import numpy as np
import argparse

# 生成一个适用于PairwiseScoring的Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

class QueryPairwiseDataset(Dataset):
    """
    输入:
      - query_sift: (B, 128)
      - centroids:  (C, 128)
      - label:      (B, C)  (top-100 概率分布, 每行和为1)
    输出(__getitem__):
      - q_i:        (128,)             已按全局最大值归一
      - sim_prior:  (C,)               由归一后的 q 与 C 矩阵乘积并按行 L2 归一
      - label_i:    (C,)
    备注:
      - 归一后的 centroids 保存在 self.centroids_norm, 可直接传给 PairwiseScoring
    """
    def __init__(self, query_sift, centroids, label, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

        # tensors
        self.query = torch.as_tensor(query_sift, dtype=torch.float32)    # (B,128)
        self.label = torch.as_tensor(label, dtype=torch.float32)         # (B,C)
        self.centroids = torch.as_tensor(centroids, dtype=torch.float32) # (C,128)

        # 全局最大值归一
        self.query = self.query / (self.query.max() + self.eps)                  # (B,128)
        self.centroids_norm = self.centroids / (self.centroids.max() + self.eps) # (C,128)

        # 相似度先验：矩阵乘法 (缩放后的点积)，再按类维做 L2 归一
        sim_dot = torch.matmul(self.query, self.centroids_norm.T)        # (B,C)
        self.sim_prior = F.normalize(sim_dot, p=2, dim=1)                # (B,C)

        # 可选：确保 label 行和为1（若你的 label 已经是分布可注释掉）
        row_sum = self.label.sum(dim=1, keepdim=True).clamp_min(self.eps)
        self.label = self.label / row_sum

    def __len__(self):
        return self.query.size(0)

    def __getitem__(self, idx):
        return self.query[idx], self.sim_prior[idx], self.label[idx],self.centroids_norm

    # 方便在训练循环中拿到归一后的 C，传入 PairwiseScoring
    @property
    def centroids_for_model(self):
        return self.centroids_norm

def load_data_and_build_loaders(
    query_path: str,
    centroids_path: str,
    label_path: str,
    batch_size: int = 64,
    val_ratio: float = 0.2,
    shuffle: bool = True,
    eps: float = 1e-12,
    num_workers: int = 0,
    pin_memory: bool = False,
    seed: int = 42,
):
    # 读取数据
    query = np.loadtxt(query_path).astype(np.float32)     # (N,128)
    centroids = np.loadtxt(centroids_path).astype(np.float32)  # (C,128)
    label = np.loadtxt(label_path).astype(np.float32)     # (N,C)

    # 划分训练/验证
    q_tr, q_val, y_tr, y_val = train_test_split(
        query, label, test_size=val_ratio, random_state=seed, shuffle=True
    )

    # 构造 Dataset（两个集合共用同一组质心）
    train_dataset = QueryPairwiseDataset(q_tr, centroids, y_tr, eps=eps)
    val_dataset = QueryPairwiseDataset(q_val, centroids, y_val, eps=eps)

    def collate_fn(batch):
        if len(batch[0]) == 4:
            q, sim, y, C_item = zip(*batch)
            return (
                torch.stack(q, 0),       # [B,128]
                torch.stack(sim, 0),     # [B,C]
                torch.stack(y, 0),       # [B,C]
                torch.stack(C_item, 0),  # [B,C,128]
            )
        else:
            q, sim, y = zip(*batch)
            return torch.stack(q, 0), torch.stack(sim, 0), torch.stack(y, 0)

    # 构造 DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory, collate_fn=collate_fn
    )

    return train_loader, val_loader

class PairwiseScoring(nn.Module):
    def __init__(self, d=128, hidden=256):
        super().__init__()
        in_dim = d + d + d + d + 1   # q, c, q*c, |q-c|, sim_i
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 2*hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(2*hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, hidden//4),
            nn.ReLU(),
            nn.Linear(hidden//4, 1) 
        )
        self.alpha = nn.Parameter(torch.tensor(0.0))  # 相似度先验权重
        self.tau = nn.Parameter(torch.tensor(1.0))    # 先验温度

    def forward(self, q, C, sim):  # q:[B,128], C:[187,128]或[B,187,128], sim:[B,187]
        B = q.size(0)
        if C.dim() == 2:
            C = C.unsqueeze(0).expand(B, -1, -1)      # [B,187,128]
        q_exp = q.unsqueeze(1).expand(-1, C.size(1), -1)  # [B,187,128]

        feat = torch.cat([
            q_exp, C, q_exp * C, (q_exp - C).abs(), sim.unsqueeze(-1)], dim=-1)                                      # [B,187,in_dim]

        s = self.mlp(feat).squeeze(-1)                  # [B,187]

        # 使用温标后的对数先验进行融合，提升稳定性与匹配性
        temp = self.tau.abs() + 1e-6
        prior_prob = F.softmax(sim / temp, dim=-1)
        log_prior = torch.log(prior_prob + 1e-12)
        logits = s + self.alpha * log_prior
        prob = F.softmax(logits, dim=-1)
        return prob, logits


class EarlyStopper:
    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = 'max'):
        assert mode in ('max', 'min')
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.wait = 0

    def step(self, value: float) -> bool:
        # 返回 True 表示应早停
        if self.best is None:
            self.best = value
            self.wait = 0
            return False
        improved = (value - self.best) > self.min_delta if self.mode == 'max' else (self.best - value) > self.min_delta
        if improved:
            self.best = value
            self.wait = 0
            return False
        else:
            self.wait += 1
            return self.wait >= self.patience

    
def run_epoch(model, loader, device, optimizer, criterion, train: bool, topk: int = 100, ce_weight: float = 0.1):
    if train:
        model.train()
    else:
        model.eval()

    total_loss, total_batches = 0.0, 0
    total_hits, total_elems = 0.0, 0  # 仅验证时统计

    with torch.set_grad_enabled(train):
        for batch in loader:
            if len(batch) == 4:
                q, sim, y, C_batch = batch
            else:
                q, sim, y = batch
                base_ds = loader.dataset.dataset if isinstance(loader.dataset, torch.utils.data.Subset) else loader.dataset
                C_batch = base_ds.centroids_for_model.unsqueeze(0).expand(q.size(0), -1, -1)

            q = q.to(device)
            sim = sim.to(device)
            y = y.to(device)
            C_batch = C_batch.to(device)

            prob, logits = model(q, C_batch, sim)
            # KL 散度（软标签）+ 辅助 CE（硬标签）
            kl = criterion(F.log_softmax(logits, dim=-1), y)
            hard_targets = y.argmax(dim=-1)
            ce = F.cross_entropy(logits, hard_targets)
            loss = kl + ce_weight * ce

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                # 计算 val_accuracy：预测前k与标签前k下标的交集比例
                k = min(topk, prob.size(-1))
                pred_topk = prob.topk(k, dim=-1).indices      # [B,k]
                true_topk = y.topk(k, dim=-1).indices         # [B,k]
                matches = (pred_topk.unsqueeze(-1) == true_topk.unsqueeze(-2)).any(dim=-1).float()  # [B,k]
                total_hits += matches.sum().item()
                total_elems += matches.numel()

            total_loss += loss.item()
            total_batches += 1

    avg_loss = total_loss / max(1, total_batches)
    if train:
        return avg_loss, None
    else:
        val_accuracy = total_hits / max(1, total_elems)
        return avg_loss, val_accuracy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--query_path', type=str, default='query_sift.txt')
    parser.add_argument('--centroids_path', type=str, default='centroids.txt')
    parser.add_argument('--label_path', type=str, default='label.txt')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--val_ratio', type=float, default=0.2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eps', type=float, default=1e-12)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--topk', type=int, default=100)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--min_delta', type=float, default=0.0)
    parser.add_argument('--monitor', type=str, default='acc', choices=['acc', 'loss'])  # 早停依据
    parser.add_argument('--ce_weight', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    train_loader, val_loader = load_data_and_build_loaders(
        query_path=args.query_path,
        centroids_path=args.centroids_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        shuffle=True,
        eps=args.eps,
    )

    model = PairwiseScoring(d=128, hidden=256).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.KLDivLoss(reduction='batchmean')

    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_state = None

    mode = 'max' if args.monitor == 'acc' else 'min'
    stopper = EarlyStopper(patience=args.patience, min_delta=args.min_delta, mode=mode)

    # 余弦退火调度器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    for epoch in range(1, args.epochs + 1):
        train_loss, _ = run_epoch(model, train_loader, args.device, optimizer, criterion, train=True, topk=args.topk, ce_weight=args.ce_weight)
        val_loss, val_acc = run_epoch(model, val_loader, args.device, optimizer, criterion, train=False, topk=args.topk, ce_weight=args.ce_weight)

        best_val_loss = min(best_val_loss, val_loss)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        monitor_value = val_acc if args.monitor == 'acc' else val_loss
        should_stop = stopper.step(monitor_value)

        print(f'Epoch {epoch:02d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f} | val_acc@{args.topk}={val_acc:.6f} | best_val_acc={best_val_acc:.6f} | patience={stopper.wait}/{args.patience}')

        scheduler.step()

        if should_stop:
            print('Early stopping triggered.')
            break

    # 可选：恢复最佳权重
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_acc

if __name__ == '__main__':
    main()

# 多行注释
"""
conda activate elpis_torch
cd ~/PycharmProjects/PredictLeafNode/src/model

python PairwiseScoring.py \
  --query_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_merged/sift_merged_query.txt \
  --centroids_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_merged/leafsize10K/leaf_center.txt \
  --label_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_merged/leafsize10K/knn_distributions.txt \
  --batch_size 512 \
  --val_ratio 0.2 \
  --epochs 50 \
  --lr 1e-3 \
  --seed 42 \
  --eps 1e-12 \
  --topk 100 \
  --patience 100 \
  --min_delta 0.0 \
  --monitor loss \
  --device cuda
"""
