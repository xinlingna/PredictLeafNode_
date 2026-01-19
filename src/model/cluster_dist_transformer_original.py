import os
import math
import random
import argparse
from typing import Optional, Tuple
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 尝试导入 optuna
try:
    import optuna
    from optuna.trial import TrialState
except ImportError:
    optuna = None

from src.model.plot   import plot_epoch1_batch_metrics, plot_epoch_metrics
from src.model.allLoss import *

# --- 原有的辅助函数保持不变 (set_seed, load_centroids, dataset 类等) ---
# 为了节省篇幅，假设以下函数/类定义保持不变，请在实际使用时保留原有定义：
# set_seed, load_centroids, build_loaders, make_synth_dataset, save_npz
# topk_recall, topk_ordered_accuracy, NPZClusterDataset, SubsetByIndex
# PositionalEncoding, ClusterDistTransformer, evaluate_batch, evaluate

# ... [在此处插入你原代码中 set_seed 到 evaluate 的所有定义] ...
# 设置随机种子以确保实验可重复
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# 加载质心，质心形状(K, D)
def load_centroids(centroids_path: str) -> np.ndarray:
    arr = np.load(centroids_path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):  # npz file
        if "centroids" in arr.files:           # prefer "centroids" key if exists
            c = arr["centroids"]
        else:
            key = arr.files[0]                 # use the first array in npz if key not specified
            c = arr[key]
    else:
        c = arr
    c = c.astype(np.float32)
    assert c.ndim == 2, f"centroids should be 2D, got {c.shape}"
    return c

def build_loaders(
    train_npz: str,                       # train_npz: (queries, targets)
    val_npz: Optional[str],
    batch_size: int,
    num_workers: int,
    normalize: bool,
    val_split: float,                    # when val_npz is None
    seed: int,
    centroids: Optional[np.ndarray],     # centroids: (K, D)
    train_first_n=None
):
    full_train = NPZClusterDataset(train_npz, normalize=normalize, centroids=centroids)
    
    if train_first_n is not None:
        n = min(train_first_n, len(full_train))
        full_train = SubsetByIndex(full_train, list(range(n)))
    
    if val_npz is None and val_split > 0:           # split training set into train/val
        n_total = len(full_train)
        n_val = int(n_total * val_split)
        n_train = n_total - n_val
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
    elif val_npz is None:                          # no validation
        train_ds, val_ds = full_train, None
    else:                                          # train/val from separate files
        train_ds = full_train
        mean, std = (None, None)
        if full_train.norm_stats is not None:
            mean, std = full_train.norm_stats
        val_ds = NPZClusterDataset(val_npz, normalize=normalize, mean=mean, std=std, centroids=centroids)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = None if val_ds is None else DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def make_synth_dataset(n: int, K: int, D: int, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # Synthetic data with shared centroids across all samples
    centroids = np.random.randn(K, D).astype(np.float32)
    queries = np.random.randn(n, D).astype(np.float32) + noise * np.random.randn(n, D).astype(np.float32)
    targets = np.zeros((n, K), dtype=np.float32)
    for i in range(n):
        dists = np.linalg.norm(centroids - queries[i][None, :], axis=1) + 1e-6
        inv = 1.0 / dists
        alpha = inv / inv.mean()
        targets[i] = np.random.dirichlet(alpha.clip(0.1, 100.0)).astype(np.float32)
    return queries, centroids, targets


def save_npz(path: str, queries: np.ndarray, targets: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, queries=queries, targets=targets)


def topk_recall(pred_prob: torch.Tensor, label_prob: torch.Tensor, k: int = 10) -> float:
    """
    Compute Top-K recall over a batch.
    pred_prob: (B, K)
    label_prob: (B, K)
    Returns average recall@k in [0,1].
    """
    k = min(k, pred_prob.size(1))  # 安全裁剪
    topk_pred = torch.topk(pred_prob, k, dim=1).indices  # (B, k)
    top_true = torch.topk(label_prob, k, dim=1).indices  # (B, k)
    recall_sum = 0.0
    B = pred_prob.size(0)
    for i in range(B):
        intersect = len(set(topk_pred[i].tolist()) & set(top_true[i].tolist()))
        recall_sum += intersect / float(k)
    return recall_sum / float(B)

def topk_ordered_accuracy(pred_prob: torch.Tensor, label_prob: torch.Tensor, k: int = 10) -> float:
    """
    Compute Top-K ordered accuracy.
    For each sample, take the top-k indices (descending) from the predicted and target
    probability distributions and compare them position-wise. Returns the batch-mean
    fraction of matches in [0, 1].
    """
    if pred_prob.ndim != 2 or label_prob.ndim != 2:
        raise ValueError("pred_prob and label_prob must be 2D tensors of shape (B, K)")
    if pred_prob.size() != label_prob.size():
        raise ValueError(f"Shape mismatch: pred_prob={pred_prob.size()}, label_prob={label_prob.size()}")
    B, K = pred_prob.size()
    k = min(k, K)
    pred_topk = torch.topk(pred_prob, k, dim=1).indices  # (B, k)
    true_topk = torch.topk(label_prob, k, dim=1).indices  # (B, k)
    matches = (pred_topk == true_topk).float()  # (B, k)
    per_sample_acc = matches.mean(dim=1)  # (B,)
    return float(per_sample_acc.mean().item())

class NPZClusterDataset(Dataset):
    """ 
    Each dataset consists of:
        Training set: concatenate the query with all centroids along the row dimension, resulting in a shape of (K+1, D)
        Labels: the distribution of the query's top 100 among all centroids
    """
    def __init__(
        self,
        npz_path: str,
        normalize: bool = False,                 # normalize queries and centroids if True
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None,  # global shared centroids [K, D], required
    ):
        data = np.load(npz_path)
        # Only support queries + targets/labels
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

        # Shape checks
        N, D = self.queries.shape
        Nk, K = self.targets.shape
        Kc, Dc = self.centroids.shape
        assert N == Nk, f"N mismatch: queries={N}, targets={Nk}"
        assert K == Kc, f"K mismatch: targets={K}, centroids={Kc}"
        assert D == Dc, f"D mismatch: queries={D}, centroids={Dc}"

        # Normalization: use the same mean/std for queries and centroids
        if normalize:
            if mean is None or std is None:
                flat = np.concatenate([self.queries, self.centroids], axis=0)
                mean = flat.mean(axis=0, keepdims=True)
                std = flat.std(axis=0, keepdims=True) + 1e-6
            self.queries = (self.queries - mean) / std
            self.centroids = (self.centroids - mean) / std
            self.norm_stats = (mean, std)
        else:
            self.norm_stats = None

    def __len__(self):
        return self.targets.shape[0]

    def __getitem__(self, idx):
        q = self.queries[idx]  # [D]
        x = np.vstack([q[None, :], self.centroids])  # [K+1, D]
        y = self.targets[idx]  # [K]
        return x.astype(np.float32), y.astype(np.float32)
    
class SubsetByIndex(Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.dataset[self.indices[idx]]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class ClusterDistTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,  # hidden dimension
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 4096,
        use_type_embed: bool = True,
        score_type: str = "bilinear",  # ["bilinear", "mlp"]
        pos_exist: bool = False,
        use_gating: bool = False       # gating mechanism
    ):
        super().__init__()
        self.pos_exist = pos_exist
        self.use_gating = use_gating
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len) # （max_len, d_model）
        self.type_embed = nn.Embedding(2, d_model) if use_type_embed else None

        # Gating mechanism layer
        # 为了 TorchScript 兼容性，总是初始化这些层
        if use_gating:
            self.gate_linear = nn.Linear(d_model, d_model)
            self.gate_activation = nn.Sigmoid()
        else:
            # 占位符，不会被使用
            self.gate_linear = nn.Identity()
            self.gate_activation = nn.Identity()

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
            # 为了 TorchScript 兼容性，即使不使用也要初始化 scorer
            self.scorer = nn.Identity()  # 占位符
        elif score_type == "mlp":
            self.scorer = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            # 为了 TorchScript 兼容性，即使不使用也要初始化 query_head 和 centroid_head
            self.query_head = nn.Identity()  # 占位符
            self.centroid_head = nn.Identity()  # 占位符
        else:
            raise ValueError("score_type must be 'bilinear' or 'mlp'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K+1, D]
        bsz, seqlen, _ = x.size()
        t = self.input_proj(x)                   # [B, L, d_model]
        if self.type_embed is not None:          # add type embeddings if used
            type_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=x.device)
            if seqlen > 1:
                type_ids[:, 1:] = 1
            t = t + self.type_embed(type_ids)
        if self.pos_exist:                        # add positional encodings if used
            t = self.pos_enc(t)
        h = self.encoder(t)  # [B, L, d_model]

        # Apply gating mechanism if enabled
        if self.use_gating:
            gate = self.gate_activation(self.gate_linear(h))  # [B, L, d_model]
            h = h * gate  # Element-wise multiplication, gated output

        q = h[:, 0, :]  # [B, d_model]
        c = h[:, 1:, :]  # [B, K, d_model]

        if self.score_type == "bilinear":
            qh = self.query_head(q)     # [B, d_model]
            ch = self.centroid_head(c)  # [B, K, d_model]
            logits = torch.einsum("bd,bkd->bk", qh, ch) / math.sqrt(self.d_model) # [B, K]
        else:
            q_expand = q.unsqueeze(1).expand(-1, c.size(1), -1)  # [B, K, d_model]
            logits = self.scorer(torch.cat([q_expand, c], dim=-1)).squeeze(-1)  # [B, K]

        return logits  # raw logits; apply log_softmax/softmax outside as needed


@torch.no_grad()
def evaluate_batch(
    model: nn.Module, 
    x: torch.Tensor, 
    y: torch.Tensor, 
    device: torch.device, 
    topk: int = 10, 
    loss_type: str = 'kld',
    # 新增损失函数相关参数
    listmle_topm: Optional[int] = None,
    listnet_pred_temp: float = 1.0,
    listnet_tgt_temp: Optional[float] = None,
    pair_num_pos: int = 1,
    pair_num_neg: int = 20,
    pair_margin: float = 0.1
) -> Tuple[float, float, float, float, float]:
    """
    评估单个batch的指标
    """
    # Temporarily switch to eval mode and restore the previous mode afterwards
    was_training = model.training
    model.eval()
    x = x.to(device)
    y = y.to(device)
    logits = model(x) # (B,K)
    log_probs = torch.log_softmax(logits, dim=-1)
    preds = torch.softmax(logits, dim=-1)
    # choose loss according to loss_type (keep consistent with train())
    if loss_type == "kld":
        kldiv = nn.KLDivLoss(reduction="batchmean")
        loss_val = kldiv(log_probs, y)  # KL(target || pred)
    elif loss_type == "kld_reverse":
        # KL(pred || target) with gradient through model: sum p*(log p - log q)
        y_log = torch.log(y + 1e-8)
        loss_val = (preds * (log_probs - y_log)).sum(dim=-1).mean()
    elif loss_type == "mse":
        mse_loss_fn = nn.MSELoss()
        loss_val = mse_loss_fn(preds, y)
    elif loss_type == "hybrid":
        hl = HybridLoss(mse_weight=0.7, kl_weight=0.3, temperature=1.0, smoothing=0.01)
        loss_val, _ = hl(logits, y)
    elif loss_type == "recall_focused_loss":
        # Use logits to maintain consistency with training
        loss_val = recall_focused_loss_batched(logits, y, topk)
    elif loss_type == "listnet":
        loss_val = listnet_top1_loss(
            logits, y,
            pred_temp=listnet_pred_temp,
            tgt_temp=listnet_tgt_temp,
            assume_targets_prob=True
        )
    elif loss_type == "listmle":
        loss_val = listmle_loss(
            logits, y,
            use_topm=listmle_topm
        )
    elif loss_type == "pairwise_hinge":
        loss_val = pairwise_hinge_loss(
            logits, y,
            num_pos=pair_num_pos,
            num_neg=pair_num_neg,
            margin=pair_margin
        )
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    mae = torch.mean(torch.abs(preds - y))
    mse = torch.mean((preds - y) ** 2)
    recall = topk_recall(preds, y, k=topk)
    ordered_acc = topk_ordered_accuracy(preds, y, k=topk)
    
    # Restore previous training mode
    if was_training:
        model.train()
    return loss_val.item(), mae.item(), mse.item(), ordered_acc, recall


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    topk: int = 10,
    loss_type: str = "kld",
    # 新增损失函数相关参数
    listmle_topm: Optional[int] = None,
    listnet_pred_temp: float = 1.0,
    listnet_tgt_temp: Optional[float] = None,
    pair_num_pos: int = 1,
    pair_num_neg: int = 20,
    pair_margin: float = 0.1
) -> Tuple[float, float, float, float, float]:
    # Preserve current mode and switch to eval for metric computation
    was_training = model.training
    model.eval()
    mae_meter, mse_meter, loss_meter = 0.0, 0.0, 0.0
    total = 0
    recall_meter = 0.0
    ordered_acc_meter = 0.0
    # Prepare reusable loss constructors where applicable (standard if/elif)
    kldiv = None
    mse_loss_fn = None
    ce = None
    hl = None
    if loss_type == "kld":
        kldiv = nn.KLDivLoss(reduction="batchmean")
    elif loss_type == "mse":
        mse_loss_fn = nn.MSELoss()
    elif loss_type == "hybrid":
        hl = HybridLoss(mse_weight=0.7, kl_weight=0.3, temperature=1.0, smoothing=0.01)
    # elif loss_type == "CrossEntropyLoss":
    #     ce = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.softmax(logits, dim=-1)

        # compute loss according to loss_type (consistent with train()/evaluate_batch)
        if loss_type == "kld":
            loss_val = kldiv(log_probs, y)  # KL(target || pred)
        elif loss_type == "kld_reverse":
            # KL(pred || target): sum p*(log p - log q)
            y_log = torch.log(y + 1e-8)
            loss_val = (preds * (log_probs - y_log)).sum(dim=-1).mean()
        elif loss_type == "mse":
            loss_val = mse_loss_fn(preds, y)
        elif loss_type == "hybrid":
            loss_val, _ = hl(logits, y)
        elif loss_type == "recall_focused_loss":
            loss_val = recall_focused_loss_batched(logits, y, topk)
        elif loss_type == "listnet":
            loss_val = listnet_top1_loss(
                logits, y,
                pred_temp=listnet_pred_temp,
                tgt_temp=listnet_tgt_temp,
                assume_targets_prob=True
            )
        elif loss_type == "listmle":
            loss_val = listmle_loss(
                logits, y,
                use_topm=listmle_topm
            )
        elif loss_type == "pairwise_hinge":
            loss_val = pairwise_hinge_loss(
                logits, y,
                num_pos=pair_num_pos,
                num_neg=pair_num_neg,
                margin=pair_margin
            )
        else:
            raise ValueError(f"Invalid loss type: {loss_type}")

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
        # print(f"loss={loss_val.item():.6f} | mae={mae.item():.6f} | mse={mse.item():.6f} | ordered_acc@{topk}={ordered_acc:.4f} | recall@{topk}={recall:.4f}")
    # Restore previous mode
    if was_training:
        model.train()
    return loss_meter / total, mae_meter / total, mse_meter / total, ordered_acc_meter / total, recall_meter / total
# ==============================================================================
# 修改 1: Train 函数增加 trial 参数用于 Pruning (剪枝)
# ==============================================================================
def train(
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
    listmle_topm: Optional[int] = None,
    listnet_pred_temp: float = 1.0,
    listnet_tgt_temp: Optional[float] = None,
    pair_num_pos: int = 1,
    pair_num_neg: int = 20,
    pair_margin: float = 0.1,
    # [新增] optuna trial 对象
    trial: Optional["optuna.trial.Trial"] = None
) -> float: # [修改] 返回最佳验证集 loss
    
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp and torch.cuda.is_available())
    
    # 损失函数初始化逻辑保持不变...
    if loss_type == "kld":
        lossFunc = nn.KLDivLoss(reduction="batchmean")
    elif loss_type == "kld_reverse":
        lossFunc = None 
    elif loss_type == "mse":
        lossFunc = nn.MSELoss()
    elif loss_type == "hybrid":
        lossFunc = HybridLoss(mse_weight=0.7, kl_weight=0.3, temperature=1.0, smoothing=0.01)
    elif loss_type == "recall_focused_loss":
        lossFunc = recall_focused_loss_batched
    elif loss_type in ["listnet", "listmle", "pairwise_hinge"]:
        lossFunc = None
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    best_val_metric = float("inf") # 默认优化目标是最小化 Loss
    
    # 为了 Optuna，如果不保存文件，可以跳过目录创建
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    epoch_history = []
    epoch1_batch_metrics = [] # 保持原逻辑

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_ordacc = 0.0
        running_recall = 0.0
        n = 0
        
        # Epoch 1 详细记录逻辑保持不变...
        if ep == 1 and val_loader is not None:
             batch_count = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(set_to_none=True)
            
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                # 损失计算逻辑保持不变...
                if loss_type == "kld_reverse":
                    y_log = torch.log(y + 1e-8)
                    loss = (probs * (log_probs - y_log)).sum(dim=-1).mean()
                elif loss_type == "kld":
                    loss = lossFunc(log_probs, y)
                elif loss_type == "mse":
                    loss = lossFunc(probs, y)
                elif loss_type == "hybrid":
                    loss, _ = lossFunc(logits, y)
                elif loss_type == "recall_focused_loss":
                    loss = lossFunc(logits, y, eval_topk)
                elif loss_type == "listnet":
                    loss = listnet_top1_loss(logits, y, pred_temp=listnet_pred_temp, tgt_temp=listnet_tgt_temp, assume_targets_prob=True)
                elif loss_type == "listmle":
                    loss = listmle_loss(logits, y, use_topm=listmle_topm)
                elif loss_type == "pairwise_hinge":
                    loss = pairwise_hinge_loss(logits, y, num_pos=pair_num_pos, num_neg=pair_num_neg, margin=pair_margin)
                else:
                    raise ValueError(f"Invalid loss type: {loss_type}")

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

            # Epoch 1 batch logging logic (保持原样)...
            if ep == 1 and val_loader is not None:
                batch_count += 1
                if batch_count % 10 == 0:
                    # (此处省略 evaluate_batch 调用以节省篇幅，逻辑与原代码一致)
                    pass

            # Log interval logic...
            if log_interval > 0:
                step_idx = n // x.size(0)
                if step_idx % log_interval == 0:
                    lr_now = opt.param_groups[0]['lr']
                    # print(f"Epoch {ep:03d} | Step {step_idx} | lr={lr_now:.6e} | loss={loss.item():.6f}")

        sched.step()
        train_loss = running / n
        train_ordacc = running_ordacc / n
        train_recall = running_recall / n

        current_val_metric = train_loss # 默认 fallback

        if val_loader is not None:
            val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate(
                model, val_loader, device, topk=eval_topk, loss_type=loss_type,
                listmle_topm=listmle_topm, listnet_pred_temp=listnet_pred_temp,
                listnet_tgt_temp=listnet_tgt_temp, pair_num_pos=pair_num_pos,
                pair_num_neg=pair_num_neg, pair_margin=pair_margin
            )
            
            # 定义优化的目标指标 (这里默认使用 KLD)
            current_val_metric = val_kld

            # 非 Optuna 模式下的打印
            if trial is None:
                print(f"epoch {ep:03d} | train_loss={train_loss:.6f} | val_loss={val_kld:.6f} | recall={val_recall:.4f}")

            # 保存逻辑
            if current_val_metric < best_val_metric:
                best_val_metric = current_val_metric
                if save_dir:
                    torch.save({"model": model.state_dict(), "epoch": ep, "val_kld": val_kld},
                               os.path.join(save_dir, save_name))

            # ================= [Optuna Pruning] =================
            if trial is not None:
                # 向 Optuna 报告当前的中间值
                trial.report(current_val_metric, ep)
                # 检查是否需要剪枝 (比如效果明显比别的 trial 差)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            # ====================================================

        else:
            if trial is None:
                 print(f"epoch {ep:03d} | train_loss={train_loss:.6f}")

    # 训练结束后绘制 (仅在非 Optuna 模式或 Optuna 选定最佳后调用，防止生成大量图片)
    if trial is None and val_loader is not None and save_dir:
        # plot_epoch_metrics... (保持原样)
        pass

    return best_val_metric


# ==============================================================================
# 修改 2: 定义 Optuna Objective Function (用户可在此处配置搜索空间)
# ==============================================================================
def optuna_objective(trial, args, base_centroids, device):
    """
    全离散参数版本的 Objective Function
    """
    
    # -------------------------------------------------------------
    # [用户配置区] 所有参数均改为 suggest_categorical
    # -------------------------------------------------------------
    
    # === 1. 模型架构参数 ===
    
    # d_model: 只有 3 种选择
    param_d_model = trial.suggest_categorical("d_model", [128, 256])
    
    # nhead: 依赖于 d_model，必须能整除
    # 我们先定义可能的 head 数，然后根据当前的 d_model 动态筛选
    possible_heads = [4, 8]
    valid_heads = [h for h in possible_heads if param_d_model % h == 0]
    
    # 如果 valid_heads 为空（虽然上面的组合不会空），做一个防错兜底
    if not valid_heads:
        valid_heads = [1] # fallback
        
    param_nhead = trial.suggest_categorical("nhead", valid_heads)
    
    # num_layers: 指定具体的层数列表
    param_num_layers = trial.suggest_categorical("num_layers", [4, 6, 8])
    
    # dim_ff: Feed Forward 层的维度
    param_dim_ff = trial.suggest_categorical("dim_ff", [256, 512])
    
    # dropout: 固定为 0.1, 0.2, 0.3 等特定档位
    param_dropout = trial.suggest_categorical("dropout", [0.1, 0.2, 0.3])
    
    # === 2. 训练超参数 ===
    
    # lr: 学习率通常按对数标度取离散点
    param_lr = trial.suggest_categorical("lr", [1e-2, 1e-3, 5e-4])
    
    # batch_size: 显存允许范围内的离散值
    param_batch_size = trial.suggest_categorical("batch_size", [512, 1024])
    
    # weight_decay: 几个常见的权重衰减值
    param_weight_decay = trial.suggest_categorical("weight_decay", [1e-2, 1e-3, 1e-4, 0.0])
    
    # === 3. 损失函数 (保持命令行输入，或者也在这里离散化) ===
    # param_loss_type = trial.suggest_categorical("loss_type", ["kld", "mse"])
    param_loss_type = args.loss_type 
    
    # -------------------------------------------------------------
    # 后续逻辑保持不变...
    # -------------------------------------------------------------
    
    # 1. 构建 DataLoader
    train_loader, val_loader = build_loaders(
        args.train_npz, args.val_npz,
        batch_size=param_batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        centroids=base_centroids,
        train_first_n=args.train_first_n
    )
    
    if val_loader is None:
        raise ValueError("Optuna optimization requires a validation set.")

    # 2. 构建模型
    K, D = base_centroids.shape
    model = ClusterDistTransformer(
        input_dim=D,
        d_model=param_d_model,
        nhead=param_nhead,
        num_layers=param_num_layers,
        dim_feedforward=param_dim_ff,
        dropout=param_dropout,
        max_len=args.max_len,
        score_type=args.score_type,
        pos_exist=args.pos_exist,
        use_type_embed=args.use_type_embed,
        use_gating=args.use_gating
    ).to(device)

    # 3. 运行训练
    best_loss = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs, 
        lr=param_lr,
        weight_decay=param_weight_decay,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        save_dir=None, 
        eval_topk=args.topk,
        log_interval=0,
        loss_type=param_loss_type,
        trial=trial,
        listmle_topm=args.listmle_topm,
        listnet_pred_temp=args.listnet_pred_temp,
        listnet_tgt_temp=args.listnet_tgt_temp,
        pair_num_pos=args.pair_num_pos,
        pair_num_neg=args.pair_num_neg,
        pair_margin=args.pair_margin
    )
    
    return best_loss


def parse_args():
    p = argparse.ArgumentParser(description="Transformer for predicting NN cluster distribution")
    # ... [原有的所有 add_argument 保持不变] ...
    # 这里为了演示，我简略写出原有的，你需要保留你的完整定义
    p.add_argument("--train_npz", type=str, default=None)
    p.add_argument("--val_npz", type=str, default=None)
    p.add_argument("--test_npz", type=str, default=None)
    p.add_argument("--centroids_path", type=str, default=None)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--normalize", action="store_true")
    p.add_argument("--gen_synth", action="store_true")
    p.add_argument("--synth_n", type=int, default=20000)
    p.add_argument("--synth_K", type=int, default=64)
    p.add_argument("--synth_D", type=int, default=128)
    p.add_argument("--save_dir", type=str, default="./results")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--experiment_id", type=str, default=None)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--nhead", type=int, default=8)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--dim_ff", type=int, default=512)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--score_type", type=str, default="bilinear")
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--pos_exist", action="store_true")
    p.add_argument("--use_type_embed", action="store_true")
    p.add_argument("--use_gating", action="store_true")
    p.add_argument("--loss_type", type=str, default="kld")
    p.add_argument("--listmle_topm", type=int, default=None)
    p.add_argument("--listnet_pred_temp", type=float, default=1.0)
    p.add_argument("--listnet_tgt_temp", type=float, default=None)
    p.add_argument("--pair_num_pos", type=int, default=1)
    p.add_argument("--pair_num_neg", type=int, default=20)
    p.add_argument("--pair_margin", type=float, default=0.1)
    p.add_argument("--train_first_n", type=int, default=None)
    p.add_argument("--test_first_n", type=int, default=None)

    # ===================== Optuna Args =====================
    p.add_argument("--use_optuna", action="store_true", help="Enable Optuna hyperparameter optimization")
    p.add_argument("--optuna_trials", type=int, default=50, help="Number of trials for Optuna")
    p.add_argument("--optuna_db", type=str, default=None, help="SQL storage URL (e.g., sqlite:///optuna.db) for resuming studies")
    p.add_argument("--optuna_study_name", type=str, default="nn_cluster_transformer", help="Name of the Optuna study")

    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. 准备数据 (如果是合成数据，先生成)
    if args.gen_synth:
        queries, centroids, targets = make_synth_dataset(args.synth_n, args.synth_K, args.synth_D)
        # ... (保持原样生成文件代码) ...
        train_npz = os.path.join(args.save_dir, "synth_train.npz")
        val_npz = os.path.join(args.save_dir, "synth_val.npz")
        test_npz = os.path.join(args.save_dir, "synth_test.npz")
        # Split 80/10/10 logic...
        n = queries.shape[0]
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        save_npz(train_npz, queries[:n_train], targets[:n_train])
        save_npz(val_npz, queries[n_train:n_train + n_val], targets[n_train:n_train + n_val])
        save_npz(test_npz, queries[n_train + n_val:], targets[n_train + n_val:])
        centroids_path = os.path.join(args.save_dir, "synth_centroids.npy")
        np.save(centroids_path, centroids)
        args.train_npz, args.val_npz, args.test_npz = train_npz, val_npz, test_npz
        args.centroids_path = centroids_path

    if args.train_npz is None:
        raise ValueError("You must provide --train_npz or use --gen_synth")
    if args.centroids_path is None:
        raise ValueError("You must provide --centroids_path")

    # 全局加载一次 Centroids
    C = load_centroids(args.centroids_path)
    K, D = C.shape
    print(f"Detected K={K}, D={D}")

    # ==============================================================================
    # 逻辑分支 1: 运行 Optuna 调参
    # ==============================================================================
    if args.use_optuna:
        if optuna is None:
            raise ImportError("Please install optuna: pip install optuna")
        
        print("Starting Optuna Hyperparameter Optimization...")
        
        # 定义 Sampler (TPESampler 是默认且高效的)
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        
        # 定义 Pruner (Hyperband 高效剪枝)
        pruner = optuna.pruners.HyperbandPruner(min_resource=3, max_resource=args.epochs, reduction_factor=3)

        study = optuna.create_study(
            direction="minimize", # 目标是最小化 loss
            study_name=args.optuna_study_name,
            storage=args.optuna_db, # 如果不为 None，支持断点续传
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )
        
        # ================= [新增] 定义回调函数：实时保存最佳结果 =================
        def save_best_callback(study, frozen_trial):
            # 检查当前刚刚结束的这个 trial 是否是目前的 Best Trial
            if study.best_trial.number == frozen_trial.number:
                print(f"  [New Best Found] Trial {frozen_trial.number} - Loss: {frozen_trial.value:.6f}")
                
                # 立即写入 Best Params 文件
                os.makedirs(args.save_dir, exist_ok=True)
                txt_path = os.path.join(args.save_dir, "optuna_best_params.txt")
                with open(txt_path, "w") as f:
                    f.write(f"Best Trial ID: {frozen_trial.number}\n")
                    f.write(f"Best Loss: {frozen_trial.value}\n")
                    f.write("Params:\n")
                    for key, value in frozen_trial.params.items():
                        f.write(f"  {key}: {value}\n")
        # ======================================================================

        # 包装 objective，注入 args 和 data
        func = lambda trial: optuna_objective(trial, args, C, device)
        
        try:
            # 加入 callbacks 参数实现实时保存
            study.optimize(func, n_trials=args.optuna_trials, callbacks=[save_best_callback])
        except KeyboardInterrupt:
            print("Optimization interrupted by user.")
            
        print("Number of finished trials: ", len(study.trials))
        
        # ================= [新增] 结束后保存所有 Trial 的详细记录 =================
        if len(study.trials) > 0:
            df = study.trials_dataframe()
            csv_path = os.path.join(args.save_dir, "optuna_all_trials.csv")
            df.to_csv(csv_path, index=False)
            print(f"All trials saved to {csv_path}")

            print("Best trial:")
            trial = study.best_trial
            print("  Value: ", trial.value)
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
            
            # 双重保险：结束后再次保存最佳参数（防止 callback 漏掉最后一次）
            txt_path = os.path.join(args.save_dir, "optuna_best_params.txt")
            with open(txt_path, "w") as f:
                f.write(f"Best Trial ID: {trial.number}\n")
                f.write(f"Best Loss: {trial.value}\n")
                f.write("Params:\n")
                for key, value in trial.params.items():
                    f.write(f"  {key}: {value}\n")
            print(f"Best params saved to {txt_path}")
            
        return # Optuna 模式下运行完即退出

    # ==============================================================================
    # 逻辑分支 2: 正常训练 (原有逻辑)
    # ==============================================================================
    
    ckpt_name = (
       f"model_d{args.d_model}_L{args.num_layers}_H{args.nhead}_ff{args.dim_ff}_topk{args.topk}"
       f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{args.score_type}_normalize{args.normalize}"
       f"_pos{args.pos_exist}_gating{args.use_gating}_loss_type{args.loss_type}.pt"
    )
    subdir_name = os.path.splitext(ckpt_name)[0]
    if args.experiment_id:
       subdir_name = f"{args.experiment_id}_{subdir_name}"
    
    ckpt_dir = os.path.dirname(args.centroids_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    result_dir = os.path.join(ckpt_dir, subdir_name)
    os.makedirs(result_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(
        args.train_npz, args.val_npz, batch_size=args.batch_size, num_workers=args.num_workers,
        normalize=args.normalize, val_split=args.val_split, seed=args.seed, centroids=C, train_first_n=args.train_first_n
    )

    model = ClusterDistTransformer(
        input_dim=D, d_model=args.d_model, nhead=args.nhead, num_layers=args.num_layers,
        dim_feedforward=args.dim_ff, dropout=args.dropout, max_len=args.max_len,
        score_type=args.score_type, pos_exist=args.pos_exist, use_type_embed=args.use_type_embed,
        use_gating=args.use_gating
    ).to(device)
    
    # 正常模式下，这里传入 save_dir 用于保存详细日志
    train(
        model=model, train_loader=train_loader, val_loader=val_loader, device=device,
        epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay, grad_clip=args.grad_clip,
        amp=not args.no_amp, save_dir=result_dir, save_name=ckpt_name, eval_topk=args.topk,
        log_interval=args.log_interval, loss_type=args.loss_type,
        listmle_topm=args.listmle_topm, listnet_pred_temp=args.listnet_pred_temp,
        listnet_tgt_temp=args.listnet_tgt_temp, pair_num_pos=args.pair_num_pos,
        pair_num_neg=args.pair_num_neg, pair_margin=args.pair_margin,
        trial=None # 正常模式不传 trial
    )
    
    # Evaluation (if test set is provided)
    if args.test_npz:
        best_path = os.path.join(result_dir, ckpt_name)
        if os.path.exists(best_path):
           ckpt = torch.load(best_path, map_location=device)
           model.load_state_dict(ckpt["model"])
        
        # Reuse training normalization stats
        mean_std = None
        if args.normalize:
            base_ds = train_loader.dataset
            if hasattr(base_ds, "dataset"): base_ds = base_ds.dataset
            mean_std = getattr(base_ds, "norm_stats", None)
        mean, std = (None, None) if mean_std is None else mean_std

        test_ds = NPZClusterDataset(args.test_npz, normalize=args.normalize, mean=mean, std=std, centroids=C)
        if args.test_first_n is not None:
            n = min(args.test_first_n, len(test_ds))
            test_ds = SubsetByIndex(test_ds, list(range(n)))
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        
        kld, mae, mse, ordacc, recall = evaluate(
            model, test_loader, device, topk=args.topk, loss_type=args.loss_type,
            listmle_topm=args.listmle_topm, listnet_pred_temp=args.listnet_pred_temp,
            listnet_tgt_temp=args.listnet_tgt_temp, pair_num_pos=args.pair_num_pos,
            pair_num_neg=args.pair_num_neg, pair_margin=args.pair_margin
        )
        print(f"[TEST] kld={kld:.6f} | recall@{args.topk}={recall:.4f}")

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

if __name__ == "__main__":
    main()

"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --val_split 0.01 \
  --topk 10 \
  --epochs 2 \
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
  --use_gating \
  --loss_type kld \
  --experiment_id gist1M_leaf20K

'''
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/sift1M_learn/leafsize20K/train_sift.npz \
  --centroids_path input/Training_data/sift1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/sift1M_learn/leafsize20K/test_sift.npz \
  --val_split 0.01 \
  --topk 100 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20251020_sift1M_leaf20K_test
"""

""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/train_deep.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/test_deep.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/centroids.npy \
  --val_split 0.05 \
  --topk 20 \
  --epochs 20 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20251118_deep2M_leaf20K
"""

""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/train_deep.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/test_deep.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/centroids.npy \
  --val_split 0.05 \
  --topk 40 \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 256 \
  --dropout 0.05 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20251118_deep2M_leaf20K
"""

""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/train_deep.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/test_deep.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/centroids.npy \
  --val_split 0.05 \
  --topk 40 \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 256 \
  --dropout 0.05 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20251118_deep2M_leaf60K
"""

""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/train_deep.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/test_deep.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/centroids.npy \
  --val_split 0.05 \
  --topk 40 \
  --epochs 20 \
  --batch_size 128 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 256 \
  --dropout 0.05 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20251118_deep2M_leaf60K
"""


""" 
conda activate elpis_torch
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/train_sift10M.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/test_sift10M.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/centroids.npy \
  --val_split 0.05 \
  --topk 100 \
  --epochs 20 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 128 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 256 \
  --dropout 0.05 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20260114_sift10M_leaf50K
"""

""" 
conda activate elpis_torch
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/train_sift10M.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/test_sift10M.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/centroids.npy \
  --val_split 0.05 \
  --topk 30 \
  --epochs 2 \
  --batch_size 1280 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 200 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --test_first_n 2000 \
  --experiment_id 20260114_sift10M_leaf20W_allLearnSamples_2KQuerySamples
"""


""" 
conda activate elpis_torch
python -m src.model.cluster_dist_transformer_original \
  --train_npz      /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/train_sift10M.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/test_sift10M.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/centroids.npy \
  --val_split 0.05 \
  --topk 20 \
  --epochs 2 \
  --batch_size 2560 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 4 \
  --num_layers 8 \
  --dim_ff 512 \
  --dropout 0.3 \
  --score_type bilinear \
  --log_interval 200 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20260115_sift10M_leaf40W
"""

# sift1M leafsize=20K
""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/sift1M/leafsize20K/train_sift1M.npz \
  --centroids_path input/Training_data/sift1M/leafsize20K/centroids.npy \
  --test_npz input/Training_data/sift1M/leafsize20K/test_sift1M.npz \
  --val_split 0.01 \
  --topk 25 \
  --epochs 10 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --log_interval 100 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20260116_sift1M_leaf20K
"""

#  optuna

""" 
python -m src.model.cluster_dist_transformer_original \
  --train_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/train_sift10M.npz \
  --test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/test_sift10M.npz \
  --centroids_path /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/centroids.npy \
  --val_split 0.05 \
  --topk 20 \
  --epochs 2 \
  --pos_exist \
  --use_gating \
  --loss_type kld \
  --experiment_id 20260115_sift10M_leaf40W_OPTUNA \
  --use_optuna \
  --optuna_trials 500
"""