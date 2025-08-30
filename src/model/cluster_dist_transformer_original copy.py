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


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# (K, D)
def load_centroids(centroids_path: str) -> np.ndarray:
    arr = np.load(centroids_path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):  # npz file
        if "centroids" in arr.files:
            c = arr["centroids"]
        else:
            key = arr.files[0]  # use the first array in npz if key not specified
            c = arr[key]
    else:
        c = arr
    c = c.astype(np.float32)
    assert c.ndim == 2, f"centroids should be 2D, got {c.shape}"
    return c

def build_loaders(
    train_npz: str,  # (queries, targets)
    val_npz: Optional[str],
    batch_size: int,
    num_workers: int,
    normalize: bool,
    val_split: float,
    seed: int,
    centroids: Optional[np.ndarray],  # centroids: (K, D)
):
    full_train = NPZClusterDataset(train_npz, normalize=normalize, centroids=centroids)
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
    def __init__(
        self,
        npz_path: str,
        normalize: bool = False,
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
        use_gating: bool = False  # 是否使用门控机制
    ):
        super().__init__()
        self.pos_exist = pos_exist
        self.use_gating = use_gating
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len) # （max_len, d_model）
        self.type_embed = nn.Embedding(2, d_model) if use_type_embed else None
        
        # 门控机制层
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
        # x: [B, K+1, D]
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

        # 应用门控机制
        if self.use_gating:
            gate = self.gate_activation(self.gate_linear(h))  # [B, L, d_model]
            h = h * gate  # 元素级别相乘，门控输出

        q = h[:, 0, :]  # [B, d_model]
        c = h[:, 1:, :]  # [B, K, d_model]

        if self.score_type == "bilinear":
            qh = self.query_head(q)  # [B, d_model]
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
    log_interval: int = 100, # steps between batch logs (0 to disable)
    loss_type: str = "kld",
    # 新增损失函数相关参数
    listmle_topm: Optional[int] = None,
    listnet_pred_temp: float = 1.0,
    listnet_tgt_temp: Optional[float] = None,
    pair_num_pos: int = 1,
    pair_num_neg: int = 20,
    pair_margin: float = 0.1
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) #定义一个学习率调度器，按余弦曲线在训练过程中逐渐降低学习率
    scaler = torch.amp.GradScaler(enabled=amp and torch.cuda.is_available()) # amp=True（用户允许混合精度）
    
    if loss_type == "kld":
        lossFunc = nn.KLDivLoss(reduction="batchmean")
    elif loss_type == "kld_reverse":
        lossFunc = None  # handled explicitly in loop for proper gradients
    elif loss_type == "mse":
        lossFunc = nn.MSELoss()
    elif loss_type == "hybrid": # mse kld混合损失
        lossFunc = HybridLoss(mse_weight=0.7, kl_weight=0.3, temperature=1.0, smoothing=0.01)
    elif loss_type == "recall_focused_loss":
        lossFunc = recall_focused_loss_batched
    elif loss_type == "listnet":
        lossFunc = None  # handled explicitly in loop
    elif loss_type == "listmle":
        lossFunc = None  # handled explicitly in loop
    elif loss_type == "pairwise_hinge":
        lossFunc = None  # handled explicitly in loop
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")

    best_val = float("inf")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # record validation metrics for plotting
    epoch_history = []
    
    # 记录epoch=1时每个batch的验证准确率
    epoch1_batch_metrics = []
    
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_ordacc = 0.0
        running_recall = 0.0
        n = 0
        
        # 在epoch=1时，记录每个batch后的验证准确率
        if ep == 1 and val_loader is not None:
            batch_count = 0
        

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                probs = torch.softmax(logits, dim=-1)
                log_probs = torch.log_softmax(logits, dim=-1)
                if loss_type == "kld_reverse":
                    # Reverse KL: KL(pred || target) = sum p*(log p - log q)
                    y_log = torch.log(y + 1e-8)
                    loss = (probs * (log_probs - y_log)).sum(dim=-1).mean()
                elif loss_type == "kld":
                    loss = lossFunc(log_probs, y)  # KL(target || pred)
                elif loss_type == "mse":
                    loss = lossFunc(probs, y) # 标量
                elif loss_type == "hybrid":     # mse kld混合损失
                    loss, _ = lossFunc(logits, y)
                elif loss_type == "recall_focused_loss": # 专门针对recall优化的损失函数
                    loss = lossFunc(logits, y, eval_topk)  # 使用logits确保梯度正确传播
                elif loss_type == "listnet":
                    loss = listnet_top1_loss(
                        logits, y,
                        pred_temp=listnet_pred_temp,
                        tgt_temp=listnet_tgt_temp,
                        assume_targets_prob=True
                    )
                elif loss_type == "listmle":
                    loss = listmle_loss(
                        logits, y,
                        use_topm=listmle_topm
                    )
                elif loss_type == "pairwise_hinge":
                    loss = pairwise_hinge_loss(
                        logits, y,
                        num_pos=pair_num_pos,
                        num_neg=pair_num_neg,
                        margin=pair_margin
                    )
                else:
                    raise ValueError(f"Invalid loss type: {loss_type}")
                train_ordacc=topk_ordered_accuracy(probs, y, k=eval_topk)
                train_recall=topk_recall(probs, y, k=eval_topk)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0) # 累积训练损失总和
            running_ordacc+=train_ordacc * x.size(0)
            running_recall+=train_recall * x.size(0)
            n += x.size(0)
            
            # 在epoch=1时，每个batch后评估验证集
            if ep == 1 and val_loader is not None:
                batch_count += 1

                if batch_count % 10 == 0: # 每10个batch评估一次
                    val_batch_metrics = []
                    for val_x, val_y in val_loader:
                        val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate_batch(
                            model, val_x, val_y, device, topk=eval_topk, loss_type=loss_type,
                            listmle_topm=listmle_topm,
                            listnet_pred_temp=listnet_pred_temp,
                            listnet_tgt_temp=listnet_tgt_temp,
                            pair_num_pos=pair_num_pos,
                            pair_num_neg=pair_num_neg,
                            pair_margin=pair_margin
                        )
                        val_batch_metrics.append((val_kld, val_mae, val_mse, val_ordacc, val_recall))
                    
                    # 计算平均指标
                    avg_val_kld = sum(m[0] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_mae = sum(m[1] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_mse = sum(m[2] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_ordacc = sum(m[3] for m in val_batch_metrics) / len(val_batch_metrics)
                    avg_val_recall = sum(m[4] for m in val_batch_metrics) / len(val_batch_metrics)
                    
                    epoch1_batch_metrics.append({
                        # test metrics per batch of epoch 1
                        'batch': batch_count,
                        'val_kld': avg_val_kld,
                        'val_mae': avg_val_mae,
                        'val_mse': avg_val_mse,
                        'val_ordacc': avg_val_ordacc,
                        'val_recall': avg_val_recall,
                        
                        # train metrics per batch of epoch 1
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
                    print(f"Epoch {ep:03d} | Step {step_idx} | lr={lr_now:.6e} | loss={loss.item():.6f}")
        sched.step()
        train_loss = running / n           # average KLDLoss per epoch
        train_ordacc = running_ordacc / n
        train_recall = running_recall / n

        if val_loader is None:
            epoch_history.append({
                "train_loss": train_loss,
                "train_ordacc": train_ordacc,
                "train_recall": train_recall
            })
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f}")
        elif val_loader is not None:
            val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate(
                model, val_loader, device, topk=eval_topk, loss_type=loss_type,
                listmle_topm=listmle_topm,
                listnet_pred_temp=listnet_pred_temp,
                listnet_tgt_temp=listnet_tgt_temp,
                pair_num_pos=pair_num_pos,
                pair_num_neg=pair_num_neg,
                pair_margin=pair_margin
            )
            epoch_history.append({
                "train_loss": train_loss,
                "train_ordacc": train_ordacc,
                "train_recall": train_recall,
                "val_kld":val_kld,
                "val_ordacc":val_ordacc,
                "val_recall":val_recall
            })
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f} | val_kld={val_kld:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | ord_acc@{eval_topk}={val_ordacc:.4f} | recall@{eval_topk}={val_recall:.4f}")
            if val_kld < best_val and save_dir: # 保存最好的模型：最小化KL散度
                best_val = val_kld
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "val_kld": val_kld},
                    os.path.join(save_dir, save_name),
                )

    # 绘制epoch=1时每个batch的验证准确率变化图
    if len(epoch1_batch_metrics) > 0 and save_dir:
        plot_epoch1_batch_metrics(epoch1_batch_metrics, save_dir, save_name, eval_topk)

    if val_loader is not None and save_dir:
        plot_epoch_metrics(save_dir, save_name, epoch_history)



def parse_args():
    p = argparse.ArgumentParser(description="Transformer for predicting NN cluster distribution")
    # data
    p.add_argument("--train_npz", type=str, default=None, help="Training npz (contains queries and targets)")
    p.add_argument("--val_npz", type=str, default=None, help="Validation npz (optional)")
    p.add_argument("--test_npz", type=str, default=None, help="Test npz (optional)")
    p.add_argument("--centroids_path", type=str, default=None, help="Global shared centroids (.npy or .npz), shape [K, D]")
    p.add_argument("--val_split", type=float, default=0.1, help="Split ratio from training set when no val_npz is provided")
    p.add_argument("--normalize", action="store_true", help="Apply feature-wise standardization")
    # synth
    p.add_argument("--gen_synth", action="store_true", help="Generate and use synthetic data to validate the pipeline")
    p.add_argument("--synth_n", type=int, default=20000, help="Number of synthetic samples")
    p.add_argument("--synth_K", type=int, default=64, help="Number of clusters (K) for synthetic data")
    p.add_argument("--synth_D", type=int, default=128, help="Feature dimension (D) for synthetic data")
    # train
    p.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    p.add_argument("--batch_size", type=int, default=256, help="Batch size")
    p.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
    p.add_argument("--grad_clip", type=float, default=1.0, help="Gradient clipping (max norm)")
    p.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_dir", type=str, default="./results", help="Directory to save synthetic data and models")
    # model
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


    p.add_argument("--pos_exist", action="store_true", help="Whether to use positional encoding")
    p.add_argument("--use_type_embed", action="store_true", help="Whether to use type embedding")
    p.add_argument("--use_gating", action="store_true", help="Whether to use gating mechanism in transformer")
    p.add_argument("--loss_type", type=str,
                   choices=["kld","kld_reverse","mse","hybrid","recall_focused_loss",
                            "listnet","listmle","pairwise_hinge"],
                   default="kld")
    
    # 与新 loss 相关的超参（可选）
    p.add_argument("--listmle_topm", type=int, default=None, help="ListMLE 仅用前 m 个目标")
    p.add_argument("--listnet_pred_temp", type=float, default=1.0)
    p.add_argument("--listnet_tgt_temp", type=float, default=None)
    p.add_argument("--pair_num_pos", type=int, default=1)
    p.add_argument("--pair_num_neg", type=int, default=20)
    p.add_argument("--pair_margin", type=float, default=0.1)
    
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.gen_synth:
        queries, centroids, targets = make_synth_dataset(args.synth_n, args.synth_K, args.synth_D)
        train_npz = os.path.join(args.save_dir, "synth_train.npz")
        val_npz = os.path.join(args.save_dir, "synth_val.npz")
        test_npz = os.path.join(args.save_dir, "synth_test.npz")
        # Split 80/10/10
        n = queries.shape[0]
        n_train = int(n * 0.8)
        n_val = int(n * 0.1)
        save_npz(train_npz, queries[:n_train], targets[:n_train])
        save_npz(val_npz, queries[n_train:n_train + n_val], targets[n_train:n_train + n_val])
        save_npz(test_npz, queries[n_train + n_val:], targets[n_train + n_val:])
        # Save shared centroids and wire it up
        centroids_path = os.path.join(args.save_dir, "synth_centroids.npy")
        np.save(centroids_path, centroids)
        args.train_npz, args.val_npz, args.test_npz = train_npz, val_npz, test_npz
        args.centroids_path = centroids_path

    if args.train_npz is None:
        raise ValueError("You must provide --train_npz or use --gen_synth")
    
    # Load global centroids (required)
    if args.centroids_path is None:
        raise ValueError("You must provide --centroids_path (global centroids) when using queries-only data.")
    C = load_centroids(args.centroids_path)
    
    # Determine dimensions solely from centroids
    K, D = C.shape
    print(f"Detected K={K}, D={D}")
    # Save checkpoints next to the centroids file
    ckpt_dir = os.path.dirname(args.centroids_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Build a descriptive checkpoint name using key hyperparameters
    ckpt_name = (
        f"model_d{args.d_model}_L{args.num_layers}_H{args.nhead}_ff{args.dim_ff}_topk{args.topk}"
        f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{args.score_type}_normalize{args.normalize}"
        f"_pos{args.pos_exist}_gating{args.use_gating}_loss_type{args.loss_type}_use_type_embed{args.use_type_embed}.pt"
    )
    # Create a subdirectory under ckpt_dir named after ckpt_name without the .pt suffix
    subdir_name = os.path.splitext(ckpt_name)[0]
    result_dir = os.path.join(ckpt_dir, subdir_name)
    os.makedirs(result_dir, exist_ok=True)

    train_loader, val_loader = build_loaders(
        args.train_npz,
        args.val_npz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize, # normalize the features
        val_split=args.val_split, # split ratio from training set when no val_npz is provided
        seed=args.seed,
        centroids=C,
    )

    model = ClusterDistTransformer(
        input_dim=D,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        score_type=args.score_type, # bilinear or mlp
        pos_exist=args.pos_exist,
        use_type_embed=args.use_type_embed,
        use_gating=args.use_gating  # 门控机制参数
    ).to(device)

    # epoch=0,保存模型参数（未训练的随机初始化模型）
    if args.epochs == 0:
        torch.save(
            {"model": model.state_dict(), "epoch": args.epochs},
            os.path.join(result_dir, ckpt_name),
        )
        print(f"Saved untrained model to {os.path.join(result_dir, ckpt_name)}")
    else:
        # 正常训练
        train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, # gradient clipping
            amp=not args.no_amp, # mixed precision training
            save_dir=result_dir,
            save_name=ckpt_name,
            eval_topk=args.topk,
            log_interval=args.log_interval,
            loss_type=args.loss_type,
            # 新增损失函数相关参数
            listmle_topm=args.listmle_topm,
            listnet_pred_temp=args.listnet_pred_temp,
            listnet_tgt_temp=args.listnet_tgt_temp,
            pair_num_pos=args.pair_num_pos,
            pair_num_neg=args.pair_num_neg,
            pair_margin=args.pair_margin
        )

    # Evaluation (if test set is provided)
    if args.test_npz:
        # Reload the best checkpoint if available
        best_path = os.path.join(result_dir, ckpt_name)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded checkpoint from {best_path} (epoch={ckpt.get('epoch', 'unknown')})")

        # Reuse training normalization stats
        mean_std = None
        if args.normalize:
            base_ds = train_loader.dataset
            if hasattr(base_ds, "dataset"):  # Subset
                base_ds = base_ds.dataset
            mean_std = getattr(base_ds, "norm_stats", None)
        mean, std = (None, None) if mean_std is None else mean_std

        test_ds = NPZClusterDataset(args.test_npz, normalize=args.normalize, mean=mean, std=std, centroids=C)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        kld, mae, mse, ordacc, recall = evaluate(
            model, test_loader, device, topk=args.topk, loss_type=args.loss_type,
            listmle_topm=args.listmle_topm,
            listnet_pred_temp=args.listnet_pred_temp,
            listnet_tgt_temp=args.listnet_tgt_temp,
            pair_num_pos=args.pair_num_pos,
            pair_num_neg=args.pair_num_neg,
            pair_margin=args.pair_margin
        )
        print(f"[TEST] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | ord_acc@{args.topk}={ordacc:.4f} | recall@{args.topk}={recall:.4f}")


if __name__ == "__main__":
    main()


"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
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
  --loss_type kld_reverse

# ListMLE损失函数版本 - 基于最大似然估计的列表排序损失
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
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
  --loss_type listmle \
  --listmle_topm 20

# ListNet损失函数版本 - 基于概率分布的排序损失，支持温度参数
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.01 \
  --topk 10 \
  --epochs 10 \
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
  --loss_type listnet \
  --listnet_pred_temp 1.5 \
  --listnet_tgt_temp 1.0

# Pairwise Hinge损失函数版本 - 成对比较的铰链损失，适用于排序任务
python -m src.model.cluster_dist_transformer_original \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/gist1M_learn/leafsize20K/test_gist.npz \
  --val_split 0.01 \
  --topk 10 \
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
  --log_interval 10 \
  --pos_exist \
  --use_gating \
  --loss_type pairwise_hinge \
  --pair_num_pos 2 \
  --pair_num_neg 15 \
  --pair_margin 0.2
'''


'''
python -m src.model.cluster_dist_transformer \
  --train_npz input/Training_data/sift1M_learn/leafsize10K/train_sift.npz \
  --centroids_path input/Training_data/sift1M_learn/leafsize10K/centroids.npy \
  --val_split 0.1 \
  --topk 10 \
  --epochs 150 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear

python -m src.model.cluster_dist_transformer \
  --train_npz input/Training_data/sift1M_learn/leafsize20K/train_sift.npz \
  --centroids_path input/Training_data/sift1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --topk 10 \
  --epochs 150 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear
"""