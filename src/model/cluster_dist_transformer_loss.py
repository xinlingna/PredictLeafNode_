import argparse
import math
import os
import random
import time
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


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
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.type_embed = nn.Embedding(2, d_model) if use_type_embed else None

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
        t = self.pos_enc(t)
        h = self.encoder(t)  # [B, L, d_model]

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


def topk_recall(pred_prob: torch.Tensor, label_prob: torch.Tensor, k: int = 10) -> float:
    """
    Compute Top-K recall over a batch.
    pred_prob: (B, K)
    label_prob: (B, K)
    Returns average recall@k in [0,1].
    """
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

def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized batch version:
    - Penalize mass on zero-target clusters (push probs on negatives -> 0)
    - Enforce descending order among non-zero targets to match target order
    - Enforce positives outrank negatives (with all negatives, no sampling)

    logits, y: [B, K]
    """
    B, K = logits.shape
    device = logits.device

    probs = torch.softmax(logits, dim=1)     # [B,K]
    pos_mask = (y > 0)                       # [B,K]
    neg_mask = ~pos_mask                     # [B,K]

    # 1) Negative mass penalty
    neg_mass = (probs * neg_mask.float()).sum(dim=1).mean()

    # Select positive candidates (all positives, assume K small ~100)
    P = K
    y_pos_only = y.masked_fill(~pos_mask, float("-inf"))   # exclude non-positives
    pos_idx = torch.topk(y_pos_only, k=P, dim=1).indices   # [B,P]
    pos_valid = pos_mask.gather(1, pos_idx).float()        # [B,P], which gathered are truly positive

    logits_pos = logits.gather(1, pos_idx)  # [B,P]
    y_pos_vals = y.gather(1, pos_idx)       # [B,P]

    # 2) Positive-positive ranking: match target order
    pos_order = torch.argsort(y_pos_vals, dim=1, descending=True)  # [B,P]
    logits_pos_sorted = torch.gather(logits_pos, 1, pos_order)     # [B,P]
    valid_sorted = torch.gather(pos_valid, 1, pos_order)           # [B,P]

    diffs_pp = logits_pos_sorted.unsqueeze(2) - logits_pos_sorted.unsqueeze(1)  # [B,P,P]
    tri = torch.triu(torch.ones((P, P), dtype=torch.bool, device=device), diagonal=1)  # [P,P]
    pair_valid = (valid_sorted.unsqueeze(2) * valid_sorted.unsqueeze(1)).bool()       # [B,P,P]
    mask_pp = tri.unsqueeze(0) & pair_valid
    pos_rank_terms = torch.nn.functional.softplus(-diffs_pp)
    pos_rank_loss = (pos_rank_terms * mask_pp.float()).sum() / mask_pp.float().sum().clamp_min(1.0)

    # 3) Positive-negative ranking with all negatives (no sampling)
    diffs_pn = logits_pos.unsqueeze(2) - logits.unsqueeze(1)  # [B, P, K]
    mask_pn = pos_valid.unsqueeze(2) * neg_mask.float().unsqueeze(1)  # [B, P, K]
    pos_neg_terms = torch.nn.functional.softplus(-diffs_pn)
    pos_neg_loss = (pos_neg_terms * mask_pn).sum() / mask_pn.sum().clamp_min(1.0)

    return neg_mass + pos_rank_loss + pos_neg_loss


def pairwise_rank_loss_per(logits: torch.Tensor, y: torch.Tensor, topk: int = 10, neg_samples: int = 20) -> torch.Tensor:
    """
    Enhanced pairwise loss matching your goals:
    - Drive probabilities on zero-target clusters to 0 (support consistency)
    - Among non-zero (positive) clusters, enforce the same descending order as targets

    Args:
      logits: [B, K]
      y:      [B, K] (probability targets; non-zero entries define the positive support)
      topk:   if > 0, use only top-k positives (by target) for positive ranking to reduce cost, otherwise use all positives
      neg_samples: number of sampled negative classes (from zero-target set) per item for pos-vs-neg ranking
    Returns:
      scalar loss tensor
    """
    B, K = logits.size()
    probs = torch.softmax(logits, dim=-1)
    losses = []
    for b in range(B):
        pos_idx = (y[b] > 0).nonzero(as_tuple=True)[0]  # positive support (non-zero targets)
        neg_idx = (y[b] <= 0).nonzero(as_tuple=True)[0] # negative support (zero-target clusters)

        # 1) Penalize probability mass on negatives → encourage probs at zero-target clusters to be 0
        neg_mass_b = probs[b, neg_idx].sum() if neg_idx.numel() > 0 else logits.new_zeros(())

        # If no positives, only apply negative mass penalty
        if pos_idx.numel() < 1:
            losses.append(neg_mass_b)
            continue

        # Optionally restrict to top-k positives by target prob to reduce cost
        if topk > 0 and pos_idx.numel() > topk:
            vals = y[b, pos_idx]
            order = torch.argsort(vals, descending=True)[:topk]
            pos_idx = pos_idx[order]

        # 2) Rank agreement within positives (non-zero targets): enforce target-descending order in logits
        vals_pos = y[b, pos_idx]
        order_pos = torch.argsort(vals_pos, descending=True)
        pos_sorted = pos_idx[order_pos]
        s_pos_sorted = logits[b, pos_sorted]  # [P]
        if s_pos_sorted.numel() >= 2:
            diff_pp = s_pos_sorted.unsqueeze(1) - s_pos_sorted.unsqueeze(0)  # [P,P]
            mask = torch.triu(torch.ones_like(diff_pp, dtype=torch.bool), diagonal=1)
            diffs = diff_pp[mask]
            pos_rank_loss_b = torch.nn.functional.softplus(-diffs).mean() if diffs.numel() > 0 else logits.new_zeros(())
        else:
            pos_rank_loss_b = logits.new_zeros(())

        # 3) Positives should outrank negatives (optional but helpful)
        if neg_idx.numel() > 0:
            if neg_idx.numel() > neg_samples:
                sel = torch.randint(0, neg_idx.numel(), (neg_samples,), device=logits.device)
                neg_samp = neg_idx[sel]
            else:
                neg_samp = neg_idx
            s_pos = logits[b, pos_idx].unsqueeze(1)  # [P,1]
            s_neg = logits[b, neg_samp].unsqueeze(0)  # [1,M]
            diff_pn = s_pos - s_neg
            pos_neg_loss_b = torch.nn.functional.softplus(-diff_pn).mean()
        else:
            pos_neg_loss_b = logits.new_zeros(())

        # Combine the three parts
        loss_b = neg_mass_b + pos_rank_loss_b + pos_neg_loss_b
        losses.append(loss_b)

    if not losses:
        return logits.new_zeros(())
    return torch.stack(losses).mean()


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, topk: int = 10) -> Tuple[float, float, float, float, float, float]:
    model.eval()
    kldiv = nn.KLDivLoss(reduction="batchmean")
    mae_meter, mse_meter, kld_meter, rank_meter = 0.0, 0.0, 0.0, 0.0 # 
    total = 0
    recall_meter = 0.0
    ordered_acc_meter = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.softmax(logits, dim=-1)
        loss_kld = kldiv(log_probs, y)
        loss_rank = pairwise_rank_loss(logits, y)
        mae = torch.mean(torch.abs(preds - y))
        mse = torch.mean((preds - y) ** 2)


        k = min(topk, preds.size(1))
        recall = topk_recall(preds, y, k=k)
        ordered_acc = topk_ordered_accuracy(preds, y, k=k)

        bs = x.size(0)
        kld_meter += loss_kld.item() * bs
        rank_meter += loss_rank.item() * bs
        mae_meter += mae.item() * bs
        mse_meter += mse.item() * bs
        recall_meter += recall * bs
        ordered_acc_meter += ordered_acc * bs
        total += bs
        # print(f"test kld={loss_kld.item():.6f} | rank={loss_rank.item():.6f} | mae={mae.item():.6f} | mse={mse.item():.6f} | recall@{topk}={recall:.4f}")
    return (
        kld_meter / total,
        rank_meter / total,
        mae_meter / total,
        mse_meter / total,
        recall_meter / total,
        ordered_acc_meter / total,
    )


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
    rank_lambda: float = 0.0,
    rank_topk: int = 10,
    log_interval: int = 100,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay) # 优化器
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) # 学习率衰减
    scaler = torch.amp.GradScaler(enabled=amp and device.type == "cuda") # 混合精度训练
    kldiv = nn.KLDivLoss(reduction="batchmean") # loss

    best_val = float("inf")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running = 0.0
        running_kl = 0.0
        running_rank = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss_kl = kldiv(log_probs, y)
                loss = loss_kl
                if rank_lambda > 0.0:
                    loss_rank = pairwise_rank_loss(logits, y)
                    loss = loss + rank_lambda * loss_rank
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0) # 累积训练损失loss总和
            running_kl += loss_kl.item() * x.size(0) # 累积训练KL损失loss_kl总和
            if rank_lambda > 0.0:
                running_rank += (loss_rank.item() if 'loss_rank' in locals() else 0.0) * x.size(0) # 累积训练loss_rank排名损失总和
            n += x.size(0)

            # Print batch results (interval)
            if log_interval > 0:
                step_idx = n // x.size(0)
                if step_idx % log_interval == 0:
                    batch_loss = loss.item()
                    batch_kl = loss_kl.item()
                    batch_rank = loss_rank.item() if rank_lambda > 0.0 else 0.0
                    lr_now = opt.param_groups[0]['lr']
                    print(f"Epoch {ep:03d} | Step {step_idx} | lr={lr_now:.6e} | loss={batch_loss:.6f} | kl={batch_kl:.6f} | rank={batch_rank:.6f}")
        sched.step()
        train_loss = running / n
        train_kl = running_kl / n
        train_rank = (running_rank / n) if rank_lambda > 0.0 else 0.0

        if val_loader is not None:
            # use the same top-k as ranking loss for validation recall to keep logic consistent
            val_kld, val_rank, val_mae, val_mse, val_recall, val_ordacc = evaluate(model, val_loader, device, topk=rank_topk)
            if rank_lambda > 0.0:
                print(f"epoch {ep:03d} | train_loss={train_loss:.6f} | train_kl={train_kl:.6f} | train_rank={train_rank:.6f} | val_kld={val_kld:.6f} | val_rank={val_rank:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | recall@{rank_topk}={val_recall:.4f} | ord_acc@{rank_topk}={val_ordacc:.4f}")
            else:
                print(f"epoch {ep:03d} | train_kl={train_kl:.6f} | val_kld={val_kld:.6f} | val_rank={val_rank:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | recall@{rank_topk}={val_recall:.4f} | ord_acc@{rank_topk}={val_ordacc:.4f}")
            # choose metric consistent with training objective
            val_metric = val_kld if rank_lambda <= 0.0 else (val_kld + rank_lambda * val_rank)
            if val_metric < best_val and save_dir:
                best_val = val_metric
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "val_kld": val_kld, "val_rank": val_rank, "val_metric": val_metric},
                    os.path.join(save_dir, save_name),
                )
        # epoch end summary
        elapsed = time.time() - epoch_start
        total_samples = n
        throughput = total_samples / elapsed if elapsed > 0 else 0.0
        if val_loader is None:
            print(f"epoch {ep:03d} done in {elapsed:.1f}s | samples/s={throughput:.1f} | train_loss={train_loss:.6f} | kl={train_kl:.6f}")
        else:
            print(f"epoch {ep:03d} done in {elapsed:.1f}s | samples/s={throughput:.1f} | train_loss={train_loss:.6f} | kl={train_kl:.6f} | rank={train_rank:.6f} | val_metric={best_val:.6f}")

    if val_loader is not None and save_dir:
        print(f"best val_metric: {best_val:.6f} (saved to {os.path.join(save_dir, save_name)})")


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


def parse_args():
    p = argparse.ArgumentParser(description="Transformer for predicting NN cluster distribution")
    # data
    p.add_argument("--train_npz", type=str, default=None, help="Training npz (contains queries and targets)")
    p.add_argument("--val_npz", type=str, default=None, help="Validation npz (optional)")
    p.add_argument("--test_npz", type=str, default=None, help="Test npz (optional)")
    p.add_argument("--centroids_path", type=str, default=None, help="Global shared centroids (.npy or .npz), shape [K, D]")
    p.add_argument("--save_dir", type=str, default="runs", help="Directory to save outputs when using --gen_synth")
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
    # ranking loss
    p.add_argument("--rank_lambda", type=float, default=0.0, help="Weight for pairwise ranking loss (0 to disable)")
    p.add_argument("--rank_topk", type=int, default=10, help="Top-k positives (by target prob) for ranking loss")
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

    train_loader, val_loader = build_loaders(
        args.train_npz,
        args.val_npz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
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
        score_type=args.score_type,
    ).to(device)

    # Build a descriptive checkpoint name using key hyperparameters
    ckpt_name = (
        f"model_d{args.d_model}_L{args.num_layers}_H{args.nhead}_ff{args.dim_ff}"
        f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}_{args.score_type}"
        f"_rank{args.rank_lambda}_tk{args.rank_topk}.pt"
    )

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
        save_dir=ckpt_dir,
        save_name=ckpt_name,
        rank_lambda=args.rank_lambda,
        rank_topk=args.rank_topk,
        log_interval=args.log_interval,
    )

    # Evaluation (if test set is provided)
    if args.test_npz:
        # Reload the best checkpoint if available
        best_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint from {best_path} (val_metric={ckpt.get('val_metric', ckpt.get('val_kld', None))})")

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
        kld, rank, mae, mse, recall, ordacc = evaluate(model, test_loader, device, topk=args.topk)
        print(f"[TEST] kld={kld:.6f} | rank={rank:.6f} | mae={mae:.6f} | mse={mse:.6f} | recall@{args.topk}={recall:.4f} | ord_acc@{args.topk}={ordacc:.4f}")


if __name__ == "__main__":
    main()


"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.cluster_dist_transformer_loss \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --topk 10 \
  --epochs 50 \
  --batch_size 512 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 4 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --rank_lambda 0.5 \
  --rank_topk 10 \
  --log_interval 100
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