import argparse
import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# --------------------
# Utilities
# --------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_centroids(centroids_path: str) -> np.ndarray:
    """Load global centroids [K, D] from .npy or .npz (first array or 'centroids')."""
    arr = np.load(centroids_path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        if "centroids" in arr.files:
            c = arr["centroids"]
        else:
            c = arr[arr.files[0]]
    else:
        c = arr
    c = c.astype(np.float32)
    assert c.ndim == 2, f"centroids should be [K, D], got {c.shape}"
    return c


class NPZClusterDataset(Dataset):
    """
    Expects an npz with:
      - queries: [N, D]
      - targets or labels: [N, K] (probabilities over clusters)
    And a shared global centroids array [K, D].
    Each item returns:
      x: [K+1, D] = [query; centroids]
      y: [K]
    """

    def __init__(
        self,
        npz_path: str,
        normalize: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        centroids: Optional[np.ndarray] = None,
    ):
        data = np.load(npz_path)
        if "queries" not in data:
            raise ValueError(f"{npz_path} must contain 'queries' [N, D]")
        tgt = data.get("targets", data.get("labels", None))
        if tgt is None:
            raise ValueError(f"{npz_path} must contain 'targets' [N, K] or 'labels'")

        self.queries = data["queries"].astype(np.float32)  # [N, D]
        self.targets = tgt.astype(np.float32)                # [N, K]

        if centroids is None:
            raise ValueError("centroids (global [K, D]) is required")
        self.centroids = centroids.astype(np.float32)        # [K, D]

        N, D = self.queries.shape
        Nk, K = self.targets.shape
        Kc, Dc = self.centroids.shape
        assert N == Nk, f"N mismatch: queries={N}, targets={Nk}"
        assert K == Kc, f"K mismatch: targets={K}, centroids={Kc}"
        assert D == Dc, f"D mismatch: queries={D}, centroids={Dc}"

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

# --------------------
# Top-K metrics
# --------------------

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

# --------------------
# Set Transformer blocks (Lee et al., 2019)
# --------------------

class MAB(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_ff, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        # X: [B, N, d], Y: [B, M, d]
        H, _ = self.attn(X, Y, Y, need_weights=False)
        H = self.ln1(X + self.dropout(H))
        Z = self.ff(H)
        Z = self.ln2(H + self.dropout(Z))
        return Z


class SAB(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.0):
        super().__init__()
        self.mab = MAB(d_model, nhead, dim_ff, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.mab(X, X)


class ISAB(nn.Module):
    """
    Induced Set Attention Block for scalability: O(Nm) with m inducing points.
    """

    def __init__(self, d_model: int, nhead: int, dim_ff: int, m: int, dropout: float = 0.0):
        super().__init__()
        self.I = nn.Parameter(torch.randn(m, d_model) / math.sqrt(d_model))
        self.mab1 = MAB(d_model, nhead, dim_ff, dropout)
        self.mab2 = MAB(d_model, nhead, dim_ff, dropout)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        B = X.size(0)
        I = self.I.unsqueeze(0).expand(B, -1, -1)  # [B, m, d]
        H = self.mab1(I, X)  # [B, m, d]
        Z = self.mab2(X, H)  # [B, N, d]
        return Z


class SetEncoder(nn.Module):
    """
    Permutation-equivariant encoder over tokens without positional encodings.
    Distinguishes query vs centroid via a small type embedding.
    Optionally uses ISAB for large K.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_type_embed: bool = True,
        use_isab: bool = True,
        induced_m: int = 64,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(2, d_model) if use_type_embed else None

        blocks = []
        for _ in range(num_layers):
            if use_isab:
                blocks.append(ISAB(d_model, nhead, dim_feedforward, m=induced_m, dropout=dropout))
            else:
                blocks.append(SAB(d_model, nhead, dim_feedforward, dropout))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, K+1, D] -> tokens: [query; centroids]
        bsz, seqlen, _ = x.shape
        h = self.input_proj(x)
        if self.type_embed is not None:
            type_ids = torch.zeros((bsz, seqlen), dtype=torch.long, device=x.device)
            if seqlen > 1:
                type_ids[:, 1:] = 1  # 0 for query, 1 for centroids
            h = h + self.type_embed(type_ids)
        for blk in self.blocks:
            h = blk(h)
        return h  # [B, K+1, d]


class ClusterDistSetTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        use_type_embed: bool = True,
        use_isab: bool = True,
        induced_m: int = 64,
        score_type: str = "bilinear",  # ["bilinear", "mlp"]
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = SetEncoder(
            input_dim=input_dim,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            use_type_embed=use_type_embed,
            use_isab=use_isab, # 是否使用ISAB
            induced_m=induced_m, # ISAB的 inducing points 数量
        )

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
        h = self.encoder(x)  # [B, K+1, d]
        q = h[:, 0, :]       # [B, d]
        c = h[:, 1:, :]      # [B, K, d]
        if self.score_type == "bilinear":
            qh = self.query_head(q)              # [B, d]
            ch = self.centroid_head(c)          # [B, K, d]
            logits = torch.einsum("bd,bkd->bk", qh, ch) / math.sqrt(self.d_model)
        else:
            q_expand = q.unsqueeze(1).expand(-1, c.size(1), -1)  # [B, K, d]
            logits = self.scorer(torch.cat([q_expand, c], dim=-1)).squeeze(-1)  # [B, K]
        return logits


# --------------------
# Train / Eval
# --------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, topk: int = 10) -> Tuple[float, float, float, float, float]:
    model.eval()
    kldiv = nn.KLDivLoss(reduction="batchmean")
    mae_meter, mse_meter, kld_meter = 0.0, 0.0, 0.0
    recall_meter, ordacc_meter = 0.0, 0.0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.softmax(logits, dim=-1)
        loss_kld = kldiv(log_probs, y)
        mae = torch.mean(torch.abs(preds - y))
        mse = torch.mean((preds - y) ** 2)
        
        # 计算topk指标（使用torch.no_grad()确保不影响梯度）
        with torch.no_grad():
            recall = topk_recall(preds, y, k=topk)
            ordacc = topk_ordered_accuracy(preds, y, k=topk)
        
        bs = x.size(0)
        kld_meter += loss_kld.item() * bs
        mae_meter += mae.item() * bs
        mse_meter += mse.item() * bs
        recall_meter += recall * bs
        ordacc_meter += ordacc * bs
        total += bs
    return kld_meter / total, mae_meter / total, mse_meter / total, ordacc_meter / total, recall_meter / total


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
    eval_topk: int = 10,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    best_val = float("inf")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        running_recall = 0.0
        running_ordacc = 0.0
        n = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = kldiv(log_probs, y)
                
                # 计算训练时的topk指标（使用torch.no_grad()确保不影响梯度）
                with torch.no_grad():
                    preds = torch.softmax(logits, dim=-1)
                    train_recall = topk_recall(preds, y, k=eval_topk)
                    train_ordacc = topk_ordered_accuracy(preds, y, k=eval_topk)
            
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0)
            running_recall += train_recall * x.size(0)
            running_ordacc += train_ordacc * x.size(0)
            n += x.size(0)
        sched.step()
        train_loss = running / n
        train_recall = running_recall / n
        train_ordacc = running_ordacc / n

        if val_loader is not None:
            val_kld, val_mae, val_mse, val_ordacc, val_recall = evaluate(model, val_loader, device, topk=eval_topk)
            print(
                f"epoch {ep:03d} | train_kld={train_loss:.6f} | train_recall@{eval_topk}={train_recall:.4f} | train_ordacc@{eval_topk}={train_ordacc:.4f} | "
                f"val_kld={val_kld:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | "
                f"val_recall@{eval_topk}={val_recall:.4f} | val_ordacc@{eval_topk}={val_ordacc:.4f}"
            )
            if val_kld < best_val and save_dir:
                best_val = val_kld
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "val_kld": val_kld, "val_recall": val_recall, "val_ordacc": val_ordacc},
                    os.path.join(save_dir, "best.pt"),
                )
        else:
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f} | train_recall@{eval_topk}={train_recall:.4f} | train_ordacc@{eval_topk}={train_ordacc:.4f}")

    if val_loader is not None and save_dir:
        print(f"best val_kld: {best_val:.6f} (saved to {os.path.join(save_dir, 'best.pt')})")


def build_loaders(
    train_npz: str,
    val_npz: Optional[str],
    batch_size: int,
    num_workers: int,
    normalize: bool,
    val_split: float,
    seed: int,
    centroids: Optional[np.ndarray],
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


# --------------------
# Synthetic data (optional)
# --------------------

def make_synth_dataset(n: int, K: int, D: int, noise: float = 0.05) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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


# --------------------
# CLI
# --------------------

def parse_args():
    p = argparse.ArgumentParser(description="Set Transformer for predicting NN cluster distribution")
    # data
    p.add_argument("--train_npz", type=str, default=None, help="Training npz (contains queries and targets)")
    p.add_argument("--val_npz", type=str, default=None, help="Validation npz (optional)")
    p.add_argument("--test_npz", type=str, default=None, help="Test npz (optional)")
    p.add_argument("--centroids_path", type=str, default=None, help="Global shared centroids (.npy or .npz), shape [K, D]")
    p.add_argument("--val_split", type=float, default=0.1, help="Split ratio from training set when no val_npz is provided")
    p.add_argument("--normalize", action="store_true", help="Apply feature-wise standardization")
    p.add_argument("--save_dir", type=str, default="./checkpoints", help="Directory to save checkpoints")
    # synth
    p.add_argument("--gen_synth", action="store_true", help="Use synthetic data to validate the pipeline")
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
    p.add_argument("--eval_topk", type=int, default=10, help="Top-k for recall and ordered accuracy evaluation")
    # model
    p.add_argument("--d_model", type=int, default=256, help="Hidden size")
    p.add_argument("--nhead", type=int, default=8, help="Number of attention heads")
    p.add_argument("--num_layers", type=int, default=4, help="Number of encoder layers")
    p.add_argument("--dim_ff", type=int, default=512, help="Feedforward hidden size")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout")
    p.add_argument("--use_isab", action="store_true", help="Use ISAB blocks (recommended for large K)")
    p.add_argument("--induced_m", type=int, default=64, help="Number of inducing points for ISAB")
    p.add_argument("--score_type", type=str, default="bilinear", choices=["bilinear", "mlp"], help="Scoring function type")
    p.add_argument("--no_amp", action="store_true", help="Disable mixed precision")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Optional synthetic data for quick sanity check
    if args.gen_synth:
        queries, centroids, targets = make_synth_dataset(args.synth_n, args.synth_K, args.synth_D)
        train_npz = os.path.join(args.save_dir, "synth_train.npz")
        val_npz = os.path.join(args.save_dir, "synth_val.npz")
        test_npz = os.path.join(args.save_dir, "synth_test.npz")
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
        raise ValueError("You must provide --centroids_path (global centroids)")

    C = load_centroids(args.centroids_path)
    K, D = C.shape
    print(f"Detected K={K}, D={D}")

    train_loader, val_loader = build_loaders(
        args.train_npz,
        args.val_npz,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        centroids=C,
    )

    model = ClusterDistSetTransformer(
        input_dim=D,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        dropout=args.dropout,
        use_type_embed=True,
        use_isab=args.use_isab,
        induced_m=args.induced_m,
        score_type=args.score_type,
    ).to(device)

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
        save_dir=args.save_dir,
        eval_topk=args.eval_topk,
    )

    # Evaluation (if test set is provided)
    if args.test_npz:
        best_path = os.path.join(args.save_dir, "best.pt")
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint from {best_path} (val_kld={ckpt.get('val_kld', None)})")

        mean_std = None
        if args.normalize:
            base_ds = train_loader.dataset
            if hasattr(base_ds, "dataset"):  # Subset
                base_ds = base_ds.dataset
            mean_std = getattr(base_ds, "norm_stats", None)
        mean, std = (None, None) if mean_std is None else mean_std

        test_ds = NPZClusterDataset(args.test_npz, normalize=args.normalize, mean=mean, std=std, centroids=C)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
        kld, mae, mse, ordacc, recall = evaluate(model, test_loader, device, topk=args.eval_topk)
        print(f"[TEST] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | ordacc@{args.eval_topk}={ordacc:.4f} | recall@{args.eval_topk}={recall:.4f}")


if __name__ == "__main__":
    main()


"""
Example usage

conda activate <your_env>
cd /home/xln/PycharmProjects/PredictLeafNode/

# Train with real data
python -m src.model.cluster_dist_setTransformer \
  --train_npz /path/to/train.npz \
  --val_npz /path/to/val.npz \
  --test_npz /path/to/test.npz \
  --centroids_path /path/to/centroids.npy \
  --normalize \
  --epochs 20 \
  --batch_size 256 \
  --score_type bilinear \
  --use_isab \
  --induced_m 64 \
  --eval_topk 10 \
  --save_dir ./checkpoints_set
  
python -m src.model.cluster_dist_setTransformer \
  --train_npz input/Training_data/sift1M_learn/leafsize20K/train_sift.npz \
  --centroids_path input/Training_data/sift1M_learn/leafsize20K/centroids.npy \
  --test_npz input/Training_data/sift1M_learn/leafsize20K/test_sift.npz \
  --normalize \
  --val_split 0.1 \
  --epochs 50 \
  --batch_size 256 \
  --lr 1e-3 \
  --weight_decay 1e-2 \
  --d_model 256 \
  --nhead 8 \
  --num_layers 6 \
  --dim_ff 512 \
  --dropout 0.1 \
  --score_type bilinear \
  --use_isab \
  --induced_m 512 \
  --eval_topk 10

# Or generate synthetic data for a quick check
python -m src.model.cluster_dist_setTransformer --gen_synth --use_isab --induced_m 32 --eval_topk 5
"""
