"""
cluster_dist_transformer_v2.py

Architecture: Cross-Attention (Q=query, K=V=centroids).
Key changes vs v1:
  1. Dataset: each centroid token = [centroid_coords(D) || L2_dist(1)] → D+1 dims
  2. Model:   forward(query[B,D], centroid_feat[B,K,D+1]) — two explicit inputs
              Manual MHA (all nn.Linear) → fully TorchScript + INT8-quantizable
  3. Training: AMP autocast + GradScaler, batch_size=1024, num_workers=8, prefetch_factor=4
  4. Optuna:  SuccessiveHalvingPruner (ASHA) for aggressive early pruning
"""
import os
import math
import random
import argparse
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
import torch.quantization
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import optuna
except ImportError:
    optuna = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_centroids(path: str) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)
    if isinstance(arr, np.lib.npyio.NpzFile):
        c = arr["centroids"] if "centroids" in arr.files else arr[arr.files[0]]
    else:
        c = arr
    c = c.astype(np.float32)
    assert c.ndim == 2, f"centroids must be 2-D, got {c.shape}"
    return c


def _parse_dataset_leafsize(npz_path: str) -> Tuple[str, str]:
    parts = os.path.normpath(npz_path).split(os.sep)
    dataset: Optional[str] = None
    leafsize: Optional[str] = None
    for i, p in enumerate(parts):
        if p.lower().startswith(("sift", "deep", "gist")):
            dataset = p
            if i + 1 < len(parts) and parts[i + 1].lower().startswith("leafsize"):
                leafsize = parts[i + 1]
            break
    if dataset is None or leafsize is None:
        raise ValueError(f"Cannot parse dataset/leafsize from: {npz_path}")
    return dataset, leafsize


def build_optuna_param_filename(npz_path: str) -> str:
    ds, ls = _parse_dataset_leafsize(npz_path)
    return f"{ds}_{ls}_optuna_best_params.txt"


def build_csv_param_filename(npz_path: str) -> str:
    ds, ls = _parse_dataset_leafsize(npz_path)
    return f"{ds}_{ls}_optuna_all_trials.csv"


def topk_recall(pred: torch.Tensor, label: torch.Tensor, k: int = 10) -> float:
    k = min(k, pred.size(1))
    top_pred = torch.topk(pred, k, dim=1).indices
    top_true = torch.topk(label, k, dim=1).indices
    recall = 0.0
    for i in range(pred.size(0)):
        recall += len(set(top_pred[i].tolist()) & set(top_true[i].tolist())) / float(k)
    return recall / float(pred.size(0))


def topk_ordered_accuracy(pred: torch.Tensor, label: torch.Tensor, k: int = 10) -> float:
    k = min(k, pred.size(1))
    return float(((torch.topk(pred, k, dim=1).indices == torch.topk(label, k, dim=1).indices)
                  .float().mean(dim=1).mean()).item())


# ---------------------------------------------------------------------------
# Dataset  (returns query [D], centroid_feat [K, D+1], target [K])
# ---------------------------------------------------------------------------

class NPZClusterDatasetV2(Dataset):
    """
    __getitem__ returns:
      query      : [D]        raw query coordinates
      cent_feat  : [K, D+1]   each centroid's coords + L2-dist-from-query
      target     : [K]        probability distribution label
    """
    def __init__(
        self,
        npz_path: str,
        centroids: np.ndarray,
        normalize: bool = False,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        train_first_n: Optional[int] = None,
    ):
        data = np.load(npz_path)
        if "queries" not in data:
            raise ValueError(f"{npz_path} must contain 'queries'")
        tgt = data.get("targets", data.get("labels", None))
        if tgt is None:
            raise ValueError(f"{npz_path} must contain 'targets' or 'labels'")

        self.queries: np.ndarray = data["queries"].astype(np.float32)  # [N, D]
        self.targets: np.ndarray = tgt.astype(np.float32)              # [N, K]
        self.centroids: np.ndarray = centroids.astype(np.float32)      # [K, D]

        N, D = self.queries.shape
        K, Dc = self.centroids.shape
        assert self.targets.shape == (N, K)
        assert D == Dc

        if train_first_n is not None:
            n = min(train_first_n, N)
            self.queries = self.queries[:n]
            self.targets = self.targets[:n]

        if normalize:
            if mean is None or std is None:
                flat = np.concatenate([self.queries, self.centroids], axis=0)
                mean = flat.mean(axis=0, keepdims=True)
                std = flat.std(axis=0, keepdims=True) + 1e-6
            self.queries = (self.queries - mean) / std
            self.centroids = (self.centroids - mean) / std
            self.norm_stats: Optional[Tuple[np.ndarray, np.ndarray]] = (mean, std)
        else:
            self.norm_stats = None

    def __len__(self) -> int:
        return self.targets.shape[0]

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.queries[idx], self.targets[idx]


def build_loaders(
    train_npz: str,
    val_npz: Optional[str],
    centroids: np.ndarray,
    batch_size: int = 1024,
    num_workers: int = 8,
    normalize: bool = False,
    val_split: float = 0.05,
    seed: int = 42,
    train_first_n: Optional[int] = None,
    prefetch_factor: int = 4,
    persistent_workers: bool = False,
) -> Tuple[DataLoader, Optional[DataLoader], np.ndarray]:

    full_train = NPZClusterDatasetV2(train_npz, centroids, normalize, train_first_n=train_first_n)

    if val_npz is None and val_split > 0:
        n_val = int(len(full_train) * val_split)
        n_train = len(full_train) - n_val
        g = torch.Generator().manual_seed(seed)
        train_ds, val_ds = random_split(full_train, [n_train, n_val], generator=g)
        val_dataset = val_ds
    elif val_npz is not None:
        train_ds = full_train
        mean, std = full_train.norm_stats if full_train.norm_stats else (None, None)
        val_dataset: Optional[Dataset] = NPZClusterDatasetV2(val_npz, centroids, normalize, mean=mean, std=std)
    else:
        train_ds = full_train
        val_dataset = None

    pf = prefetch_factor if num_workers > 0 else None
    pw = persistent_workers and num_workers > 0
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
        prefetch_factor=pf, persistent_workers=pw,
    )
    val_loader: Optional[DataLoader] = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True,
            prefetch_factor=pf, persistent_workers=pw,
        )
    return train_loader, val_loader, full_train.centroids


def _make_cent_feat(q: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
    """Build centroid features [B, K, D+1] on GPU: centroids coords + L2 dist from each query."""
    diff = C[None, :, :] - q[:, None, :]                              # [B, K, D]
    dist = diff.norm(dim=-1, keepdim=True)                             # [B, K, 1]
    return torch.cat([C.expand(q.size(0), -1, -1), dist], dim=-1)     # [B, K, D+1]


# ---------------------------------------------------------------------------
# Model  (TorchScript-compatible manual Attention layers)
# ---------------------------------------------------------------------------

class SelfAttnLayer(nn.Module):
    """
    Pre-norm self-attention for a sequence of N tokens (used for centroid encoder).
    All nn.Linear → TorchScript + INT8 compatible.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale: float = float(math.sqrt(self.head_dim))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, N, d_model]
        B, N, _ = x.shape
        x_n = self.norm1(x)
        Q  = self.q_proj(x_n).view(B, N, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,N,Dh]
        Kp = self.k_proj(x_n).view(B, N, self.nhead, self.head_dim).transpose(1, 2)
        Vp = self.v_proj(x_n).view(B, N, self.nhead, self.head_dim).transpose(1, 2)

        attn = torch.matmul(Q, Kp.transpose(-2, -1)) / self.scale  # [B,H,N,N]
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, Vp).transpose(1, 2).reshape(B, N, self.d_model)
        x = x + self.drop(self.out_proj(out))
        x = x + self.drop(self.ff2(self.act(self.ff1(self.norm2(x)))))
        return x


class CrossAttnLayer(nn.Module):
    """
    Pre-norm cross-attention: query (Q=[B,1,d]) attends to centroids (K=V=[B,K,d]).
    All nn.Linear → TorchScript + INT8 compatible.
    """
    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float):
        super().__init__()
        assert d_model % nhead == 0
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.scale: float = float(math.sqrt(self.head_dim))

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, dim_ff)
        self.ff2 = nn.Linear(dim_ff, d_model)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        # q:  [B, 1, d_model]   kv: [B, K, d_model]
        B = q.shape[0]
        K = kv.shape[1]

        q_n = self.norm1(q)
        Q  = self.q_proj(q_n).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)  # [B,H,1,Dh]
        Kp = self.k_proj(kv).view(B, K, self.nhead, self.head_dim).transpose(1, 2)   # [B,H,K,Dh]
        Vp = self.v_proj(kv).view(B, K, self.nhead, self.head_dim).transpose(1, 2)   # [B,H,K,Dh]

        attn = torch.matmul(Q, Kp.transpose(-2, -1)) / self.scale  # [B,H,1,K]
        attn = torch.softmax(attn, dim=-1)
        attn = self.drop(attn)
        out = torch.matmul(attn, Vp).transpose(1, 2).reshape(B, 1, self.d_model)
        q = q + self.drop(self.out_proj(out))
        q = q + self.drop(self.ff2(self.act(self.ff1(self.norm2(q)))))
        return q


class QueryCentroidModel(nn.Module):
    """
    Static-Dynamic separated architecture:

    STATIC PATH  (centroid coords, never change per index)
      centroid_coords [K, D]  →  coord_proj  →  SelfAttn × n  →  c_static [K, d_model]
      Cached once per epoch in training; once per index build in C++.

    DYNAMIC PATH  (L2 distance from query to each centroid, changes per sample)
      l2_dists [B, K, 1]  →  dist_proj  →  c_dist [B, K, d_model]

    COMBINE + CROSS-ATTEND
      c = c_static + c_dist                         [B, K, d_model]
      query → query_proj → CrossAttn(c) × n  →  q  [B, 1, d_model]
      bilinear(q, c) → logits                       [B, K]

    forward(query, centroid_feat, c_static_cache=None)
      query          : [B, D]
      centroid_feat  : [B, K, D+1]  (last dim = L2 dist; first D dims = coords)
      c_static_cache : [1, K, d_model] or None
        When provided, skips the expensive SelfAttn (use precompute_static()).
    """
    def __init__(
        self,
        query_dim: int,
        centroid_dim: int,           # D+1
        d_model: int = 128,
        nhead: int = 8,
        num_centroid_layers: int = 2,
        num_cross_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.d_model = d_model
        self.coord_dim = centroid_dim - 1   # D  (strip the L2 dim)
        self.scale: float = float(math.sqrt(d_model))

        # Static path
        self.coord_proj = nn.Linear(centroid_dim - 1, d_model)
        cent_layers: List[SelfAttnLayer] = []
        for _ in range(num_centroid_layers):
            cent_layers.append(SelfAttnLayer(d_model, nhead, dim_feedforward, dropout))
        self.centroid_layers = nn.ModuleList(cent_layers)

        # Dynamic path (L2 distance: 1 → d_model)
        self.dist_proj = nn.Linear(1, d_model)

        # Query path
        self.query_proj = nn.Linear(query_dim, d_model)

        # Cross-attention
        cross_layers: List[CrossAttnLayer] = []
        for _ in range(num_cross_layers):
            cross_layers.append(CrossAttnLayer(d_model, nhead, dim_feedforward, dropout))
        self.cross_layers = nn.ModuleList(cross_layers)

        self.norm_out = nn.LayerNorm(d_model)
        self.query_head    = nn.Linear(d_model, d_model, bias=False)
        self.centroid_head = nn.Linear(d_model, d_model, bias=False)

    def precompute_static(self, centroid_coords: torch.Tensor) -> torch.Tensor:
        """
        Run the static centroid self-attention.
        Call once per epoch (training) or once per index build (C++ inference).
        centroid_coords: [K, D] or [1, K, D]
        Returns: [1, K, d_model]  (batch dim = 1 for broadcasting)
        """
        if centroid_coords.dim() == 2:
            centroid_coords = centroid_coords.unsqueeze(0)   # [1, K, D]
        c = self.coord_proj(centroid_coords)                 # [1, K, d_model]
        for layer in self.centroid_layers:
            c = layer(c)
        return c  # [1, K, d_model]

    def forward(self, query: torch.Tensor, centroid_feat: torch.Tensor,
                c_static_cache: Optional[torch.Tensor] = None) -> torch.Tensor:
        # query:         [B, D]
        # centroid_feat: [B, K, D+1]   last channel = L2 dist, first D = coords

        # --- static path (self-attn on coords, or use cache) ---
        if c_static_cache is not None:
            c_static = c_static_cache                        # [1, K, d_model]
        else:
            coords = centroid_feat[:1, :, :-1]              # [1, K, D]  (same for all batch)
            c_static = self.precompute_static(coords)        # [1, K, d_model]

        # --- dynamic path (per-sample L2 distance) ---
        l2 = centroid_feat[:, :, -1:]                        # [B, K, 1]
        c_dist = self.dist_proj(l2)                          # [B, K, d_model]

        c = c_static + c_dist                                # [B, K, d_model]  broadcast

        # --- cross-attention ---
        q = self.query_proj(query).unsqueeze(1)              # [B, 1, d_model]
        for layer in self.cross_layers:
            q = layer(q, c)

        q = self.norm_out(q).squeeze(1)                      # [B, d_model]

        qh = self.query_head(q)                              # [B, d_model]
        ch = self.centroid_head(c)                           # [B, K, d_model]
        logits = torch.einsum("bd,bkd->bk", qh, ch) / self.scale  # [B, K]

        return logits


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    C_gpu: torch.Tensor,
    topk: int = 10,
) -> Tuple[float, float, float]:
    """Returns (kld, recall@topk, ordered_acc@topk)."""
    was_training = model.training
    model.eval()
    kld_fn = nn.KLDivLoss(reduction="batchmean")
    kld_sum = 0.0
    recall_sum = 0.0
    ord_sum = 0.0
    total = 0

    for q, y in loader:
        q = q.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        c_feat = _make_cent_feat(q, C_gpu)

        logits = model(q, c_feat)
        log_p = torch.log_softmax(logits, dim=-1)
        p = torch.softmax(logits, dim=-1)

        bs = q.size(0)
        kld_sum += kld_fn(log_p, y).item() * bs
        recall_sum += topk_recall(p, y, k=topk) * bs
        ord_sum += topk_ordered_accuracy(p, y, k=topk) * bs
        total += bs

    if was_training:
        model.train()
    return kld_sum / total, recall_sum / total, ord_sum / total


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def _refresh_static_cache(model: nn.Module,
                          C_gpu: torch.Tensor) -> Optional[torch.Tensor]:
    """Precompute static centroid embeddings once. Returns [1, K, d_model] or None."""
    if not hasattr(model, "precompute_static"):
        return None
    with torch.no_grad():
        return model.precompute_static(C_gpu)  # type: ignore[operator]


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader],
    device: torch.device,
    C_gpu: torch.Tensor,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    grad_clip: float = 1.0,
    amp: bool = True,
    save_dir: Optional[str] = None,
    save_name: str = "best.pt",
    eval_topk: int = 10,
    trial: Optional[object] = None,
    patience: int = 7,
) -> float:

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # BF16 has FP32 dynamic range → no overflow; FP16 can overflow on large embeddings
    amp_enabled = amp and device.type == "cuda"
    amp_dtype = torch.bfloat16 if (amp_enabled and torch.cuda.is_bf16_supported()) else torch.float16
    if amp_enabled:
        print(f"AMP enabled: dtype={amp_dtype}")
    # GradScaler only needed for FP16 (BF16 has sufficient dynamic range without scaling)
    scaler = torch.amp.GradScaler(enabled=amp_enabled and amp_dtype == torch.float16)
    kld_fn = nn.KLDivLoss(reduction="batchmean")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    best_val = float("inf")
    no_improve = 0

    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        nan_batches = 0

        for q, y in train_loader:
            q = q.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=amp_enabled):
                c_feat = _make_cent_feat(q, C_gpu)
                logits = model(q, c_feat)
                loss = kld_fn(torch.log_softmax(logits, dim=-1), y)

            scaler.scale(loss).backward()
            if grad_clip > 0.0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()

            loss_val = loss.item()
            if math.isfinite(loss_val):
                running += loss_val * q.size(0)
                n += q.size(0)
            else:
                nan_batches += 1

        sched.step()
        train_loss = running / n if n > 0 else float("nan")
        if nan_batches > 0:
            print(f"  [!] epoch {ep}: {nan_batches} NaN/Inf batches skipped (GradScaler discarded these)")

        if val_loader is not None:
            val_kld, val_recall, _ = evaluate(model, val_loader, device, C_gpu, topk=eval_topk)

            if trial is None:
                print(f"epoch {ep:03d} | train={train_loss:.6f} | val={val_kld:.6f} | recall@{eval_topk}={val_recall:.4f}")

            if val_kld < best_val:
                best_val = val_kld
                no_improve = 0
                if save_dir:
                    torch.save({"model": model.state_dict(), "epoch": ep, "val_kld": val_kld},
                               os.path.join(save_dir, save_name))
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience and trial is None:
                    print(f"  [Early stop] no improvement for {patience} epochs, best val={best_val:.6f}")
                    break

            if trial is not None:
                trial.report(val_kld, ep)  # type: ignore[union-attr]
                if trial.should_prune():   # type: ignore[union-attr]
                    import optuna as _optuna
                    raise _optuna.exceptions.TrialPruned()
        else:
            if trial is None:
                print(f"epoch {ep:03d} | train={train_loss:.6f}")
            best_val = train_loss

    return best_val


# ---------------------------------------------------------------------------
# Optuna objective
# ---------------------------------------------------------------------------

def optuna_objective(trial, args, centroids: np.ndarray, device: torch.device) -> float:
    D = centroids.shape[1]

    d_model              = trial.suggest_categorical("d_model",              [64, 128, 256])
    valid_heads          = [h for h in [4, 8] if d_model % h == 0]
    nhead                = trial.suggest_categorical("nhead",                valid_heads)
    num_centroid_layers  = trial.suggest_categorical("num_centroid_layers",  [1, 2, 3, 4])
    num_cross_layers     = trial.suggest_categorical("num_cross_layers",     [1, 2, 3, 4])
    dim_ff               = trial.suggest_categorical("dim_ff",               [128, 256, 512])
    dropout              = trial.suggest_categorical("dropout",              [0.0, 0.05, 0.1])
    lr                   = trial.suggest_categorical("lr",                   [1e-2, 1e-3, 5e-4])
    weight_decay         = trial.suggest_categorical("weight_decay",         [1e-2, 1e-3])
    batch_size           = trial.suggest_categorical("batch_size",           [256, 512, 1024, 2048])

    # num_workers=0 avoids nested multiprocessing conflicts between Optuna and DataLoader
    train_loader, val_loader, C_norm = build_loaders(
        args.train_npz, args.val_npz, centroids,
        batch_size=batch_size,
        num_workers=0,
        normalize=args.normalize,
        val_split=args.val_split,
        seed=args.seed,
        train_first_n=args.train_first_n,
    )
    if val_loader is None:
        raise ValueError("Optuna requires a validation set (--val_split or --val_npz).")

    C_gpu = torch.from_numpy(C_norm).to(device)

    model = QueryCentroidModel(
        query_dim=D, centroid_dim=D + 1,
        d_model=d_model, nhead=nhead,
        num_centroid_layers=num_centroid_layers,
        num_cross_layers=num_cross_layers,
        dim_feedforward=dim_ff, dropout=dropout,
    ).to(device)

    return train(
        model, train_loader, val_loader, device, C_gpu,
        epochs=args.epochs, lr=lr, weight_decay=weight_decay,
        grad_clip=args.grad_clip, amp=not args.no_amp,
        save_dir=None, eval_topk=args.topk, trial=trial,
    )


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Cross-Attention cluster-dist predictor v2")
    p.add_argument("--train_npz", type=str, required=True)
    p.add_argument("--test_npz",  type=str, default=None)
    p.add_argument("--val_npz",   type=str, default=None)
    p.add_argument("--centroids_path", type=str, required=True)
    p.add_argument("--val_split",  type=float, default=0.05)
    p.add_argument("--normalize",  action="store_true")
    p.add_argument("--save_dir",   type=str, default="./results")
    p.add_argument("--epochs",     type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-3)
    p.add_argument("--grad_clip",  type=float, default=1.0)
    p.add_argument("--num_workers",type=int, default=8)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--experiment_id", type=str, default=None)
    p.add_argument("--d_model",             type=int, default=128)
    p.add_argument("--nhead",               type=int, default=8)
    p.add_argument("--num_centroid_layers", type=int, default=2)
    p.add_argument("--num_cross_layers",    type=int, default=2)
    p.add_argument("--dim_ff",              type=int, default=256)
    p.add_argument("--dropout",    type=float, default=0.05)
    p.add_argument("--no_amp",     action="store_true")
    p.add_argument("--topk",       type=int, default=20)
    p.add_argument("--log_interval", type=int, default=0)
    p.add_argument("--train_first_n", type=int, default=None)
    p.add_argument("--test_first_n",  type=int, default=None)
    p.add_argument("--patience",    type=int, default=7)
    p.add_argument("--eval_only",   action="store_true", help="Skip training; load existing checkpoint and evaluate on test set")
    p.add_argument("--use_optuna",    action="store_true")
    p.add_argument("--optuna_trials", type=int, default=100)
    p.add_argument("--optuna_db",     type=str, default=None)
    p.add_argument("--optuna_study_name", type=str, default="cross_attn_v2")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    C = load_centroids(args.centroids_path)
    K, D = C.shape
    print(f"Detected K={K}, D={D}  →  centroid_feat_dim={D+1}")

    # ── Branch 1: Optuna search ────────────────────────────────────────────
    if args.use_optuna:
        if optuna is None:
            raise ImportError("pip install optuna")

        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=5, reduction_factor=2, min_early_stopping_rate=0,
        )
        sampler = optuna.samplers.TPESampler(seed=args.seed)
        study = optuna.create_study(
            direction="minimize",
            study_name=args.optuna_study_name,
            storage=args.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        def _save_best(study, frozen_trial):
            if study.best_trial.number == frozen_trial.number:
                print(f"  [Best] Trial {frozen_trial.number}  loss={frozen_trial.value:.6f}")
                os.makedirs(args.save_dir, exist_ok=True)
                path = os.path.join(args.save_dir, build_optuna_param_filename(args.train_npz))
                with open(path, "w") as f:
                    f.write(f"Best Trial ID: {frozen_trial.number}\n")
                    f.write(f"Best Loss: {frozen_trial.value}\n")
                    f.write("Params:\n")
                    for k, v in frozen_trial.params.items():
                        f.write(f"  {k}: {v}\n")

        print(f"Starting Optuna ({args.optuna_trials} trials, ASHA pruning)...")
        try:
            study.optimize(
                lambda t: optuna_objective(t, args, C, device),
                n_trials=args.optuna_trials,
                callbacks=[_save_best],
            )
        except KeyboardInterrupt:
            print("Interrupted.")

        print(f"Finished trials: {len(study.trials)}")
        df = study.trials_dataframe()
        os.makedirs(args.save_dir, exist_ok=True)
        csv_path = os.path.join(args.save_dir, build_csv_param_filename(args.train_npz))
        df.to_csv(csv_path, index=False)
        print(f"All trials → {csv_path}")

        # Final save of best params
        best = study.best_trial
        path = os.path.join(args.save_dir, build_optuna_param_filename(args.train_npz))
        with open(path, "w") as f:
            f.write(f"Best Trial ID: {best.number}\n")
            f.write(f"Best Loss: {best.value}\n")
            f.write("Params:\n")
            for k, v in best.params.items():
                f.write(f"  {k}: {v}\n")
        return

    # ── Branch 2: Normal training ──────────────────────────────────────────
    ckpt_name = (
        f"v2_d{args.d_model}_Lc{args.num_centroid_layers}_Lx{args.num_cross_layers}"
        f"_H{args.nhead}_ff{args.dim_ff}"
        f"_bs{args.batch_size}_ep{args.epochs}_lr{args.lr}_wd{args.weight_decay}.pt"
    )
    subdir = os.path.splitext(ckpt_name)[0]
    if args.experiment_id:
        subdir = f"{args.experiment_id}_{subdir}"
    ckpt_dir = os.path.dirname(args.centroids_path)
    result_dir = os.path.join(ckpt_dir, subdir)
    os.makedirs(result_dir, exist_ok=True)

    model = QueryCentroidModel(
        query_dim=D, centroid_dim=D + 1,
        d_model=args.d_model, nhead=args.nhead,
        num_centroid_layers=args.num_centroid_layers,
        num_cross_layers=args.num_cross_layers,
        dim_feedforward=args.dim_ff, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model params: {n_params:,}")

    # Compute (possibly normalized) centroids and norm stats for test set.
    # Done before branching so --eval_only also has correct stats.
    mean_v: Optional[np.ndarray] = None
    std_v:  Optional[np.ndarray] = None
    if args.eval_only:
        if args.normalize:
            _ds = NPZClusterDatasetV2(args.train_npz, C, normalize=True)
            C_norm = _ds.centroids
            mean_v, std_v = _ds.norm_stats  # type: ignore[assignment]
        else:
            C_norm = C
    else:
        train_loader, val_loader, C_norm = build_loaders(
            args.train_npz, args.val_npz, C,
            batch_size=args.batch_size, num_workers=args.num_workers,
            normalize=args.normalize, val_split=args.val_split, seed=args.seed,
            train_first_n=args.train_first_n,
            persistent_workers=True,
        )
        if args.normalize:
            base = train_loader.dataset
            while hasattr(base, "dataset"):
                base = base.dataset  # type: ignore[union-attr]
            if hasattr(base, "norm_stats") and base.norm_stats is not None:
                mean_v, std_v = base.norm_stats  # type: ignore[assignment]
        C_gpu = torch.from_numpy(C_norm).to(device)
        train(
            model, train_loader, val_loader, device, C_gpu,
            epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
            grad_clip=args.grad_clip, amp=not args.no_amp,
            save_dir=result_dir, save_name=ckpt_name, eval_topk=args.topk,
            patience=args.patience,
        )

    C_gpu = torch.from_numpy(C_norm).to(device)

    # ── Evaluation on test set ─────────────────────────────────────────────
    if args.test_npz:
        ckpt_path = os.path.join(result_dir, ckpt_name)
        ckpt: dict = {}
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint (epoch {ckpt['epoch']}, val_kld={ckpt['val_kld']:.6f})")

        test_ds = NPZClusterDatasetV2(args.test_npz, C, normalize=args.normalize, mean=mean_v, std=std_v)
        if args.test_first_n is not None:
            from torch.utils.data import Subset
            test_ds = Subset(test_ds, list(range(min(args.test_first_n, len(test_ds)))))  # type: ignore[assignment]

        pf = 4 if args.num_workers > 0 else None
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True, prefetch_factor=pf,
        )
        kld, recall, ord_acc = evaluate(model, test_loader, device, C_gpu, topk=args.topk)
        print(f"[TEST] kld={kld:.6f} | recall@{args.topk}={recall:.4f} | ord_acc={ord_acc:.4f}")

        # Save test summary txt (format mirrors original training code stdout)
        summary_path = os.path.join(result_dir, ckpt_name.replace(".pt", "_test_result.txt"))
        with open(summary_path, "w") as f:
            f.write(f"[TEST] kld={kld:.6f} | recall@{args.topk}={recall:.4f} | ord_acc={ord_acc:.4f}\n")
            f.write(f"Best checkpoint: epoch {ckpt.get('epoch', '?')}, val_kld={ckpt.get('val_kld', float('nan')):.6f}\n")
            f.write(f"Experiment: {args.experiment_id}\n")
            f.write(f"Params: d_model={args.d_model} nhead={args.nhead} "
                    f"num_centroid_layers={args.num_centroid_layers} num_cross_layers={args.num_cross_layers} "
                    f"dim_ff={args.dim_ff} lr={args.lr} batch_size={args.batch_size} "
                    f"dropout={args.dropout} normalize={args.normalize}\n")
        print(f"Result → {summary_path}")

        # Save predictions
        pred_path = os.path.join(result_dir, ckpt_name.replace(".pt", "_pred.txt"))
        model.eval()
        with torch.no_grad(), open(pred_path, "w") as f:
            for q, _ in test_loader:
                q = q.to(device)
                probs = torch.softmax(model(q, _make_cent_feat(q, C_gpu)), dim=-1).cpu()
                for row in probs:
                    f.write(" ".join(f"{v.item():.6f}" for v in row) + "\n")
        print(f"Predictions → {pred_path}")

        # TorchScript export (FP32 + INT8)
        print("Exporting TorchScript models...")
        model_cpu = model.cpu().eval()

        # FP32 scripted
        scripted_fp32 = torch.jit.script(model_cpu)
        fp32_path = os.path.join(result_dir, ckpt_name.replace(".pt", "_scripted_fp32.pt"))
        scripted_fp32.save(fp32_path)
        print(f"FP32 TorchScript → {fp32_path}")

        # INT8 dynamic quantization → script
        try:
            quantized = torch.quantization.quantize_dynamic(
                model_cpu, {nn.Linear}, dtype=torch.qint8
            )
            scripted_int8 = torch.jit.script(quantized)
            int8_path = os.path.join(result_dir, ckpt_name.replace(".pt", "_scripted_int8.pt"))
            scripted_int8.save(int8_path)
            print(f"INT8 TorchScript  → {int8_path}")
        except Exception as e:
            print(f"[Warning] INT8 export failed: {e}")


if __name__ == "__main__":
    main()


# =============================================================================
# Tensor dimension flow (forward pass)
# =============================================================================
# Input:
#   query       [B, D=96]
#   centroid_feat [B, K=162, D+1=97]
#
# query_proj (Linear 96→128):
#   [B, 96]  →  [B, 128]  →  unsqueeze(1)  →  [B, 1, 128]   ← q
#
# centroid_proj (Linear 97→128):
#   [B, 162, 97]  →  [B, 162, 128]                            ← c
#
# CrossAttnLayer × num_layers (e.g. 3):
#   norm1(q):              [B, 1, 128]
#   q_proj(q_n):           [B, 1, 128]  → reshape → [B, 8, 1, 16]    (H=8, Dh=16)
#   k_proj(c):             [B, 162, 128] → reshape → [B, 8, 162, 16]
#   v_proj(c):             [B, 162, 128] → reshape → [B, 8, 162, 16]
#   attn = Q@K^T / √16:    [B, 8, 1, 162]
#   softmax(attn):         [B, 8, 1, 162]
#   attn @ V:              [B, 8, 1, 16]  → reshape → [B, 1, 128]
#   out_proj + residual:   [B, 1, 128]
#   FFN + residual:        [B, 1, 128]   ← q (updated)
#
# norm_out + squeeze:      [B, 128]
#
# query_head (Linear 128→128):  [B, 128]   ← qh
# centroid_head (Linear 128→128): [B, 162, 128] ← ch
#
# einsum("bd,bkd->bk") / 128:   [B, 162]   ← logits
#
# Complexity per forward:
#   Self-Attn v1: O((K+1)^2 * D) = O(163^2 * 96) ≈ 2.55M ops/sample
#   Cross-Attn v2: O(1 * K * D)  = O(162 * 96)    ≈ 15.6K ops/sample  (~163× reduction)
