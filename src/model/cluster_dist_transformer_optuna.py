import argparse
import math
import os
import random
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

try:
    import optuna  # hyperparameter tuning
except Exception:
    optuna = None


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


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, topk: int = 10) -> Tuple[float, float, float, float]:
    model.eval()
    kldiv = nn.KLDivLoss(reduction="batchmean")
    mae_meter, mse_meter, kld_meter = 0.0, 0.0, 0.0 # 
    total = 0
    recall_meter = 0.0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        log_probs = torch.log_softmax(logits, dim=-1)
        preds = torch.softmax(logits, dim=-1)
        loss_kld = kldiv(log_probs, y)
        mae = torch.mean(torch.abs(preds - y))
        mse = torch.mean((preds - y) ** 2)
        recall = topk_recall(preds, y, k=topk)
        bs = x.size(0)
        kld_meter += loss_kld.item() * bs
        mae_meter += mae.item() * bs
        mse_meter += mse.item() * bs
        recall_meter += recall * bs
        total += bs
        print(f"kld={loss_kld.item():.6f} | mae={mae.item():.6f} | mse={mse.item():.6f} | recall@{topk}={recall:.4f}")
    return kld_meter / total, mae_meter / total, mse_meter / total, recall_meter / total


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
    trial: Optional[object] = None,
):
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler(enabled=amp)
    kldiv = nn.KLDivLoss(reduction="batchmean")

    print(
        f"Start training | epochs={epochs} | lr={lr} | weight_decay={weight_decay} | "
        f"grad_clip={grad_clip} | amp={'on' if amp else 'off'} | "
        f"save_dir={save_dir} | save_name={save_name}"
    )

    best_val = float("inf")
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    for ep in range(1, epochs + 1):
        model.train()
        running = 0.0
        n = 0
        # show current learning rate (first param group)
        try:
            current_lr = opt.param_groups[0].get("lr", lr)
        except Exception:
            current_lr = lr
        print(f"[Epoch {ep}/{epochs}] lr={current_lr:.6g}")
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", enabled=amp and device.type == "cuda"):
                logits = model(x)
                log_probs = torch.log_softmax(logits, dim=-1)
                loss = kldiv(log_probs, y)
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
            running += loss.item() * x.size(0) # accumulate training loss
            n += x.size(0)
        sched.step()
        train_loss = running / n

        if val_loader is not None:
            val_kld, val_mae, val_mse, val_recall = evaluate(model, val_loader, device, topk=10)
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f} | val_kld={val_kld:.6f} | val_mae={val_mae:.6f} | val_mse={val_mse:.6f} | recall@10={val_recall:.4f}")
            if val_kld < best_val and save_dir: # save_dir =None when tuning
                best_val = val_kld
                torch.save(
                    {"model": model.state_dict(), "epoch": ep, "val_kld": val_kld},
                    os.path.join(save_dir, save_name),
                )
                print(f"Saved checkpoint: {os.path.join(save_dir, save_name)} (val_kld={val_kld:.6f})")
            # report to Optuna and prune
            if trial is not None:
                trial.report(val_kld, step=ep)
                if trial.should_prune():
                    print(f"Trial #{getattr(trial, 'number', '?')} pruned at epoch {ep} (val_kld={val_kld:.6f})")
                    raise optuna.TrialPruned()
        else:
            print(f"epoch {ep:03d} | train_kld={train_loss:.6f}")

    if val_loader is not None and save_dir:
        print(f"best val_kld: {best_val:.6f} (saved to {os.path.join(save_dir, save_name)})")
    return best_val if val_loader is not None else train_loss


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
    # tuning
    p.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    p.add_argument("--tune_trials", type=int, default=30, help="Number of Optuna trials")
    p.add_argument("--tune_timeout", type=int, default=0, help="Timeout seconds for tuning (0 means no timeout)")
    p.add_argument("--tune_epochs", type=int, default=50, help="Epochs per trial during tuning")
    p.add_argument("--tune_sampler", type=str, default="tpe", choices=["tpe", "random", "grid"], help="Optuna sampler")
    p.add_argument("--tune_pruner", type=str, default="median", choices=["median", "hb", "none"], help="Optuna pruner")
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

    # optional: Optuna hyperparameter tuning
    best_hparams = {
        "d_model": args.d_model,
        "nhead": args.nhead,
        "num_layers": args.num_layers,
        "dim_ff": args.dim_ff,
        "dropout": args.dropout,
        "score_type": args.score_type,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "grad_clip": args.grad_clip,
    }

    if args.tune:
        if optuna is None:
            raise RuntimeError("Optuna is required for --tune. Please install it via: pip install optuna")

        # Sampler & Pruner
        is_grid = args.tune_sampler == "grid"
        if is_grid:
            # Define discrete grid for exhaustive combinations
            grid_space = {
                "d_model": [128, 256, 384],
                "nhead": [4, 6, 8, 10],
                "num_layers": [2, 3, 4, 5, 6],
                "dim_ff": [256, 512, 768, 1024],
                "dropout": [0.0, 0.1, 0.2, 0.3],
                "score_type": ["bilinear", "mlp"],
                "batch_size": [128, 256, 512],
                "lr": [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3],
                "weight_decay": [0.0, 1e-6, 1e-5, 1e-4, 1e-3],
                "grad_clip": [0.5, 1.0, 2.0],
            }
            sampler = optuna.samplers.GridSampler(search_space=grid_space)
            # Disable pruning to enumerate all combinations deterministically
            pruner = optuna.pruners.NopPruner()
            # Compute total combinations for n_trials
            try:
                import numpy as _np
                total_trials = int(_np.prod([len(v) for v in grid_space.values()]))
            except Exception:
                total_trials = None
        else:
            if args.tune_sampler == "tpe":
                sampler = optuna.samplers.TPESampler(seed=args.seed)
            else:
                sampler = optuna.samplers.RandomSampler(seed=args.seed)
            if args.tune_pruner == "median":
                pruner = optuna.pruners.MedianPruner()
            elif args.tune_pruner == "hb":
                pruner = optuna.pruners.HyperbandPruner()
            else:
                pruner = optuna.pruners.NopPruner()

        if is_grid:
            print(
                "Tuning enabled (Grid Search) | "
                f"epochs_per_trial={args.tune_epochs} | sampler=grid | pruner=none | "
                f"total_combinations={total_trials}"
            )
        else:
            print(
                "Tuning enabled | "
                f"trials={args.tune_trials} | timeout={args.tune_timeout}s | epochs_per_trial={args.tune_epochs} | "
                f"sampler={args.tune_sampler} | pruner={args.tune_pruner}"
            )

        # minimize val_kld
        study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

        def objective(trial):
            print(f"\n==== Trial #{trial.number} ====")
            if is_grid:
                # Use categorical suggestions consistent with grid definitions
                t_d_model = trial.suggest_categorical("d_model", grid_space["d_model"])
                t_nhead = trial.suggest_categorical("nhead", grid_space["nhead"])
                t_layers = trial.suggest_categorical("num_layers", grid_space["num_layers"])
                t_dim_ff = trial.suggest_categorical("dim_ff", grid_space["dim_ff"])
                t_dropout = trial.suggest_categorical("dropout", grid_space["dropout"])
                t_score = trial.suggest_categorical("score_type", grid_space["score_type"]) 
                t_batch = trial.suggest_categorical("batch_size", grid_space["batch_size"])
                t_lr = trial.suggest_categorical("lr", grid_space["lr"])
                t_wd = trial.suggest_categorical("weight_decay", grid_space["weight_decay"])
                t_clip = trial.suggest_categorical("grad_clip", grid_space["grad_clip"])
            else:
                t_d_model = trial.suggest_categorical("d_model", [128, 256, 512,1024])
                t_nhead = trial.suggest_categorical("nhead", [4, 6, 8])
                t_layers = trial.suggest_categorical("num_layers", [2, 3, 4, 5, 6])
                t_dim_ff = trial.suggest_categorical("dim_ff", [256, 512, 768, 1024])
                t_dropout = trial.suggest_float("dropout", 0.0, 0.3)
                t_score = trial.suggest_categorical("score_type", ["bilinear", "mlp"]) 
                t_batch = trial.suggest_categorical("batch_size", [128, 256, 512])
                t_lr = trial.suggest_float("lr", 1e-5, 3e-3, log=True)
                t_wd = trial.suggest_float("weight_decay", 1e-6, 3e-3, log=True)
                t_clip = trial.suggest_categorical("grad_clip", [0.5, 1.0, 2.0])

            print(
                "Trial params | "
                f"d_model={t_d_model}, nhead={t_nhead}, num_layers={t_layers}, dim_ff={t_dim_ff}, dropout={float(t_dropout):.3f}, "
                f"score_type={t_score}, batch_size={t_batch}, lr={float(t_lr):.2e}, weight_decay={float(t_wd):.2e}, grad_clip={t_clip}"
            )

            # Validate constraints early to avoid runtime assertion
            if int(t_d_model) % int(t_nhead) != 0:
                print(
                    f"Skip invalid combo: d_model % nhead != 0 (d_model={t_d_model}, nhead={t_nhead}) -> objective=inf"
                )
                try:
                    trial.set_user_attr("invalid_combo", "d_model % nhead != 0")
                except Exception:
                    pass
                return float("inf")

            tr_loader, vl_loader = build_loaders(
                args.train_npz,
                args.val_npz, # optional,if none, split from training set
                batch_size=t_batch,
                num_workers=args.num_workers,
                normalize=args.normalize,
                val_split=args.val_split,
                seed=args.seed,
                centroids=C,
            )

            mdl = ClusterDistTransformer(
                input_dim=D,
                d_model=t_d_model,
                nhead=t_nhead,
                num_layers=t_layers,
                dim_feedforward=t_dim_ff,
                dropout=t_dropout,
                max_len=args.max_len,
                score_type=t_score,
            ).to(device)

            best_val_kld = train(
                model=mdl,
                train_loader=tr_loader,
                val_loader=vl_loader,
                device=device,
                epochs=args.tune_epochs,
                lr=t_lr,
                weight_decay=t_wd,
                grad_clip=t_clip,
                amp=not args.no_amp,
                save_dir=None,
                save_name="_",
                trial=trial,
            )
            return best_val_kld

        if is_grid:
            n_trials_to_run = total_trials
        else:
            n_trials_to_run = args.tune_trials
        study.optimize(
            objective,
            n_trials=n_trials_to_run,
            timeout=None if args.tune_timeout <= 0 else args.tune_timeout,
        )
        print("\nBest trial:")
        print(f"  value (val_kld): {study.best_value:.6f}")
        print("  params:")
        for k, v in study.best_trial.params.items():
            print(f"    {k}: {v}")
        # merge best parameters
        best_hparams.update(study.best_trial.params)

    # train with best (or default) hyperparameters and save
    train_loader, val_loader = build_loaders(
        args.train_npz,
        args.val_npz,
        batch_size=best_hparams["batch_size"],
        num_workers=args.num_workers,
        normalize=args.normalize,
        val_split=args.val_split, # split ratio from training set when no val_npz is provided
        seed=args.seed,
        centroids=C,
    )
    # dataset summary
    try:
        train_count = len(train_loader.dataset)
    except Exception:
        train_count = None
    try:
        val_count = None if val_loader is None else len(val_loader.dataset)
    except Exception:
        val_count = None
    print(
        "Dataset summary | "
        f"train_samples={train_count} | val_samples={val_count} | batch_size={best_hparams['batch_size']} | device={device}"
    )

    model = ClusterDistTransformer(
        input_dim=D,
        d_model=best_hparams["d_model"],
        nhead=best_hparams["nhead"],
        num_layers=best_hparams["num_layers"],
        dim_feedforward=best_hparams["dim_ff"],
        dropout=best_hparams["dropout"],
        max_len=args.max_len,
        score_type=best_hparams["score_type"],
    ).to(device)

    # Build a descriptive checkpoint name using key hyperparameters
    ckpt_name = (
        f"model_d{best_hparams['d_model']}_L{best_hparams['num_layers']}_H{best_hparams['nhead']}_ff{best_hparams['dim_ff']}"
        f"_bs{best_hparams['batch_size']}_ep{args.epochs}_lr{best_hparams['lr']}_wd{best_hparams['weight_decay']}_{best_hparams['score_type']}"
        + ("_tuned.pt" if args.tune else ".pt")
    )

    print("\nStart final training with best hyperparameters...")
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=best_hparams["lr"],
        weight_decay=best_hparams["weight_decay"],
        grad_clip=best_hparams["grad_clip"],
        amp=not args.no_amp,
        save_dir=ckpt_dir,
        save_name=ckpt_name,
    )

    # Evaluation (if test set is provided)
    if args.test_npz:
        # Reload the best checkpoint if available
        best_path = os.path.join(ckpt_dir, ckpt_name)
        if os.path.exists(best_path):
            ckpt = torch.load(best_path, map_location=device)
            model.load_state_dict(ckpt["model"])
            print(f"Loaded best checkpoint from {best_path} (val_kld={ckpt.get('val_kld', None)})")

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
        kld, mae, mse, recall = evaluate(model, test_loader, device, topk=args.topk)
        print(f"[TEST] kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | recall@{args.topk}={recall:.4f}")


if __name__ == "__main__":
    main()


"""
##### 已经测试过的指令
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.cluster_dist_transformer_optuna \
  --train_npz input/Training_data/gist1M_learn/leafsize20K/train_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize20K/centroids.npy \
  --val_split 0.1 \
  --tune \
  --tune_sampler grid \
  --tune_pruner none \
  --tune_epochs 100 \
  --tune_timeout 0 \
  --epochs 150
screen1:leafsize10K
screen2:leafsize20K
"""


"""
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