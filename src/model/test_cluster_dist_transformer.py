import argparse
import os
import re
import torch

from .cluster_dist_transformer import (
    NPZClusterDataset,
    ClusterDistTransformer,
    evaluate,
    load_centroids,
    set_seed,
)


def parse_hparams_from_ckpt(ckpt_path: str):
    """
    Try to parse key hyperparameters from checkpoint filename.
    Expected pattern:
      model_d{d_model}_L{num_layers}_H{nhead}_ff{dim_ff}_bs{batch}_ep{epochs}_lr{lr}_wd{wd}_{score_type}.pt
    Returns dict or None if not matched.
    """
    name = os.path.basename(ckpt_path)
    pat = re.compile(
        r"model_d(\d+)_L(\d+)_H(\d+)_ff(\d+)_bs(\d+)_ep(\d+)_lr([0-9.eE+-]+)_wd([0-9.eE+-]+)_(\w+)\.pt$"
    )
    m = pat.match(name)
    if not m:
        return None
    d_model = int(m.group(1))
    num_layers = int(m.group(2))
    nhead = int(m.group(3))
    dim_ff = int(m.group(4))
    score_type = m.group(9)
    return {
        "d_model": d_model,
        "num_layers": num_layers,
        "nhead": nhead,
        "dim_ff": dim_ff,
        "score_type": score_type,
    }


def build_model(D: int, args, parsed):
    d_model = args.d_model if args.d_model is not None else parsed["d_model"]
    nhead = args.nhead if args.nhead is not None else parsed["nhead"]
    num_layers = args.num_layers if args.num_layers is not None else parsed["num_layers"]
    dim_ff = args.dim_ff if args.dim_ff is not None else parsed["dim_ff"]
    score_type = args.score_type if args.score_type is not None else parsed["score_type"]
    return ClusterDistTransformer(
        input_dim=D,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        score_type=score_type,
    )


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained ClusterDistTransformer on a test set")
    p.add_argument("--ckpt_path", type=str, required=True, help="Path to checkpoint .pt file")
    p.add_argument("--test_npz", type=str, required=True, help="Test npz containing queries/targets")
    p.add_argument("--centroids_path", type=str, required=True, help="Centroids .npy/.npz path [K,D]")
    p.add_argument("--normalize", action="store_true", help="Apply feature-wise standardization")
    p.add_argument("--topk", type=int, default=10, help="Top-K for recall metric")
    p.add_argument("--seed", type=int, default=42)
    # Optional overrides if filename parsing fails or you want to override
    p.add_argument("--d_model", type=int, default=None)
    p.add_argument("--nhead", type=int, default=None)
    p.add_argument("--num_layers", type=int, default=None)
    p.add_argument("--dim_ff", type=int, default=None)
    p.add_argument("--score_type", type=str, default=None, choices=["bilinear", "mlp"])
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--max_len", type=int, default=4096)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse hyperparams from filename when possible
    parsed = parse_hparams_from_ckpt(args.ckpt_path)
    if parsed is None and any(v is None for v in [args.d_model, args.nhead, args.num_layers, args.dim_ff, args.score_type]):
        raise ValueError("Cannot parse hyperparameters from ckpt name. Please provide --d_model --nhead --num_layers --dim_ff --score_type explicitly.")

    # Load centroids to determine D
    C = load_centroids(args.centroids_path)
    K, D = C.shape

    # Build model and load weights
    model = build_model(D, args, parsed or {}).to(device)
    ckpt = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build test loader (compute mean/std from test if normalize=True)
    test_ds = NPZClusterDataset(args.test_npz, normalize=args.normalize, centroids=C)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

    # Evaluate
    kld, mae, mse, recall = evaluate(model, test_loader, device, topk=args.topk)
    print(f"[TEST] K={K}, D={D} | kld={kld:.6f} | mae={mae:.6f} | mse={mse:.6f} | recall@{args.topk}={recall:.4f}")

    # Dump predicted probabilities to file next to ckpt: <ckpt_basename>_pred.txt
    out_pred = os.path.join(os.path.dirname(args.ckpt_path), os.path.basename(args.ckpt_path).rsplit('.', 1)[0] + "_pred.txt")
    with torch.no_grad(), open(out_pred, "w") as f:
        for x, _ in test_loader:
            x = x.to(device)
            probs = torch.softmax(model(x), dim=-1).cpu()
            for row in probs:
                f.write(" ".join(f"{v.item():.6f}" for v in row) + "\n")
    print(f"Saved predictions -> {out_pred}")


if __name__ == "__main__":
    main()

'''
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.model.test_cluster_dist_transformer \
  --ckpt_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/model_d512_L4_H8_ff1024_bs256_ep150_lr0.001_wd0.01_bilinear.pt \
  --test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/test_gist.npz \
  --centroids_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/centroids.npy \
  --topk 10 \
  --seed 42


python -m src.model.test_cluster_dist_transformer \
  --ckpt_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/model_d256_L4_H8_ff512_bs256_ep150_lr0.001_wd0.01_bilinear.pt \
  --test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/test_gist.npz \
  --centroids_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/centroids.npy \
  --topk 10 \
  --seed 42

python -m src.model.test_cluster_dist_transformer \
  --ckpt_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/model_d512_L4_H8_ff512_bs256_ep150_lr0.001_wd0.01_bilinear.pt\
  --test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/test_gist.npz \
  --centroids_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/centroids.npy \
  --topk 10 \
  --seed 42
'''




# gist1M-leafsize10K
'''
python -m src.model.test_cluster_dist_transformer \
  --ckpt_path input/Training_data/gist1M_learn/leafsize10K/model_d256_L4_H8_ff512_bs256_ep150_lr0.001_wd0.01_bilinear.pt\
  --test_npz input/Training_data/gist1M_learn/leafsize10K/test_gist.npz \
  --centroids_path input/Training_data/gist1M_learn/leafsize10K/centroids.npy \
  --topk 10 \
  --seed 42
'''# sift1M-leafsize20K
'''
python -m src.model.test_cluster_dist_transformer \
  --ckpt_path input/Training_data/sift1M_learn/leafsize10K/model_d256_L4_H8_ff512_bs256_ep150_lr0.001_wd0.01_bilinear.pt \
  --test_npz input/Training_data/sift1M_learn/leafsize10K/test_sift.npz \
  --centroids_path input/Training_data/sift1M_learn/leafsize10K/centroids.npy \
  --topk 10 \
  --seed 42
'''

