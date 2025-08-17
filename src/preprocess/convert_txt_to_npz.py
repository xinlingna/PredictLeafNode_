import argparse
import os
import numpy as np


def load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    # Default: whitespace-delimited. Consider a more efficient loader for very large files.
    arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries_txt", type=str, required=True, help="Query vectors .txt, shape [N, D]")
    ap.add_argument("--centroids_txt", type=str, required=True, help="Centroids .txt, shape [K, D]")
    ap.add_argument("--labels_txt", type=str, required=True, help="Labels .txt, shape [N, K]; rows will be normalized")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz containing 'queries' and 'targets'")
    ap.add_argument("--out_centroids", type=str, required=True, help="Output .npy for centroids [K, D]")
    ap.add_argument("--no_norm_labels", action="store_true", help="Do not normalize labels per row")
    args = ap.parse_args()

    queries = load_txt_matrix(args.queries_txt, dtype=np.float32)   # [N, D]
    centroids = load_txt_matrix(args.centroids_txt, dtype=np.float32)  # [K, D]
    labels = load_txt_matrix(args.labels_txt, dtype=np.float32)     # [N, K]

    N, D = queries.shape
    Kc, Dc = centroids.shape
    Nl, K = labels.shape

    assert N == Nl, f"N mismatch: queries={N}, labels={Nl}"
    assert D == Dc, f"D mismatch: queries={D}, centroids={Dc}"
    assert K == Kc, f"K mismatch: labels={K}, centroids={Kc}"

    if not args.no_norm_labels:
        rs = labels.sum(axis=1, keepdims=True) + 1e-12
        labels = labels / rs

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_centroids), exist_ok=True)

    np.savez_compressed(args.out_npz, queries=queries, targets=labels)  # save (queries, targets)
    np.save(args.out_centroids, centroids.astype(np.float32))  # save centroids
    
    print(f"Saved queries+targets -> {args.out_npz} (queries {queries.shape}, targets {labels.shape})")
    print(f"Saved centroids      -> {args.out_centroids} (centroids {centroids.shape})")


if __name__ == "__main__":
    main()

# 多行注释
"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.preprocess.convert_txt_to_npz \
  --queries_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/leaf_center.txt \
  --labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/knn_distributions.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/train_gist.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/centroids.npy

python -m src.preprocess.convert_txt_to_npz \
  --queries_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/leaf_center.txt \
  --labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/knn_distributions.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/train_gist.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/centroids.npy
"""