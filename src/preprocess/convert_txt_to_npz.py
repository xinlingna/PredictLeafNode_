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
    ap.add_argument("--learn_txt", type=str, required=True, help="Query vectors .txt, shape [N1, D]")
    ap.add_argument("--learn_labels_txt", type=str, required=True, help="Labels .txt, shape [N1, K]; rows will be normalized")
    ap.add_argument("--query_txt", type=str, required=True, help="Query vectors .txt, shape [N2, D]")
    ap.add_argument("--query_labels_txt", type=str, required=True, help="Labels .txt, shape [N2, K]; rows will be normalized")
    ap.add_argument("--centroids_txt", type=str, required=True, help="Centroids .txt, shape [K, D]")
    ap.add_argument("--out_npz", type=str, required=True, help="Output .npz containing 'queries' and 'targets'")
    ap.add_argument("--out_test_npz", type=str, required=True, help="Output .npz for test set (queries/targets)")
    ap.add_argument("--out_centroids", type=str, required=True, help="Output .npy for centroids [K, D]")
    ap.add_argument("--no_norm_labels", action="store_true", help="Do not normalize labels per row")
    args = ap.parse_args()

    learn = load_txt_matrix(args.learn_txt, dtype=np.float32)   # [N1, D]
    learn_labels = load_txt_matrix(args.learn_labels_txt, dtype=np.float32)     # [N1, K]
    centroids = load_txt_matrix(args.centroids_txt, dtype=np.float32)  # [K, D]

    query = load_txt_matrix(args.query_txt, dtype=np.float32)   # [N2, D]
    query_labels = load_txt_matrix(args.query_labels_txt, dtype=np.float32)     # [N2, K]


    N1, D = learn.shape
    N2, K = learn_labels.shape
    Kc, Dc = centroids.shape
    assert N1 == N2, f"N mismatch: learn={N1}, learn_labels={N2}"
    assert D == Dc, f"D mismatch: learn={D}, centroids={Dc}"
    assert K == Kc, f"K mismatch: labels={K}, centroids={Kc}"

    N1, D = query.shape
    N2, K = query_labels.shape
    Kc, Dc = centroids.shape
    assert N1 == N2, f"N mismatch: query={N1}, query_labels={N2}"
    assert D == Dc, f"D mismatch: query={D}, centroids={Dc}"
    assert K == Kc, f"K mismatch: query_labels={K}, centroids={Kc}"

    if not args.no_norm_labels:
        rs = learn_labels.sum(axis=1, keepdims=True) + 1e-12
        learn_labels = learn_labels / rs
        rs = query_labels.sum(axis=1, keepdims=True) + 1e-12
        query_labels = query_labels / rs

    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test_npz), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_centroids), exist_ok=True)

    np.savez_compressed(args.out_npz, queries=learn, targets=learn_labels)  # save (queries, targets)
    np.savez_compressed(args.out_test_npz, queries=query, targets=query_labels)  # save (queries, targets)
    np.save(args.out_centroids, centroids.astype(np.float32))  # save centroids
    
    print(f"Saved queries+targets -> {args.out_npz} (queries {learn.shape}, targets {learn_labels.shape})")
    print(f"Saved queries+targets -> {args.out_test_npz} (queries {query.shape}, targets {query_labels.shape})")
    print(f"Saved centroids      -> {args.out_centroids} (centroids {centroids.shape})")


if __name__ == "__main__":
    main()

# 多行注释
"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --learn_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/knn_distributions.txt \
  --query_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_query.txt \
  --query_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/knn_distributions_query.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/leaf_center.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/train_gist.npz \
  --out_test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/test_gist.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/centroids.npy

python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --learn_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/knn_distributions.txt \
  --query_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_query.txt \
  --query_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/knn_distributions_query.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/leaf_center.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/train_gist.npz \
  --out_test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/test_gist.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10k/centroids.npy


python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/sift_learn.txt \
  --learn_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/knn_distributions.txt \
  --query_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/sift_query.txt \
  --query_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/knn_distributions_query.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/leaf_center.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/train_sift.npz \
  --out_test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/test_sift.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize20K/centroids.npy

"""