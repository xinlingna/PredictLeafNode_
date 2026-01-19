""" import argparse
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

    # 对标签learn_labels 和 query_labels 进行归一化
    if not args.no_norm_labels:
        rs = learn_labels.sum(axis=1, keepdims=True)
        learn_labels = learn_labels / rs
        rs = query_labels.sum(axis=1, keepdims=True)
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
    main() """

# 多行注释
"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --learn_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/dynamicK/knn_distributions_k50.txt \
  --query_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_query.txt \
  --query_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/knn_distributions_query_k50.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/leaf_center.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/train_gist_top50.npz \
  --out_test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/test_gist_top50.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20K/centroids_top50.npy

python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_learn.txt \
  --learn_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/knn_distributions.txt \
  --query_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/gist_query.txt \
  --query_labels_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/knn_distributions_query.txt \
  --centroids_txt /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/leaf_center.txt \
  --out_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/train_gist.npz \
  --out_test_npz /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/test_gist.npz \
  --out_centroids /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize10K/centroids.npy


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
import argparse
import os
import numpy as np


def load_txt_matrix(path: str, dtype=np.float32) -> np.ndarray:
    # 默认：空格分隔的 txt，每行一个向量
    arr = np.loadtxt(path, dtype=dtype)
    if arr.ndim == 1:
        arr = arr[None, :]
    return arr


def load_bin_matrix(path: str, n: int, d: int, dtype=np.float32) -> np.ndarray:
    """
    读取无头 bin 文件：
      - 连续存储 n * d 个值
      - 每个维度占 4 字节（float32）
      - 布局为行优先： [row0_d0, ..., row0_d(d-1), row1_d0, ...]
    """
    data = np.fromfile(path, dtype=dtype)
    expected = n * d
    if data.size != expected:
        raise ValueError(
            f"Bin size mismatch for {path}: expected {expected} elements "
            f"(n={n}, d={d}), but got {data.size}"
        )
    return data.reshape(n, d)


def load_matrix_auto(path: str, n: int | None, d: int | None, dtype=np.float32) -> np.ndarray:
    """
    统一入口：
      - 若同时提供 n 和 d，则按 bin 读取
      - 若都不提供，则按 txt 读取
      - 若只给一个，报错
    """
    if (n is None) and (d is None):
        # txt 模式
        return load_txt_matrix(path, dtype=dtype)
    if (n is None) != (d is None):
        raise ValueError(
            f"For file {path}, you must provide BOTH n and d for bin mode, "
            f"or neither for txt mode."
        )
    # bin 模式
    return load_bin_matrix(path, n, d, dtype=dtype)


def main():
    ap = argparse.ArgumentParser()
    # 必选：文件路径（可以是 txt 或 bin）
    ap.add_argument("--learn_txt", type=str, required=True,
                    help="Learn vectors file (.txt or .bin), shape [N1, D]")
    ap.add_argument("--learn_labels_txt", type=str, required=True,
                    help="Learn labels file (.txt or .bin), shape [N1, K]")
    ap.add_argument("--query_txt", type=str, required=True,
                    help="Query vectors file (.txt or .bin), shape [N2, D]")
    ap.add_argument("--query_labels_txt", type=str, required=True,
                    help="Query labels file (.txt or .bin), shape [N2, K]")
    ap.add_argument("--centroids_txt", type=str, required=True,
                    help="Centroids .txt, shape [K, D] (仍按 txt 读取)")
    ap.add_argument("--out_npz", type=str, required=True,
                    help="Output .npz containing 'queries' and 'targets' for learn set")
    ap.add_argument("--out_test_npz", type=str, required=True,
                    help="Output .npz containing 'queries' and 'targets' for query set")
    ap.add_argument("--out_centroids", type=str, required=True,
                    help="Output .npy for centroids [K, D]")
    ap.add_argument("--no_norm_labels", action="store_true",
                    help="Do not normalize labels per row")

    # 新增：若上述 4 个文件为 bin，则需要指定 N 和 D（每行向量数、每个向量维度）
    # learn vectors
    ap.add_argument("--learn_n", type=int, default=None,
                    help="If learn_txt is .bin: number of learn vectors N1")
    ap.add_argument("--learn_d", type=int, default=None,
                    help="If learn_txt is .bin: learn vector dim D")
    # learn labels
    ap.add_argument("--learn_labels_n", type=int, default=None,
                    help="If learn_labels_txt is .bin: number of label rows N1")
    ap.add_argument("--learn_labels_d", type=int, default=None,
                    help="If learn_labels_txt is .bin: labels dim K")
    # query vectors
    ap.add_argument("--query_n", type=int, default=None,
                    help="If query_txt is .bin: number of query vectors N2")
    ap.add_argument("--query_d", type=int, default=None,
                    help="If query_txt is .bin: query vector dim D")
    # query labels
    ap.add_argument("--query_labels_n", type=int, default=None,
                    help="If query_labels_txt is .bin: number of label rows N2")
    ap.add_argument("--query_labels_d", type=int, default=None,
                    help="If query_labels_txt is .bin: labels dim K")

    args = ap.parse_args()

    # ----- 读取数据 -----
    # learn: [N1, D]
    learn = load_matrix_auto(
        args.learn_txt, args.learn_n, args.learn_d, dtype=np.float32
    )
    # learn_labels: [N1, K]
    learn_labels = load_matrix_auto(
        args.learn_labels_txt, args.learn_labels_n, args.learn_labels_d, dtype=np.float32
    )
    # # centroids 仍按 txt 读取: [K, D]
    # centroids = load_txt_matrix(args.centroids_txt, dtype=np.float32)
    # 保证严格每行按空格切分
    with open(args.centroids_txt, "r") as f:
        centroids = []
        for line in f:
            nums = line.strip().split()
            if len(nums) == 0:
                continue
            centroids.append([float(x) for x in nums])
    
    centroids = np.array(centroids, dtype=np.float32)


    # query: [N2, D]
    query = load_matrix_auto(
        args.query_txt, args.query_n, args.query_d, dtype=np.float32
    )
    # query_labels: [N2, K]
    query_labels = load_matrix_auto(
        args.query_labels_txt, args.query_labels_n, args.query_labels_d, dtype=np.float32
    )

    # ----- 一致性检查（保持你原来的逻辑） -----
    N1, D = learn.shape
    N2, K = learn_labels.shape
    Kc, Dc = centroids.shape
    print("========== SHAPE CHECK ==========")
    print(f"learn:               N1={N1}, D={D}")
    print(f"learn_labels:        N2={N2}, K={K}")
    print(f"centroids:           Kc={Kc}, Dc={Dc}")
    print("=================================")

    assert N1 == N2, f"N mismatch: learn={N1}, learn_labels={N2}"
    assert D == Dc, f"D mismatch: learn={D}, centroids={Dc}"
    assert K == Kc, f"K mismatch: labels={K}, centroids={Kc}"

    N1_q, D_q = query.shape
    N2_q, K_q = query_labels.shape
    Kc, Dc = centroids.shape
    assert N1_q == N2_q, f"N mismatch: query={N1_q}, query_labels={N2_q}"
    assert D_q == Dc, f"D mismatch: query={D_q}, centroids={Dc}"
    assert K_q == Kc, f"K mismatch: query_labels={K_q}, centroids={Kc}"

    # ----- 标签归一化（沿用原逻辑） -----
    if not args.no_norm_labels:
        rs = learn_labels.sum(axis=1, keepdims=True)
        learn_labels = learn_labels / rs
        rs = query_labels.sum(axis=1, keepdims=True)
        query_labels = query_labels / rs

    # ----- 保存输出 -----
    os.makedirs(os.path.dirname(args.out_npz), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_test_npz), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_centroids), exist_ok=True)

    np.savez_compressed(args.out_npz,
                        queries=learn,
                        targets=learn_labels)
    np.savez_compressed(args.out_test_npz,
                        queries=query,
                        targets=query_labels)
    np.save(args.out_centroids, centroids.astype(np.float32))

    print(f"Saved queries+targets -> {args.out_npz} (queries {learn.shape}, targets {learn_labels.shape})")
    print(f"Saved queries+targets -> {args.out_test_npz} (queries {query.shape}, targets {query_labels.shape})")
    print(f"Saved centroids      -> {args.out_centroids} (centroids {centroids.shape})")


if __name__ == "__main__":
    main()

# 多行注释
"""
示例（原 txt 版本不变）：
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /path/to/gist_learn.txt \
  --learn_labels_txt /path/to/knn_distributions_k50.txt \
  --query_txt /path/to/gist_query.txt \
  --query_labels_txt /path/to/knn_distributions_query_k50.txt \
  --centroids_txt /path/to/leaf_center.txt \
  --out_npz /path/to/train_gist_top50.npz \
  --out_test_npz /path/to/test_gist_top50.npz \
  --out_centroids /path/to/centroids_top50.npy

示例（learn / query / labels 为 bin 文件，float32 无头）：
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /path/to/learn.bin \
  --learn_n 100000 --learn_d 128 \
  --learn_labels_txt /path/to/learn_labels.bin \
  --learn_labels_n 100000 --learn_labels_d 50 \
  --query_txt /path/to/query.bin \
  --query_n 10000 --query_d 128 \
  --query_labels_txt /path/to/query_labels.bin \
  --query_labels_n 10000 --query_labels_d 50 \
  --centroids_txt /path/to/leaf_center.txt \
  --out_npz /path/to/train_bin_top50.npz \
  --out_test_npz /path/to/test_bin_top50.npz \
  --out_centroids /path/to/centroids_top50.npy
"""

""" 
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/elpis/data/real/Deep1B/learn.bin \
  --learn_n 200000 --learn_d 96 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/elpis/data/real/Deep1B/deep2M_query.bin \
  --query_n 2000 --query_d 96 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/train_deep.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/test_deep.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize20K/centroids.npy
"""


""" 
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/elpis/data/real/Deep1B/learn.bin \
  --learn_n 200000 --learn_d 96 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/elpis/data/real/Deep1B/deep2M_query.bin \
  --query_n 2000 --query_d 96 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/train_deep.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/test_deep.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/deep2M/leafsize60K/centroids.npy
"""

# sift10M leafsize50K
""" 
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_learn.bin \
  --learn_n 1000000 --learn_d 128 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_query.bin \
  --query_n 10000 --query_d 128 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/train_deep.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/test_deep.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize50K/centroids.npy
"""

# sift10M leafsize20W (115)
""" 
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_learn.bin \
  --learn_n 1000000 --learn_d 128 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_query.bin \
  --query_n 10000 --query_d 128 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/train_sift10M.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/test_sift10M.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize20W/centroids.npy
"""


# sift10M leafsize40W
""" 
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_learn.bin \
  --learn_n 1000000 --learn_d 128 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/sift10M_query.bin \
  --query_n 10000 --query_d 128 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/train_sift10M.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/test_sift10M.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/sift10M/leafsize40W/centroids.npy
"""



# deep50M leafsize100W
""" 
conda activate elpis_torch
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_learn.bin \
  --learn_n 500000 --learn_d 96 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_query.bin \
  --query_n 10000 --query_d 96 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/train_deep50M.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/test_deep50M.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize100W/centroids.npy
"""

# deep50M leafsize50W 162
""" 
conda activate elpis_torch
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_learn.bin \
  --learn_n 500000 --learn_d 96 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_query.bin \
  --query_n 10000 --query_d 96 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/train_deep50M.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/test_deep50M.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize50W/centroids.npy
"""


# deep50M leafsize20W
""" 
conda activate elpis_torch
python -m src.preprocess.convert_txt_to_npz \
  --learn_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_learn.bin \
  --learn_n 500000 --learn_d 96 \
  --learn_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/learn_knn_distributions_k100.txt \
  --query_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/deep50M_query.bin \
  --query_n 10000 --query_d 96 \
  --query_labels_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/query_knn_distributions_k100.txt \
  --centroids_txt /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/leaf_centroid.txt \
  --out_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/train_deep50M.npz \
  --out_test_npz /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/test_deep50M.npz \
  --out_centroids /home/xln/PredictLeafNode/input/Training_data/deep50M/leafsize20W/centroids.npy
"""