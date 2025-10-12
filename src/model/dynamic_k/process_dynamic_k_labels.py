#!/usr/bin/env python3
"""
处理动态K值标签数据和查询向量，转换为符合DynamicKClusterDistTransformer要求的NPZ格式

支持两种处理模式：

1. 标签模式 (labels)：
   输入：多个txt文件，每个文件包含一个K值下的最近邻聚类分布
   输出：单个NPZ文件，包含所有K值的标签，可直接作为模型的train_labels_path使用
   
   数据格式：
   - 输入txt文件：每行为一个查询的聚类分布计数 [M个聚类]
   - 输出NPZ文件：包含 targets_k{K} 数组，每个数组形状为 [N, M]，N为查询数量，M为聚类数量

2. 向量模式 (vectors)：
   输入：单个txt文件，每行为一个查询向量，维度值用空格分隔
   输出：单个NPZ文件，包含查询向量数组
   
   数据格式：
   - 输入txt文件：每行为一个向量，如 "1.2 3.4 5.6 7.8"
   - 输出NPZ文件：包含一个数组，形状为 [N, D]，N为查询数量，D为向量维度

使用示例：

# 处理标签文件
python src/model/dynamic_k/process_dynamic_k_labels.py \
    --mode labels \
    --input_dir input/Training_data/gist1M_learn/leafsize20K/dynamicK \
    --output_path output/train_labels.npz \
    --k_values 1 10 20 50 100

# 处理查询向量文件  
python src/model/dynamic_k/process_dynamic_k_labels.py \
    --mode vectors \
    --input_txt input/query_vectors.txt \
    --output_path output/query_vectors.npz \
    --array_name "queries"
"""

import os
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
import re


def load_distribution_file(file_path: str, normalize: bool = True) -> np.ndarray:
    """
    加载单个K值的分布文件
    
    Args:
        file_path: 分布文件路径
        normalize: 是否归一化为概率分布
    
    Returns:
        distributions: [N, M] 形状的数组，N为查询数量，M为聚类数量
    """
    print(f"Loading distribution file: {file_path}")
    
    # 读取文本文件
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # 解析数据
    distributions = []
    for line_idx, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue
            
        # 解析每行的数字
        counts = list(map(int, line.split()))
        distributions.append(counts)
        
        # 每10万行打印一次进度
        if (line_idx + 1) % 100000 == 0:
            print(f"  Processed {line_idx + 1} lines")
    
    distributions = np.array(distributions, dtype=np.float32)
    print(f"  Loaded shape: {distributions.shape}")
    
    if normalize:
        # 归一化为概率分布
        row_sums = distributions.sum(axis=1, keepdims=True)
        
        # 处理全零行（如果有的话）
        zero_rows = (row_sums.flatten() == 0)
        if zero_rows.any():
            print(f"  Warning: Found {zero_rows.sum()} zero-sum rows, using uniform distribution")
            distributions[zero_rows] = 1.0
            row_sums[zero_rows] = distributions.shape[1]
        
        # 归一化
        distributions = distributions / row_sums
        
        # 验证归一化结果
        new_sums = distributions.sum(axis=1)
        if not np.allclose(new_sums, 1.0, atol=1e-6):
            print(f"  Warning: Normalization check failed, sum range: [{new_sums.min():.6f}, {new_sums.max():.6f}]")
        else:
            print(f"  Successfully normalized to probability distributions")
    
    return distributions


def extract_k_value_from_filename(filename: str) -> int:
    """
    从文件名中提取K值
    
    Args:
        filename: 文件名，如 "knn_distributions_k10.txt"
    
    Returns:
        k_value: 提取的K值
    """
    # 使用正则表达式提取K值
    match = re.search(r'k(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract K value from filename: {filename}")


def process_query_vectors_txt_to_npz(
    input_txt_path: str,
    output_npz_path: str,
    array_name: str = "queries",
    dtype: str = "float32",
    max_vectors: int = None
) -> None:
    """
    将查询向量txt文件转换为npz格式
    
    Args:
        input_txt_path: 输入txt文件路径，每行为一个查询向量，维度值用空格分隔
        output_npz_path: 输出npz文件路径
        array_name: npz文件中数组的键名，默认为"queries"
        dtype: 数据类型，默认为"float32"
        max_vectors: 最大向量数量限制（用于测试）
    
    输入格式示例：
        1.2 3.4 5.6 7.8 9.0
        2.1 4.3 6.5 8.7 0.9
        ...
    
    输出格式：
        NPZ文件包含一个数组，形状为 [N, D]，N为查询数量，D为向量维度
    """
    print(f"Processing query vectors from: {input_txt_path}")
    
    if not os.path.exists(input_txt_path):
        raise FileNotFoundError(f"Input file not found: {input_txt_path}")
    
    # 读取txt文件
    vectors = []
    with open(input_txt_path, 'r') as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            
            # 解析每行的浮点数
            try:
                vector = list(map(float, line.split()))
                vectors.append(vector)
                
                # 每10万行打印一次进度
                if (line_idx + 1) % 100000 == 0:
                    print(f"  Processed {line_idx + 1} lines")
                
                # 如果设置了最大向量数量限制
                if max_vectors is not None and len(vectors) >= max_vectors:
                    print(f"  Reached maximum vector limit: {max_vectors}")
                    break
                    
            except ValueError as e:
                print(f"  Warning: Skipping invalid line {line_idx + 1}: {e}")
                continue
    
    if not vectors:
        raise ValueError(f"No valid vectors found in {input_txt_path}")
    
    # 转换为numpy数组
    vectors_array = np.array(vectors, dtype=getattr(np, dtype))
    print(f"  Loaded {len(vectors)} vectors")
    print(f"  Vector shape: {vectors_array.shape}")
    print(f"  Data type: {vectors_array.dtype}")
    
    # 验证向量维度一致性
    if len(vectors) > 1:
        dims = [len(v) for v in vectors[:100]]  # 检查前100个向量的维度
        if len(set(dims)) > 1:
            print(f"  Warning: Inconsistent vector dimensions found: {set(dims)}")
            print(f"  Using first vector dimension: {dims[0]}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_npz_path), exist_ok=True)
    
    # 保存为NPZ格式
    print(f"\nSaving to NPZ format: {output_npz_path}")
    save_data = {array_name: vectors_array}
    np.savez_compressed(output_npz_path, **save_data)
    
    print(f"✅ Successfully saved query vectors to: {output_npz_path}")
    print(f"   Array name: '{array_name}'")
    print(f"   Array shape: {vectors_array.shape}")
    print(f"   File size: {os.path.getsize(output_npz_path) / 1024 / 1024:.2f} MB")
    
    # 验证保存的文件
    print(f"\n--- Verification ---")
    loaded = np.load(output_npz_path)
    print(f"NPZ file contains keys: {list(loaded.keys())}")
    for key in loaded.keys():
        array = loaded[key]
        print(f"  {key}: {array.shape}, dtype: {array.dtype}")
        if array.size > 0:
            print(f"    Value range: [{array.min():.6f}, {array.max():.6f}]")
            print(f"    Sample (first vector): {array[0][:5]}{'...' if array.shape[1] > 5 else ''}")


def process_dynamic_k_labels(
    input_dir: str,
    output_path: str,
    k_values: List[int] = None,
    normalize: bool = True,
    max_queries: int = None
) -> None:
    """
    处理动态K值标签文件
    
    Args:
        input_dir: 输入目录，包含多个K值的分布文件
        output_path: 输出NPZ文件路径
        k_values: 要处理的K值列表，如果为None则自动检测
        normalize: 是否归一化为概率分布
        max_queries: 最大查询数量限制（用于测试）
    """
    input_dir = Path(input_dir)
    
    # 自动检测K值文件
    if k_values is None:
        k_files = {}
        for file_path in input_dir.glob("knn_distributions_k*.txt"):
            try:
                k_val = extract_k_value_from_filename(file_path.name)
                k_files[k_val] = file_path
            except ValueError as e:
                print(f"Skipping file {file_path.name}: {e}")
        
        k_values = sorted(k_files.keys())
        print(f"Auto-detected K values: {k_values}")
    else:
        # 手动指定K值
        k_files = {}
        for k_val in k_values:
            file_path = input_dir / f"knn_distributions_k{k_val}.txt"
            if file_path.exists():
                k_files[k_val] = file_path
            else:
                raise FileNotFoundError(f"Distribution file not found: {file_path}")
    
    if not k_files:
        raise ValueError(f"No valid distribution files found in {input_dir}")
    
    print(f"\nProcessing {len(k_files)} K values: {sorted(k_files.keys())}")
    
    # 处理每个K值的分布文件
    all_distributions = {}
    
    for k_val in sorted(k_files.keys()):
        file_path = k_files[k_val]
        print(f"\n--- Processing K={k_val} ---")
        
        # 加载分布数据
        distributions = load_distribution_file(str(file_path), normalize=normalize)
        
        # 如果设置了最大查询数量限制
        if max_queries is not None and distributions.shape[0] > max_queries:
            print(f"  Limiting to first {max_queries} queries")
            distributions = distributions[:max_queries]
        
        # 存储到字典中
        all_distributions[k_val] = distributions
        
        print(f"  Final shape for K={k_val}: {distributions.shape}")
    
    # 验证所有K值的查询数量一致
    query_counts = [dist.shape[0] for dist in all_distributions.values()]
    cluster_counts = [dist.shape[1] for dist in all_distributions.values()]
    
    if len(set(query_counts)) > 1:
        raise ValueError(f"Inconsistent query counts across K values: {query_counts}")
    if len(set(cluster_counts)) > 1:
        raise ValueError(f"Inconsistent cluster counts across K values: {cluster_counts}")
    
    n_queries = query_counts[0]
    n_clusters = cluster_counts[0]
    
    print(f"\n--- Summary ---")
    print(f"Total queries: {n_queries}")
    print(f"Total clusters: {n_clusters}")
    print(f"K values: {sorted(all_distributions.keys())}")
    
    # 保存为NPZ格式
    print(f"\nSaving to NPZ format: {output_path}")
    
    # 创建输出目录
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 准备保存的数据
    save_data = {}
    for k_val, distributions in all_distributions.items():
        key = f"targets_k{k_val}"
        save_data[key] = distributions
        print(f"  {key}: {distributions.shape} (sum_range: [{distributions.sum(axis=1).min():.6f}, {distributions.sum(axis=1).max():.6f}])")
    
    # 保存NPZ文件
    np.savez_compressed(output_path, **save_data)
    
    print(f"\n✅ Successfully saved multi-K labels to: {output_path}")
    print(f"   File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    # 验证保存的文件
    print(f"\n--- Verification ---")
    loaded = np.load(output_path)
    print(f"NPZ file contains keys: {list(loaded.keys())}")
    for key in loaded.keys():
        print(f"  {key}: {loaded[key].shape}, dtype: {loaded[key].dtype}")


def main():
    parser = argparse.ArgumentParser(description="Process dynamic K value labels or query vectors for transformer training")
    
    # 添加模式选择
    parser.add_argument("--mode", type=str, choices=["labels", "vectors"], default="labels",
                       help="Processing mode: 'labels' for K-value distributions, 'vectors' for query vectors")
    
    # 标签处理相关参数
    parser.add_argument("--input_dir", type=str,
                       help="Input directory containing knn_distributions_k*.txt files (for labels mode)")
    parser.add_argument("--k_values", nargs='+', type=int, default=None,
                       help="K values to process (default: auto-detect, for labels mode)")
    parser.add_argument("--no_normalize", action="store_true",
                       help="Skip normalization to probability distributions (for labels mode)")
    
    # 向量处理相关参数
    parser.add_argument("--input_txt", type=str,
                       help="Input txt file containing query vectors (for vectors mode)")
    parser.add_argument("--array_name", type=str, default="queries",
                       help="Array name in NPZ file (for vectors mode)")
    parser.add_argument("--dtype", type=str, default="float32",
                       help="Data type for vectors (for vectors mode)")
    
    # 通用参数
    parser.add_argument("--output_path", type=str, required=True,
                       help="Output NPZ file path")
    parser.add_argument("--max_queries", type=int, default=None,
                       help="Maximum number of queries/vectors to process (for testing)")
    
    args = parser.parse_args()
    
    if args.mode == "labels":
        # 处理标签
        if not args.input_dir:
            parser.error("--input_dir is required for labels mode")
        
        process_dynamic_k_labels(
            input_dir=args.input_dir,
            output_path=args.output_path,
            k_values=args.k_values,
            normalize=not args.no_normalize,
            max_queries=args.max_queries
        )
    
    elif args.mode == "vectors":
        # 处理查询向量
        if not args.input_txt:
            parser.error("--input_txt is required for vectors mode")
        
        process_query_vectors_txt_to_npz(
            input_txt_path=args.input_txt,
            output_npz_path=args.output_path,
            array_name=args.array_name,
            dtype=args.dtype,
            max_vectors=args.max_queries
        )


if __name__ == "__main__":
    main()


'''
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode/
python src/model/dynamic_k/process_dynamic_k_labels.py \
    --mode labels \
    --input_dir input/Training_data/siftsmall/siftsmall5H/query \
    --output_path input/Training_data/siftsmall/siftsmall5H/query/train_labels.npz \
    --k_values 1 10 20 50 100

python src/model/dynamic_k/process_dynamic_k_labels.py \
    --mode vectors \
    --input_txt input/Training_data/gist1M_learn/gist_learn.txt \
    --output_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/train_queries.npz

python src/model/dynamic_k/process_dynamic_k_labels.py \
    --mode vectors \
    --input_txt input/Training_data/gist1M_learn/gist_query.txt \
    --output_path input/Training_data/gist1M_learn/leafsize20K/dynamicK/test_queries.npz
'''