import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from pathlib import Path
import argparse

class QueryDataset(Dataset):
    def __init__(self, query_sift, centroids, label):
        super().__init__()
        eps = 1e-12
        self.query_sift = torch.tensor(query_sift, dtype=torch.float32)     # (N, 128)
        self.query_sift  = self.query_sift / (self.query_sift.max() + eps)        # (N, 128)

        self.label = torch.tensor(label, dtype=torch.float32)               # (N, C)
        self.centroids = torch.tensor(centroids, dtype=torch.float32)       # (C, 128)

        eps = 1e-12
        normalized_query = self.query_sift / (self.query_sift.max() + eps)        # (N, 128)
        normalized_centroids = self.centroids / (self.centroids.max() + eps)      # (C, 128)
        # 预计算欧氏距离 ，结果 shape = (N, C)
        # 注意这里取负号，越小距离 → 越高“相似度”
        self.similarity = -torch.cdist(normalized_query, normalized_centroids, p=2)  # (N, C)


       # 计算余弦相似度：矩阵乘法 (N, 128) @ (128, C) → (N, C)
        # normalized_query = F.normalize(self.query_sift, p=2, dim=1)     # (N, 128)
        # normalized_centroids = F.normalize(self.centroids, p=2, dim=1)  # (C, 128)
        # self.similarity_cosine = torch.matmul(normalized_query, normalized_centroids.T)  # (N, C)

        eps = 1e-12
        normalized_query = self.query_sift / (self.query_sift.max() + eps)        # (N, 128)
        normalized_centroids = self.centroids / (self.centroids.max() + eps)      # (C, 128)
        self.similarity_cosine = torch.matmul(normalized_query, normalized_centroids.T)  # (N, C)
        self.similarity_cosine = F.normalize(self.similarity_cosine, p=2, dim=1)  # (N, C)   /////////

        

    def __len__(self):
        return self.query_sift.shape[0]

    def __getitem__(self, idx):
        query = self.query_sift[idx]            # (128,)
        similarity_cosine=self.similarity_cosine[idx] # (C,)
        label_vec = self.label[idx]             # (C,)
        return query, similarity_cosine, label_vec


class QueryDataset_l2_cosine(Dataset):
    def __init__(self, query_sift, centroids, label):
        super().__init__()

        self.query_sift = torch.tensor(query_sift, dtype=torch.float32)     # (N, 128)
        self.label = torch.tensor(label, dtype=torch.float32)               # (N, C)
        self.centroids = torch.tensor(centroids, dtype=torch.float32)       # (C, 128)

        # 预计算欧氏距离 ，结果 shape = (N, C)
        # 注意这里取负号，越小距离 → 越高“相似度”
        normalized_query = F.normalize(self.query_sift, p=2, dim=1)     # (N, 128)
        normalized_centroids = F.normalize(self.centroids, p=2, dim=1)  # (C, 128)
        self.similarity = -torch.cdist(normalized_query, normalized_centroids, p=2)  # (N, C)


       # 计算余弦相似度：矩阵乘法 (N, 128) @ (128, C) → (N, C)
        normalized_query = F.normalize(self.query_sift, p=2, dim=1)     # (N, 128)
        normalized_centroids = F.normalize(self.centroids, p=2, dim=1)  # (C, 128)
        self.similarity_cosine = torch.matmul(normalized_query, normalized_centroids.T)  # (N, C)
        # 归一化
        self.similarity_cosine = F.normalize(self.similarity_cosine, p=2, dim=1)  # (N, C)   /////////
        

    def __len__(self):
        return self.query_sift.shape[0]

    def __getitem__(self, idx):
        query = self.query_sift[idx]            # (128,)
        similarity = self.similarity[idx]         # (C,)
        similarity_cosine=self.similarity_cosine[idx] # (C,)
        label_vec = self.label[idx]             # (C,)
        return query, similarity, similarity_cosine,label_vec

def load_data_and_build_loaders(
    query_path="query_sift.txt",
    centroids_path="centroids.txt",
    label_path="label.txt",
    batch_size=64,
    val_ratio=0.2,
    shuffle=True
):
    # 加载原始数据（.txt → numpy）
    query_sift = np.loadtxt(query_path)
    centroids = np.loadtxt(centroids_path)
    label = np.loadtxt(label_path)

    # 拆分训练集与验证集
    train_q, val_q, train_l, val_l = train_test_split(query_sift, label, test_size=val_ratio, random_state=42)

    # 封装为 Dataset
    train_dataset = QueryDataset(train_q, centroids, train_l)
    val_dataset = QueryDataset(val_q, centroids, val_l)

    # DataLoader 封装
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader



################ 测试函数 #####################
def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Query Dataset Loader")
    
    # Add arguments for command-line parameters
    parser.add_argument('--query_path', type=str, default="query_sift.txt", help='Path to the query sift data')
    parser.add_argument('--centroids_path', type=str, default="centroids.txt", help='Path to the centroids data')
    parser.add_argument('--label_path', type=str, default="label.txt", help='Path to the label data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the training data')
    
    # Parse arguments
    args = parser.parse_args()

    # Load data and create DataLoader
    train_loader, val_loader = load_data_and_build_loaders(
        query_path=args.query_path,
        centroids_path=args.centroids_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        shuffle=args.shuffle
    )

    # Example: Print out the first batch's query, similarity, and label shape to confirm data loading
    for query, similarity, label in train_loader:
        print(f"Query shape: {query.shape}, Similarity shape: {similarity.shape}, Label shape: {label.shape}")
        break  # Just print the first batch as a test

if __name__ == '__main__':
    main()
    