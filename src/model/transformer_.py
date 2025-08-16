import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from ..preprocess.bulid_loaders import load_data_and_build_loaders
import math

""" 
# 可以考虑 cosine值
# 
"""

class DistributionPredictor_l2_cosine(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_classes=183, num_layers=4):
        super().__init__()
        
        self.vector_dim=input_dim-num_classes
        self.hidden_dim=hidden_dim
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 标量
        # 输入投影层：将拼接向量映射到 transformer 的 d_model
        self.sift_proj=nn.Linear(self.vector_dim,hidden_dim//2)
        self.sim_proj=nn.Linear(num_classes,hidden_dim//2)
        self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # 构建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=0.1,
            batch_first=True  # shape: (batch_size, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：预测每个类出现的次数（soft regression）
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.BatchNorm1d(hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
            # nn.BatchNorm1d(num_classes),
        )

    def forward(self, sift_vec, similarity_vec):
        """
        输入：
            sift_vec: (B, 128)
            similarity_vec: (B, 183)
        输出：
            类别直方图分布预测: (B, num_classes)
        """
        sift_vec=self.sift_proj(sift_vec)
        similarity_vec=self.sim_proj(similarity_vec)
        x = torch.cat([sift_vec, similarity_vec], dim=-1)  # (B, 311)
        # batch_size=x.shape[0]
        x = self.input_proj(x).unsqueeze(1)  # (B, 1, hidden_dim)
        x=self.norm(x)
        # reshape_hd=math.sprt(self.hidden_dim)
        # x=self.norm(x).reshape(-1,reshape_hd,reshape_hd)


        residual=x
        x = self.transformer(x)  # (B, 1, hidden_dim)
        # x=self.alpha*x+(1-self.alpha)*residual
        x=x.squeeze(1)
        # x = x.reshpae(batch_size,1,self.hidden_dim)  # 去掉 sequence 维度

        logits = self.output_layer(x)  # (B, num_classes)

        # 可选激活：预测的是top-100类的soft频率分布 → Softmax 或 Soft ReLU
        return logits
        
def topk_recall(pred_prob, label_prob, k=10):
    """
    计算 Top-K recall: 预测出的 top-k 类别中有多少真实 top-100 类别
    pred_dist: (B, C)
    label_dist: (B, C)
    """
    topk_pred = torch.topk(pred_prob, k, dim=1).indices  # (B, k)
    top_true = torch.topk(label_prob, k, dim=1).indices  # (B, k) 用真实分布 top-k 当目标

    recall_count = 0
    for i in range(pred_prob.size(0)):
        intersect = len(set(topk_pred[i].tolist()) & set(top_true[i].tolist()))
        recall_count += intersect / k

    return recall_count / pred_prob.size(0)  # average recall@k


def train_one_epoch(model, train_loader, optimizer, criterion, device,label_process="proportional"):
    model.train()
    total_loss = 0

    for sift_vec, sim_vec,label_dist in train_loader:
        sift_vec = sift_vec.to(device)
        sim_vec = sim_vec.to(device)
        # sim_cosine = sim_cosine.to(device)
        label_dist = label_dist.to(device)

        optimizer.zero_grad()
        pred_logits = model(sift_vec, sim_vec)

        # KL 散度损失
        log_probs = F.log_softmax(pred_logits, dim=-1)
        if label_process=="proportional":
            soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)
        elif label_process=="softmax":
            soft_labels = F.softmax(label_dist, dim=1)

        loss = criterion(log_probs, soft_labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_one_epoch(model, val_loader, device,label_process="proportional"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sift_vec, sim_vec,label_dist in val_loader:
            sift_vec = sift_vec.to(device)
            sim_vec = sim_vec.to(device)
            # sim_cosine = sim_cosine.to(device)
            label_dist = label_dist.to(device)
            if label_process=="proportional":
                label_dist = label_dist / label_dist.sum(dim=1, keepdim=True)
            elif label_process=="softmax":
               label_dist = F.softmax(label_dist, dim=1)

            pred_logits = model(sift_vec, sim_vec)
            all_preds.append(pred_logits)
            all_labels.append(label_dist)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    recall_at_10 = topk_recall(all_preds, all_labels, k=10)
    recall_at_30 = topk_recall(all_preds, all_labels, k=30)

    return recall_at_10, recall_at_30


def train_model(model, train_loader, val_loader, num_epochs=500, lr=1e-3, device='cuda'):
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL散度用于软标签

    # 初始化最大指标记录
    best_recall10 = 0.0
    best_recall30 = 0.0
    best_epoch_recall10 = 0
    best_epoch_recall30 = 0

    for epoch in range(num_epochs):
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        recall_at_10, recall_at_30 = evaluate_one_epoch(model, val_loader, device)

        # 更新最大 recall@10
        if recall_at_10 > best_recall10:
            best_recall10 = recall_at_10
            best_epoch_recall10 = epoch + 1  # epoch 从 0 开始，显示时加 1

        # 更新最大 recall@30
        if recall_at_30 > best_recall30:
            best_recall30 = recall_at_30
            best_epoch_recall30 = epoch + 1

        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} | Recall@10: {recall_at_10:.4f} | Recall@30: {recall_at_30:.4f}")
        print(f"📈 Best Recall@10: {best_recall10:.4f} at Epoch {best_epoch_recall10}")
        print(f"📈 Best Recall@30: {best_recall30:.4f} at Epoch {best_epoch_recall30}")

    # 输出训练中表现最好的指标
    print("\n✅ Training Complete.")
    print(f"📈 Best Recall@10: {best_recall10:.4f} at Epoch {best_epoch_recall10}")
    print(f"📈 Best Recall@30: {best_recall30:.4f} at Epoch {best_epoch_recall30}")



def main():
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Train the SIFT Distribution Predictor Model")
    
    # Add arguments for command-line parameters
    parser.add_argument('--query_path', type=str, default="query_sift.txt", help='Path to the query sift data')
    parser.add_argument('--centroids_path', type=str, default="centroids.txt", help='Path to the centroids data')
    parser.add_argument('--label_path', type=str, default="label.txt", help='Path to the label data')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for DataLoader')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation set ratio')
    parser.add_argument('--shuffle', type=bool, default=True, help='Whether to shuffle the training data')
    parser.add_argument('--num_epochs', type=int, default=500, help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for the optimizer')
    parser.add_argument('--device', type=str, default='cuda', choices=['cpu', 'cuda'], help='Device to run the model on')
    
    # Add arguments for hidden_dim and num_layers
    parser.add_argument('--hidden_dim', type=int, default=256, help='Hidden dimension size for the transformer model')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of layers in the transformer model')

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

    # Dynamically set input_dim and num_classes based on the data
    # input_dim is the size of sift_vec (e.g., 128 for the SIFT feature size)
    input_dim = train_loader.dataset.query_sift.shape[1] + train_loader.dataset.label.shape[1]  # (SIFT + similarity)
    
    # num_classes is the number of unique classes in the label dataset
    num_classes = train_loader.dataset.label.shape[1]  # This assumes label is one-hot encoded

    # Initialize the model with the dynamically determined input_dim and num_classes
    model =DistributionPredictor_l2_cosine(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers
    )

    # Train the model
    train_model(model, train_loader, val_loader, num_epochs=args.num_epochs, lr=args.lr, device=args.device)



if __name__ == '__main__':
    main()


      

