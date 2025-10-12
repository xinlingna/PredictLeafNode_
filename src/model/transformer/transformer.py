from stringprep import b1_set
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from ..preprocess.bulid_loaders import load_data_and_build_loaders
import math
import matplotlib.pyplot as plt
import os

""" 
# 可以考虑 cosine值
"""

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # CUDA随机性
    torch.cuda.manual_seed_all(seed)          # 多GPU情况下
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # 避免非确定性卷积

class DistributionPredictor(nn.Module):
    def __init__(self, 
                 input_dim,
                 dropout, 
                 hidden_dim=256, 
                 num_classes=187, 
                 num_layers=4, 
                 alpha_Exists=False,
                 dual_branch_fusion=True,
                 sim_only=False):
        super().__init__()

        self.hidden_dim=hidden_dim
        self.sim_only = sim_only
        self.sim_scale = nn.Parameter(torch.tensor(1.0))  # sim温度
        self.film = nn.Sequential(
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim)  # 拆成 gamma,beta
        )
        self.res_gate = nn.Parameter(torch.tensor(0.0))  # 残差门控
        
        self.hidden_dim=hidden_dim
        self.dual_branch_fusion=dual_branch_fusion
        self.alpha_Exists=alpha_Exists
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 标量
        # 输入投影层：将拼接向量映射到 transformer 的 d_model
        if self.sim_only:
            # 仅使用 sim 分支
            self.sim_proj = nn.Linear(num_classes, hidden_dim)
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        elif self.dual_branch_fusion==True:
            self.vector_dim=input_dim-num_classes
            self.sift_proj=nn.Linear(self.vector_dim,hidden_dim//2)
            self.sim_proj=nn.Linear(num_classes,hidden_dim//2)
            self.input_proj = nn.Linear(hidden_dim, hidden_dim)
        else:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)

        # 构建 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=4,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True  # shape: (batch_size, seq_len, d_model)
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 输出层：预测每个类出现的次数（soft regression）
        # 改善：移除 BatchNorm1d（小 batch 不稳定），改用 LayerNorm + Dropout 增强泛化
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            # nn.Dropout(p=0.2),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, sift_vec, similarity_vec):
        # sim 标准化 + 温度
        similarity_vec = (similarity_vec - similarity_vec.mean(dim=1, keepdim=True)) / (similarity_vec.std(dim=1, keepdim=True) + 1e-6)
        similarity_vec = similarity_vec * F.softplus(self.sim_scale)

        if self.sim_only:
            # 仅 sim 分支
            sim_vec = self.sim_proj(similarity_vec)
            x = self.input_proj(sim_vec).unsqueeze(1)
        else:
            if self.dual_branch_fusion:
                sift_vec = self.sift_proj(sift_vec)
                sim_vec  = self.sim_proj(similarity_vec)

                # FiLM: 用 sift 生成 gamma,beta 调制 sim
                film_params = self.film(sift_vec)              # [B, hidden_dim]
                gamma, beta = torch.chunk(film_params, 2, dim=-1)
                sim_vec = (1 + torch.tanh(gamma)) * sim_vec + beta

                # x = torch.cat([sift_vec, sim_vec], dim=-1)
                x = torch.cat([sim_vec, sift_vec], dim=-1)
                x = self.input_proj(x).unsqueeze(1)
            else:
                # 不做双分支投影，直接拼接原始输入后投影
                x_in = torch.cat([sift_vec, similarity_vec], dim=-1)
                x = self.input_proj(x_in).unsqueeze(1)
        x = self.norm(x)

        residual = x
        x = self.transformer(x)
        gate = torch.sigmoid(self.res_gate)
        x = gate * x + (1 - gate) * residual
        x = x.squeeze(1)
        logits = self.output_layer(x)
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


def train_one_epoch(model, train_loader, optimizer, criterion, device,label_process):
    model.train()
    total_loss = 0

    for sift_vec, sim_vec, label_dist in train_loader:
        sift_vec = sift_vec.to(device)
        sim_vec = sim_vec.to(device)
        label_dist = label_dist.to(device)

        optimizer.zero_grad()
        pred_logits = model(sift_vec, sim_vec)

        # KL 散度损失
        log_probs = F.log_softmax(pred_logits, dim=-1)
        if label_process=="proportional":
            soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)
        elif label_process=="softmax":
            soft_labels = F.softmax(label_dist, dim=1)
        elif label_process=="proportional_weight":
            topk=10
            weight_factor = 0.5
            weighted_label = label_dist.clone()

            topk_indices = torch.topk(weighted_label, k=topk, dim=1).indices  # (B, 10)
            mask = torch.zeros_like(weighted_label).scatter(1, topk_indices, 1.0)  # (B, C), 1.0 at top-k positions
            weighted_label = weighted_label * (1 - mask * (1 - weight_factor))  # 只有 top-k 被乘以 0.5
            label_dist= label_dist+weighted_label
            soft_labels =  label_dist /  label_dist.sum(dim=1, keepdim=True)   # 避免除以0
            

        loss = criterion(log_probs, soft_labels)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)
    return avg_loss


def evaluate_one_epoch(model, val_loader, criterion,device,label_process):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for sift_vec, sim_vec, label_dist in val_loader:
            sift_vec = sift_vec.to(device)
            sim_vec = sim_vec.to(device)
            label_dist = label_dist.to(device)
            label_dist = label_dist / label_dist.sum(dim=1, keepdim=True)
            

            pred_logits = model(sift_vec, sim_vec)
            all_preds.append(pred_logits)
            all_labels.append(label_dist)


            log_probs = F.log_softmax(pred_logits, dim=-1)
            if label_process=="proportional":
                soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)
            elif label_process=="softmax":
                soft_labels = F.softmax(label_dist, dim=1)
            elif label_process=="proportional_weight":
                topk=10
                weight_factor = 0.8
                weighted_label = label_dist.clone()

                topk_indices = torch.topk(weighted_label, k=topk, dim=1).indices  # (B, 10)
                mask = torch.zeros_like(weighted_label).scatter(1, topk_indices, 1.0)  # (B, C), 1.0 at top-k positions
                # weighted_label = weighted_label * (1 - mask * (1 - weight_factor))  # 只有 top-k 被乘以 0.5
                # label_dist= label_dist+weighted_label
                # soft_labels =  label_dist /  label_dist.sum(dim=1, keepdim=True)   # 避免除以0

                label_dist = label_dist * (1 + weight_factor * mask)
                soft_labels =  label_dist /  label_dist.sum(dim=1, keepdim=True)   # 避免除以0

                

            loss = criterion(log_probs, soft_labels)
            total_loss += loss.item()

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    recall_at_1 = topk_recall(all_preds, all_labels, k=1)
    recall_at_5 = topk_recall(all_preds, all_labels, k=5)
    recall_at_10 = topk_recall(all_preds, all_labels, k=10)
    recall_at_30 = topk_recall(all_preds, all_labels, k=30)

    return recall_at_1,recall_at_5,recall_at_10, recall_at_30,total_loss/len(val_loader)


def train_model(model, train_loader, val_loader, num_epochs=500, lr=1e-3, device='cuda',label_process="proportional", early_stop_patience=30):
    
    # 绘制 accuracy-epoch图像
    recall1_list = []
    recall5_list = []
    recall10_list = []
    recall30_list = []
    
    # early-stop（基于 Recall@10 改善）
    early_stop_counter = 0

        
    model.to(device)
    # 优化器：AdamW + 权重衰减更稳定
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    # 学习率调度器：验证集损失无提升则降低学习率
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-6
    )
    criterion = nn.KLDivLoss(reduction='batchmean')  # KL散度用于软标签

    # 初始化最大指标记录
    best_recall1 = 0.0
    best_recall5 = 0.0
    best_recall10 = 0.0
    best_recall30 = 0.0
    best_epoch_recall1 = 0
    best_epoch_recall5 = 0
    best_epoch_recall10 = 0
    best_epoch_recall30 = 0

    actual_epochs=num_epochs-1
    for epoch in range(num_epochs):
        # 训练 1 epoch
        model.train()
        total_loss = 0.0
        for sift_vec, sim_vec, label_dist in train_loader:
            sift_vec = sift_vec.to(device)
            sim_vec = sim_vec.to(device)
            label_dist = label_dist.to(device)

            optimizer.zero_grad()
            pred_logits = model(sift_vec, sim_vec)

            log_probs = F.log_softmax(pred_logits, dim=-1)
            if label_process=="proportional":
                soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)
            elif label_process=="softmax":
                soft_labels = F.softmax(label_dist, dim=1)
            elif label_process=="proportional_weight":
                # 与验证阶段保持一致：强调 top-k
                topk = 10
                weight_factor = 0.8
                topk_indices = torch.topk(label_dist, k=topk, dim=1).indices
                mask = torch.zeros_like(label_dist).scatter(1, topk_indices, 1.0)
                label_dist = label_dist * (1 + weight_factor * mask)
                soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)
            else:
                soft_labels = label_dist / label_dist.sum(dim=1, keepdim=True)

            loss = criterion(log_probs, soft_labels)

            loss.backward()
            # 梯度裁剪，抑制梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # 验证
        recall_at_1,recall_at_5,recall_at_10, recall_at_30,val_avg_loss = evaluate_one_epoch(model, val_loader, criterion,device,label_process)
        # 调度器根据验证损失调整学习率
        scheduler.step(val_avg_loss)
        
        # 记录最优指标
        improved_recall10 = False
        if recall_at_1 > best_recall1:
            best_recall1 = recall_at_1
            best_epoch_recall1 = epoch + 1
        if recall_at_5 > best_recall5:
            best_recall5 = recall_at_5
            best_epoch_recall5 = epoch + 1
        if recall_at_10 > best_recall10:
            best_recall10 = recall_at_10
            best_epoch_recall10 = epoch + 1
            improved_recall10 = True
        if recall_at_30 > best_recall30:
            best_recall30 = recall_at_30
            best_epoch_recall30 = epoch + 1

        # Early stopping（基于 Recall@10 是否提升）
        if improved_recall10:
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        if early_stop_counter >= early_stop_patience:
            actual_epochs = epoch
            print(f"\n Early stopping triggered at epoch {epoch+1}. No improvement in Recall@10 for {early_stop_patience} consecutive epochs.")
            break


        print(f"[Epoch {epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f}| Recall@1: {recall_at_1:.4f} | Recall@5: {recall_at_5:.4f} | Recall@10: {recall_at_10:.4f} | Recall@30: {recall_at_30:.4f}")
        print(f"📈 Best Recall@1: {best_recall1:.4f} at Epoch {best_epoch_recall1}")
        print(f"📈 Best Recall@5: {best_recall5:.4f} at Epoch {best_epoch_recall5}")
        print(f"📈 Best Recall@10: {best_recall10:.4f} at Epoch {best_epoch_recall10}")
        print(f"📈 Best Recall@30: {best_recall30:.4f} at Epoch {best_epoch_recall30}")
        
        recall1_list.append(recall_at_1)
        recall5_list.append(recall_at_5)
        recall10_list.append(recall_at_10)
        recall30_list.append(recall_at_30)
        

    # 输出训练中表现最好的指标
    print("\n✅ Training Complete.")
    print(f"📈 Best Recall@1: {best_recall1:.4f} at Epoch {best_epoch_recall1}")
    print(f"📈 Best Recall@5: {best_recall5:.4f} at Epoch {best_epoch_recall5}")
    print(f"📈 Best Recall@10: {best_recall10:.4f} at Epoch {best_epoch_recall10}")
    print(f"📈 Best Recall@30: {best_recall30:.4f} at Epoch {best_epoch_recall30}")
    
    
    # 📊 绘制 Recall@K 随 Epoch 的变化曲线并标注最高点
    # 使用实际采样到的 epoch 数，避免 x/y 维度不一致
    actual_epochs = len(recall1_list)
    epochs = list(range(1, actual_epochs + 1))
    
    plt.figure(figsize=(10, 6))
    
    # 定义曲线和标签的元组列表
    recall_data = [
        (recall1_list, 'Recall@1'),
        (recall5_list, 'Recall@5'),
        (recall10_list, 'Recall@10'),
        (recall30_list, 'Recall@30'),
    ]
    
    for recall_values, label in recall_data:
        plt.plot(epochs, recall_values, label=label)
        
        # 找到最大值和对应的 epoch
        max_val = max(recall_values)
        max_epoch = epochs[recall_values.index(max_val)]
    
        # 标注最大点
        plt.annotate(f'{max_val:.4f}',
                     xy=(max_epoch, max_val),
                     xytext=(max_epoch, max_val + 0.01),
                     textcoords='data',
                     ha='center',
                     fontsize=8,
                     arrowprops=dict(arrowstyle='->', lw=0.8))
    
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall@K over Training Epochs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("recall_curve.png")
    plt.show()


def main():
    
    set_seed(42)
    # Initialize argument parser && Add arguments for command-line parameters
    parser = argparse.ArgumentParser(description="Train the SIFT Distribution Predictor Model")
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
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout of the model')
    parser.add_argument('--dual_branch_fusion', type=bool, default=True, help='ddual_branch_fusion of  the model')
    parser.add_argument('--label_process', type=str, default="proportional", help='the way label_process')
    parser.add_argument('--sim_only', action='store_true', help='Use only similarity vector as input (ignore sift_vec)')
    parser.add_argument('--similarity_type', type=str, default='cosine', choices=['cosine','l2','dot'], help='How to compute similarity vector in dataloader')
    
    parser.add_argument('--early_stop_patience', type=int, default=20, help='Number of epochs with no Recall@10 improvement to stop early')


    # Parse arguments
    args = parser.parse_args()

    # Load data and create DataLoader
    train_loader, val_loader = load_data_and_build_loaders(
        query_path=args.query_path,
        centroids_path=args.centroids_path,
        label_path=args.label_path,
        batch_size=args.batch_size,
        val_ratio=args.val_ratio,
        shuffle=args.shuffle,
        similarity_type=args.similarity_type
    )

    # Dynamically set input_dim and num_classes based on the data
    # input_dim is the size of sift_vec (e.g., 128 for the SIFT feature size)
    if args.sim_only:
        input_dim = train_loader.dataset.label.shape[1]  # only similarity dimension
    else:
        input_dim = train_loader.dataset.query_sift.shape[1] + train_loader.dataset.label.shape[1]  # (SIFT + similarity)
    
    # num_classes is the number of unique classes in the label dataset
    num_classes = train_loader.dataset.label.shape[1]

    # Initialize the model with the dynamically determined input_dim and num_classes
    model = DistributionPredictor(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=num_classes,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dual_branch_fusion=args.dual_branch_fusion,
        sim_only=args.sim_only
    )

    # Train the model
    train_model(model, 
                train_loader, 
                val_loader, 
                num_epochs=args.num_epochs, 
                lr=args.lr, 
                device=args.device, 
                label_process=args.label_process,
                early_stop_patience=args.early_stop_patience
                )
    
    
    # save the model to the same directory as query_path
    model_name = f"model_hd{args.hidden_dim}_nl{args.num_layers}_lp_{args.label_process}_st_{args.similarity_type}_so_{args.sim_only}.pth"
    save_dir = os.path.dirname(os.path.abspath(args.centroids_path))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, model_name)
    torch.save(model.state_dict(), save_path)
    print(f"✅ 模型已保存到 {save_path}")




if __name__ == '__main__':
    main()


'''    
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode 
python -m src.model.transformer \
--query_path input/Training_data/sift1M_learn/sift_learn.txt \
--centroids_path input/Training_data/sift1M_learn/leafsize10K/leaf_center.txt \
--label_path input/Training_data/sift1M_learn/leafsize10K/knn_distributions.txt \
--batch_size 256 \
--val_ratio 0.15 \
--num_epochs 150 \
--lr 0.0003 \
--device cuda \
--hidden_dim 256 \
--num_layers 4 \
--dropout 0.2 \
--dual_branch_fusion True \
--label_process proportional_weight \
--early_stop_patience 20 \
--similarity_type l2 \
--sim_only
'''
