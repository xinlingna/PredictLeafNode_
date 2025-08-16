import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from ..preprocess.bulid_loaders import load_data_and_build_loaders
import math
import matplotlib.pyplot as plt
import optuna


""" 
# 可以考虑 cosine值
# 
"""

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)              # CUDA随机性
    torch.cuda.manual_seed_all(seed)          # 多GPU情况下
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    # 避免非确定性卷积

class SIFTDistributionPredictor(nn.Module):
    def __init__(self, input_dim,
                 dropout, 
                 hidden_dim=256, 
                 num_classes=183, 
                 num_layers=4, 
                 alpha_Exists=False,
                 dual_branch_fusion=True):
        super().__init__()
        
        self.hidden_dim=hidden_dim
        self.dual_branch_fusion=dual_branch_fusion
        self.alpha_Exists=alpha_Exists
        self.alpha = nn.Parameter(torch.tensor(1.0))  # 标量
        # 输入投影层：将拼接向量映射到 transformer 的 d_model
        if self.dual_branch_fusion==True:
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
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
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
        if self.dual_branch_fusion:
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
        if self.alpha_Exists:
            x=self.alpha*x+(1-self.alpha)*residual
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


def evaluate_one_epoch(model, val_loader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for sift_vec, sim_vec, label_dist in val_loader:
            sift_vec = sift_vec.to(device)
            sim_vec = sim_vec.to(device)
            label_dist = label_dist.to(device)
            label_dist = label_dist / label_dist.sum(dim=1, keepdim=True)
            

            pred_logits = model(sift_vec, sim_vec)
            all_preds.append(pred_logits)
            all_labels.append(label_dist)

    all_preds = torch.cat(all_preds, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    recall_at_1 = topk_recall(all_preds, all_labels, k=1)
    recall_at_5 = topk_recall(all_preds, all_labels, k=5)
    recall_at_10 = topk_recall(all_preds, all_labels, k=10)
    recall_at_30 = topk_recall(all_preds, all_labels, k=30)

    return recall_at_1,recall_at_5,recall_at_10, recall_at_30


def train_model(model, train_loader, val_loader, num_epochs=500, lr=1e-3, device='cuda',label_process="proportional", early_stop_patience=10):
    
    # 绘制 accuracy-epoch图像
    recall1_list = []
    recall5_list = []
    recall10_list = []
    recall30_list = []
    
    # early-stop
    early_stop_counter = 0
    best_recall10_for_earlystop = 0.0

        
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
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
        avg_loss = train_one_epoch(model, train_loader, optimizer, criterion, device,label_process)
        recall_at_1,recall_at_5,recall_at_10, recall_at_30 = evaluate_one_epoch(model, val_loader, device)
        
        # 更新最大 recall@1
        if recall_at_1 > best_recall1:
            best_recall1 = recall_at_1
            best_epoch_recall1 = epoch + 1  # epoch 从 0 开始，显示时加 1
        # 更新最大 recall@5
        if recall_at_5 > best_recall5:
            best_recall5 = recall_at_5
            best_epoch_recall5 = epoch + 1  # epoch 从 0 开始，显示时加 1
        # 更新最大 recall@10
        if recall_at_10 > best_recall10:
            best_recall10 = recall_at_10
            best_epoch_recall10 = epoch + 1  # epoch 从 0 开始，显示时加 1

        # 更新最大 recall@30
        if recall_at_30 > best_recall30:
            best_recall30 = recall_at_30
            best_epoch_recall30 = epoch + 1
            
        # Early stopping check based on Recall@10
        if recall_at_10 > best_recall10_for_earlystop:
            best_recall10_for_earlystop = recall_at_10
            early_stop_counter = 0
        else:
            early_stop_counter += 1
        
        if early_stop_counter >= early_stop_patience:
            actual_epochs=epoch
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
    
    return best_recall1, best_recall5, best_recall10, best_recall30
    
    
"""     # 📊 绘制 Recall@K 随 Epoch 的变化曲线并标注最高点
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
    plt.show() """
 

def parse_args():
    parser = argparse.ArgumentParser(description="Auto-tune SIFT Distribution Predictor")
    parser.add_argument('--query_path', type=str, required=True, help='Path to the query sift data')
    parser.add_argument('--centroids_path', type=str, required=True, help='Path to the centroids data')
    parser.add_argument('--label_path', type=str, required=True, help='Path to the label data')
    parser.add_argument('--n_trials', type=int, default=300, help='Number of Optuna trials')
    return parser.parse_args()

def objective(trial, query_path, centroids_path, label_path):
    # 超参数搜索空间
    hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
    num_layers = trial.suggest_int("num_layers", 2, 6)
    dropout = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-4, 1e-2)
    label_process = trial.suggest_categorical("label_process", ["proportional", "softmax", "proportional_weight"])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    dual_branch_fusion = trial.suggest_categorical("dual_branch_fusion", [True, False])

    # 数据加载
    train_loader, val_loader = load_data_and_build_loaders(
        query_path=query_path,
        centroids_path=centroids_path,
        label_path=label_path,
        batch_size=batch_size,
        val_ratio=0.2,
        shuffle=True
    )

    input_dim = train_loader.dataset.query_sift.shape[1] + train_loader.dataset.label.shape[1]
    num_classes = train_loader.dataset.label.shape[1]

    # 模型构建
    model = SIFTDistributionPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        dual_branch_fusion=dual_branch_fusion,
    )

    # 模型训练与评估
    _, _, recall_10, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=300,
        lr=lr,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        label_process=label_process,
        early_stop_patience=10
    )

    return recall_10

if __name__ == "__main__":
    args = parse_args()

    study = optuna.create_study(
        direction="maximize",
        study_name="sift_recall10_optimization"
    )

    # lambda 把命令行参数绑定进 objective
    study.optimize(lambda trial: objective(trial, args.query_path, args.centroids_path, args.label_path),
                   n_trials=args.n_trials)

    print("\n✅ Best Trial:")
    print(f"Recall@10: {study.best_value:.4f}")
    print("Best hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")
    
    # Create filename from parameter names
    param_names = "_".join(study.best_params.keys())
    filename = f"best_params_{param_names}.txt"
    
    # Save best parameters to file
    with open(filename, 'w') as f:
        for key, value in study.best_params.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\nSaved best parameters to: {filename}")

""" 
python -m src.model.transformer_optuna \
  --query_path input/Training_data/sift1M_merged/sift_merged_query.txt \
  --centroids_path input/Training_data/sift1M_merged/leafsize10K/leaf_center.txt \
  --label_path input/Training_data/sift1M_merged/leafsize10K/knn_distributions.txt \
  --n_trials 3000  screen -r test2
  
  
python -m src.model.transformer_optuna \
  --query_path input/Training_data/sift1M_merged/sift_merged_query.txt \
  --centroids_path input/Training_data/sift1M_merged/leafsize10K/leaf_centroid.txt \
  --label_path input/Training_data/sift1M_merged/leafsize10K/knn_distributions.txt \
  --n_trials 3000  screen -r test1

"""