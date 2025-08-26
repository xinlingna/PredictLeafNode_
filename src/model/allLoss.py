import torch
import torch.nn as nn

class HybridLoss(nn.Module):
    def __init__(self, mse_weight=0.7, kl_weight=0.3, temperature=1.0, smoothing=0.01, debug=False):
        super().__init__()
        self.mse_weight = mse_weight
        self.kl_weight = kl_weight
        self.temperature = temperature
        self.smoothing = smoothing
        self.debug = debug
        
        self.mse = nn.MSELoss()
        self.kl = nn.KLDivLoss(reduction='batchmean')
    
    def forward(self, logits, target):
        # 温度softmax
        preds = torch.softmax(logits / self.temperature, dim=-1)
        log_preds = torch.log_softmax(logits / self.temperature, dim=-1)
        
        # 标签平滑
        K = target.size(-1)
        smoothed_target = target * (1 - self.smoothing) + self.smoothing / K
        
        # 计算两种损失
        mse_loss = self.mse(preds, smoothed_target)
        kl_loss = self.kl(log_preds, smoothed_target)
        if self.debug:
            print(f"HybridLoss ||  mse_loss: {mse_loss}, kl_loss: {kl_loss}")
        
        total_loss = self.mse_weight * mse_loss + self.kl_weight * kl_loss
        
        return total_loss, {
            'mse': mse_loss.item(),
            'kl': kl_loss.item(),
            'total': total_loss.item()
        }


def temperature_softmax(logits, temperature=1.0):
    return torch.softmax(logits / temperature, dim=-1)

def pairwise_rank_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Vectorized batch version:
    - Penalize mass on zero-target clusters (push probs on negatives -> 0)
    - Enforce descending order among non-zero targets to match target order
    - Enforce positives outrank negatives (with all negatives, no sampling)

    logits, y: [B, K]
    """
    B, K = logits.shape
    device = logits.device

    probs = torch.softmax(logits, dim=1)     # [B,K]
    pos_mask = (y > 0)                       # [B,K]
    neg_mask = ~pos_mask                     # [B,K]

    # 1) Negative mass penalty
    neg_mass = (probs * neg_mask.float()).sum(dim=1).mean()

    # Select positive candidates (all positives, assume K small ~100)
    P = K
    y_pos_only = y.masked_fill(~pos_mask, float("-inf"))   # exclude non-positives
    pos_idx = torch.topk(y_pos_only, k=P, dim=1).indices   # [B,P]
    pos_valid = pos_mask.gather(1, pos_idx).float()        # [B,P], which gathered are truly positive

    logits_pos = logits.gather(1, pos_idx)  # [B,P]
    y_pos_vals = y.gather(1, pos_idx)       # [B,P]

    # 2) Positive-positive ranking: match target order
    pos_order = torch.argsort(y_pos_vals, dim=1, descending=True)  # [B,P]
    logits_pos_sorted = torch.gather(logits_pos, 1, pos_order)     # [B,P]
    valid_sorted = torch.gather(pos_valid, 1, pos_order)           # [B,P]

    diffs_pp = logits_pos_sorted.unsqueeze(2) - logits_pos_sorted.unsqueeze(1)  # [B,P,P]
    tri = torch.triu(torch.ones((P, P), dtype=torch.bool, device=device), diagonal=1)  # [P,P]
    pair_valid = (valid_sorted.unsqueeze(2) * valid_sorted.unsqueeze(1)).bool()       # [B,P,P]
    mask_pp = tri.unsqueeze(0) & pair_valid
    pos_rank_terms = torch.nn.functional.softplus(-diffs_pp)
    pos_rank_loss = (pos_rank_terms * mask_pp.float()).sum() / mask_pp.float().sum().clamp_min(1.0)

    # 3) Positive-negative ranking with all negatives (no sampling)
    diffs_pn = logits_pos.unsqueeze(2) - logits.unsqueeze(1)  # [B, P, K]
    mask_pn = pos_valid.unsqueeze(2) * neg_mask.float().unsqueeze(1)  # [B, P, K]
    pos_neg_terms = torch.nn.functional.softplus(-diffs_pn)
    pos_neg_loss = (pos_neg_terms * mask_pn).sum() / mask_pn.sum().clamp_min(1.0)

    return neg_mass + pos_rank_loss + pos_neg_loss

def recall_focused_loss(logits: torch.Tensor, y: torch.Tensor, topk: int = 22) -> torch.Tensor:
    """
    专门针对 recall 优化的损失函数（基于 logits）- 优化版本。
    - 目标：提高预测 top-k 中真实 top-k 的召回比例。
    - 方法：对真实 top-k 中缺失的类别（missing）与预测 top-k 但不在真实 top-k 的类别（wrong）构造对比损失，
        鼓励 missing 的 logit 高于 wrong 的 logit。
    注意：
    - 优化版本使用向量化操作，避免Python循环和集合操作
    - 该损失通过对选中的 logits 计算 softplus(margin - (missing - wrong))，可反向传递到 logits。
    - topk 的索引选择是离散的（不可导），属于常见的排序近似策略。
    """
    B, K = logits.shape
    device = logits.device

    if topk >= K or topk <= 0:
        topk = K - 1
    
    # 获取真实top-k和预测top-k类别
    y_topk_vals, y_topk = torch.topk(y, k=topk, dim=1)  # [B, topk]
    pred_topk_vals, pred_topk = torch.topk(logits, k=topk, dim=1)  # [B, topk]
    
    # 创建掩码来标识真实top-k和预测top-k
    true_mask = torch.zeros_like(y, dtype=torch.bool)  # [B, K]
    pred_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, K]
    
    # 使用scatter来设置掩码
    true_mask.scatter_(1, y_topk, True)
    pred_mask.scatter_(1, pred_topk, True)
    
    # 计算缺失（在真实top-k中但不在预测top-k中）和错误（在预测top-k中但不在真实top-k中）
    missing_mask = true_mask & (~pred_mask)  # [B, K]
    wrong_mask = pred_mask & (~true_mask)    # [B, K]
    
    # 获取缺失和错误预测的logits
    missing_logits = logits * missing_mask.float()  # [B, K]
    wrong_logits = logits * wrong_mask.float()       # [B, K]
    
    # 对每个batch计算对比损失
    recall_loss = torch.tensor(0.0, device=device, dtype=logits.dtype)
    margin = 1.0
    
    # 向量化计算：对所有缺失vs错误的组合计算损失
    for b in range(B):
        missing_indices = missing_mask[b].nonzero(as_tuple=True)[0]  # 缺失类别的索引
        wrong_indices = wrong_mask[b].nonzero(as_tuple=True)[0]     # 错误类别的索引
        
        if len(missing_indices) > 0 and len(wrong_indices) > 0:
            # 广播计算：missing[i] - wrong[j] for all i,j pairs
            missing_vals = logits[b, missing_indices].unsqueeze(1)  # [n_missing, 1]
            wrong_vals = logits[b, wrong_indices].unsqueeze(0)      # [1, n_wrong]
            
            # 计算所有对的损失
            pairwise_diff = missing_vals - wrong_vals  # [n_missing, n_wrong]
            pairwise_loss = torch.nn.functional.softplus(margin - pairwise_diff)
            recall_loss = recall_loss + pairwise_loss.sum()
    
    return recall_loss / B


def recall_focused_loss_batched(
    logits: torch.Tensor,
    y: torch.Tensor,
    topk: int = 22,
    margin: float = 1.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    批处理（向量化）版本的 recall_focused_loss（基于 logits）。
    - 对每个样本：找出真实 top-k 与 预测 top-k 的差集，构造 missing vs wrong 的成对比较：
        loss = softplus(margin - (logits_missing - logits_wrong))
    - 完全向量化实现，避免 Python for 循环。

    参数:
    - logits: [B, K] 预测的原始分数（未 softmax）
    - y: [B, K] 目标分布或分数（用于获取真实 top-k）
    - topk: 取前 k 个类别
    - margin: softplus 的间隔超参
    - reduction: 'mean' | 'sum' | 'none'，对 batch 的归约方式（默认 mean）

    返回:
    - 一个标量（mean/sum）或 [B]（none）
    """
    B, K = logits.shape
    device = logits.device

    if topk >= K or topk <= 0:
        topk = K - 1

    # 真实 top-k 和 预测 top-k（基于 logits）
    _, y_topk = torch.topk(y, k=topk, dim=1)          # [B, topk]
    _, pred_topk = torch.topk(logits, k=topk, dim=1)  # [B, topk]

    true_mask = torch.zeros_like(y, dtype=torch.bool)       # [B, K]
    pred_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, K]
    true_mask.scatter_(1, y_topk, True)
    pred_mask.scatter_(1, pred_topk, True)

    missing_mask = true_mask & (~pred_mask)  # [B, K]
    wrong_mask = pred_mask & (~true_mask)    # [B, K]

    # 构造成对比较的掩码：i 为 missing，j 为 wrong
    pair_mask = missing_mask.unsqueeze(2) & wrong_mask.unsqueeze(1)  # [B, K, K]

    # 所有 (i, j) 的分数差：logits_i - logits_j
    diffs = logits.unsqueeze(2) - logits.unsqueeze(1)  # [B, K, K]

    # 仅保留 missing vs wrong 的组合
    pairwise_loss = torch.nn.functional.softplus(margin - diffs) * pair_mask.float()  # [B, K, K]

    # 每个样本的总损失（与原函数一致，不按对数归一）
    loss_per_sample = pairwise_loss.sum(dim=(1, 2))  # [B]

    if reduction == "none":
        return loss_per_sample
    elif reduction == "sum":
        return loss_per_sample.sum()
    else:  # mean
        return loss_per_sample.mean()


def recall_focused_loss_v1(logits: torch.Tensor, y: torch.Tensor, topk: int = 22) -> torch.Tensor:
    """
    原始版本的recall_focused_loss - 保留用于比较
    """
    B, K = logits.shape
    device = logits.device

    if topk >= K or topk <= 0:
        topk = K - 1
    
    # 获取真实top-k类别
    y_topk = torch.topk(y, k=topk, dim=1).indices  # [B, topk]
    
    # 获取预测的top-k类别
    pred_topk = torch.topk(logits, k=topk, dim=1).indices  # [B, topk]
    
    # 计算每个样本的recall损失
    recall_loss = torch.zeros((), device=device, dtype=logits.dtype)
    margin = 1.0  # margin参数
    
    for b in range(B):
        # 计算真实top-k中不在预测top-k中的类别
        true_set = set(y_topk[b].tolist())
        pred_set = set(pred_topk[b].tolist())
        missing_in_pred = true_set - pred_set
        
        if len(missing_in_pred) > 0:
            # 对于缺失的类别，我们希望它们的logits更高
            missing_indices = list(missing_in_pred)
            missing_logits = logits[b, missing_indices]  # [len(missing)]
            
            # 对于预测top-k中的类别，我们希望它们的logits更低（如果不在真实top-k中）
            pred_only = pred_set - true_set
            if len(pred_only) > 0:
                pred_only_indices = list(pred_only)
                pred_only_logits = logits[b, pred_only_indices]  # [len(pred_only)]
                
                # 使用对比损失：缺失类别的logits应该比错误预测的logits更高
                for missing_logit in missing_logits:
                    for wrong_logit in pred_only_logits:
                        recall_loss = recall_loss + torch.nn.functional.softplus(margin - (missing_logit - wrong_logit))
    
    return recall_loss / B