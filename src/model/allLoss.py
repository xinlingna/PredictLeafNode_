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
            print(f"pairwise_diff: {pairwise_diff}")
            pairwise_loss = torch.nn.functional.softplus(margin - pairwise_diff) # 计算所有对的损失, 希望缺少和错误logit的差值大于margin
            recall_loss = recall_loss + pairwise_loss.sum()
    
    return recall_loss / B


def recall_focused_loss_batched(
    logits: torch.Tensor,
    y: torch.Tensor,
    topk: int = 22,
    margin: float = 0.5,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    批处理（向量化）版本的 recall_focused_loss（基于 logits）。
    - 对每个样本：找出真实 top-k 与 预测 top-k 的差集，构造 missing vs wrong 的成对比较：
        loss = softplus(margin - (logits_missing - logits_wrong))
    - 完全向量化实现，避免 Python for 循环。
    - 注意：这个函数接受 logits 作为输入，确保梯度能正确传播到模型参数。

    参数:
    - logits: [B, K] 预测的原始分数（未 softmax），梯度会通过这个张量传播
    - y: [B, K] 目标分布或分数（用于获取真实 top-k）
    - topk: 取前 k 个类别
    - margin: softplus 的间隔超参
    - reduction: 'mean' | 'sum' | 'none'，对 batch 的归约方式（默认 mean）

    返回:
    - 一个标量（mean/sum）或 [B]（none），梯度可以回传到 logits
    """
    B, K = logits.shape
    device = logits.device

    if topk >= K or topk <= 0:
        topk = K - 1

    # 真实 top-k（基于目标分布 y）和预测 top-k（基于 logits）
    _, y_topk = torch.topk(y, k=topk, dim=1)          # [B, topk]
    _, pred_topk = torch.topk(logits, k=topk, dim=1)  # [B, topk] - 这里梯度会流向 logits

    true_mask = torch.zeros_like(y, dtype=torch.bool)       # [B, K]
    pred_mask = torch.zeros_like(logits, dtype=torch.bool)  # [B, K]
    true_mask.scatter_(1, y_topk, True)
    pred_mask.scatter_(1, pred_topk, True)

    missing_mask = true_mask & (~pred_mask)  # [B, K] - 真实top-k中但预测top-k中没有的
    wrong_mask = pred_mask & (~true_mask)    # [B, K] - 预测top-k中但真实top-k中没有的

    # 构造成对比较的掩码：i 为 missing，j 为 wrong
    pair_mask = missing_mask.unsqueeze(2) & wrong_mask.unsqueeze(1)  # [B, K, K]

    # 所有 (i, j) 的分数差：logits_i - logits_j
    diffs = logits.unsqueeze(2) - logits.unsqueeze(1)  # [B, K, K] - 梯度从这里传播
    # print(f"diffs: {diffs}")

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


#==============================================================================================================
import torch.nn.functional as F
from typing import Optional

# ========= Ranking-Optimized Losses (diff-safe) =========
def _safe_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # 完全可微（对 x），用于生成概率/权重；若全 0 行，仍会产生有定义的输出（但注意这种行通常应在上游避免）
    s = x.sum(dim=-1, keepdim=True)
    return x / (s + eps)



# 函数的本质：对logits做softmax，然后计算logits和targets的交叉熵
# 交叉熵 = 熵 + KL散度
def listnet_top1_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pred_temp: float = 1.0,
    tgt_temp: Optional[float] = None,
    assume_targets_prob: bool = True,
    eps: float = 1e-12,
) -> torch.Tensor:
    """
    ListNet (Top-1)：
    - 对 logits 做 softmax 得到 P_pred（可微）
    - 目标分布 P_tgt 可由 targets 归一化或 softmax 得到（不需要对 targets 求梯度）
    - 损失使用 (P_pred + eps).log()，避免 clamp 对梯度的截断
    """
    # 预测分布（对 logits 完全可微）
    P_pred = F.softmax(logits / pred_temp, dim=-1)

    # 目标分布（默认认为 targets 近似概率；否则用 softmax 得到概率）
    if assume_targets_prob and tgt_temp is None:
        P_tgt = _safe_normalize_rows(targets, eps=eps)
    else:
        tt = 1.0 if tgt_temp is None else tgt_temp
        P_tgt = F.softmax(targets / tt, dim=-1)

    loss = -(P_tgt * torch.log(P_pred + eps)).sum(dim=-1).mean()
    return loss  # 对 logits 的梯度可通过 softmax 与 log 链式传递


# Plackett–Luce 模型 定义排序概率
def listmle_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    use_topm: Optional[int] = None,
) -> torch.Tensor:
    """
    ListMLE：
    - 用 targets 排序得到置换 π（与 logits 无关，不阻断对 logits 的梯度）
    - 再对 logits 进行 gather 与 logsumexp（完全可微）
    - 支持 use_topm 仅用前 m 个目标，降算量
    """
    # 目标排序（对 logits 无梯度依赖）
    _, pi = torch.sort(targets, dim=-1, descending=True)  # [B, K]
    if use_topm is not None:
        pi = pi[:, :use_topm]  # [B, m]

    # 对 logits gather 后参与后续可微运算
    s_sorted = torch.gather(logits, dim=1, index=pi)  # [B, m or K]
    B, M = s_sorted.shape

    # 从右到左累计 logsumexp（可微）
    # lse_suffix[:, j] = logsumexp(s_sorted[:, j:])
    lse_suffix = []
    running = None
    for j in range(M - 1, -1, -1):
        curr = s_sorted[:, j:j+1]
        running = curr if running is None else torch.logsumexp(
            torch.cat([curr, running], dim=1), dim=1, keepdim=True)
        lse_suffix.append(running)
    lse_suffix = torch.cat(lse_suffix[::-1], dim=1)  # [B, M]

    loss = -(s_sorted - lse_suffix).sum(dim=1).mean() # 逼迫模型打分函数让真实排序概率尽可能高
    return loss  # 对 logits 的梯度通过 gather->logsumexp 路径正常回传

def pairwise_hinge_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_pos: int = 1,
    num_neg: int = 20,
    margin: float = 0.1,
) -> torch.Tensor:
    """
    Pairwise Hinge：
    - 正样本取 targets 的 top-(num_pos)（与 logits 无关）
    - 负样本从非正集合按 (1 - targets) 采样（与 logits 无关）
    - 对 logits 的梯度来自 gather 的分数差 diff 与 ReLU
    - 当 num_pos >= K（无负样本可采）时，返回 0*logits.sum()，保持计算图连通且梯度为 0
    """
    B, K = logits.shape
    P = min(num_pos, K)

    if P >= K:
        # 没有负样本可用：返回与 logits 相连通的 0，避免图断裂
        return logits.sum() * 0.0

    # 正样本索引
    _, pos_idx = torch.topk(targets, k=P, dim=-1)  # [B, P]

    # 构造负样本候选的权重（不依赖 logits），并做归一化（数值安全）
    mask = torch.ones(B, K, dtype=torch.bool, device=logits.device) # 构建一个元素值全是True的矩阵，形状为[B,K]
    mask.scatter_(1, pos_idx, False) # 将pos_idx位置的元素设置为False，其他位置保持True
    neg_weights = (1.0 - targets).clamp_min(0.0) * mask # 将targets中大于0的元素设置为0，其他位置保持1.0-targets
    neg_weights = _safe_normalize_rows(neg_weights)  # 若某行全 0，这里会均匀（不过通常上游应避免）

    # 负样本采样（不影响对 logits 的梯度）
    N = min(num_neg, max(1, K - P))
    neg_idx = torch.multinomial(neg_weights, num_samples=N, replacement=True)  # [B, N]

    # 从 logits 中 gather（可微）
    s_pos = torch.gather(logits, 1, pos_idx)  # [B, P]
    s_neg = torch.gather(logits, 1, neg_idx)  # [B, N]

    # pairwise margin：ReLU(margin - (s_p - s_n)) （对 logits 可微，分段线性）
    diff = s_pos.unsqueeze(2) - s_neg.unsqueeze(1)  # 构造正负对 [B, P, N]
    loss = F.relu(margin - diff).mean()
    return loss
# ========= /Ranking-Optimized Losses =========


import torch
import torch.nn.functional as F

def _safe_normalize_rows(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # 对每一行做归一化；若行和为 0，则回退为均匀分布（避免除零）
    row_sum = x.sum(dim=1, keepdim=True)
    zero_row = row_sum <= eps
    # 均匀分布（仅在零行处启用）
    uniform = torch.full_like(x, 1.0 / x.size(1))
    normed = x / row_sum.clamp_min(eps)
    return torch.where(zero_row, uniform, normed)

def pairwise_hinge_loss_strict_pos(
    logits: torch.Tensor,
    targets: torch.Tensor,
    num_pos: int = 1,     # 每行从正样本集合中最多取多少个用于配对（>0 才能被选）
    num_neg: int = 20,    # 每行从负样本集合中最多采多少个用于配对（==0 才能被选）
    margin: float = 0.1,
) -> torch.Tensor:
    """
    修改要点：
    - 正样本：严格定义为 targets > 0
    - 负样本：严格定义为 targets == 0
    - 正样本选择不依赖 logits；从正集合中选至多 num_pos 个（按 targets 值挑大的）
    - 负样本只从 neg_mask 中采样；权重固定为 1（也可按需要改成别的先验）
    - 仅对“有效行”（同时有正有负）计算 pairwise hinge；其他行贡献为 0
    - 使用 pair_mask 只对有效的正-负对计算 ReLU(margin - (s_p - s_n)) 并做平均
    """
    assert logits.shape == targets.shape, "logits 和 targets 形状需一致 [B, K]"
    B, K = logits.shape

    device = logits.device
    pos_mask = (targets > 0)
    neg_mask = (targets == 0)

    # 仅保留同时有正且有负的行；若全无有效行，返回与 logits 相连通的 0
    valid_rows = pos_mask.any(dim=1) & neg_mask.any(dim=1)
    if valid_rows.sum() == 0:
        return logits.sum() * 0.0

    # 过滤到有效行
    logits_v  = logits[valid_rows]          # [Bv, K]
    targets_v = targets[valid_rows]         # [Bv, K]
    pos_mask_v = pos_mask[valid_rows]       # [Bv, K]
    neg_mask_v = neg_mask[valid_rows]       # [Bv, K]
    Bv = logits_v.size(0)

    # ---------- 选正样本索引（只在 pos_mask 内部） ----------
    # 用 -inf 屏蔽负位置后 topk；这样绝不会选到非正位置
    # 对于正样本数量不足 num_pos 的行，topk 仍会返回 K 中的若干索引，
    # 但我们稍后会用 pos_valid 掩掉无效位置，避免参与损失。
    pos_scores = torch.where(pos_mask_v, targets_v, torch.tensor(float('-inf'), device=device))
    P = min(num_pos, K)  # 统一一个上界，实际有效个数由 pos_valid 决定
    P = max(P, 1)        # 至少取 1，方便张量形状；无效位置后续会被掩蔽
    _, pos_idx = torch.topk(pos_scores, k=P, dim=1)  # [Bv, P]
    s_pos = torch.gather(logits_v, 1, pos_idx)       # [Bv, P]
    pos_valid = torch.gather(pos_mask_v, 1, pos_idx) # [Bv, P] 仅真正的正样本为 True

    # ---------- 负样本采样（严格在 neg_mask 内） ----------
    # 负样本权重：只给 neg_mask 处 1，其余 0；行归一化后用 multinomial 采样
    neg_weights = neg_mask_v.float()
    neg_weights = _safe_normalize_rows(neg_weights)
    N = min(num_neg, K)
    N = max(N, 1)
    neg_idx = torch.multinomial(neg_weights, num_samples=N, replacement=True)  # [Bv, N]
    s_neg = torch.gather(logits_v, 1, neg_idx)        # [Bv, N]
    neg_valid = torch.gather(neg_mask_v, 1, neg_idx)  # [Bv, N] 理论上全 True，但保守起见保留

    # ---------- 计算 pairwise hinge ----------
    # diff[b, i, j] = s_pos[b, i] - s_neg[b, j]
    diff = s_pos.unsqueeze(2) - s_neg.unsqueeze(1)    # [Bv, P, N]
    pair_mask = (pos_valid.unsqueeze(2) & neg_valid.unsqueeze(1))  # [Bv, P, N]
    hinge = F.relu(margin - diff)                     # [Bv, P, N]

    # 只对有效的正-负对计入损失；若极端情况下没有有效对，则返回 0*
    valid_counts = pair_mask.sum()
    if valid_counts == 0:
        return logits.sum() * 0.0

    loss = (hinge * pair_mask.float()).sum() / valid_counts.float()
    return loss

