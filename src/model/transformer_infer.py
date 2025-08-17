import argparse
import numpy as np
import torch
import torch.nn.functional as F

from .transformer import DistributionPredictor


def compute_cosine_similarity_matrix(query_sift: torch.Tensor, centroids: torch.Tensor) -> torch.Tensor:
    """
    计算查询向量与中心向量的余弦相似度矩阵。

    输入:
      - query_sift: 张量 [Q, D]
      - centroids:  张量 [C, D]
    输出:
      - sim:       张量 [Q, C]
    """
    query_sift = F.normalize(query_sift, p=2, dim=1)
    centroids = F.normalize(centroids, p=2, dim=1)
    return torch.matmul(query_sift, centroids.t())


def load_trained_model(
    model_path: str,
    input_dim: int,
    num_classes: int,
    hidden_dim: int = 256,
    num_layers: int = 4,
    dropout: float = 0.3,
    dual_branch_fusion: bool = True,
    device: str = "cuda",
):
    """构建模型并加载已训练权重，切换为 eval 模式。"""
    model = DistributionPredictor(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
        num_layers=num_layers,
        dropout=dropout,
        dual_branch_fusion=dual_branch_fusion,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_probabilities(
    model: torch.nn.Module,
    query_sift: torch.Tensor,
    centroids: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 128,
) -> torch.Tensor:
    """
    给定查询向量与中心向量，输出每个查询的类别概率分布。

    参数:
      - model:      已加载好的模型
      - query_sift: [Q, D]
      - centroids:  [C, D]
    返回:
      - probs:      [Q, C]
    """
    model.eval()
    query_sift = query_sift.to(device)
    centroids = centroids.to(device)

    # 预先计算所有查询与所有中心的相似度，避免重复计算
    all_sim = compute_cosine_similarity_matrix(query_sift, centroids)

    probs_chunks = []
    with torch.no_grad():
        total = query_sift.size(0)
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            batch_sift = query_sift[start:end]
            batch_sim = all_sim[start:end]
            logits = model(batch_sift, batch_sim)
            batch_probs = F.softmax(logits, dim=-1)
            probs_chunks.append(batch_probs)

    return torch.cat(probs_chunks, dim=0)


def main():
    parser = argparse.ArgumentParser(description="Transformer 推理：加载已训练模型，对每个 query 预测概率")
    parser.add_argument("--query_path", type=str, required=True, help="query SIFT 向量文件路径 (空格分隔)")
    parser.add_argument("--centroids_path", type=str, required=True, help="centroids 向量文件路径 (空格分隔)")
    parser.add_argument("--model_path", type=str, required=True, help="已训练模型 .pth 路径")
    parser.add_argument("--output_path", type=str, default="pred_probs.txt", help="输出概率文件路径")

    parser.add_argument("--hidden_dim", type=int, default=256, help="模型 hidden_dim (需与训练时一致)")
    parser.add_argument("--num_layers", type=int, default=4, help="Transformer 编码层数 (需与训练时一致)")
    parser.add_argument("--dropout", type=float, default=0.3, help="dropout (需与训练时一致)")
    parser.add_argument("--dual_branch_fusion", type=bool, default=True, help="是否启用双分支融合 (需与训练时一致)")

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"], help="推理设备")
    parser.add_argument("--infer_batch_size", type=int, default=1024, help="推理 batch size")

    args = parser.parse_args()

    # 读取 query 与 centroids，按空格分隔
    query_array = np.loadtxt(args.query_path, dtype=np.float32)
    centroids_array = np.loadtxt(args.centroids_path, dtype=np.float32)

    # 保证二维
    if query_array.ndim == 1:
        query_array = np.expand_dims(query_array, axis=0)

    query_sift = torch.from_numpy(query_array)
    centroids = torch.from_numpy(centroids_array)

    if centroids.ndim != 2 or query_sift.ndim != 2:
        raise ValueError("query_sift 与 centroids 必须为二维矩阵 [Q,D] 与 [C,D]")
    if query_sift.shape[1] != centroids.shape[1]:
        raise ValueError(
            f"查询维度({query_sift.shape[1]}) 与 中心维度({centroids.shape[1]}) 不一致"
        )

    num_classes = centroids.shape[0]
    input_dim = query_sift.shape[1] + num_classes

    # 加载模型
    model = load_trained_model(
        model_path=args.model_path,
        input_dim=input_dim,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        dual_branch_fusion=args.dual_branch_fusion,
        device=args.device,
    )

    # 推理
    probs = predict_probabilities(
        model=model,
        query_sift=query_sift,
        centroids=centroids,
        device=args.device,
        batch_size=args.infer_batch_size,
    )

    # 保存到文本：每行一个 query 的概率分布
    np.savetxt(args.output_path, probs.cpu().numpy(), fmt="%.6f")
    print(f"✅ 推理完成，概率已保存到 {args.output_path}")


if __name__ == "__main__":
    main()



"""
conda activate elpis_torch
cd /home/xln/PycharmProjects/PredictLeafNode
python -m src.model.transformer_infer \
  --query_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/sift_query.txt \
  --centroids_path  /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/leafsize10K/leaf_center.txt \
  --model_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/model_hd512_nl4_lp_proportional_weight.pth \
  --output_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/sift1M_learn/model_hd512_nl4_lp_proportional_weight_pred_probs.txt \
  --hidden_dim 512 \
  --num_layers 4 \
  --dropout 0.3 \
  --dual_branch_fusion True \
  --device cuda \
  --infer_batch_size 128

"""
