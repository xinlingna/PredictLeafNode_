
<!-- sift1M -->
cd ~/PycharmProjects/PredictLeafNode
conda activate elpis_torch
python -m src.model.transformer \
  --query_path input/Training_data/sift1M/sift_query.txt \
  --centroids_path input/Training_data/sift1M/leafsize10000/leaf_centroids.txt \
  --label_path input/Training_data/sift1M/leafsize10000/knn_distributions.txt \
  --batch_size 64 \
  --val_ratio 0.2 \
  --num_epochs 100 \
  --lr 0.001 \
  --device cuda \
  --hidden_dim 256 \
  --num_layers 4\
  --dropout 0.3\
  --dual_branch_fusion True\
  --label_process proportional_weight

<!-- sift-merged -->
cd ~/PycharmProjects/PredictLeafNode
python -m src.model.transformer \
  --query_path input/Training_data/sift1M_merged/sift_merged_query.txt \
  --centroids_path input/Training_data/sift1M_merged/leafsize10K/leaf_center.txt \
  --label_path input/Training_data/sift1M_merged/leafsize10K/knn_distributions.txt \
  --batch_size 256 \
  --val_ratio 0.2 \
  --num_epochs 100 \
  --lr 0.001 \
  --device cuda \
  --hidden_dim 512 \
  --num_layers 4\
  --dropout 0.3\
  --dual_branch_fusion True\
  --label_process proportional_weight


//////////////////////////////////////2025.8.16
cd ~/PycharmProjects/PredictLeafNode
python -m src.model.transformer \
  --query_path /home/xln/PycharmProjects/PredictLeafNode//input/Training_data/gist1M_learn/gist_learn.txt \
  --centroids_path /home/xln/PycharmProjects/PredictLeafNode//input/Training_data/gist1M_learn/leafsize20k/leaf_center.txt \
  --label_path /home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M_learn/leafsize20k/knn_distributions.txt \
  --batch_size 1024 \
  --val_ratio 0.2 \
  --num_epochs 150 \
  --lr 0.001 \
  --device cuda \
  --hidden_dim 512 \
  --num_layers 4\
  --dropout 0.3\
  --dual_branch_fusion True\
  --label_process proportional_weight



python -m src.model.transformer \
  --query_path input/Training_data/sift1M_merged/sift_merged_query.txt \
  --centroids_path input/Training_data/sift1M_merged/leafsize10K/leaf_center.txt \
  --label_path input/Training_data/sift1M_merged/leafsize10K/knn_distributions.txt \
  --batch_size 256 \
  --val_ratio 0.15 \
  --num_epochs 150 \
  --lr 0.0003 \
  --device cuda \
  --hidden_dim 512 \
  --num_layers 4 \
  --dropout 0.2 \
  --dual_branch_fusion True \
  --label_process proportional_weight \
  --early_stop_patience 25


<!-- siftsmall-merged -->
cd ~/PycharmProjects/PredictLeafNode
python -m src.model.transformer \
  --query_path input/Training_data/siftsmall_merged/siftsmall_merged_query.txt \
  --centroids_path input/Training_data/siftsmall_merged/leafsize1K/leaf_center.txt \
  --label_path input/Training_data/siftsmall_merged/leafsize1K/knn_distributions.txt \
  --batch_size 64 \
  --val_ratio 0.2 \
  --num_epochs 100 \
  --lr 0.001 \
  --device cuda \
  --hidden_dim 256 \
  --num_layers 4\
  --dropout 0.3\
  --dual_branch_fusion True\
  --label_process proportional

<!-- gist-merged -->
cd ~/PycharmProjects/PredictLeafNode
python -m src.model.transformer \
  --query_path input/Training_data/gist1M_merged/gist_merged_query.txt \
  --centroids_path input/Training_data/gist1M_merged/leafsize10K/leaf_center.txt \
  --label_path input/Training_data/gist1M_merged/leafsize10K/knn_distributions.txt \
  --batch_size 64 \
  --val_ratio 0.2 \
  --num_epochs 100 \
  --lr 0.001 \
  --device cuda \
  --hidden_dim 256 \
  --num_layers 4\
  --dropout 0.3\
  --dual_branch_fusion True\
  --label_process proportional