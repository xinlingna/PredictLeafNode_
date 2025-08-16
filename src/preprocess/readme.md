
<!-- bulid_loaders -->
python bulid_loaders.py \
--query_path ../../input/Training_data/sift1M/sift_query.txt \
--centroids_path ../../input/Training_data/sift1M/leafsize10000/leaf_centroids.txt \
--label_path ../../input/Training_data/sift1M/leafsize10000/knn_distributions.txt \
--batch_size 64 \
--val_ratio 0.2 \
--shuffle True
