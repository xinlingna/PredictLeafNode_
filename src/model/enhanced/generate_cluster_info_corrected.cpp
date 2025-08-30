// 修正后的聚类信息生成函数（不包含representative_vectors）
void Hercules::generate_cluster_info_files_corrected(const char* output_path) {
	if (this->num_leaf_node <= 0) {
		std::cerr << "Error: num_leaf_node is not set or is zero." << std::endl;
		return;
	}

	// Buffers indexed by mapped leaf index (leafId2Idx ordering)
	std::vector<float> sizes(this->num_leaf_node, 0.0f);  // 修改：使用float而不是int
	std::vector<float> variances(this->num_leaf_node, 0.0f);
	std::vector<float> densities(this->num_leaf_node, 0.0f);
	std::vector<float> intra_distances(this->num_leaf_node, 0.0f);

	// Iterate over leaves and compute stats, but write into mapped index
	for (int i = 0; i < this->num_leaf_node; ++i) {
		Node *leaf = this->leaves[i];
		int rec_vecs_number = leaf->file_buffer->buffered_list_size + leaf->file_buffer->disk_count;

		// Map to output index
		auto it = this->leafId2Idx.find(leaf->id);
		if (it == this->leafId2Idx.end()) {
			std::cerr << "Error: leaf->id " << leaf->id << " not found in leafId2Idx map!" << std::endl;
			return;
		}
		int out_idx = it->second;
		sizes[out_idx] = static_cast<float>(rec_vecs_number);  // 修改：转换为float

		// Load time series
		VectorWithIndex* rec = leaf->getTS(index);
		if (rec == nullptr) {
			std::cerr << "Error: Failed to get time series from leaf node." << std::endl;
			return;
		}

		// compute centroid for this leaf (use existing centroids_center if available)
		std::vector<float> centroid(this->timeseries_size, 0.0f);
		bool have_centroid = false;
		if (this->centroids_centroids != nullptr) {
			// centroids_centroids is ordered according to leafId2Idx; use out_idx
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_centroids[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		} else if (this->centroids_center != nullptr) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				centroid[d] = this->centroids_center[out_idx].ts_centroid[d];
			}
			have_centroid = true;
		}

		// If no precomputed centroid, compute mean
		if (!have_centroid) {
			for (int r = 0; r < rec_vecs_number; ++r) {
				for (int d = 0; d < this->timeseries_size; ++d) {
					centroid[d] += rec[r].ts[d];
				}
			}
			for (int d = 0; d < this->timeseries_size; ++d) centroid[d] /= std::max(1, rec_vecs_number);
		}

		// compute per-dimension variance and intra distances
		std::vector<double> var_per_dim(this->timeseries_size, 0.0);
		double intra_sum = 0.0;
		for (int r = 0; r < rec_vecs_number; ++r) {
			// distance to centroid
			double dist = 0.0;
			for (int d = 0; d < this->timeseries_size; ++d) {
				double diff = rec[r].ts[d] - centroid[d];
				var_per_dim[d] += diff * diff;
				dist += diff * diff;
			}
			intra_sum += sqrt(dist);
		}

		float mean_var = 0.0f; // Mean variance across dimensions
		if (rec_vecs_number > 1) {
			for (int d = 0; d < this->timeseries_size; ++d) {
				var_per_dim[d] /= (rec_vecs_number - 1);
				mean_var += static_cast<float>(var_per_dim[d]);
			}
			mean_var /= this->timeseries_size;
		} else {
			mean_var = 0.0f;
		}
		variances[out_idx] = mean_var;

		if (rec_vecs_number > 0) {
			intra_distances[out_idx] = static_cast<float>(intra_sum / rec_vecs_number);
		} else {
			intra_distances[out_idx] = 0.0f;
		}

		// density heuristic: size / (variance^D + eps)
		double denom = pow(std::max(1e-6f, mean_var), this->timeseries_size);
		densities[out_idx] = static_cast<float>(rec_vecs_number / (denom + 1e-9));

		// free loaded rec
		for (int j = 0; j < rec_vecs_number; j++) {
			free(rec[j].ts);
			free(rec[j].ts_index);
		}
		free(rec);
	}

	// base dir derived from index_path_txt
	std::string base_dir = this->index->index_setting->index_path_txt;
	if (base_dir.back() != '/' && base_dir.back() != '\\') base_dir += "/";

	// cluster_sizes -> float32 (修改：直接保存float)
	{
		std::ofstream ofs(base_dir + "cluster_sizes.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_sizes.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(sizes.data()), sizes.size() * sizeof(float));
		std::cout << "cluster_sizes.bin written: " << sizes.size() << " floats" << std::endl;
	}

	// variances
	{
		std::ofstream ofs(base_dir + "cluster_variances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_variances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(variances.data()), variances.size() * sizeof(float));
		std::cout << "cluster_variances.bin written: " << variances.size() << " floats" << std::endl;
	}

	// densities
	{
		std::ofstream ofs(base_dir + "cluster_densities.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open cluster_densities.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(densities.data()), densities.size() * sizeof(float));
		std::cout << "cluster_densities.bin written: " << densities.size() << " floats" << std::endl;
	}

	// intra distances
	{
		std::ofstream ofs(base_dir + "intra_distances.bin", std::ios::binary);
		if (!ofs.is_open()) { std::cerr << "cannot open intra_distances.bin" << std::endl; return; }
		ofs.write(reinterpret_cast<const char*>(intra_distances.data()), intra_distances.size() * sizeof(float));
		std::cout << "intra_distances.bin written: " << intra_distances.size() << " floats" << std::endl;
	}

	std::cout << "All cluster info files generated successfully!" << std::endl;
}
