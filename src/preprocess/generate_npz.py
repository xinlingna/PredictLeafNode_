import numpy as np
from pathlib import Path
import os

# Raw query feature + similarity between query and centriod
def create(query, centroids, label, output_dir, processing_way):

    ########### calculate probability of data ############################
    # 处理label的第一种方式：加权并归一化
    if processing_way=="l2_sum":
        Top1bid_label = np.argsort(-label, axis=1)[:,0: int(0.25*centroids.shape[0])]  # 获取每个query的top-25%的聚簇编号
        label = label.astype(float) 
        
        TDeduct=5
        for i in range(query.shape[0]):
            label[i, Top1bid_label[i]] += TDeduct  # 对Top1bid_label[i]位置的聚簇加权，使得被选中的概率更大
            label[i] /= label[i].sum()   
    elif processing_way=="l2":
        label=label/ np.sum(label, axis=1, keepdims=True)
            
    elif processing_way=="softmax":
        # 处理label的第二种方式：softmax
        label = np.exp(label)
        label = label / np.sum(label, axis=1, keepdims=True)

    #########  Query Centriod Distance ############################
    RawdataDistance = np.zeros((query.shape[0], query.shape[1]+centroids.shape[0]), dtype=float)
    dist = np.zeros((query.shape[0], centroids.shape[0]), dtype=float)
    sim  = np.zeros((query.shape[0], centroids.shape[0]), dtype=float)
    for i in range (query.shape[0]):
        dist[i, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((centroids.shape[0], 1)), [query[i, :]]) - centroids), axis=-1))
        maxdist = np.max(dist[i])
        sim_score = ( maxdist-dist[i])/ maxdist
        SumSim = np.sum(sim_score)
        sim[i] = sim_score/SumSim                       # similarity feature between query and centriod
        RawdataDistance[i,0:query.shape[1]] = query[i] # query raw data
        RawdataDistance[i,query.shape[1]:RawdataDistance.shape[1]] = sim[i] # similarity feature between query and centriod

    idx = np.argsort(dist) # 返回query到质心的距离下标，按照顺序排列

    # fn:保存的文件位置
    # prob[i]： query[i]的top-100的聚簇分布
    # dist[i]: query[i]与所有聚簇质心的距离
    # idx[i,j]: 第 i 个 query 到所有质心中，第 j 近的是哪个质心的编号
    # RawdataDistance[i] : concatation query[i](D) and similarity(M)
    # np.savez_compressed(fn,  label = label, dist = dist, EU_idx = idx, sim = sim, RawdataDistance = RawdataDistance)
    output_file = os.path.join(output_dir, f"output_training_file_{processing_way}.npz")
    np.savez_compressed(output_file, label = label, EU_dist = dist, EU_idx = idx, query=query,sim = sim, RawdataDistance = RawdataDistance)

    return 0
  
def CreateTraining(centroids_file, query_file, label_file,processing_way):
    # load codebook
    centroids = np.loadtxt(centroids_file, dtype=float)
    centroids /= np.max(centroids) 
    
    #Training Data
    query_sift = np.loadtxt(query_file, dtype=float)
    query_sift /= np.max(query_sift) 


    label = np.loadtxt(label_file, dtype=int) # top-100分布的聚簇的编号

    output_dir = os.path.dirname(centroids_file)
    create(query_sift, centroids, label, output_dir,processing_way)
    print("创建完成：{}",output_dir)

def CreateTesting(centroids_file, query_file, label_file, processing_way):
    # # load codebook
    centroids = np.loadtxt(centroids_file, dtype=float)
    centroids /= np.max(centroids) 
    # centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    
    #Testing Data
    # query_sift = np.loadtxt(query_file, dtype=float)
    query_sift /= np.max(query_sift) 
    query_sift = query_sift / np.linalg.norm(query_sift, axis=1, keepdims=True)
    
    label = np.loadtxt(label_file, dtype=int)     
    
    output_dir = os.path.dirname(centroids_file)
    create(query_sift, centroids, label, output_dir,processing_way)


# base_dir = Path(__file__).resolve().parent.parent.parent
def main():
    
    # Training
    centroids_file = '../../input/Training_data/sift1M/leafsize10000/leaf_centroids.txt'
    query_file = '../../input/Training_data/sift1M/sift_query.txt'
    label_file = '../../input/Training_data/sift1M/leafsize10000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
    
    
    # Training
    centroids_file = '../../input/Training_data/sift1M/leafsize5000/leaf_centroids.txt'
    query_file = '../../input/Training_data/sift1M/sift_query.txt'
    label_file = '../../input/Training_data/sift1M/leafsize5000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
    
    
    # Training
    centroids_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/leafsize5000/leaf_centroids.txt'
    query_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/gist_query.txt'
    label_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/leafsize5000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
    
    # Training
    centroids_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/leafsize10000/leaf_centroids.txt'
    query_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/gist_query.txt'
    label_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/gist1M/leafsize10000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
    

    # Training
    centroids_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/leafsize1000/leaf_centroids.txt'
    query_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/siftsmall_query.txt'
    label_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/leafsize1000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
                   
    
    # Training
    centroids_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/leafsize2000/leaf_centroids.txt'
    query_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/siftsmall_query.txt'
    label_file = '/home/xln/PycharmProjects/PredictLeafNode/input/Training_data/siftsmall/leafsize2000/knn_distributions.txt'
    CreateTraining(centroids_file, query_file, label_file,processing_way="softmax")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2")
    CreateTraining(centroids_file, query_file, label_file,processing_way="l2_sum")
    
if __name__ == "__main__":
    main()


