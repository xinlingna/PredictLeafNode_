import numpy as np

# Raw query feature + similarity between query and centriod


# 	统计 top-k 中各聚簇的出现频率（即出现次数越多认为越重要）
#  需要计算每个query的top-100在每个叶子节点的分布
def create(query, codebook, label,fn):


    ########### calculate probability of data ############################
    Top1bid_label = label[:,0] 
    label = np.zeros((query.shape[0], codebook.shape[0]), dtype=float)
    for i in range(query.shape[0]):
        uniset = np.unique(label[i, :])
        TDeduct = 0        
        for j in range(uniset.shape[0]):
            bid = uniset[j]
            freq = len(np.where(label[i,:] == bid)[0])
            label[i, bid] = freq  

        label[i, Top1bid_label[i]] += TDeduct  # 对top-1位置的聚簇加权，使得被选中的概率更大
            
        label[i] /= label.shape[1]           
        
    #########  Query Centriod Distance ############################
    RawdataDistance = np.zeros((query.shape[0], query.shape[1]+codebook.shape[0]), dtype=float)
    dist = np.zeros((query.shape[0], codebook.shape[0]), dtype=float)
    sim  = np.zeros((query.shape[0], codebook.shape[0]), dtype=float)
    for i in range (query.shape[0]):
        dist[i, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((codebook.shape[0], 1)), [query[i, :]]) - codebook), axis=-1))
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
    np.savez_compressed(label = label, dist = dist, EU_idx = idx, sim = sim, RawdataDistance = RawdataDistance)

    return 0
  
def CreateTraining():
    # load codebook
    Codebook = np.loadtxt('./input/FirstStageCodebook.txt', dtype=float)
    Codebook /= np.max(Codebook) 
    
    #Training Data
    query_sift = np.loadtxt('./input/Train_data/Train.txt', dtype=float)
    query_sift /= np.max(query_sift) 
    bid_label = np.loadtxt('./input/Train_data/FirstStageGT_BID_Train.txt', dtype=int) # top-100分布的聚簇的编号，可能有重复
    fn = './output/prob_dist_train_weightingProb'
    create(query_sift, Codebook, bid_label,fn)    

def CreateTesting():
    # 加载聚类的质心
    Codebook = np.loadtxt('./input/FirstStageCodebook.txt', dtype=float)
    Codebook /= np.max(Codebook) 
    
    #Testing Data
    query_sift = np.loadtxt('./input/Test_data/Test.txt', dtype=float)
    query_sift /= np.max(query_sift) 
    bid_label = np.loadtxt('./input/Test_Data/FirstStageGT_BID_Test.txt', dtype=int)      
    fn = './output/prob_dist_test_weightingProb'         
    create(query_sift, Codebook, bid_label,fn)    



