import numpy as np

# 基于欧氏距离计算聚簇的相似度后归一化成概率分布，并对概率进行修正
#  codebook：一级聚簇
def create(query, centroids, ID,CID, fn): 
    
    dist = np.zeros((query.shape[0], centroids.shape[0]), dtype=float)
    sim  = np.zeros((query.shape[0], centroids.shape[0]), dtype=float)
    for i in range (query.shape[0]):
        dist[i, :] = np.sqrt(np.sum(np.square(np.dot(np.ones((centroids.shape[0], 1)), [query[i, :]]) - centroids), axis=-1))
        maxdist = np.max(dist[i])
        sim_score = ( maxdist-dist[i])/ maxdist
        SumSim = np.sum(sim_score)
        sim[i] = sim_score/SumSim
    
    sim2  = np.zeros((query.shape[0], centroids.shape[0]), dtype=float)
    for i in range (query.shape[0]):
        TDeduct = 0
        THD0 = 0
        for j in range (centroids.shape[0]):
            sim2[i,j] = sim[i,j]         
            TDeduct += sim[i,j] * 0.07
            THD0 += sim[i,j] * 0.01
     
        sim2[i,ID[i]] += TDeduct  # 对某些位置的相似度加权，使得被选中的概率更大
        sim2[i,CID[i]] += THD0  # sim2:batch_size * FirstCodebookNum
    
    idx = np.argsort(-sim2)
    idxTopk = idx[:,0:0.25*centroids.shape[0]]      # Top-25%的下标

    np.savez_compressed(fn,  prob = sim, idxTopk = idxTopk)
    # np.savez_compressed(fn,  prob = sim2, idxTopk = idxTopk)

    return 0


# prob_train.npz:  ['prob', 'idxTop100']
def CreateTraining():
    Codebook = np.loadtxt('./input/FirstStageCodebook.txt', dtype=float)
    Query = np.loadtxt('./input/Train_data/Train.txt', dtype=float)
    ID = np.loadtxt('./input/Train_data/FirstStageGT_BID_Train.txt', dtype=int) # top-100分布的聚簇的编号，可能有重复
    ID = ID[:,0]
    CID = np.loadtxt('./input/Sample/SortedTrain_BID_First_Level.txt', dtype=int)
    CID = CID[:,0]    
    fn = './output/prob_train'
    create(Query, Codebook, ID,CID, fn)    


# prob_test.npz:  ['prob', 'idxTop100']
# prob: (10000, 4096)
# idxTop100: (10000, 100)
def CreateTesting():    
    Codebook = np.loadtxt('./input/FirstStageCodebook.txt', dtype=float)
    Query = np.loadtxt('./input/Test_data/Test.txt', dtype=float)
    ID = np.loadtxt('./input/Test_data/FirstStageGT_BID_Test.txt', dtype=int)
    ID = ID[:,0]
    CID = np.loadtxt('./input/Sample/SortedTrain_BID_Second_Level.txt', dtype=int)
    CID = CID[:,0]    
    fn = './output/prob_test'

    # Codebook: first stage codebook
    create(Query, Codebook, ID,CID, fn)    

