from keras.models import load_model
import numpy as np
np.random.seed(1337)  # for reproducibility

from time import time

def Testing(test_file,test_label_file):
    start_time = time()

    BatchSize = 1000

    # load model
    model = load_model('My_Model_RawData_WeightingProb.h5')

    # y_test
    npzfile = np.load(test_label_file) # './output/prob_test.npz'
    y_test = np.array(npzfile['prob'])

    # x_test
    query = np.loadtxt(test_file, dtype=float)
    query /=  np.max(query)
    x_test = query

    # prediction
    p_test = model.predict(x_test)
    x_idx = np.argsort(-x_test)
    y_idx = np.argsort(-y_test)
    p_idx = np.argsort(-p_test)

    np.savez_compressed('./output/(te)sort_bid_RawData_weightingProb', x_idx=x_idx, y_idx=y_idx, p_idx=p_idx, p_prob = p_test)
    np.savez_compressed('./output/(te)subcluster_retrieve_prob_RawData_weightingProb', x_prob=x_test, y_prob=y_test, p_prob=p_test)
    np.savetxt('input/Train_data/Traininng_Set/p_idx_Test.txt', p_idx, delimiter=' ', fmt='%d')

    # evaluate test data under the trained model
    score = model.evaluate(x_test, y_test, batch_size=BatchSize) # loss  accuracy
    print('\n* evaluation for test data - loss: %0.4f - acc: %0.4f' %(score[0], score[1]))
