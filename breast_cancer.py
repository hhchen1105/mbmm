from sklearn import cluster, datasets, mixture
import numpy as np
import pandas as pd
import pickle
import os

import MBMM
from MBMM import MBMM

import random
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from scipy.optimize import linear_sum_assignment
from sklearn import metrics
import warnings

## Calculate accuracy reference from: https://github.com/sharmaroshan/MNIST-Using-K-means  
def infer_cluster_labels(pred_labels, actual_labels):
    """
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    """
    
    inferred_labels = {}
    n_clusters = len(set(actual_labels))
    #n_clusters = len(np.unique(pred_labels))
    for i in range(n_clusters):

        # find index of points in cluster
        labels = []
        index = np.where(pred_labels == i)

        # append actual labels for each point in cluster
        labels.append(actual_labels[index])

        # determine most common label
        if len(labels[0]) == 1:
            counts = np.bincount(labels[0])
        else:
            counts = np.bincount(np.squeeze(labels))

        # assign the cluster to a value in the inferred_labels dictionary
        if len(counts) > 0:        
            if np.argmax(counts) in inferred_labels:
                # append the new number to the existing array at this slot
                inferred_labels[np.argmax(counts)].append(i)
            else:
                # create a new array in this slot
                inferred_labels[np.argmax(counts)] = [i]    

    return inferred_labels  
    
    
def infer_data_labels(X_labels, cluster_labels):
    """
    Determines label for each array, depending on the cluster it has been assigned to.
    returns: predicted labels for each array
    """
    predicted_labels = np.array([-1 for i in range(len(X_labels))])
    
    for i, cluster in enumerate(X_labels):
        for key, value in cluster_labels.items():
            if cluster in value:
                predicted_labels[i] = key
                
    return predicted_labels

def load_data():
    my_dataset = datasets.load_breast_cancer()
    data = my_dataset.data
    target = my_dataset.target

    data_2d = pd.read_csv('data/breast_cancer_2d.csv').to_numpy() 
    #data_2d = np.load('breast_cancer_autoencoder.npy')
    lower, upper = 0.01, 0.99
    data = lower + (data - np.min(data))*(upper-lower)/(np.max(data)-np.min(data))
    #data_2d = lower + (data_2d - np.min(data_2d))*(upper-lower)/(np.max(data_2d)-np.min(data_2d))
    return data, data_2d, target

def initial_param(data, data_2d):
    # ============
    # Initial parameters
    # ============
    MBMM_param = np.array([[  18.4304407 ,   20.43840706,   40.27655399,  177.95326911,
                  14.4969193 ,   14.49508325,   14.4850757 ,   14.47619695,
                  14.52384832,   14.48736256,   14.56286897,   14.87205442,
                  15.14051679,   21.48343479,   14.46831484,   14.47429577,
                  14.47556931,   14.46945111,   14.47341827,   14.46859549,
                  18.85124516,   22.29473345,   43.41399334,  218.77739231,
                  14.50803049,   14.53328519,   14.52710944,   14.49421237,
                  14.55657632,   14.49233357, 1383.32450673],
               [   3.68966275,    3.83704773,    9.19853026,   74.59825957,
                   2.73010962,    2.73228991,    2.73335744,    2.72959039,
                   2.7348392 ,    2.72796936,    2.75841463,    2.78583864,
                   2.95888474,    6.47719107,    2.72518268,    2.72637963,
                   2.72693926,    2.7256005 ,    2.72590435,    2.72507588,
                   3.88906989,    4.20717636,   10.64297778,  125.70029092,
                   2.73214003,    2.7428036 ,    2.74735402,    2.73431498,
                   2.74134964,    2.72936184,  222.54686006]])

    MBMM_2d_param =  np.array([[9.83834984, 3.98573002, 8.08905886],
                             [5.64924405, 3.40372068, 4.14833321]])

    param = [(data, {'n_clusters': 2, 'quantile': .242, 'eps': .011, 'linkage': "complete", 
                     'affinity': "cosine", 'MBMM_param': MBMM_param}),
            (data_2d, {'n_clusters': 2, 'quantile': .214, 'eps': .031, 'linkage': "complete", 
                       'affinity': "cosine", 'MBMM_param': MBMM_2d_param})]

    return param


def clustering_accuracy(y, yhat):
    distinct_y = list(set(y))
    distinct_yhat = list(set(yhat))
    relabeled_y = [distinct_y.index(_) for _ in y]
    relabeled_yhat = [distinct_yhat.index(_) for _ in yhat]

    W = np.zeros((len(distinct_y), len(distinct_yhat)))
    for i, j in zip(relabeled_y, relabeled_yhat):
        W[i,j] += 1
    M = W.max() - W
    y_idx, yhat_idx = linear_sum_assignment(M)
    return W[y_idx, yhat_idx].sum()/len(y)


if __name__ == "__main__":
    data, data_2d, target = load_data()
               
    parameters = initial_param(data, data_2d)
    
    for i_dataset, (dataset, params) in enumerate(parameters):  

        kmeans = cluster.KMeans(n_clusters=params['n_clusters'])

        dbscan = cluster.DBSCAN(eps=params['eps'])
        aggolmarative = cluster.AgglomerativeClustering(
            linkage=params['linkage'],
            affinity=params['affinity'],
            n_clusters=params['n_clusters'],
        )

        gmm = mixture.GaussianMixture(n_components=params['n_clusters'])

        mbmm = MBMM(C=params['n_clusters'], n_runs=100, param=params['MBMM_param'])
        
        clustering_algorithms = [
            ('MBMM', mbmm),
            ('K-means', kmeans),
            ("DBSCAN", dbscan),
            ("AgglomerativeClustering", aggolmarative),
            ('GMM', gmm),
        ]
               
        #print result
        if i_dataset == 0:     
            print('Original data:')
        else:
             print('2-dim data:')
                
        for algo_name, algorithm in clustering_algorithms:
            filename = 'models/{}-{}-{}.pck'.format('breast_cancer', algo_name, i_dataset)
            if os.path.exists(filename):
                with open(filename, 'rb') as f:
                    algorithm = pickle.load(f)
            else:
                algorithm.fit(dataset)
                with open(filename, 'wb') as f:
                    pickle.dump(algorithm, f)

            if hasattr(algorithm, 'labels_'):
                train_predict_y = algorithm.labels_.astype(int)
            else:
                train_predict_y = algorithm.predict(dataset)        
                
            #cluster_labels = infer_cluster_labels(train_predict_y, target)
            #train_predicted_labels = infer_data_labels(train_predict_y, cluster_labels)
            #acc = np.round(np.count_nonzero(target == train_predicted_labels)/len(target), 3)
            acc = clustering_accuracy(target, train_predict_y)

            ari_value = np.round(metrics.adjusted_rand_score(target, train_predict_y), 3)

            ami_value = np.round(metrics.adjusted_mutual_info_score(target, train_predict_y), 3)
                   
            print(algo_name, {'Accuracy':acc, 'ARI':ari_value, "AMI":ami_value})
