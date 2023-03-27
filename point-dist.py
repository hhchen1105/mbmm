import pickle
import os
from sklearn import datasets
import numpy as np
import scipy
import matplotlib.pyplot as plt
import time
import warnings

import imp
import MBMM
imp.reload(MBMM)
from MBMM import MBMM


np.random.seed(0)


def scale(X):
    #X = np.array(X[0])
    X = (X - np.min(X)) / (np.max(X) - np.min(X))

    for i in range(len(X[:,0])):
        if X[i][0] == 0.0:
            X[i][0] += 1e-1
        if X[i][0] == 1.0:
            X[i][0] -= 1e-1
        if X[i][1] == 0.0: 
            X[i][1] += 1e-1
        if X[i][1] == 1.0:
            X[i][1] -= 1e-1
    return X

# ============
# Generate datasets. We choose the size big enough to see the scalability
# of the algorithms, but not too big to avoid too long running times
# ============
n_samples = 500
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.3,
                                      noise=.05)

# ============
# Set up cluster parameters
# ============
#plt.figure(figsize=((9 * 2 + 3)*1.1, 13/6*7))
#plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05, hspace=.01)

plot_num = 1

default_base = {'quantile': .2,
                'eps': .05,
                'damping': .9,
                'preference': -5,
                'n_neighbors': 8,
                'n_clusters': 3,
                'min_samples': 20,
                'xi': 0.05,
                'min_cluster_size': 0.1,
                'threshold': 0.05}


MBMM_init_para = [np.array([[0.5,0.5,2.],[6.,6.,6.]]),]

datasets = [(noisy_circles, {'damping': .85, 'preference': -20,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}),]

for i_dataset, (dataset, algo_params) in enumerate(datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    #X = StandardScaler().fit_transform(X)
    X = scale(X)
    ref_point = X[0,:]

    mbmm = MBMM(C = params['n_clusters'], n_runs = 200, param = MBMM_init_para[i_dataset])
    
    clustering_algorithms = (
        ('MBMM', mbmm),
    )
   
    for name, algorithm in clustering_algorithms:
        filename = os.path.join('pickles', name+'.pkl')
        if not os.path.isfile(filename):
            t0 = time.time()

            # catch warnings related to kneighbors_graph
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="the number of connected components of the " +
                    "connectivity matrix is [0-9]{1,2}" +
                    " > 1. Completing it to avoid stopping the tree early.",
                    category=UserWarning)
                warnings.filterwarnings(
                    "ignore",
                    message="Graph is not fully connected, spectral embedding" +
                    " may not work as expected.",
                    category=UserWarning)
                algorithm.fit(X)

                print("Parameters: ", algorithm.get_params())
            with open(filename, 'wb') as f:
                pickle.dump(algorithm, f)
        else:
            with open(filename, 'rb') as f:
                algorithm = pickle.load(f)
        
        y_pred = algorithm.predict_proba(X)
        distances = np.linalg.norm(y_pred[0,:] - y_pred, axis=1)
        #print(distances.shape)
        #print([y.shape for y in y_pred])
        distances = [scipy.stats.entropy(y_pred[0,:], y) for y in y_pred]

        #plt.subplot(len(datasets), len(clustering_algorithms), plot_num)

        # plot x, y, distances
        cm = plt.cm.get_cmap('winter')
        vmax = np.max(distances)
        sc = plt.scatter(X[:, 0], X[:, 1], c=distances, vmin=0, vmax=vmax, cmap=cm, marker='x')
        plt.colorbar(sc)
        plt.scatter(ref_point[0], ref_point[1], c='red')
        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
plt.savefig('point-dist.pdf')

