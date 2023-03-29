import os
import pickle
import warnings

import numpy as np
import matplotlib.pyplot as plt

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice


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
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)

# Anisotropicly distributed data
#upper left to lower right
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]

#upper right to lower left
X, y = datasets.make_blobs(n_samples=n_samples, random_state=170)
transformation = [[-0.6, -0.6], [0.4, 0.8]]
X_aniso2 = np.dot(X, transformation)
aniso2 = (X_aniso2, y)


# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)



# ============
# Set up cluster parameters
# ============
plt.figure(figsize=((9 * 2 + 3)*1.1, 15/6*7))
plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.95, wspace=.05,
                    hspace=.01)

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


MBMM_init_para = [
    np.array([[2.,9.,3.],[9.,2.,3.],[8.,8.,8.]]), # blob
    np.array([[2.,9.,3.],[9.,2.,3.],[8.,8.,8.]]), # varied
    np.array([[2.,9.,3.],[9.,2.,3.],[4.,0.8,1.3]]),  # aniso2
    np.array([[0.5,0.5,2.],[6.,6.,6.]]),    # noisy_circles
    np.array([[0.5,0.5,2.],[6.,6.,6.]]),    # noisy_moons
]


syn_datasets = [
    (blobs, {}, 'blobs'),
    (varied, {'eps': .03, 'n_neighbors': 2,
              'min_samples': 5, 'xi': 0.035, 'min_cluster_size': .2}, 'varied'),
    (aniso2, {'eps': .02, 'n_neighbors': 2,
             'min_samples': 20, 'xi': 0.04, 'min_cluster_size': .2}, 'aniso2'),
    (noisy_circles, {'damping': .85, 'preference': -20,
                     'quantile': .2, 'n_clusters': 2,
                     'min_samples': 20, 'xi': 0.25}, 'noisy_circles'),
    (noisy_moons, {'damping': .86, 'preference': -6, 'n_clusters': 2}, 'noisy_moons'),
]

for i_dataset, (dataset, algo_params, dataset_name) in enumerate(syn_datasets):
    # update parameters with dataset-specific values
    params = default_base.copy()
    params.update(algo_params)

    X, y = dataset

    # normalize dataset for easier parameter selection
    #X = StandardScaler().fit_transform(X)
    X = scale(X)

    # connectivity matrix for structured Ward
    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
    dbscan = cluster.DBSCAN(eps=params['eps'])
    average_linkage = cluster.AgglomerativeClustering(
        linkage="average", affinity="cityblock",
        n_clusters=params['n_clusters'], connectivity=connectivity)
    gmm = mixture.GaussianMixture(
        n_components=params['n_clusters'], covariance_type='full')
    mbmm = MBMM(C = params['n_clusters'], n_runs = 200, param = MBMM_init_para[i_dataset])
    
    
    clustering_algorithms = (
        ('MBMM', mbmm),
        ('KMeans', two_means),
        ('AC', average_linkage),
        ('DBSCAN', dbscan),
        ('GMM', gmm),
    )

    for algo_name, algorithm in clustering_algorithms:
        filename = 'models/{}-{}.pck'.format(dataset_name, algo_name)
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                algorithm = pickle.load(f)
        else:
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
            with open(filename, 'wb') as f:
                pickle.dump(algorithm, f)

        if hasattr(algorithm, 'labels_'):
            y_pred = algorithm.labels_.astype(int)
        else:
            y_pred = algorithm.predict(X)

        plt.subplot(len(syn_datasets), len(clustering_algorithms), plot_num)
        if i_dataset == 0:
            plt.title(algo_name, size=30)

        colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                             '#f781bf', '#a65628', '#984ea3',
                                             '#999999', '#e41a1c', '#dede00']),
                                      int(max(y_pred) + 1))))
        # add black color for outliers (if any)
        colors = np.append(colors, ["#000000"])
        plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

        plt.xlim(0.0, 1.0)
        plt.ylim(0.0, 1.0)
        plt.xticks(())
        plt.yticks(())
        plot_num += 1
figure_filename = "visualize-cluster.pdf"
plt.savefig(figure_filename)
print("Clustering results are saved in the file: '{}'".format(figure_filename))


