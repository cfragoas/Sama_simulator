from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # kmeans sklearn library - https://realpython.com/k-means-clustering-python/
from sklearn.cluster import AgglomerativeClustering  # Hierarchical clustering skleran library - https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
from sklearn.neighbors import NearestCentroid  # to find centroids in methods that won't use then
from sklearn.mixture import GaussianMixture
import numpy as np
import matplotlib.pyplot as plt


class Cluster:
    def __init__(self):
        self.features = None
        self.scaled_features = None
        self.scaler = None
        self.cluster_method = None
        self.centroids = None
        self.labels = None

    def set_features(self, grid):
        self.features = np.where(grid != 0)

    def scaling(self, grid):
        self.set_features(grid)
        # self.features = np.where(grid != 0)
        features_ = []
        for i in range(len(self.features[0])):
            features_.append([self.features[0][i], self.features[1][i]])
        self.features = np.array(features_)
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(self.features)

    def k_means(self, grid, n_clusters, plot=False):
        self.centroids, self.labels = None, None
        if self.scaled_features is None:
            self.scaling(grid)
        kmeans = KMeans(init="random", n_clusters=n_clusters, n_init=10, max_iter=300)
        kmeans.fit(self.scaled_features)
        self.cluster_method = 'k_means'
        self.centroids = self.scaler.inverse_transform(kmeans.cluster_centers_)
        self.labels = kmeans.labels_
        if plot:
            self.plot()

    def hierarchical_clustering(self, grid, n_clusters, plot=False):
        if n_clusters == 1:
            self.k_means(grid=grid, n_clusters=n_clusters, plot=plot)
            return
        self.centroids, self.labels = None, None
        if self.scaled_features is None:
            self.scaling(grid)
        hier_clust = AgglomerativeClustering(n_clusters, affinity='euclidean', linkage='ward')
        hier_clust.fit(self.scaled_features)
        self.cluster_method = 'Hierarchical Clustering'
        clf = NearestCentroid()
        clf.fit(self.scaled_features, hier_clust.labels_)
        self.centroids = self.scaler.inverse_transform(clf.centroids_)
        self.labels = hier_clust.labels_
        if plot:
            self.plot()

    def gaussian_mixture_model(self, grid, n_clusters, plot=False):
        if n_clusters == 1:
            # print('n_clusters = 1 -> using k_means instead')
            self.k_means(grid=grid, n_clusters=n_clusters, plot=plot)
            return
        self.centroids, self.labels = None, None
        if self.scaled_features is None:
            self.scaling(grid)
        gmm = GaussianMixture(n_components=n_clusters)
        gmm.fit(self.scaled_features)
        self.labels = gmm.predict(self.scaled_features)
        clf = NearestCentroid()
        clf.fit(self.scaled_features, self.labels)
        self.centroids = self.scaler.inverse_transform(clf.centroids_)

        if plot:
            self.plot()

    def random(self, grid, n_clusters, plot=False):  # not clustering!!!
        # self.features = np.where(grid != 0)
        self.scaling(grid=grid)
        x_size = grid.shape[0]
        y_size = grid.shape[1]
        self.centroids = np.ndarray(shape=(n_clusters, 2))
        for i in range(n_clusters):
            # self.centroids[i] = [np.random.randint(0, x_size), np.random.randint(0, y_size)]
            self.centroids[i] = np.random.uniform(0, x_size-1, 2)

        print('')

    def plot(self):
        plt.scatter(self.features[:, 0], self.features[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='^', c='k')
        plt.title(self.cluster_method)
        plt.show()