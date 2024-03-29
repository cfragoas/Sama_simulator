from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans  # kmeans sklearn library - https://realpython.com/k-means-clustering-python/
from sklearn.cluster import AgglomerativeClustering  # Hierarchical clustering skleran library - https://www.analyticsvidhya.com/blog/2019/05/beginners-guide-hierarchical-clustering/
from sklearn.neighbors import NearestCentroid  # to find centroids in methods that won't use then
from sklearn.mixture import GaussianMixture
from util.data_management import convert_file_path_os
from pandas import read_csv
import numpy as np
import csv
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
        # max_value = grid.max()  # finding the maximum value to loop
        x = []
        y = []
        for value in range(1, grid.max().astype(int) + 1):
            x_, y_ = (np.where(grid == value))
            x = np.hstack((x, x_))
            y = np.hstack((y, y_))
        self.features = [x, y]
        # self.features = np.where(grid != 0)

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

    def random(self, grid, n_clusters, plot=False):  # not clustering!!! - todo make this piece of shit runs with a rectangular shape
        # self.features = np.where(grid != 0)
        self.scaling(grid=grid)
        x_size = grid.shape[0]
        y_size = grid.shape[1]
        self.centroids = np.ndarray(shape=(n_clusters, 2))
        xy_min=[0,0]
        xy_max = [x_size, y_size]
        self.centroids = np.random.default_rng().integers(low=xy_min, high=xy_max, size=(n_clusters, 2))
        # for i in range(n_clusters):
            # self.centroids[i] = [np.random.randint(0, x_size), np.random.randint(0, y_size)]
            # self.centroids[i] = np.random.uniform(0, x_size-1, 2)

        if plot:
            self.plot()

    def from_file(self, name_file, grid=None):
        # this function will pick the bs coordinates from a csv file and return the cell number of BSs to the simulation
        file_path = convert_file_path_os('inputs\\' + name_file)
        try:
            self.centroids = np.array(read_csv(file_path, delimiter=';')).astype('float64')  # reading the csv file
        except:
            raise TypeError('BS cvs format cannot be imported or converted to numpy matrix - please check ' + name_file +
                            'file')
        if np.sum(np.isnan(self.centroids)) != 0:
            raise TypeError('csv BS data is inconsistent - please check ' + name_file + 'file')

        n_cells = self.centroids.shape[0]

        # if grid is not None:
        #     self.scaling(grid)

        return n_cells


    def check_centers(self, lines, columns):
        # simply checks if the centroids are inside the limits of other data (lines and columns size)
        check_lines = self.centroids[:, 0] > lines
        check_columns = self.centroids[:, 1] > columns

        if np.sum(check_lines) != 0:
            raise ValueError('BS center outside the X limit of the grid - please check the BS allocation configuration')
        if np.sum(check_columns) != 0:
            raise ValueError('BS center outside the Y limit of the grid - please check the BS allocation configuration')

    def plot(self):
        plt.scatter(self.features[:, 0], self.features[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='^', c='k')
        plt.title(self.cluster_method)
        plt.show()