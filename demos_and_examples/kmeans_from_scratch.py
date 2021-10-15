# original source: https://medium.com/@rishit.dagli/build-k-means-from-scratch-in-python-e46bf68aa875
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

class K_Means_XP:
    def __init__(self, k=3, tol=0.0000000001, max_iter=10000):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self,data=None, predetermined_centroids=None):
        if data is None:
            features, true_labels = make_blobs(n_samples=100, centers=3, cluster_std=4)
            data = features
        else:
            X,Y = np.where(data==1)
            data = np.column_stack((X, Y))
        self.centroids = {}
        self.features = data


        for i in range(self.k):
            if predetermined_centroids is not None and i <= len(predetermined_centroids)-1:
                self.centroids[i] = predetermined_centroids[i]
            else:
                self.centroids[i] = [np.max(data)/2, np.max(data)/2] + 3*np.random.rand(1)


        for i in range(self.max_iter):
            self.classifications = {}

            for i in range(self.k):
                self.classifications[i] = []

            for featureset in data:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                if predetermined_centroids is not None:
                    if classification > len(predetermined_centroids)-1:
                        self.centroids[classification] = np.average(self.classifications[classification],axis=0)
                else:
                    self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    # print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                X, Y = list(), list()
                self.centroids_dict = self.centroids
                for centroid in self.centroids:
                    X.append(self.centroids[centroid][0])
                    Y.append(self.centroids[centroid][1])
                self.centroids = np.column_stack((X, Y))
                break


    def predict(self,data=None):
        self.labels = list()
        if data is None:
            data = self.features
        for data in self.features:
            distances = [np.linalg.norm(data-self.centroids_dict[centroid]) for centroid in self.centroids_dict]
            self.labels.append(distances.index(min(distances)))
        return self.labels

    def plot(self):
        plt.scatter(self.features[:, 0], self.features[:, 1], c=self.labels)
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], marker='^', c='k')
        plt.title('Kmeans from scratch')
        plt.show()