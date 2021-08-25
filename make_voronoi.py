import numpy as np
import math
import matplotlib.pyplot as plt

# Voronoi class represents the voronoi regions associated with n_centers
# It can be used to calculate a per pixel classifications of the regions

class Voronoi:
    def __init__(self, centers, lines, columns):
        self.centers = np.round(centers)
        self.lines = lines
        self.columns = columns
        self.n_centers = len(centers)

        # functions variables
        self.weights = None
        self.typ = None

        # calculated variables
        self.std_voronoi_map = None
        self.power_voronoi_map = None
        self.dist_mtx = None

        # 'private' variables
        self.__coord_map = np.indices((self.lines, self.columns)).transpose((1, 2, 0))  # coordinates map used to calculate the distance

    def distance_matrix(self, plot=False):
        # calculated euclidean distance from n_centers to each pixel
        self.dist_mtx = np.ndarray(shape=(self.n_centers, self.lines, self.columns))

        a_b = np.ndarray(shape=(self.n_centers, 2))
        for line in self.__coord_map:
            for coord in line:
                a_b[:, 0] = self.centers[:, 0] - coord[0]
                a_b[:, 1] = self.centers[:, 1] - coord[1]

                self.dist_mtx[:, coord[0], coord[1]] = np.linalg.norm(a_b, axis=1)  # euclidean distance using L2 norm

        self.dist_mtx = np.where(self.dist_mtx == 0, 1, self.dist_mtx)

        if plot:
            for i in range(self.n_centers):
                plt.matshow(self.dist_mtx[i])
                plt.title('Distance matrix: ' + str(i))
                plt.show()

        return self.dist_mtx

    def generate_voronoi(self, plot=False):
        # returns per pixel classification of voronoi regions of n_centers
        if self.dist_mtx is None:  # this is so we don't recalculate the matrix and save time
            self.distance_matrix()

        # fill_completion_map = np.zeros(shape=(self.lines, self.columns))

        fill_completion_map = np.argmin(self.dist_mtx, axis=0)

        # for line in self.__coord_map:
        #     for coord in line:
        #         superpos = self.dist_mtx[:, coord[0], coord[1]]
        #
        #         # if np.argmin(superpos) != 1000000:  # need to remove this workaround
        #         #     fill_completion_map[coord[1], coord[0]] = np.argmin(superpos)
        #         fill_completion_map[coord[0], coord[1]] = np.argmin(superpos)

        if plot:
            plt.matshow(fill_completion_map, origin='lower')
            plt.title('Voronoi mapping')
            for i in range(self.n_centers):
                plt.plot(self.centers[i][1], self.centers[i][0], '.', color='black')
            plt.show()

        self.std_voronoi_map = fill_completion_map

        return self.std_voronoi_map

    def generate_power_voronoi(self, weights, typ=None, plot=False):
        if self.dist_mtx is None:  # this is so we don't recalculate the matrix and save time
            self.distance_matrix()
        self.distance_matrix()

        ### implementing the multiplicatively and additively weighted voronoi diagram ###
        if typ is not None:
            self.typ = typ
        self.weights = weights

        # generating random weights to all centroids
        if type(weights) == str:
            if weights == 'random':
                min_weight = min(self.lines, self.columns)/20
                max_weight = min(self.lines, self.columns)/4
                weights = np.random.randint(min_weight, max_weight, size=self.n_centers)
                print('initial weghts ', weights)

        #  initializing the matrices
        fill_completion_map = np.zeros(shape=(self.lines, self.columns))

        for line in self.__coord_map:
            for coord in line:
                # initializing the matrices that will stores all distance values for each pixel
                superpos = np.zeros(shape=self.n_centers)  # need to remove this workaround
                for centroid in range(self.n_centers):
                    if self.typ == 'multi':
                        superpos[centroid] = self.dist_mtx[centroid, coord[0], coord[1]] / weights[centroid]
                    elif self.typ == 'add':
                        superpos[centroid] = self.dist_mtx[centroid, coord[0], coord[1]] + weights[centroid]  # log because exp will generate enourmous values

                if np.argmin(superpos) != 10 ** 8:  # need to remove this workaround
                    fill_completion_map[coord[0], coord[1]] = np.argmin(superpos)

        if plot:
            plt.matshow(fill_completion_map, origin='lower')
            plt.title('Power voronoi mapping')
            for i in range(self.n_centers):
                plt.plot(self.centers[i][1], self.centers[i][0], '.', color='black')
            plt.show()

        self.power_voronoi_map = fill_completion_map

        return self.power_voronoi_map
