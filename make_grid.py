import numpy as np
import math
from random import randint, gauss, gammavariate
import sys
import matplotlib.pyplot as plt
from util.util_funcs import highestPowerOf2

class Grid:
    def __init__(self):
        self.lines = None
        self.columns = None
        self.grid = None
        self.dist_mtx = None


        # variables calculated inside the class
        self.centers_set = None

    def make_grid(self, lines=None, columns=None):
        self.lines = lines
        self.columns = columns
        self.grid = np.zeros(shape=(lines, columns))

    def clear_grid(self):
        # self.grid = np.zeros(shape=(self.lines, self.columns))
        self.grid[:] = 0

    def make_points(self, dist_type, samples, n_centers, random_centers=True, plot=False):  # generating centers for the distributions
        centers_set = set()  # workaround to generate unique values
        if random_centers:
            while len(centers_set) < n_centers:
                x, y = 7, 0
                while (x, y) == (7, 0):
                    x, y = randint(0, self.lines - 1), randint(0, self.columns - 1)
                # that will make sure we don't add (7, 0) to cords_set
                centers_set.add((x, y))
            self.centers_set = list(centers_set)

        else:
            # Create a grid of points in x-y space
            if n_centers == 1:
                xvals = [np.round(self.columns / 2)]
                yvals = [np.round(self.lines / 2)]
            elif n_centers % 2 != 0:
                sys.exit('n_centers must be a even number!!!')
            else:
                n = highestPowerOf2(n_centers)  # number of lines
                if n % 2 != 0:
                    n -= 1
                if n == 0:
                    n = 2
                xvals = np.round(np.linspace((self.columns / (n_centers / n)) / 2,
                                             self.columns - (self.columns / (n_centers / n)) / 2, int(n_centers / n)))
                yvals = np.round(np.linspace(self.lines / (2 * n), self.lines * (1 - 1 / (2 * n)), int(n)))

            centers_set_ = np.row_stack([(x, y) for x in xvals for y in yvals]).astype(int)
            for i in range(len(centers_set_)):
                centers_set.add((centers_set_[i][0], centers_set_[i][1]))
            self.centers_set = list(centers_set)

            # # Apply linear transform
            # a = np.column_stack([[2, 1], [-1, 1]])
            # print(a)
            # uvgrid = np.dot(a, xygrid)

        if dist_type == 'gaussian':
            mu = 0
            sigma = min(self.lines, self.columns) / 10
            for i in range(n_centers):
                for _ in range(samples):
                    [x, y] = [self.centers_set[i][0] + round(gauss(mu, sigma)), self.centers_set[i][1] + round(gauss(mu, sigma))]
                    if (x < self.lines and y < self.columns) and (x >= 0 and y >= 0):
                        self.grid[x, y] += 1

        elif dist_type == 'gamma':
            alpha = min(self.columns, self.lines) / 10
            beta = 1
            for i in range(n_centers):
                for _ in range(samples):
                    [x, y] = [self.centers_set[i][0] + round(gammavariate(alpha, beta)),
                              self.centers_set[i][1] + round(gammavariate(alpha, beta))]
                    self.grid[x, y] += 1

        if plot:
            plt.matshow(self.grid, origin='lower')
            plt.title('Grid with random points')
            plt.show()

    def distance_matrix(self, coordinates):
        n_coord = len(coordinates)
        self.dist_mtx = np.ndarray(shape=(n_coord, self.lines, self.columns))
        coord_map = np.indices((self.lines, self.columns)).transpose(
            (1, 2, 0))  # coordinates map used to calculate the distance

        for i in range(1, n_coord):
            map = np.empty(shape=(self.lines, self.columns))
            for line in coord_map:
                for coord in line:
                    map[coord[0], coord[1]] = math.dist((coordinates[i][0], coordinates[i][1]), (coord[0], coord[1]))
                    self.dist_mtx[i] = map



