import numpy as np

import models
from make_grid import Grid
from clustering import Cluster
from make_voronoi import Voronoi
import matplotlib.pyplot as plt
from models.propagation.prop_models import generate_path_loss_map, generate_capcity_map

grid = Grid()  # grid object
grid.make_grid(100, 100)  # creating a grid with x, y dimensions
grid.make_points(dist_type='gaussian', samples=500, n_centers=4, random_centers=False)  # distributing points aring centers in the grid

plt.matshow(grid.grid)
plt.show()

# input("Press Enter to continue...")
cluster = Cluster()
cluster.hierarchical_clustering(grid=grid.grid, n_clusters=8, plot=True)  # clustering

print(cluster.centroids)

# Voronoi Generation
voronoi = Voronoi(cluster.centroids, grid.lines, grid.columns)  # voronoi object
# voronoi.distance_matrix(plot=True)  # plot distance matrix if needed
print(voronoi.n_centers)
print(voronoi.centers)
voronoi.generate_voronoi(plot=True)  # generating standard voronoi regions
weights = np.bincount(cluster.labels)/15
print(weights)
voronoi.generate_power_voronoi(weights=weights, typ='add', plot=True)
voronoi.generate_power_voronoi(weights='random', typ='add', plot=True)  # generating power voronoi regions

# testing Macel positions algorithm
# macel = Macel(grid=grid, prop_model='fs', criteria=-40, cell_size=10, log=True)
# macel.set_tx(frequency=3.5, tx_power=50, tx_height=35, bw=2.8)
# macel.set_rx(rx_height=1.5)
# plt.matshow(macel.elev_angle_map)
# macel.adjust_cell_number(min_n_cell=2)

path_loss_map = generate_path_loss_map(eucli_dist_map=voronoi.dist_mtx, centroids=voronoi.n_centers,
                                               cell_size=10, prop_model='fs', frequency=1,
                                               htx=30, hrx=1.5, plot=False)

snr_map = generate_SNR_map(ptx=50, centroids=voronoi.n_centers, path_loss_map=path_loss_map, bw=20, plot=False)
# testing with the best SNR per pixel
best_snr_map = np.max(snr_map, axis=0)
plt.matshow(best_snr_map)
plt.colorbar()
plt.show()
capacity_map = 20E6 * np.log2(1 + 10**(best_snr_map/10))
plt.matshow(capacity_map)
plt.colorbar()
plt.show()

#testing elevation angles
coord_map = np.indices((voronoi.lines, voronoi.columns)).transpose((1, 2, 0))  # coordinates map used to calculate the distance
elev_map = np.ndarray(shape=(voronoi.n_centers, voronoi.lines, voronoi.columns))
for i in range(voronoi.n_centers):
    for line in coord_map:
        for coord in line:
            elev_map[i][coord[0], coord[1]] = models.elevation_angle(i, coord, 30, 1.5, 10, voronoi.dist_mtx)
    plt.matshow(elev_map[i], origin='lower')
    plt.colorbar()
    plt.show()

#testing azimuth
az_map = np.ndarray(shape=(voronoi.n_centers, voronoi.lines, voronoi.columns))
for i in range(voronoi.n_centers):
    for line in coord_map:
        for coord in line:
            az_map[i][coord[0], coord[1]] = models.azimuth_angle_clockwise( [45, 30], [coord[0], coord[1]])
    plt.matshow(az_map[i], origin='lower')
    plt.colorbar()
    plt.show()


capacity_map = generate_capcity_map(best_snr_map, bw=20, plot=True)