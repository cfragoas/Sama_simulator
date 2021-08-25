import matplotlib.pyplot as plt
from make_grid import Grid
from make_voronoi import Voronoi
from clustering import Cluster
from base_station import BaseStation
from prop_models import generate_azimuth_map, generate_elevation_map, generate_gain_map, generate_path_loss_map, \
    generate_rx_power_map, generate_snr_map, generate_capcity_map, generate_euclidian_distance
from macel import Macel
import numpy as np
from demos_and_examples.kmeans_from_scratch import K_Means
import time


grid = Grid()  # grid object
grid.make_grid(1000, 1000)  # creating a grid with x, y dimensions
grid.make_points(dist_type='gaussian', samples=500, n_centers=4, random_centers=False, plot=False)  # distributing points aring centers in the grid

# plt.matshow(grid.grid, origin='lower')
# plt.title('grid samples')
# plt.colorbar()
# plt.show()

bs = BaseStation(frequency=3.5, tx_power=3, tx_height=30, bw=20, n_sectors=3, antenna='ITU1336', gain=10, downtilts=[10, 20, 30], plot=False)
cell_size = 10


# input("Press Enter to continue...")
cluster = Cluster()
start = time.time()
cluster.k_means(grid=grid.grid, n_clusters=1,plot=False)
cluster.hierarchical_clustering(grid=grid.grid, n_clusters=1, plot=False)  # clustering
cluster.gaussian_mixture_model(grid=grid.grid, n_clusters=3, plot=False)
cluster = K_Means(k=4)
cluster.fit(data=grid.grid, predetermined_centroids=np.array([[250, 250], [400, 800]]))
# cluster.fit(data=grid.grid)
cluster.predict()
cluster.plot()
print(cluster.centroids)

print('hierarchical clustering time: ' + str(start - time.time()))

# euclidian distance test
dist = generate_euclidian_distance(lines=grid.lines, columns=grid.columns, centers=cluster.centroids, samples=cluster.features, plot=False)
print(dist.shape)
# Voronoi Generation

voronoi = Voronoi(cluster.centroids, grid.lines, grid.columns)  # voronoi object

# voronoi.distance_matrix(plot=True)  # plot distance matrix if needed
print('')
print(str(voronoi.n_centers) + ' Voronoi Centers with coordinates: ')
print(voronoi.centers)

start_time = time.time()
voronoi.generate_voronoi(plot=False)  # generating standard voronoi regions
print('voronoi generation running time: ', str(start_time - time.time()))

start_time = time.time()
az_map = generate_azimuth_map(lines=grid.lines, columns=grid.columns, centroids=voronoi.centers)
print('az map running time: ', str(start_time - time.time()))

start_time = time.time()
elev_map = generate_elevation_map(htx=bs.tx_height, hrx=1.5, d_euclid=voronoi.dist_mtx, cell_size=cell_size, plot=True)
print('elev map running time: ', str(start_time - time.time()))

start_time = time.time()
gain_map = generate_gain_map(antenna=bs.antenna, elevation_map=elev_map, azimuth_map=az_map, sectors_hor_pattern=bs.sectors_hor_pattern, sectors_ver_pattern=bs.sectors_ver_pattern)
print('gain map running time: ', str(start_time - time.time()))

# plt.matshow(gain_map[0][0])
# plt.title('gain map')
# plt.colorbar()
# plt.show()

start_time = time.time()
path_loss_map = generate_path_loss_map(eucli_dist_map=voronoi.dist_mtx, cell_size=cell_size, prop_model='FS',
                                       frequency=bs.frequency,  htx=bs.tx_height, hrx=1.5, plot=False)
print('path loss running time: ', str(start_time - time.time()))

start_time = time.time()
rx_power_map = generate_rx_power_map(path_loss_map=path_loss_map,  azimuth_map=az_map, elevation_map=elev_map, base_station=bs)
print('rx pw running time: ', str(start_time - time.time()))


plt.matshow(rx_power_map[0][0], origin='lower')
plt.title('rx power map')
plt.colorbar()
plt.show()
plt.matshow(rx_power_map[1][0], origin='lower')
plt.title('rx power map')
plt.colorbar()
plt.show()
plt.matshow(rx_power_map[2][0], origin='lower')
plt.title('rx power map')
plt.colorbar()
plt.show()

snr_map, snr_map_uni, snr_samples, snr_grid = generate_snr_map(base_station=bs,
                                                                    rx_power_map=rx_power_map,
                                                                    samples=cluster.features, unified=True)
plt.matshow(snr_map[0], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()
plt.matshow(snr_map[1], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()
plt.matshow(snr_map[2], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()

cap_map = generate_capcity_map(snr_map=snr_map, bw=bs.bw)
plt.matshow(cap_map[0], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()
plt.matshow(cap_map[1], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()
plt.matshow(cap_map[2], origin='lower')
plt.title('snr map')
plt.colorbar()
plt.show()