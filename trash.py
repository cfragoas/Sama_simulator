# from macel
# def adjust_cell_number(self, min_n_cell):
    #     more_cells = True
    #     n_cells = min_n_cell
    #     lines = self.grid.lines
    #     columns = self.grid.columns
    #     weight_norm_param = 15  # not using wights, for now
    #     max_iter = 100  # not using adjustable weights, for now
    #     total_coverage_perc = []
    #     snr_samples_hist = []
    #     snr_grid_hist = []
    #     cap_samples_hist = []
    #     cap_grid_hist = []
    #     n_cells_hist = []
    #
    #     if self.log:
    #         print('Initializing execution...')
    #         print(max_iter, 'iterations per cluster process')
    #
    #     while more_cells:
    #         for n in range(max_iter):
    #             self.base_station_list = []
    #             cluster = Cluster()
    #             # cluster.hierarchical_clustering(grid=self.grid.grid, n_clusters=n_cells, plot=False)  # clustering
    #             # cluster.gaussian_mixture_model(grid=self.grid.grid, n_clusters=n_cells, plot=False)
    #
    #             for i in range(n_cells):
    #                 self.base_station_list.append(self.default_base_station)  # generating different base station configurations
    #
    #             self.grid.clear_grid()
    #             self.grid.make_points(dist_type='gaussian', samples=20, n_centers=self.n_centers, random_centers=False, plot=False)
    #             cluster.gaussian_mixture_model(grid=self.grid.grid, n_clusters=n_cells, plot=False)
    #
    #             # weights = np.bincount(cluster.labels) / weight_norm_param
    #             # self.voronoi = Voronoi(centers=cluster.centroids, lines=self.grid.lines, columns=self.grid.columns)
    #             # self.voronoi.generate_voronoi(plot=False)
    #             self.azi_map = generate_azimuth_map(lines=lines, columns=columns, centroids=cluster.centroids, samples=cluster.features)
    #             dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=cluster.centroids,samples=cluster.features,plot=False)
    #             self.elev_map = generate_elevation_map(htx=self.default_base_station.tx_height, hrx=self.rx_height,
    #                                                    d_euclid=dist_map, cell_size=self.cell_size, samples=None)
    #             self.path_loss_map = generate_path_loss_map(eucli_dist_map=dist_map, cell_size=self.cell_size,
    #                                                         prop_model='fs', frequency=self.default_base_station.frequency,
    #                                                         htx=self.default_base_station.tx_height, hrx=self.rx_height, samples=None)
    #
    #             # self.adjust_elev_pattern(min_tilt=0, max_tilt=15, samples=cluster.features)  # adjusting the tilt per sector per base station
    #
    #             self.gain_map = generate_gain_map(antenna=self.default_base_station.antenna, elevation_map=self.elev_map, azimuth_map=self.azi_map, base_station_list=self.base_station_list)
    #
    #             self.rx_pw_map = generate_rx_power_map(path_loss_map=self.path_loss_map, azimuth_map=self.azi_map,elevation_map=self.elev_map,base_station=self.default_base_station, gain_map=self.gain_map)
    #             self.snr_map, snr_map_uni, snr_grid = generate_snr_map(base_station=self.default_base_station, rx_power_map=self.rx_pw_map, unified=True)
    #             self.cap_map, cap_grid_criteria, cap_grid = generate_capcity_map(snr_map=self.snr_map, bw=self.default_base_station.bw, threshold=self.criteria, unified=False)
    #
    #             # self.voronoi.generate_power_voronoi(weights=weights, typ='add')
    #             if self.log:
    #                 # print('using ', n_cells, ' with weights: ', weights, 'throught ', max_iter, ' iterations')
    #                 print('using ', n_cells, 'standard voronoi cells')
    #             # more_cells, total_coverage_perc_ = self.adjust_weights(max_iter=max_iter)
    #
    #             # snr_samples_hist.append(snr_samples)
    #             snr_grid_hist.append(np.mean(self.snr_map))
    #             # cap_samples_hist.append(cap_samples)
    #             cap_grid_hist.append(np.mean(self.cap_map))
    #             n_cells_hist.append(n_cells)
    #
    #             # print('cap samples: ' + str(np.round(cap_samples/10E6, decimals=2)) + ' Mbps')
    #             # print('cap grid: ' + str(np.round(cap_grid/10E6, decimals=2)) + ' Mbps')
    #             print('cap samples: ' + str(np.round(np.mean(self.cap_map) / 10E6, decimals=2)) + ' Mbps')
    #
    #             # if n_cells == 7:
    #             #     plt.matshow(snr_map_uni, origin='lower', cmap=plt.get_cmap('jet'))
    #             #     plt.title('snr map')
    #             #     plt.colorbar()
    #             #     plt.show()
    #
    #             if n_cells == 5:
    #                 cap_samples_hist = np.round(np.asarray(cap_samples_hist)/10E6, 2)
    #                 cap_grid_hist = np.round(np.asarray(cap_grid_hist)/10E6, 2)
    #                 snr_samples_hist = np.round(np.asarray(snr_samples_hist), 2)
    #                 snr_grid_hist = np.round(np.asarray(snr_grid_hist), 2)
    #
    #                 # plt.plot(n_cells_hist, cap_samples_hist, label='in samples capacity')
    #                 plt.plot(cap_grid_hist, label='mean grid capacity')
    #                 plt.legend()
    #                 plt.title('capacity (Mbps)x base stations')
    #                 plt.show()
    #
    #                 # plt.plot(n_cells_hist, snr_samples_hist, label='in samples mean SNR')
    #                 plt.plot(snr_grid_hist, label='mean grid SNR')
    #                 plt.legend()
    #                 plt.title('SNR (dB)x base stations')
    #                 plt.show()
    #             # total_coverage_perc.extend(perc)
    #         n_cells += 1
    #
    #     plt.plot(total_coverage_perc)
    #     plt.show()
    #
    #     return self.voronoi

#
# def adjust_weights(self, max_iter):  # NOT USED (FOR NOW)
#     fulfillment = False
#     more_cells = False
#
#     path_loss_map = generate_path_loss_map(eucli_dist_map=self.voronoi.dist_mtx, centroids=self.voronoi.n_centers,
#                                            cell_size=self.cell_size, prop_model='fs', frequency=2.8,
#                                            htx=self.tx_height, hrx=self.rx_height)
#
#     rx_power_map = self.tx_power - path_loss_map
#
#     if self.log:
#         print('Generated ', self.voronoi.n_centers, 'reception power maps')
#
#     # main loop - increases cell number and changes cell size when criteria is not match
#     while not more_cells:
#         # checking the fulfillment of the criteria
#         counter = 0
#         total_points = len(self.grid.grid[0]) * len(self.grid.grid[1])
#         total_coverate_perc = np.zeros(shape=(max_iter))
#
#         while not fulfillment:
#             counter += 1
#             fulfill_criteria_abs = np.zeros(shape=(self.voronoi.n_centers))
#             fulfill_criteria_perc = np.zeros(shape=(self.voronoi.n_centers))
#
#             total_coverage_abs = 0
#
#             for i in range(self.voronoi.n_centers):
#                 indices = np.where(self.voronoi.power_voronoi_map == i)
#                 total_points_centroid = len(indices[0])
#                 # indices = list(zip(indices[0], indices[1]))
#                 x = len(np.where(rx_power_map[i][indices[0], indices[1]] > self.criteria)[0])
#                 fulfill_criteria_abs[i] = len(np.where(rx_power_map[i][indices[0], indices[1]] > self.criteria)[0])
#                 fulfill_criteria_perc[i] = fulfill_criteria_abs[i] / total_points_centroid
#                 total_coverage_abs += fulfill_criteria_abs[i]
#
#             total_coverate_perc[counter - 1] = total_coverage_abs / total_points
#
#             if not (fulfill_criteria_perc[
#                 fulfill_criteria_perc > 0.85]).size == self.voronoi.n_centers:  # checking if the criteria is met for all cells
#                 new_weights = np.zeros(shape=(self.voronoi.n_centers))
#                 for i, weight in enumerate(self.voronoi.weights):  # changing all weights
#                     new_weight = gauss(mu=weight, sigma=weight / 10)
#                     new_weights[i] = new_weight
#                 self.voronoi.generate_power_voronoi(weights=new_weights)  # calculating new voronoi with the new weights
#             else:
#                 if self.log:
#                     print('Criteria was met in ', counter, 'iterations')
#                     print(fulfill_criteria_perc)
#                     print((fulfill_criteria_perc[fulfill_criteria_perc > 0.85]).size)
#                 return more_cells, total_coverate_perc  # return false when the criteria is met
#
#             if counter >= max_iter:
#                 more_cells = True
#                 if self.log:
#                     print('Criteria was not met. Thought ', counter, 'iterations, the best result was ',
#                           max(total_coverate_perc))
#                 return more_cells, total_coverate_perc
#
#
# def adjust_cell_number(self, min_n_cell, max_iter, max_n_cell=None, default_cells=None, processes=-1, log=False,
#                        log_lv=3):
#     more_cells = True  # variable to change value when a desired criteria is met
#     if default_cells is not None:
#         n_cells = default_cells.shape[0] + min_n_cell
#     else:
#         n_cells = min_n_cell  # minimum cell number
#     lines = self.grid.lines  # grid line size
#     columns = self.grid.columns  # grid column size
#     weight_norm_param = 15  # not using wights, for now
#     max_iter = max_iter  # maximum number of iterations per BS number
#     total_coverage_perc = []
#     snr_samples_hist = []
#     snr_grid_hist = []
#     cap_samples_hist = []
#     cap_grid_hist = []
#     n_cells_hist = []
#
#     if self.log:
#         print('Initializing execution...')
#         print(max_iter, 'iterations per cluster process')
#
#     self.generate_base_station_list(n_cells=n_cells)
#     # self.base_station_list = []
#     # for i in range(n_cells):
#     #     self.base_station_list.append(self.default_base_station)  # generating copies for different base station configurations
#
#     # creating a poll for multithreading
#     if processes == -1:  # to use all available cores -1 (using all threads may cause system instability)
#         threads = os.cpu_count()
#         p = multiprocessing.Pool(processes=threads - 1)
#     else:
#         p = multiprocessing.Pool(processes=processes)
#
#     # executing the function (optmize_cell) that will sample and calcullate and optimize the macro cell performance
#     # it will return the performance and the position of each BS per iteration in data in list form
#     data = list(
#         tqdm.tqdm(p.imap_unordered(self.optimize_cell, [(n_cells, default_cells) for i in range(max_iter)]),
#                   total=max_iter
#                   ))
#
#     data = np.array(data)
#     data = np.transpose(data)
#     perf_list = data[0]
#     centroids_list = data[1]
#     # sample_list = data[2]
#     # samples_dist = np.sum(sample_list, axis=0)
#     # plt.matshow(samples_dist)
#     # plt.colorbar()
#     # plt.show()
#
#     map_data = np.zeros(shape=(lines, columns))
#     map_data2 = np.zeros(shape=(lines, columns))
#     for i, centroids in enumerate(centroids_list):
#         for j, centroid in enumerate(centroids):
#             if j > 1:
#                 map_data[int(np.round(centroid[0], 0)), int(np.round(centroid[1], 0))] += perf_list[i] / 10E6
#                 map_data2[int(np.round(centroid[0], 0)), int(np.round(centroid[1], 0))] += 1
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(range(lines), range(columns))
#     ax.plot_surface(X, Y, map_data)
#     ax.scatter3D([100, 400], [100, 400], [10, 10], color='black', alpha=0.8, marker='x')
#     plt.show()
#
#     plt.matshow(map_data.T, origin='lower')
#     plt.set_cmap('RdYlGn')
#     plt.colorbar()
#     plt.scatter([100, 400], [100, 400])
#     plt.show()
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(range(lines), range(columns))
#     ax.plot_surface(X, Y, map_data2)
#     ax.scatter([100, 400], [100, 400], [0, 0])
#     plt.show()
#
#     plt.matshow(map_data2.T, origin='lower')
#     plt.colorbar()
#     plt.scatter([100, 400], [100, 400])
#     plt.show()
#
#     map_data_norm = np.where(map_data != 0, map_data / map_data2, 0)
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     X, Y = np.meshgrid(range(lines), range(columns))
#     ax.plot_surface(X, Y, map_data_norm)
#     ax.scatter([100, 400], [100, 400], [10, 10])
#     plt.show()
#
#     plt.matshow(map_data_norm.T, origin='lower')
#     plt.colorbar()
#     plt.scatter([100, 400], [100, 400])
#     plt.show()
#
#
# def optimize_cell(self, args):
#     grid = self.grid
#     # grid = args[0]
#     n_cells = args[0]
#     default_cells = args[1]
#     # print(default_cells)
#     lines = grid.lines
#     columns = grid.columns
#     grid.clear_grid()
#     grid.make_points(dist_type='gaussian', samples=20, n_centers=self.n_centers, random_centers=False, plot=False)
#     # cluster = Cluster()
#     # cluster.gaussian_mixture_model(grid=grid.grid, n_clusters=n_cells, plot=False)
#     # exotic cluster execution
#     cluster = K_Means(k=4)
#     cluster.fit(data=grid.grid, predetermined_centroids=np.array([[100, 100], [400, 400]]))
#     # cluster.predict()
#     # cluster.plot()
#
#     # JUST A TEST!!!
#     if hasattr(base_station_list[0].antenna, 'beamforming_id'):
#         pass
#
#     azi_map = generate_azimuth_map(lines=lines, columns=columns, centroids=cluster.centroids,
#                                    samples=cluster.features)
#     dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=cluster.centroids,
#                                            samples=cluster.features, plot=False)
#     elev_map = generate_elevation_map(htx=self.default_base_station.tx_height, hrx=self.rx_height,
#                                       d_euclid=dist_map, cell_size=self.cell_size, samples=None)
#     path_loss_map = generate_path_loss_map(eucli_dist_map=dist_map, cell_size=self.cell_size,
#                                            prop_model='fs', frequency=self.default_base_station.frequency,
#                                            htx=self.default_base_station.tx_height, hrx=self.rx_height,
#                                            samples=None)
#
#     gain_map = generate_gain_map(antenna=self.default_base_station.antenna, elevation_map=elev_map,
#                                  azimuth_map=azi_map, base_station_list=self.base_station_list)
#
#     rx_pw_map = generate_rx_power_map(path_loss_map=path_loss_map, azimuth_map=azi_map,
#                                       elevation_map=elev_map, base_station=self.default_base_station,
#                                       gain_map=gain_map)
#     snr_map, snr_map_uni, snr_grid = generate_snr_map(base_station=self.default_base_station,
#                                                       rx_power_map=rx_pw_map, unified=True)
#     cap_map, cap_grid_criteria, cap_grid = generate_capcity_map(snr_map=snr_map,
#                                                                 bw=self.default_base_station.bw,
#                                                                 threshold=self.criteria, unified=False)
#
#     if self.log:
#         # print('using ', n_cells, ' with weights: ', weights, 'throught ', max_iter, ' iterations')
#         print('using ', n_cells, 'standard voronoi cells')
#     cap = np.mean(cap_map)
#     # print('cap samples: ' + str(np.round(cap_samples/10E6, decimals=2)) + ' Mbps')
#     # print('cap grid: ' + str(np.round(cap_grid/10E6, decimals=2)) + ' Mbps')
#     # print('cap samples: ' + str(np.round(np.mean(cap_map) / 10E6, decimals=2)) + ' Mbps')
#     return cap, cluster.centroids  # , grid.grid
#
#
# def adjust_elev_pattern(self, min_tilt, max_tilt, samples):
#     import warnings
#     warnings.filterwarnings('ignore')
#     # downtilts = np.asarray(list(combinations_with_replacement(np.arange(min_tilt, max_tilt, 5), 3)))  # all possible tilts combinations
#     downtilts = np.asarray(list(itertools.product(np.arange(min_tilt, max_tilt + 1, 1), repeat=3)))
#     mean_perf = np.ndarray(shape=downtilts.shape[0])
#
#     # adjusting the samples to the calcullated voronoi cells
#     # samples = np.where(self.voronoi.std_voronoi_map[samples[samples, 0], samples[samples, 1]] == i]
#
#     for i, bs in enumerate(self.base_station_list):
#         for j, downtilt in enumerate(downtilts):
#             bs.change_downtilt(downtilt)
#             user_samples = np.where(self.voronoi.std_voronoi_map[samples[:, 0], samples[:, 1]] == i)
#             user_gain_map = generate_gain_map(antenna=bs.antenna, elevation_map=self.elev_map[i][
#                 samples[user_samples, 0], samples[user_samples, 1]],
#                                               azimuth_map=self.azi_map[i][
#                                                   samples[user_samples, 0], samples[user_samples, 1]],
#                                               sectors_ver_pattern=bs.sectors_ver_pattern,
#                                               sectors_hor_pattern=bs.sectors_hor_pattern)
#             # user_rx_pw_map = generate_rx_power_map(path_loss_map=self.path_loss_map[i][samples[user_samples, 0], samples[user_samples, 1]],
#             #                                        azimuth_map=self.azi_map[i][samples[user_samples, 0], samples[user_samples, 1]],
#             #                                        elevation_map=self.elev_map[i][samples[user_samples, 0], samples[user_samples, 1]],
#             #                                        base_station=bs, gain_map=user_gain_map)
#             interf_samples = np.where(self.voronoi.std_voronoi_map[samples[:, 0], samples[:, 1]] != i)
#             interf_gain_map = generate_gain_map(antenna=bs.antenna, elevation_map=self.elev_map[i][
#                 samples[interf_samples, 0], samples[interf_samples, 1]],
#                                                 azimuth_map=self.azi_map[i][
#                                                     samples[interf_samples, 0], samples[interf_samples, 1]],
#                                                 sectors_ver_pattern=bs.sectors_ver_pattern,
#                                                 sectors_hor_pattern=bs.sectors_hor_pattern)
#             # interf_rx_pw_map = generate_rx_power_map(path_loss_map=self.path_loss_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
#             #                                        azimuth_map=self.azi_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
#             #                                        elevation_map=self.elev_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
#             #                                        base_station=bs, gain_map=interf_gain_map)
#
#             user_gain_map_unified = np.max(10 ** (user_gain_map / 10), axis=0)
#             interf_gain_map_unified = np.max(10 ** (interf_gain_map / 10), axis=0)
#             mean_perf[j] = np.mean(user_gain_map_unified)  # - np.mean(interf_gain_map_unified)
#
#             # user_pw_map_unified = np.max(10**(user_rx_pw_map/10), axis=0)
#             # interf_pw_map_unified = np.max(10**(interf_rx_pw_map/10), axis=0)
#             # mean_perf[j] = np.mean(user_pw_map_unified) - np.mean(interf_pw_map_unified)
#
#         best_downtilt = np.argmax(mean_perf)
#         print(downtilts[best_downtilt])
#         # if best_downtilt != 0:
#         #     print(mean_perf)
#         #     print(best_downtilt)
#         #     plt.plot(mean_perf)
#         #     plt.show()
#         bs.change_downtilt(downtilts[best_downtilt])
#         self.base_station_list[i] = bs
#
#     warnings.filterwarnings('default')