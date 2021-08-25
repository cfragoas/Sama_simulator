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