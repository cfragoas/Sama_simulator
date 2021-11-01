import itertools
import multiprocessing.pool
import os
import tqdm
import numpy as np
import copy
from prop_models import generate_path_loss_map, generate_elevation_map, generate_azimuth_map, generate_gain_map, \
    generate_rx_power_map, generate_snr_map, generate_capcity_map, generate_euclidian_distance, generate_bf_gain
from user_eq import User_eq
from random import gauss
from make_voronoi import Voronoi
from clustering import Cluster
import matplotlib.pyplot as plt
from itertools import product
from matplotlib import cm
import base_station
from demos_and_examples.kmeans_from_scratch import K_Means_XP


class Macel:
    def __init__(self, grid, prop_model, cell_size, base_station, criteria=None, log=False):
        self.grid = grid  # grid object - size, points, etc
        self.n_centers = None
        self.voronoi = None  # voronoi object - voronoi cells, distance matrix, voronoi maps, etc
        self.prop_model = prop_model  # string - name of prop model to be used in prop_models
        self.criteria = criteria  # for now, received power
        self.cell_size = cell_size  # size of one side of a cell, in meters
        self.log = log  # if true, prints information about the ongoing process
        self.default_base_station = base_station  # BaseStation class variable

        self.ue = None  # the user equipment object - position and technical characteristics

        self.base_station_list = []

        # calculated variables
        self.azi_map = None
        self.elev_map = None
        self.gain_map = None
        self.ch_gain_map = None
        self.sector_map = None
        self.path_loss_map = None
        self.rx_pw_map = None
        self.snr_map = None
        self.cap_map = None
        self.cluster = None

    def set_base_station(self, base_station):  # simple function, but will include sectors and MIMO in the future
        self.default_base_station = base_station

    def generate_base_station_list(self, n_centers):
        # generating copies for different base station configurations
        self.base_station_list = []
        self.n_centers = n_centers
        for i in range(self.n_centers):
            self.base_station_list.append(copy.deepcopy(self.default_base_station))

    def set_ue(self, hrx):
        ue = User_eq(positions=self.grid.grid, height=hrx)  # creating the user equipament object
        self.ue = ue
        # self.rx_height = rx_height

    def generate_bf_gain_maps(self, az_map, elev_map, dist_map):
        # ue = np.empty(shape=(self.n_centers, elev_map.shape[1], 100))  # ARRUMAR DEPOIS ESSA GAMBIARRA
        # ue[:] = np.nan
        ue = np.empty(shape=(self.n_centers, elev_map.shape[1], self.base_station_list[0].antenna.beams))
        self.ch_gain_map = np.zeros(shape=(self.n_centers, elev_map.shape[1], self.base_station_list[0].antenna.beams + 1)) - 10000
        self.sector_map = np.ndarray(shape=(self.n_centers, elev_map.shape[1]))

        # path loss attenuation to sum with the beam gain
        att_map = generate_path_loss_map(eucli_dist_map=dist_map, cell_size=self.cell_size, prop_model=self.prop_model,
                                         frequency=self.base_station_list[0].frequency,  # todo
                                         htx=self.default_base_station.tx_height, hrx=1.5)  # LEMBRAR DE TORNAR O HRX EDITÁVEL AQUI!!!

        for bs_index, base_station in enumerate(self.base_station_list):
            lower_bound = 0
            for sector_index, higher_bound in enumerate(base_station.sectors_phi_range):
                ue_in_range = np.where((az_map[bs_index] > lower_bound) & (az_map[bs_index] <= higher_bound))
                sector_gain_map = generate_bf_gain(elevation_map=np.expand_dims(elev_map[bs_index][ue_in_range], axis=0),
                                                   azimuth_map=np.expand_dims(az_map[bs_index][ue_in_range], axis=0),
                                                   base_station_list=[base_station],
                                                   sector_index=sector_index)[0][0]
                ue[bs_index][ue_in_range, 0:sector_gain_map.shape[0]] = sector_gain_map.T
                self.sector_map[bs_index][ue_in_range] = sector_index
                self.ch_gain_map[bs_index][ue_in_range, 0:sector_gain_map.shape[0]] = (sector_gain_map - att_map[bs_index][ue_in_range]).T
                lower_bound = higher_bound

        return self.ch_gain_map, self.sector_map

    def send_ue_to_bs(self, simulation_time, time_slot):
        # set random activation indexes for all the BSs
        for bs_index, base_station in enumerate(self.base_station_list):
            ue_in_bs = np.where(self.ue.ue_bs[:, 0] == bs_index)

            for sector_index in range(base_station.n_sectors):
                ue_in_sector = np.where(self.ue.sector_map[bs_index] == sector_index)
                ue_in_bs_and_sector = np.intersect1d(ue_in_bs, ue_in_sector)
                ue_in_bs_sector_and_beam = self.ue.ue_bs[ue_in_bs_and_sector, 1]
                [beams, users_per_beams] = np.unique(ue_in_bs_sector_and_beam, return_counts=True)

                base_station.add_active_beam(beams=beams.astype(int), sector=sector_index, n_users=users_per_beams)
            base_station.generate_beam_timing(simulation_time, time_slot)  # precalculating the beam activation timings
            base_station.generate_beam_bw()  # calculating the bw for each active beam user
        return

    def place_and_configure_bs(self, n_centers, output_typ='raw', predetermined_centroids=None):
        if predetermined_centroids is not None:
            self.cluster = K_Means_XP(k=n_centers)
            self.cluster.fit(data=self.grid.grid, predetermined_centroids=predetermined_centroids)
        else:
            self.cluster = Cluster()
            self.cluster.k_means(grid=self.grid.grid, n_clusters=n_centers)
        lines = self.grid.lines
        columns = self.grid.columns
        az_map = generate_azimuth_map(lines=lines, columns=columns, centroids=self.cluster.centroids,
                                      samples=self.cluster.features)
        dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=self.cluster.centroids,
                                               samples=self.cluster.features, plot=False)
        elev_map = generate_elevation_map(htx=30, hrx=1.5, d_euclid=dist_map, cell_size=self.cell_size, samples=None)
        self.default_base_station.beam_configuration(
            az_map=self.default_base_station.beams_pointing)  # creating a beamforming configuration pointing to the the az_map points

        # =============================================================================================

        self.generate_base_station_list(n_centers)
        self.generate_bf_gain_maps(az_map=az_map, elev_map=elev_map, dist_map=dist_map)
        self.ue.acquire_bs_and_beam(ch_gain_map=self.ch_gain_map,
                                     sector_map=self.sector_map)  # calculating the best ch gain for each UE
        self.send_ue_to_bs(simulation_time=1000, time_slot=1)

        snr_cap_stats = self.simulate_ue_bs_comm(ch_gain_map=self.ch_gain_map, output_typ=output_typ)

        return (snr_cap_stats)

    def simulate_ue_bs_comm(self, ch_gain_map, output_typ='raw'):
        cap = np.zeros(shape=(self.ue.ue_bs.shape[0], self.base_station_list[0].beam_timing_sequence.shape[1]))
        snr = np.zeros(shape=(self.ue.ue_bs.shape[0], self.base_station_list[0].beam_timing_sequence.shape[1]))
        user_time = np.zeros(shape=(self.ue.ue_bs.shape[0], self.base_station_list[0].beam_timing_sequence.shape[1]))
        user_bw = np.zeros(shape=(self.ue.ue_bs.shape[0], self.base_station_list[0].beam_timing_sequence.shape[1]))
        act_beams_nmb = np.zeros(shape=(self.base_station_list.__len__(), self.base_station_list[0].beam_timing_sequence.shape[1]))
        user_per_bs = np.zeros(shape=(self.base_station_list.__len__(), self.base_station_list[0].beam_timing_sequence.shape[1]))

        snr[:] = np.nan
        cap[:] = np.nan
        user_time[:] = np.nan
        user_bw[:] = np.nan
        act_beams_nmb[:] = np.nan
        user_per_bs[:] = np.nan

        # to calculate noise power
        k = 1.380649E-23  # Boltzmann's constant (J/K)
        t = 290  # absolute temperature

        for time_index, _ in enumerate(self.base_station_list[0].beam_timing_sequence.T):
            #check the active Bs's in time_index
            for bs_index, base_station in enumerate(self.base_station_list):
                ue_in_active_beam = np.where((self.ue.ue_bs[:, 0] == bs_index)
                                             & (self.ue.ue_bs[:, 1] == base_station.beam_timing_sequence[self.ue.ue_bs[:, 2], time_index]))[0]
                pw_in_active_ue = base_station.tx_power + ch_gain_map[bs_index][ue_in_active_beam, self.ue.ue_bs[ue_in_active_beam, 1]]

                # print("main power ",pw_in_active_ue)
                interf_in_active_ue = 0
                # interference calculation
                for bs_index2, base_station2 in enumerate(self.base_station_list):
                    if bs_index2 != bs_index:
                        interf = base_station.tx_power + \
                                 ch_gain_map[bs_index2][ue_in_active_beam, base_station2.beam_timing_sequence[self.ue.sector_map[bs_index2, ue_in_active_beam], time_index]]
                        interf_in_active_ue += 10**(interf/10)
                        # print("interf ",interf)
                        # print("interf total ",interf_in_active_ue)
                # print("time slot:", time_index)
                # print("snr (dB)", 10*np.log10(10**(pw_in_active_ue/10)/interf_in_active_ue))
                bw = base_station.beam_bw[base_station.beam_timing_sequence[
                                              self.ue.sector_map[bs_index, ue_in_active_beam], time_index],
                                          self.ue.sector_map[bs_index, ue_in_active_beam]]  # user BW
                noise_power = k * t * bw * 10E6
                interf_in_active_ue += noise_power

                # metrics
                snr[ue_in_active_beam, time_index] = 10*np.log10(10**(pw_in_active_ue/10)/interf_in_active_ue)
                cap[ue_in_active_beam, time_index] = bw * 10E6 * np.log2(1+10**(pw_in_active_ue/10)/interf_in_active_ue)/(10E6)
                user_time[ue_in_active_beam, time_index] = 1
                user_bw[ue_in_active_beam, time_index] = bw
                act_beams_nmb[bs_index, time_index] = np.mean(np.count_nonzero(base_station.active_beams, axis=0))
                user_per_bs[bs_index, time_index] = np.sum(base_station.active_beams)



        # preparing output data
        mean_snr = 10*np.log10(np.nansum(10**(snr/10), axis=1))
        cap_sum = np.nansum(cap, axis=1)/(self.base_station_list[0].beam_timing_sequence.shape[1])
        mean_act_beams = np.mean(act_beams_nmb, axis=1)
        mean_user_bs = np.mean(user_per_bs, axis=1)
        user_time = np.nansum(user_time, axis=1) / (self.base_station_list[0].beam_timing_sequence.shape[1])
        positions = [np.round(self.cluster.centroids).astype(int)]


        # simple stats data
        mean_mean_snr = np.mean(mean_snr)
        std_snr = np.std(mean_snr)
        # min_mean_snr = np.min(mean_mean_snr)
        # max_mean_snr = np.max(mean_mean_snr)
        mean_cap = np.mean(cap_sum)
        std_cap = np.std(cap_sum)
        # min_mean_cap = np.min(cap_sum)
        # max_mean_cap = np.max(cap_sum)
        mean_user_time = np.mean(user_time)
        std_user_time = np.std(user_time)
        # min_user_time = np.min(user_time)
        # max_user_time = np.max(user_time)
        mean_user_bw = np.nanmean(user_bw)
        std_user_bw = np.nanstd(user_bw)
        # min_user_bw = np.min(user_bw)
        # max_user_bw = np.max(user_bw)
        #FAZER AS OUTRAS MÉTRICAS !!!!

        meet_criteria = None
        if self.criteria != None:
            meet_criteria = np.sum(cap_sum >= self.criteria)/cap_sum.shape[0]

        if meet_criteria or meet_criteria == 0:
            snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw, std_user_bw, meet_criteria]
        else:
            snr_cap_stats = [mean_mean_snr, std_snr, mean_cap, std_cap, mean_user_time, std_user_time, mean_user_bw,
                             std_user_bw]

        # preparing 'raw' data to export
        raw_data_dict = {'position': positions,'snr': mean_snr, 'cap': cap_sum, 'user_bs': mean_user_bs, 'act_beams': mean_act_beams,
                         'user_time': user_time, 'user_bw': np.nanmean(user_bw, axis=1)}

        if output_typ == 'simple':
            return snr_cap_stats
        if output_typ == 'complete':
            return snr_cap_stats, raw_data_dict
        if output_typ == 'raw':
            return raw_data_dict

    def adjust_weights(self, max_iter):  # NOT USED (FOR NOW)
        fulfillment = False
        more_cells = False

        path_loss_map = generate_path_loss_map(eucli_dist_map=self.voronoi.dist_mtx, centroids=self.voronoi.n_centers,
                                               cell_size=self.cell_size, prop_model='fs', frequency=2.8,
                                               htx=self.tx_height, hrx=self.rx_height)

        rx_power_map = self.tx_power - path_loss_map

        if self.log:
            print('Generated ', self.voronoi.n_centers, 'reception power maps')

        # main loop - increases cell number and changes cell size when criteria is not match
        while not more_cells:
            # checking the fulfillment of the criteria
            counter = 0
            total_points = len(self.grid.grid[0]) * len(self.grid.grid[1])
            total_coverate_perc = np.zeros(shape=(max_iter))

            while not fulfillment:
                counter += 1
                fulfill_criteria_abs = np.zeros(shape=(self.voronoi.n_centers))
                fulfill_criteria_perc = np.zeros(shape=(self.voronoi.n_centers))

                total_coverage_abs = 0

                for i in range(self.voronoi.n_centers):
                    indices = np.where(self.voronoi.power_voronoi_map == i)
                    total_points_centroid = len(indices[0])
                    # indices = list(zip(indices[0], indices[1]))
                    x = len(np.where(rx_power_map[i][indices[0], indices[1]] > self.criteria)[0])
                    fulfill_criteria_abs[i] = len(np.where(rx_power_map[i][indices[0], indices[1]] > self.criteria)[0])
                    fulfill_criteria_perc[i] = fulfill_criteria_abs[i]/total_points_centroid
                    total_coverage_abs += fulfill_criteria_abs[i]

                total_coverate_perc[counter - 1] = total_coverage_abs/total_points

                if not (fulfill_criteria_perc[fulfill_criteria_perc > 0.85]).size == self.voronoi.n_centers:  # checking if the criteria is met for all cells
                    new_weights = np.zeros(shape=(self.voronoi.n_centers))
                    for i, weight in enumerate(self.voronoi.weights):  # changing all weights
                        new_weight = gauss(mu=weight, sigma=weight/10)
                        new_weights[i] = new_weight
                    self.voronoi.generate_power_voronoi(weights=new_weights)  # calculating new voronoi with the new weights
                else:
                    if self.log:
                        print('Criteria was met in ', counter, 'iterations')
                        print(fulfill_criteria_perc)
                        print((fulfill_criteria_perc[fulfill_criteria_perc > 0.85]).size)
                    return more_cells, total_coverate_perc  # return false when the criteria is met

                if counter >= max_iter:
                    more_cells = True
                    if self.log:
                        print('Criteria was not met. Thought ', counter, 'iterations, the best result was ', max(total_coverate_perc))
                    return more_cells, total_coverate_perc


    def adjust_cell_number(self, min_n_cell, max_iter, max_n_cell=None, default_cells=None, processes=-1, log=False, log_lv=3):
        more_cells = True  # variable to change value when a desired criteria is met
        if default_cells is not None:
            n_cells = default_cells.shape[0] + min_n_cell
        else:
            n_cells = min_n_cell  # minimum cell number
        lines = self.grid.lines  # grid line size
        columns = self.grid.columns  # grid column size
        weight_norm_param = 15  # not using wights, for now
        max_iter = max_iter  # maximum number of iterations per BS number
        total_coverage_perc = []
        snr_samples_hist = []
        snr_grid_hist = []
        cap_samples_hist = []
        cap_grid_hist = []
        n_cells_hist = []

        if self.log:
            print('Initializing execution...')
            print(max_iter, 'iterations per cluster process')

        self.generate_base_station_list(n_cells=n_cells)
        # self.base_station_list = []
        # for i in range(n_cells):
        #     self.base_station_list.append(self.default_base_station)  # generating copies for different base station configurations

        # creating a poll for multithreading
        if processes == -1:  # to use all available cores -1 (using all threads may cause system instability)
            threads = os.cpu_count()
            p = multiprocessing.Pool(processes=threads-1)
        else:
            p = multiprocessing.Pool(processes=processes)

        # executing the function (optmize_cell) that will sample and calcullate and optimize the macro cell performance
        # it will return the performance and the position of each BS per iteration in data in list form
        data = list(
            tqdm.tqdm(p.imap_unordered(self.optimize_cell, [(n_cells, default_cells) for i in range(max_iter)]), total=max_iter
        ))

        data = np.array(data)
        data = np.transpose(data)
        perf_list = data[0]
        centroids_list = data[1]
        # sample_list = data[2]
        # samples_dist = np.sum(sample_list, axis=0)
        # plt.matshow(samples_dist)
        # plt.colorbar()
        # plt.show()

        map_data = np.zeros(shape=(lines, columns))
        map_data2 = np.zeros(shape=(lines, columns))
        for i, centroids in enumerate(centroids_list):
            for j, centroid in enumerate(centroids):
                if j > 1:
                    map_data[int(np.round(centroid[0], 0)), int(np.round(centroid[1], 0))] += perf_list[i]/10E6
                    map_data2[int(np.round(centroid[0], 0)), int(np.round(centroid[1], 0))] += 1


        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(lines), range(columns))
        ax.plot_surface(X, Y, map_data)
        ax.scatter3D([100, 400], [100, 400], [10, 10], color='black', alpha=0.8, marker='x')
        plt.show()

        plt.matshow(map_data.T, origin='lower')
        plt.set_cmap('RdYlGn')
        plt.colorbar()
        plt.scatter([100, 400], [100, 400])
        plt.show()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(lines), range(columns))
        ax.plot_surface(X, Y, map_data2)
        ax.scatter([100, 400], [100, 400], [0, 0])
        plt.show()

        plt.matshow(map_data2.T,origin='lower')
        plt.colorbar()
        plt.scatter([100, 400], [100, 400])
        plt.show()

        map_data_norm = np.where(map_data!=0, map_data/map_data2, 0)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        X, Y = np.meshgrid(range(lines), range(columns))
        ax.plot_surface(X, Y, map_data_norm)
        ax.scatter([100, 400], [100, 400], [10, 10])
        plt.show()

        plt.matshow(map_data_norm.T, origin='lower')
        plt.colorbar()
        plt.scatter([100, 400], [100, 400])
        plt.show()

    def optimize_cell(self, args):
        grid = self.grid
        # grid = args[0]
        n_cells = args[0]
        default_cells = args[1]
        # print(default_cells)
        lines = grid.lines
        columns = grid.columns
        grid.clear_grid()
        grid.make_points(dist_type='gaussian', samples=20, n_centers=self.n_centers, random_centers=False, plot=False)
        # cluster = Cluster()
        # cluster.gaussian_mixture_model(grid=grid.grid, n_clusters=n_cells, plot=False)
        # exotic cluster execution
        cluster = K_Means(k=4)
        cluster.fit(data=grid.grid, predetermined_centroids=np.array([[100, 100], [400, 400]]))
        # cluster.predict()
        # cluster.plot()

        # JUST A TEST!!!
        if hasattr(base_station_list[0].antenna, 'beamforming_id'):
            pass

        azi_map = generate_azimuth_map(lines=lines, columns=columns, centroids=cluster.centroids,
                                        samples=cluster.features)
        dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=cluster.centroids,
                                               samples=cluster.features, plot=False)
        elev_map = generate_elevation_map(htx=self.default_base_station.tx_height, hrx=self.rx_height,
                                            d_euclid=dist_map, cell_size=self.cell_size, samples=None)
        path_loss_map = generate_path_loss_map(eucli_dist_map=dist_map, cell_size=self.cell_size,
                                                prop_model='fs', frequency=self.default_base_station.frequency,
                                                htx=self.default_base_station.tx_height, hrx=self.rx_height,
                                                samples=None)

        gain_map = generate_gain_map(antenna=self.default_base_station.antenna, elevation_map=elev_map,
                                          azimuth_map=azi_map, base_station_list=self.base_station_list)

        rx_pw_map = generate_rx_power_map(path_loss_map=path_loss_map, azimuth_map=azi_map,
                                            elevation_map=elev_map, base_station=self.default_base_station,
                                            gain_map=gain_map)
        snr_map, snr_map_uni, snr_grid = generate_snr_map(base_station=self.default_base_station,
                                                            rx_power_map=rx_pw_map, unified=True)
        cap_map, cap_grid_criteria, cap_grid = generate_capcity_map(snr_map=snr_map,
                                                                    bw=self.default_base_station.bw,
                                                                    threshold=self.criteria, unified=False)

        if self.log:
            # print('using ', n_cells, ' with weights: ', weights, 'throught ', max_iter, ' iterations')
            print('using ', n_cells, 'standard voronoi cells')
        cap = np.mean(cap_map)
        # print('cap samples: ' + str(np.round(cap_samples/10E6, decimals=2)) + ' Mbps')
        # print('cap grid: ' + str(np.round(cap_grid/10E6, decimals=2)) + ' Mbps')
        # print('cap samples: ' + str(np.round(np.mean(cap_map) / 10E6, decimals=2)) + ' Mbps')
        return cap, cluster.centroids  #, grid.grid


    def adjust_elev_pattern(self, min_tilt, max_tilt, samples):
        import warnings
        warnings.filterwarnings('ignore')
        # downtilts = np.asarray(list(combinations_with_replacement(np.arange(min_tilt, max_tilt, 5), 3)))  # all possible tilts combinations
        downtilts = np.asarray(list(itertools.product(np.arange(min_tilt, max_tilt + 1, 1), repeat=3)))
        mean_perf = np.ndarray(shape=downtilts.shape[0])

        # adjusting the samples to the calcullated voronoi cells
        # samples = np.where(self.voronoi.std_voronoi_map[samples[samples, 0], samples[samples, 1]] == i]

        for i, bs in enumerate(self.base_station_list):
            for j, downtilt in enumerate(downtilts):
                bs.change_downtilt(downtilt)
                user_samples = np.where(self.voronoi.std_voronoi_map[samples[:, 0], samples[:, 1]] == i)
                user_gain_map = generate_gain_map(antenna=bs.antenna, elevation_map=self.elev_map[i][samples[user_samples, 0], samples[user_samples, 1]],
                                             azimuth_map=self.azi_map[i][samples[user_samples, 0], samples[user_samples, 1]], sectors_ver_pattern=bs.sectors_ver_pattern,
                                             sectors_hor_pattern=bs.sectors_hor_pattern)
                # user_rx_pw_map = generate_rx_power_map(path_loss_map=self.path_loss_map[i][samples[user_samples, 0], samples[user_samples, 1]],
                #                                        azimuth_map=self.azi_map[i][samples[user_samples, 0], samples[user_samples, 1]],
                #                                        elevation_map=self.elev_map[i][samples[user_samples, 0], samples[user_samples, 1]],
                #                                        base_station=bs, gain_map=user_gain_map)
                interf_samples = np.where(self.voronoi.std_voronoi_map[samples[:, 0], samples[:, 1]] != i)
                interf_gain_map = generate_gain_map(antenna=bs.antenna, elevation_map=self.elev_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
                                             azimuth_map=self.azi_map[i][samples[interf_samples, 0], samples[interf_samples, 1]], sectors_ver_pattern=bs.sectors_ver_pattern,
                                             sectors_hor_pattern=bs.sectors_hor_pattern)
                # interf_rx_pw_map = generate_rx_power_map(path_loss_map=self.path_loss_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
                #                                        azimuth_map=self.azi_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
                #                                        elevation_map=self.elev_map[i][samples[interf_samples, 0], samples[interf_samples, 1]],
                #                                        base_station=bs, gain_map=interf_gain_map)


                user_gain_map_unified = np.max(10**(user_gain_map/10), axis=0)
                interf_gain_map_unified = np.max(10**(interf_gain_map/10), axis=0)
                mean_perf[j] = np.mean(user_gain_map_unified) #- np.mean(interf_gain_map_unified)

                # user_pw_map_unified = np.max(10**(user_rx_pw_map/10), axis=0)
                # interf_pw_map_unified = np.max(10**(interf_rx_pw_map/10), axis=0)
                # mean_perf[j] = np.mean(user_pw_map_unified) - np.mean(interf_pw_map_unified)

            best_downtilt = np.argmax(mean_perf)
            print(downtilts[best_downtilt])
            # if best_downtilt != 0:
            #     print(mean_perf)
            #     print(best_downtilt)
            #     plt.plot(mean_perf)
            #     plt.show()
            bs.change_downtilt(downtilts[best_downtilt])
            self.base_station_list[i] = bs

        warnings.filterwarnings('default')



