import itertools
import multiprocessing.pool
import os
import tqdm
import numpy as np
import pandas as pd
import copy
from models.propagation.prop_models import generate_path_loss_map, generate_elevation_map, generate_azimuth_map, generate_gain_map, \
    generate_rx_power_map, generate_snr_map, generate_capcity_map, generate_euclidian_distance, generate_bf_gain
from user_eq import User_eq
from random import gauss
from clustering import Cluster
import matplotlib.pyplot as plt
from demos_and_examples.kmeans_from_scratch import K_Means_XP


class Macro_net_uplink:
    def __init__(self, grid, prop_model, cell_size, base_station, simulation_time, time_slot, t_min=None,
                 criteria=None, scheduler_typ=None, log=False):

        self.grid = grid  # grid object - size, points, etc
        self.n_centers = None
        self.voronoi = None  # voronoi object - voronoi cells, distance matrix, voronoi maps, etc
        self.prop_model = prop_model  # string - name of prop model to be used in prop_models
        self.criteria = criteria  # for now, received power
        self.cell_size = cell_size  # size of one side of a cell, in meters
        self.log = log  # if true, prints information about the ongoing process
        self.default_base_station = base_station  # BaseStation class variable
        # self.scheduling_opt = scheduling_opt  # to choose if the optimized scheduling is to be used
        self.simulation_time = simulation_time  # number of time slots (ms)
        self.time_slot = time_slot  # size of the time slot (ms)
        self.scheduler_typ = scheduler_typ  # RR (round-robin), prop-cmp (proposed complete), prop-smp (proposed simplified) or BCQI (Best Channel Quality Indicator)

        if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
            self.t_min = t_min  # minimum per beam allocated time if schdl opt is used

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

    def generate_base_station_list(self, n_centers, scheduler_typ):
        # generating copies for different base station configurations
        self.base_station_list = []
        self.n_centers = n_centers
        for i in range(self.n_centers):
            self.base_station_list.append(copy.deepcopy(self.default_base_station))

        # setting the scheduler for each bs
        for bs_index, bs in enumerate(self.base_station_list):
            bs.initialize_scheduler(scheduler_typ=scheduler_typ, time_slot=self.time_slot, t_min=self.t_min,
                                              simulation_time=self.simulation_time, bs_index=bs_index, c_target=self.criteria)

    def set_ue(self, hrx):
        ue = User_eq(positions=self.grid.grid, height=hrx)  # creating the user equipament object
        self.ue = ue
        # self.rx_height = rx_height

    def generate_ue_gain_maps(self):
        lines = self.grid.lines
        columns = self.grid.columns
        # az_map = generate_azimuth_map(lines=lines, columns=columns, centroids=self.cluster.features,
        #                               samples=self.cluster.features)
        dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=self.cluster.features,
                                               samples=self.cluster.features, plot=False)
        # elev_map = generate_elevation_map(htx=30, hrx=1.5, d_euclid=dist_map, cell_size=self.cell_size, samples=None)

        # path loss attenuation to sum with the beam gain
        att_map = generate_path_loss_map(eucli_dist_map=dist_map, cell_size=self.cell_size, prop_model=self.prop_model,
                                         frequency=self.base_station_list[0].frequency,  # todo
                                         htx=1.5,
                                         hrx=1.5)  # LEMBRAR DE TORNAR O HRX EDITÁVEL AQUI !!!

        # todo - especificar densidade espectral de potência dos ue e ver essa eq
        rx_max_pwr = 0
        ue_bw = 100
        pw_in_5mghz = rx_max_pwr + 10 * np.log10(5 / ue_bw)
        ue_in_ue_range = pw_in_5mghz + 30 - att_map > -100
        pw_in_5mghz[ue_in_ue_range] = None


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