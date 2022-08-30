import numpy as np
import pandas as pd
import copy
from models.propagation.prop_models import generate_path_loss_map, generate_elevation_map, generate_azimuth_map, generate_gain_map, \
    generate_rx_power_map, generate_snr_map, generate_capcity_map, generate_euclidian_distance, generate_bf_gain
from models.scheduler.master_scheduler import Master_scheduler
from user_eq import User_eq
from random import gauss
from clustering import Cluster
from util.metrics import Metrics
import matplotlib.pyplot as plt
from demos_and_examples.kmeans_from_scratch import K_Means_XP


class Macel:
    def __init__(self, grid, prop_model, cell_size, base_station, simulation_time, time_slot, t_min=None, bw_slot=None,
                 criteria=None, scheduler_typ=None, log=False, downlink_specs=None, uplink_specs=None,
                 output_type="complete",tdd_up_time=0):

        self.grid = grid  # grid object - size, points, etc
        self.n_centers = None
        self.voronoi = None  # voronoi object - voronoi cells, distance matrix, voronoi maps, etc
        self.prop_model = prop_model  # string - name of prop model to be used in prop_models
        self.criteria = criteria  # for now, received power
        self.cell_size = cell_size  # size of one side of a cell, in meters
        self.log = log  # if true, prints information about the ongoing process
        self.default_base_station = base_station  # BaseStation class variable
        self.simulation_time = simulation_time  # number of time slots (ms)
        self.time_slot = time_slot  # size of the time slot (ms)
        self.scheduler_typ = scheduler_typ  # RR (round-robin), prop-cmp (proposed complete), prop-smp (proposed simplified) or BCQI (Best Channel Quality Indicator)
        self.output_type = output_type  # type of the outyput to be store after a simulation in metrics class
        self.metrics = Metrics  # metrics class - to store and process simullation data
        self.downlink_specs = downlink_specs
        self.uplink_specs = uplink_specs
        if uplink_specs is not None:
            self.tdd_up_time = tdd_up_time

        # if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
        #     self.t_min = t_min  # minimum per beam allocated time if schdl opt is used

        self.t_min = t_min  # minimum per beam allocated time if schdl opt is used (prop smp or prop cmp)
        self.bw_slot = bw_slot # slot fixed bandwidth for scheduller with a queue (RR)

        self.ue = None  # the user equipment object - position and technical characteristics

        self.base_station_list = []

        # calculated variables # todo - rever o que manter aqui
        self.ue_bs_table = None
        self.azi_map = None
        self.elev_map = None
        self.gain_map = None
        self.dist_map =  None
        self.ch_gain_map = None
        self.sector_map = None
        self.path_loss_map = None
        self.rx_pw_map = None
        self.snr_map = None
        self.cap_map = None
        self.cluster = None

        self.dwn_rel_t_index = None # todo VER SE VAI USAR ISSO AQUI !!!!
        self.dwn_elapsed_time = None
        self.up_elapsed_time = None

    def set_base_station(self, base_station):  # simple function, but will include sectors and MIMO in the future
        self.default_base_station = base_station

    def generate_base_station_list(self, n_centers, up_tdd_time=0):
        # generating copies for different base station configurations
        self.base_station_list = []
        self.n_centers = n_centers
        for i in range(self.n_centers):
            self.base_station_list.append(copy.deepcopy(self.default_base_station))

        # setting the mux and scheduler for each bs
        for bs_index, bs in enumerate(self.base_station_list):
            bs.initialize_mux(simulation_time=self.simulation_time, up_tdd_time=up_tdd_time)
            # downlink_specs = {**self.downlink_specs, **{'bs_index': bs_index,
            #                                             'simulation_time':self.simulation_time,
            #                                             'tx_power': bs.tx_power}}
            # uplink_specs = {**self.uplink_specs, **{'bs_index': bs_index,
            #                                         'simulation_time':self.simulation_time,
            #                                         'tx_power': self.ue.tx_power}}
            bs.initialize_dwn_up_scheduler(downlink_specs={**self.downlink_specs, **{'bs_index': bs_index,
                                                           'simulation_time': self.simulation_time,
                                                           'tx_power': bs.tx_power, 'bw': bs.bw,
                                                           'time_slot': self.time_slot}},
                                           uplink_specs={**self.uplink_specs, **{'bs_index': bs_index,
                                                         'simulation_time': self.simulation_time,
                                                         'tx_power': self.ue.tx_power, 'bw': bs.bw,
                                                         'time_slot': self.time_slot}})
            # bs.initialize_scheduler(scheduler_typ=scheduler_typ, time_slot=self.time_slot, t_min=self.t_min,
            #                         simulation_time=self.simulation_time, bs_index=bs_index, c_target=self.criteria,
            #                         bw_slot=self.bw_slot)

    def set_ue(self, hrx=None, tx_power=None):
        if self.ue is not None:
            self.ue.positions = self.grid.grid
        else:
            ue = User_eq(positions=self.grid.grid, height=None, tx_power=None)  # creating the user equipament object
            self.ue = ue

        if hrx is not None:
            self.ue.height = hrx
        if tx_power is not None:
            self.ue.tx_power = tx_power


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

    def send_ue_to_bs(self, t_index=0, cap_defict=None, t_min=None, bs_2b_updt=None, updated_beams=None,
                      downlink=False, uplink=False):
        if cap_defict is None:
            if downlink:
                cap_defict = self.downlink_specs['criteria'] + np.zeros(shape=self.ue.dw_ue_bs.shape[0])
            if uplink:
                cap_defict = self.uplink_specs['criteria'] + np.zeros(shape=self.ue.dw_ue_bs.shape[0])

        if bs_2b_updt is None:
            bs_2b_updt = range(len(self.base_station_list))

        # set random activation indexes for all the BSs
        for bs_index, bs in enumerate(self.base_station_list):
            if bs_index in bs_2b_updt:
                bs.clear_active_beams(downlink=downlink, uplink=uplink)
                for sector_index in range(bs.n_sectors):
                    if downlink:
                        ue_in_bs_sector_and_beam = self.ue.dw_ue_bs[np.where((self.ue.dw_ue_bs[:, 0] == bs_index)
                                                                             & (self.ue.dw_ue_bs[:, 2] == sector_index)), 1]
                        [beams, users_per_beams] = np.unique(ue_in_bs_sector_and_beam, return_counts=True)
                    elif uplink:
                        ue_in_bs_sector_and_beam = self.ue.up_ue_bs[np.where((self.ue.up_ue_bs[:, 0] == bs_index)
                                                                             & (self.ue.up_ue_bs[:, 2] == sector_index)), 1]
                        [beams, users_per_beams] = np.unique(ue_in_bs_sector_and_beam, return_counts=True)

                    bs.add_active_beam(beams=beams.astype(int), sector=sector_index, n_users=users_per_beams,
                                       uplink=uplink, downlink=downlink)

        # SCHEDULING
        for bs_index, bs in enumerate(self.base_station_list):
            if bs_index in bs_2b_updt:  # todo - separar uplink e downlink e também refaser o bs_2b_updt
                if downlink:
                    bs.tdd_mux.dwn_scheduler.update_scheduler(active_beams=bs.dwn_active_beams, ue_bs=self.ue.dw_ue_bs,
                                                              t_index=t_index, c_target=cap_defict, ue_updt=True)
                elif uplink:
                    bs.tdd_mux.up_scheduler.update_scheduler(active_beams=bs.up_active_beams, ue_bs=self.ue.up_ue_bs,
                                                             t_index=t_index, c_target=cap_defict, ue_updt=True)

                # bs.tdd_mux.up_scheduler(active_beams=bs.active_beams, ue_bs=self.ue.ue_bs,
                #                                t_index=t_index, c_target=cap_defict, ue_updt=True)
            else:
                if downlink:
                    bs.tdd_mux.dwn_scheduler.update_scheduler(active_beams=bs.dwn_active_beams, ue_bs=self.ue.dw_ue_bs,
                                                              t_index=t_index, c_target=cap_defict, ue_updt=False,
                                                              updated_beams=updated_beams[bs_index])
                if uplink:
                    bs.tdd_mux.up_scheduler.update_scheduler(active_beams=bs.up_active_beams, ue_bs=self.ue.up_ue_bs,
                                                             t_index=t_index, c_target=cap_defict, ue_updt=False,
                                                             updated_beams=updated_beams[bs_index])

    def place_and_configure_bs(self, n_centers, predetermined_centroids=None, clustering=True):
        if clustering:
            if predetermined_centroids is not None:
                self.cluster = K_Means_XP(k=n_centers)
                self.cluster.fit(data=self.grid.grid, predetermined_centroids=predetermined_centroids)
            else:
                self.cluster = Cluster()
                self.cluster.k_means(grid=self.grid.grid, n_clusters=n_centers)
        else:
            if predetermined_centroids is not None:
                self.cluster = Cluster()
                self.cluster.scaling(grid=self.grid.grid)
                # self.cluster.features = self.cluster.set_features(grid=self.grid.grid)
                self.cluster.centroids = np.array(predetermined_centroids)
            else:
                self.cluster = Cluster()
                self.cluster.random(grid=self.grid.grid, n_clusters=n_centers)
        lines = self.grid.lines
        columns = self.grid.columns
        az_map = generate_azimuth_map(lines=lines, columns=columns, centroids=self.cluster.centroids,
                                      samples=self.cluster.features)
        self.dist_map = generate_euclidian_distance(lines=lines, columns=columns, centers=self.cluster.centroids,
                                               samples=self.cluster.features, plot=False)
        elev_map = generate_elevation_map(htx=30, hrx=1.5, d_euclid=self.dist_map, cell_size=self.cell_size, samples=None)
        self.default_base_station.beam_configuration(
            az_map=self.default_base_station.beams_pointing)  # creating a beamforming configuration pointing to the the az_map points

        # =============================================================================================

        # self.tdd_mux = Master_scheduler()  # todo - instanciar esta puerra
        # self.tdd_mux.create_tdd_scheduler(simulation_time=self.simulation_time, up_tdd_time=self.tdd_up_time)
        self.generate_base_station_list(n_centers=n_centers, up_tdd_time=self.tdd_up_time)
        self.generate_bf_gain_maps(az_map=az_map, elev_map=elev_map, dist_map=self.dist_map)

        self.ue.acquire_bs_and_beam(ch_gain_map=self.ch_gain_map,
                                     sector_map=self.sector_map,
                                    pw_5mhz=self.default_base_station.tx_power + 10*np.log10(5/self.default_base_station.bw))  # calculating the best ch gain for each UE

        self.metrics = Metrics()  # instancing the Metrics object
        if self.downlink_specs is not None and self.tdd_up_time != 1:
            self.metrics.store_downlink_metrics(n_ues=self.ue.dw_ue_bs.shape[0], n_bs=self.base_station_list.__len__(),
                                                simulation_time=self.simulation_time, time_slot=self.time_slot,
                                                criteria=self.downlink_specs['criteria'])  # initializing the downlink variables
            self.send_ue_to_bs(downlink=True)
        if self.uplink_specs is not None and self.tdd_up_time != 0:
            self.metrics.store_uplink_metrics(n_ues=self.ue.up_ue_bs.shape[0], n_bs=self.base_station_list.__len__(),
                                              simulation_time=self.simulation_time, time_slot=self.time_slot,
                                              criteria=self.uplink_specs['criteria'])  # initializing the uplink variables
            self.send_ue_to_bs(uplink=True)

        output = self.tdd_dwn_up_sim(output_typ=self.output_type)  # tdd scheduling that cals uplink and downlink simulations

        return output

    def tdd_dwn_up_sim(self, output_typ='complete'):
        tdd_scheduler = self.base_station_list[0].tdd_mux.tdd_scheduler  # picking any tdd scheduler because all BSs are sync
        index_list = np.where(np.ones(shape=tdd_scheduler.shape))[0]
        dwn_indexes = np.where(tdd_scheduler == 0)[0]
        up_indexes = np.where(tdd_scheduler == 1)[0]
        end_sim = False
        t_index = 0
        downlink_results = None
        uplink_results = None

        while not end_sim:
            boss = tdd_scheduler[t_index]
            # picking the next time to change between downlink and uplink
            next_boss_index = np.where((tdd_scheduler != boss) & (index_list > t_index))[0]
            if next_boss_index.size == 0:
                next_boss_index = tdd_scheduler.size
                end_sim = True
            else:
                next_boss_index = next_boss_index[0]
            scheduler_range = range(t_index, next_boss_index)
            t_index = next_boss_index
            if boss == 0:
                rel_index = np.where(np.isin(dwn_indexes, scheduler_range))[0]
                downlink_results = self.downlink_interference(ch_gain_map=self.ch_gain_map, output_typ=output_typ,
                                                              tdd_scheduler_range=scheduler_range,
                                                              rel_schdl_range=rel_index)  # downlink channel simulation
                # print(scheduler_range)
                # print('downlink')
            if boss == 1:
                rel_index = np.where(np.isin(up_indexes, scheduler_range))[0]
                uplink_results = self.uplink_interference(ch_gain_map=self.ch_gain_map, output_typ=output_typ,
                                                          tdd_scheduler_range=scheduler_range,
                                                          rel_schdl_range=rel_index)  # uplink channel simulation
                # print(scheduler_range)
                # print('uplink')

        # if downlink_results is not None and uplink_results is not None:
        return {'downlink_results': downlink_results, 'uplink_results': uplink_results}

    def uplink_interference(self, ch_gain_map, tdd_scheduler_range, rel_schdl_range, output_typ='complete'):
        if self.dwn_rel_t_index is None:
            self.dwn_rel_t_index = 0
        if self.ue_bs_table is None:  # this is a backup tab;e that stores the initial UE/BS association
            self.ue_bs_table = pd.DataFrame(copy.copy(self.ue.dw_ue_bs), columns=['bs_index', 'beam_index', 'sector_index', 'csi'])
        # ue_bs_table = pd.DataFrame(copy.copy(self.ue.up_ue_bs), columns=['bs_index', 'beam_index', 'sector_index', 'csi'])
        # elapsed_time = 0
        # count_satisfied_ue_old = 0
        rbw = 0.1  # the badwidth resolution
        n_ues = self.ue.up_ue_bs.shape[0]
        t_slot_ratio = self.simulation_time/self.time_slot

        # calculating the noise power for the selected rbw
        k = 1.380649E-23  # Boltzmann's constant (J/K)
        t = 290  # absolute temperature
        noise_power = k * t * rbw * 10E6

        if self.up_elapsed_time is None:
            self.up_elapsed_time = 0

        for tdd_t_index, time_index in enumerate(rel_schdl_range):
        # for time_index, _ in enumerate(self.base_station_list[0].scheduler.time_scheduler.beam_timing_sequence.T):
        #     v_time_index = time_index - elapsed_time  # virtual time index used after generating new beam timing sequence when needed
            v_time_index = time_index - self.up_elapsed_time

            # firstly, store all uplink Tx channels
            bs_rx_spectrum = []  # this list stores the received spectrum power per bs
            bs_occupied_spectrum = []  # this list stores ue index that uses the frequency per bs
            updated_beams = []  # this vector stores the schedulled beams in the time scheduler to inform the scheduller controler
            for bs_index, base_station in enumerate(self.base_station_list):
                # filtrating the active UEs in a active beam of a BS sector
                ue_in_active_beam = (self.ue.up_ue_bs[:, 0] == bs_index) & \
                                    (self.ue.up_ue_bs[:, 1] ==
                                     base_station.tdd_mux.up_scheduler.time_scheduler.beam_timing_sequence[
                                         self.ue.up_ue_bs[:, 2], v_time_index])
                active_ue_in_active_beam = np.where((base_station.tdd_mux.up_scheduler.freq_scheduler.user_bw !=0) &
                                                    ue_in_active_beam)

                updated_beams.append(base_station.tdd_mux.up_scheduler.time_scheduler.beam_timing_sequence[:, v_time_index])  # this stores the active beams in a time index to infor the scheduler

                ue_tx_pw = self.ue.tx_power + 10 * np.log10(rbw/base_station.tdd_mux.up_scheduler.bw)  # TODO TROCAR AQUI PELA POT DA UE (tb trocar pela densidade de potencia)

                pw_of_active_ue = ue_tx_pw + ch_gain_map[bs_index][active_ue_in_active_beam, self.ue.up_ue_bs[active_ue_in_active_beam, 1]][0]
                pw_of_active_ue = 10 ** (pw_of_active_ue/10)
                # here, we need to set the base frequency response for all UEs
                _bs_channels = np.zeros(shape=[base_station.n_sectors, int(base_station.bw // rbw)])
                _bs_occupied_spectrum = copy.copy(_bs_channels) - 1  # the -1 is to initialize the matrix with a unindexed value

                # here, we need to allocate virtual bandwidths for each of the UEs of the BS, for a BSs
                for sector_index in range(base_station.n_sectors):
                    ue_in_sector = active_ue_in_active_beam[0][self.ue.up_ue_bs[active_ue_in_active_beam[0], 2] == sector_index].astype(int)
                    bw_index = 0
                    bw = base_station.tdd_mux.up_scheduler.freq_scheduler.user_bw[ue_in_sector]
                    for ue_index, ue in enumerate(ue_in_sector):
                        _ue_bw = bw[ue_index]  # the bandwidth for a specified UE
                        n_sub_channes = (_ue_bw//rbw).astype(int)  # the number of spectrum matrix indices that will be occupied
                        _bs_channels[sector_index, range(bw_index, bw_index + n_sub_channes)] = pw_of_active_ue[ue_index]  # storing the power in the pw. spectrum
                        _bs_occupied_spectrum[sector_index, range(bw_index, bw_index + n_sub_channes)] = ue  # mapping where is needed to make interf. pw. sum (ue bandwidth)
                        bw_index += n_sub_channes

                bs_rx_spectrum.append(_bs_channels)  # it is appended for the case that different BSs have different BWs
                bs_occupied_spectrum.append(_bs_occupied_spectrum.astype(int))

                # ========= PLOTTER !!!!!!! (TO TEST!) =========
                # import matplotlib.pyplot as plt
                # for sector_index in range(base_station.n_sectors):
                #     ue_in_sector = active_ue_in_active_beam[0][self.ue.ue_bs[active_ue_in_active_beam[0], 2] == sector_index]
                #     # for ue_index, ue in enumerate(ue_in_sector):
                #     x = np.log10(bs_rx_spectrum[0][sector_index])
                #     x[np.isnan(x)] = -160
                #     plt.plot(x, label = 'ue ' + str(ue))
                #     plt.title('Sector ' + str(sector_index) + ' - ' + 'received power in dBW')
                #     plt.legend()
                #     plt.show()
                # =============================================

            # second, calculate the accumulated interference channels from the bs1 perspective
            interf_ue_channels = []
            for bs_index, base_station in enumerate(self.base_station_list):
                interf_ch_pbs = None
                for bs_index2, base_station2 in enumerate(self.base_station_list):
                    if bs_index2 != bs_index:
                        active_ue_in_active_beam2 = (self.ue.up_ue_bs[:, 0] == bs_index2) & \
                                                    (self.ue.up_ue_bs[:, 1] ==
                                                     base_station2.tdd_mux.up_scheduler.time_scheduler.beam_timing_sequence[
                                                         self.ue.up_ue_bs[:, 2], v_time_index])
                        active_ue_in_active_beam2 = np.where((base_station2.tdd_mux.up_scheduler.freq_scheduler.user_bw != 0) &
                                                             active_ue_in_active_beam2)

                        bw_pw = self.ue.tx_power + 10 * np.log10(rbw/base_station.tdd_mux.up_scheduler.bw)  # todo - arrumar aqui com a pot da UE

                        # interference channels calculated from the perspective of the active beams of the bs1
                        interf = bw_pw + ch_gain_map[bs_index][active_ue_in_active_beam2[0],
                                                               base_station.tdd_mux.up_scheduler.time_scheduler.beam_timing_sequence[
                                                            self.sector_map[bs_index2, active_ue_in_active_beam2[0]].astype(int), v_time_index]]  # channels from bs2 UEs to bs1
                        interf = 10**(interf/10)
                        # allocating the ues in bandwidth for each BS sector/antenna
                        interf_ch_pbs = np.zeros(shape=[base_station2.n_sectors, int(base_station.bw // rbw)])  # empty channel for all UEs from bs1 perspective
                        for sector_index in range(base_station.n_sectors):
                            ue_in_sector = np.array(np.where((self.sector_map[bs_index] == sector_index) & (self.ue.up_ue_bs[:, 0] == bs_index2)))
                            ue_in_sector = ue_in_sector[np.isin(ue_in_sector, active_ue_in_active_beam2)]

                            for ue_index in ue_in_sector:
                                ue_sector = self.ue.up_ue_bs[ue_index, 2]
                                _interf_channel = bs_occupied_spectrum[bs_index2][ue_sector] == ue_index  # mapping the occupied frequency for a UE in bs2
                                interf_ch_pbs[sector_index, _interf_channel] += interf[ue_index == active_ue_in_active_beam2[0]]  # storing the interference power in the mapped frequency

                if interf_ch_pbs is None:
                    interf_ch_pbs = np.zeros(shape=[base_station.n_sectors, int(base_station.bw // rbw)])  # special case when theres is no one interfering in the Tx channel
                interf_ch_pbs += noise_power  # adding the noise power
                interf_ue_channels.append(interf_ch_pbs)  # it is appended for the case that different BSs have different BWs

            # ========= PLOTTER 2  !!!!!!! (TO TEST!) =========
            # import matplotlib.pyplot as plt
            # for sector_index in range(base_station.n_sectors):
            #     x = 10 * np.log10(interf_ue_channels[0][sector_index])
            #     x[np.isnan(x)] = -160
            #     plt.plot(x)
            #     plt.title('Sector ' + str(sector_index) + ' - ' + 'received power in dBW')
            #     plt.legend()
            #     plt.show()
            # =============================================


            # finally, the SNIR/capacity calculation for in each BS sector, then separtes the parameters per UE
            bs_snr_spectrum = np.zeros(shape=[len(self.base_station_list), self.base_station_list[0].n_sectors, int(self.base_station_list[0].bw // rbw)])
            bs_cap_spectrum = copy.copy(bs_snr_spectrum)
            # snr and cap are one-liners that will store the capacity and the SNIR for one time index
            snr = np.zeros(shape=n_ues)
            snr.fill(np.nan)  # filling with NaN to avoid miscalculation
            cap = np.zeros(shape=n_ues)
            cap.fill(np.nan)
            for bs_index, base_station in enumerate(self.base_station_list):
                for sector_index in range(base_station.n_sectors):
                    rx_spectrum = bs_rx_spectrum[bs_index][sector_index]
                    interf_spectrum = interf_ue_channels[bs_index][sector_index]
                    bs_snr_spectrum[bs_index, sector_index] = rx_spectrum / interf_spectrum  # SNR = RX_pw/Interf
                    # bs_cap_spectrum[bs_index, sector_index] = np.sum(rbw * np.log2(1 + bs_snr_spectrum[bs_index, sector_index]))  # the summed capacity for a UEs bw
                    bs_cap_spectrum[bs_index, sector_index] = rbw * np.log2(1 + bs_snr_spectrum[bs_index, sector_index])  # the summed capacity for a UEs bw


                active_ue = np.unique(bs_occupied_spectrum[bs_index])
                active_ue = np.delete(active_ue, active_ue == -1)  # removing unindexed bandwidth
                for ue_index, ue in enumerate(active_ue):
                    active_beam_spectrum = bs_occupied_spectrum[self.ue.up_ue_bs[ue, 0]][self.ue.up_ue_bs[ue, 2]] == ue  # mapping the occupied frequency for a UE
                    snr[ue] = np.mean(bs_snr_spectrum[self.ue.up_ue_bs[ue, 0]][self.ue.up_ue_bs[ue, 2]][active_beam_spectrum])  # mean SNR for the mapped frequnecy
                    cap[ue] = np.sum(bs_cap_spectrum[self.ue.up_ue_bs[ue, 0]][self.ue.up_ue_bs[ue, 2]][active_beam_spectrum])  # summed capacity for all mapped frequencies
                    # cap[ue, time_index] = np.sum(rbw * np.log2(1 + ue_snr))  # todo checar aqui se aqui é em megabit ou em bit por segundo
                    # plt.plot(10 * np.log10(rx_spectrum))
                    # plt.show()
                    # plt.plot(10 * np.log10(interf_spectrum))
                    # plt.show()
                    # plt.plot(rbw * np.log2(1 + rx_spectrum/interf_spectrum))
                    # plt.show()

            # storing metrics
            self.metrics.store_uplink_metrics(cap=cap/t_slot_ratio, snr=snr/t_slot_ratio,
                                              t_index=tdd_scheduler_range[tdd_t_index],
                                              base_station_list=self.base_station_list,
                                              simulation_time=self.simulation_time, time_slot=self.time_slot)

            # self.metrics.store_uplink_metrics(cap=cap / t_slot_ratio, snr=snr / t_slot_ratio, t_index=time_index,
            #                                   base_station_list=self.base_station_list,
            #                                   simulation_time=self.simulation_time, time_slot=self.time_slot)

            bs_2b_updt = []
            # if count_satisfied_ue_old != self.metrics.up_cnt_satisfied_ue[time_index]:
            if time_index == 0:
                count_satisfied_ue_old = 0
            else:
                count_satisfied_ue_old = self.metrics.up_cnt_satisfied_ue[time_index - 1]
            if self.metrics.up_cnt_satisfied_ue[time_index] != count_satisfied_ue_old:
                bs_2b_updt = np.unique(self.ue.up_ue_bs[self.metrics.up_satisfied_ue, 0])  # is the BSs of the UEs that meet c_target
                bs_2b_updt = bs_2b_updt[bs_2b_updt >= 0]  # removing the all the UEs that already have been removed before
                count_satisfied_ue_old = copy.copy(self.metrics.up_cnt_satisfied_ue[time_index])
                self.ue.remove_ue(ue_index=self.metrics.up_satisfied_ue, uplink=True)  # removing selected UEs from the rest of simulation time
                # elapsed_time = time_index + 1  # VERIFICAR QUE AQUI TÁ ERRADO !!!!!!
                self.up_elapsed_time = time_index + 1
            # this command will redo the beam allocations and scheduling, if necessary
            self.send_ue_to_bs(t_index=time_index+1, cap_defict=self.metrics.up_cap_deficit, bs_2b_updt=bs_2b_updt,
                               updated_beams=updated_beams, uplink=True)

        if time_index == base_station.tdd_mux.up_scheduler.time_scheduler.simulation_time - 1:
            return(self.metrics.create_uplink_metrics_dataframe(output_typ=output_typ, active_ue=self.ue.active_ue,
                                                                cluster_centroids=[np.round(self.cluster.centroids).astype(int)],
                                                                ue_pos=self.cluster.features,
                                                                ue_bs_table=self.ue_bs_table,
                                                                dist_map=self.dist_map * self.cell_size,
                                                                scheduler_typ=self.scheduler_typ))

    def downlink_interference(self, ch_gain_map, tdd_scheduler_range, rel_schdl_range, output_typ='raw'):
        if self.dwn_rel_t_index is None:
            self.dwn_rel_t_index = 0
        self.sector_map = self.sector_map.astype(int)
        t_slot_ratio = self.simulation_time/self.time_slot
        if self.ue_bs_table is None:  # this is a backup tab;e that stores the initial UE/BS association
            self.ue_bs_table = pd.DataFrame(copy.copy(self.ue.dw_ue_bs), columns=['bs_index', 'beam_index', 'sector_index', 'csi'])

        # to calculate noise power
        k = 1.380649E-23  # Boltzmann's constant (J/K)
        t = 290  # absolute temperature

        if self.dwn_elapsed_time is None:
            self.dwn_elapsed_time = 0
        count_satisfied_ue_old = 0
        # for time_index, _ in enumerate(self.base_station_list[0].tdd_mux.dwn_scheduler.time_scheduler.beam_timing_sequence.T):
        for tdd_t_index, time_index in enumerate(rel_schdl_range):
            v_time_index = time_index - self.dwn_elapsed_time  # virtual time index used after generating new beam timing sequence when needed
            # print(v_time_index)
            snr = np.zeros(shape=self.ue.dw_ue_bs.shape[0])
            snr.fill(np.nan)
            cap = copy.copy(snr)
            #check the active Bs's in time_index
            updated_beams = []  # this vector stores the schedulled beams in the time scheduler to inform the scheduller controler
            for bs_index, base_station in enumerate(self.base_station_list):
                ue_in_active_beam = (self.ue.dw_ue_bs[:, 0] == bs_index) & \
                                    (self.ue.dw_ue_bs[:, 1] == base_station.tdd_mux.dwn_scheduler.time_scheduler.beam_timing_sequence[self.ue.dw_ue_bs[:, 2], v_time_index])
                active_ue_in_active_beam = np.where((base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw !=0) & ue_in_active_beam)

                updated_beams.append(base_station.tdd_mux.dwn_scheduler.time_scheduler.beam_timing_sequence[:, v_time_index])

                if base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw is None:  # uniform beam bw
                    bw = base_station.dwn_scheduler.freq_scheduler.beam_bw[base_station.tdd_mux.dwn_scheduler.time_scheduler.beam_timing_sequence[
                            self.sector_map[bs_index, active_ue_in_active_beam], v_time_index],  # AQUI OI
                        self.sector_map[bs_index, active_ue_in_active_beam]]  # user BW
                else:  # different bw for each user
                    bw = base_station.tdd_mux.dwn_scheduler.freq_scheduler.user_bw[active_ue_in_active_beam]

                pw_in_active_ue = base_station.tx_power + ch_gain_map[bs_index][active_ue_in_active_beam, self.ue.dw_ue_bs[active_ue_in_active_beam, 1]]
                interf_in_active_ue = 0
                # interference calculation
                for bs_index2, base_station2 in enumerate(self.base_station_list):
                    if bs_index2 != bs_index:
                        interf = base_station2.tx_power + \
                                 ch_gain_map[bs_index2][active_ue_in_active_beam,
                                                        base_station2.tdd_mux.dwn_scheduler.time_scheduler.beam_timing_sequence[
                                                            self.sector_map[bs_index2, active_ue_in_active_beam],
                                                            v_time_index]]  # AQUI OI
                        interf_in_active_ue += 10**(interf/10)

                noise_power = k * t * bw * 10E6
                interf_in_active_ue += noise_power

                snr[active_ue_in_active_beam] = (10 ** (pw_in_active_ue / 10)) / interf_in_active_ue
                cap[active_ue_in_active_beam] = bw * 10E6 * np.log2(1 + 10 ** (pw_in_active_ue / 10) / interf_in_active_ue) / (10E6)

            # self.dwn_rel_t_index += 1

            # storing metrics
            # print(tdd_scheduler_range[tdd_t_index])
            self.metrics.store_downlink_metrics(cap=cap / t_slot_ratio, snr=snr,
                                                t_index=tdd_scheduler_range[tdd_t_index],
                                                base_station_list=self.base_station_list,
                                                simulation_time=self.simulation_time, time_slot=self.time_slot)

            bs_2b_updt = []
            if count_satisfied_ue_old != self.metrics.dwn_cnt_satisfied_ue[time_index]:
                bs_2b_updt = np.unique(self.ue.dw_ue_bs[self.metrics.dwn_satisfied_ue, 0])  # is the BSs of the UEs that meet c_target
                bs_2b_updt = bs_2b_updt[bs_2b_updt >= 0]  # removing the all the UEs that already have been removed before
                count_satisfied_ue_old = copy.copy(self.metrics.dwn_cnt_satisfied_ue[time_index])
                self.ue.remove_ue(ue_index=self.metrics.dwn_satisfied_ue, downlink=True)  # removing selected UEs from the rest of simulation time
                self.dwn_elapsed_time = time_index + 1  # VERIFICAR QUE AQUI TÁ ERRADO !!!!!!
            # this command will redo the beam allocations and scheduling, if necessary
            self.send_ue_to_bs(t_index=time_index + 1, cap_defict=self.metrics.dwn_cap_deficit, bs_2b_updt=bs_2b_updt,
                               updated_beams=updated_beams, downlink=True)

        if time_index == base_station.tdd_mux.dwn_scheduler.time_scheduler.simulation_time-1:
            return(self.metrics.create_downlink_metrics_dataframe(output_typ='complete', active_ue=self.ue.active_ue,
                                                                  cluster_centroids=[np.round(self.cluster.centroids).astype(int)],
                                                                  ue_pos=self.cluster.features,
                                                                  ue_bs_table=self.ue_bs_table,
                                                                  dist_map=self.dist_map * self.cell_size,
                                                                  scheduler_typ=self.scheduler_typ))