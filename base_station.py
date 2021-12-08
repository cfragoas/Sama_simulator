import numpy as np
import matplotlib.pyplot as plt
import copy, warnings

# BaseStation Class represents the relationship between the station resources (antenna and transmission characteristics)
# and the space and users around it

class BaseStation:
    def __init__(self, frequency, tx_power, tx_height, bw, n_sectors, antenna, gain, downtilts, plot=False):  # simple function, but will include sectors and MIMO in the future
        self.frequency = frequency
        self.tx_power = tx_power  # tx power in dBW
        self.tx_height = tx_height
        self.bw = bw/n_sectors
        self.n_sectors = n_sectors
        self.sector_bw = self.bw / self.n_sectors

        if np.size(downtilts) == 1:  # if the downtilts variable is not array type
            downtilts = np.zeros(shape=n_sectors) + downtilts  # setting all the array values the same
        self.downtilts = downtilts  # this variable must have the shape of the same size of n_sectors

        self.antenna = antenna  # antenna object

        # initializing calculated variables
        self.sectors_pointing = None
        if not hasattr(self.antenna, 'beamforming_id'):
            self.sectors_hor_pattern = []  # rotated antenna patterns for each of the base station sectors
            self.sectors_ver_pattern = []  # tilted elevation pattern

            self.generate_ant_pattern()
            self.generate_sector_pattern(plot)
        else:
            self.beam_sector_pattern = []
            self.active_beams = None
            self.active_beams_index = None
            self.beam_timing = None
            self.beam_timing_sequence = None
            self.beam_bw = None
            if hasattr(self.antenna, 'beams'):
                self.beams = self.antenna.beams
            else:
                self.beams = None

    def beam_configuration(self, az_map, elev_map=None): # change the beam configuration according to the grid if beamforing is used
        # always in sample list!!!
        if elev_map is None:  # in case of one dimension beamforming
            elev_map = np.zeros(shape=az_map.shape)

        self.sectors_phi_range = np.arange(360 / self.n_sectors, 360.1, 360 / self.n_sectors)
        self.sectors_pointing = np.arange(360 / (2*self.n_sectors), 360, 360 / self.n_sectors)
        lower_bound = 0

        for sector, higher_bound in enumerate(self.sectors_phi_range):
            range_sector = np.where((az_map > lower_bound) & (az_map <= higher_bound))
            self.antenna.change_beam_configuration(point_phi=np.rint(az_map[range_sector]-self.sectors_pointing[sector])
                                                   , point_theta=-np.rint(elev_map[range_sector]))
            self.beam_sector_pattern.append(copy.deepcopy(self.antenna))

            # self.antenna.change_beam_configuration(point_phi=az_map[range_sector], point_theta=elev_map[range_sector])
            # self.beams = self.antenna.beams  # NOT USING - FOR FUTURE CALCULATIONS
            lower_bound = higher_bound

    def add_active_beam(self, sector, beams, n_users):
        if hasattr(self, 'active_beams'):
            if self.active_beams is None:
                self.active_beams = np.zeros(shape=(self.antenna.beams, self.n_sectors))
        else:
            print('Active beam list not found! Is the antenna object a beamforming one?')
            return
        for beam_index, beam in enumerate(beams):
            self.active_beams[beam][sector] = n_users[beam_index]

    def clean_active_beams(self):
        self.active_beams = None

    def generate_beam_timing_new(self, simulation_time, time_slot, uniform_time_dist=True):
        if uniform_time_dist:
            # trying to better adjust the beam timing to serve the users uniformly
            # beams with more users will get more time slots
            self.beam_timing = [None] * self.n_sectors
            min_ue_sector = np.zeros(self.n_sectors)
            for sector_index, sector_beams in enumerate(self.active_beams.T):
                if len(sector_beams[sector_beams != 0]) != 0:
                    min_ue_sector[sector_index] = np.min(sector_beams[sector_beams != 0])
                else:
                    print('ui')  # TESTANDO!!! APAGAR!!!
            # min_ue_sector = np.min(self.active_beams[self.active_beams != 0], axis=0)
            wighted_act_beams = np.round(self.active_beams / min_ue_sector)

            self.beam_timing = [None] * self.n_sectors
            for sector_index, sector in enumerate(self.beam_timing):
                sector = np.where(wighted_act_beams[:, sector_index] != 0)[0]
                np.random.shuffle(sector)  # randomizing the beam timing sequence
                ordened_weighted_beams = copy.deepcopy(wighted_act_beams[:, sector_index])
                wighted_act_beams[:, sector_index] = 0
                wighted_act_beams[np.arange(len(sector)), sector_index] = ordened_weighted_beams[sector]  # putting the weights in the same shuffle order of the beams
                self.beam_timing[
                    sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

            self.beam_timing_sequence = np.zeros(shape=(self.n_sectors, np.round(simulation_time/time_slot).astype(int))) + self.antenna.beams
            wighted_act_beams_bkp = copy.deepcopy(wighted_act_beams)
            for time in np.arange(0, simulation_time, time_slot):
                wighted_act_beams = self.next_active_beam_new(wighted_act_beams)  # passing the beam list with how many times each beam need to be active
                for sector_index, sector in enumerate(self.beam_timing):
                    if self.active_beams_index[sector_index].astype(int) == -1:
                        self.active_beams_index[sector_index] = 0
                        wighted_act_beams[:, sector_index] = wighted_act_beams_bkp[:, sector_index]
                    if self.beam_timing[sector_index].size != 0:
                        if len(self.beam_timing[sector_index]) - 1 < self.active_beams_index[sector_index].astype(int):
                            print('ui')
                        self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][self.active_beams_index[sector_index].astype(int)]

        else:
            # same time distribution for all beams
            self.beam_timing = [None] * self.n_sectors
            for sector_index, sector in enumerate(self.beam_timing):
                sector = np.where(self.active_beams[:, sector_index] != 0)[0]
                np.random.shuffle(sector)  # randomizing the beam timing sequence
                self.beam_timing[
                    sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

            self.beam_timing_sequence = np.zeros(
                shape=(self.n_sectors, np.round(simulation_time / time_slot).astype(int))) + self.antenna.beams
            for time in np.arange(0, simulation_time, time_slot):
                self.next_active_beam()
                for sector_index, sector in enumerate(self.beam_timing):
                    if self.beam_timing[sector_index].size != 0:
                        self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
                            self.active_beams_index[sector_index].astype(int)]

        self.beam_timing_sequence = self.beam_timing_sequence.astype(int)


    def generate_beam_timing(self, simulation_time, time_slot):
        # self.beam_timing = np.ndarray(shape=self.n_sectors)
        self.beam_timing = [None] * self.n_sectors
        for sector_index, sector in enumerate(self.beam_timing):
            sector = np.where(self.active_beams[:, sector_index] != 0)[0]
            np.random.shuffle(sector)  # randomizing the beam timing sequence
            self.beam_timing[sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

        self.beam_timing_sequence = np.zeros(shape=(self.n_sectors, np.round(simulation_time/time_slot).astype(int))) + self.antenna.beams
        for time in np.arange(0, simulation_time, time_slot):
            self.next_active_beam()
            for sector_index, sector in enumerate(self.beam_timing):
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][self.active_beams_index[sector_index].astype(int)]

        self.beam_timing_sequence = self.beam_timing_sequence.astype(int)

    def next_active_beam_new(self, beam_list=None):
        if beam_list is not None:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)
            # else:
            #     self.active_beams_index += 1

            for sector_index, beam_index in enumerate(self.active_beams_index):
                if np.sum(beam_list[:, sector_index]) != 0:
                    beam_index += 1
                    if beam_index > len(beam_list[:, sector_index]) - 1:
                        beam_index = 0
                    while beam_list[beam_index, sector_index] == 0:
                        beam_index += 1
                        if beam_index > len(beam_list[:, sector_index]) - 1:
                            beam_index = 0
                    beam_list[beam_index, sector_index] -= 1
                    # print(beam_list)
                    self.active_beams_index[sector_index] = beam_index
                else:
                    self.active_beams_index[sector_index] = -1  # informing that the beam list need to be restarted

            return beam_list

        else:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)
            else:
                self.active_beams_index += 1
                for sector_index, beam in enumerate(self.active_beams_index):
                    if beam > len(self.beam_timing[sector_index])-1:
                        self.active_beams_index[sector_index] = 0

    def next_active_beam(self):
        if self.active_beams_index is None:
            self.active_beams_index = np.zeros(shape=self.n_sectors)
        else:
            self.active_beams_index += 1
            for sector_index, beam in enumerate(self.active_beams_index):
                if beam > len(self.beam_timing[sector_index])-1:
                    self.active_beams_index[sector_index] = 0

    def generate_beam_bw(self):
        import warnings
        warnings.filterwarnings("ignore")
        self.beam_bw = np.where(self.active_beams != 0, self.bw / self.active_beams, 0)
        warnings.simplefilter('always')

    def slice_utility(self, ue_bs, bs_index):  # utility per user bw/snr
        # ue_bs -> bs|beam|sector|ch_gain
        # ue_bs = ue_bs[ue_bs[:, 0] == bs_index]
        warnings.filterwarnings("ignore")
        beam_bw = np.where(self.active_beams != 0, self.bw / self.active_beams, 0)  # minimum per beam bw
        warnings.simplefilter('always')
        bw_min = np.zeros(shape=ue_bs.shape[0])
        for ue_index, ue in enumerate(ue_bs):
            bw_min[ue_index] = beam_bw[ue[1], ue[2]] * 10E6 # minimum per user bw

        bw = 5 * 10*6  # making SNR for a bandwidth of 5MHz
        k = 1.380649E-23  # Boltzmann's constant (J/K)
        t = 290  # absolute temperature
        pw_noise_bw = k*t*bw
        tx_pw = 10**(self.tx_power/10)  # converting from dBw to watt
        snr = (tx_pw * 10**(ue_bs[:, 3]/10)) / pw_noise_bw  # signal to noise ratio (linear)
        c_target = 50 * 10E6  # todo - MUDAR ESTAR PUERRRA
        bw_need = 2**(c_target/snr) - 1  # needed bw to achieve the capacity target

        self.slice_util = np.zeros(shape=ue_bs.shape[0])
        self.slice_util = (bw_min/bw_need) * np.log2(snr)


    def beam_utility(self, ue_bs=None, bs_index=None):
        # ue_bs -> bs|beam|sector|ch_gain
        self.slice_utility(ue_bs=ue_bs, bs_index=bs_index)
        self.beam_util = np.zeros(shape=self.active_beams.shape)

        for sector_index in np.unique(ue_bs[ue_bs[:, 0] == bs_index][:, 2]).astype(int):
            for beam_index in np.unique(ue_bs[(ue_bs[:, 0] == bs_index) * (ue_bs[:, 2] == sector_index)][:, 1]).astype(int):
                    ue_in_beam_bs = np.where((ue_bs[:, 0] == bs_index) * (ue_bs[:, 1] == beam_index) * (ue_bs[:, 2] == sector_index))
                    self.beam_util[beam_index, sector_index] = np.sum(self.slice_util[ue_in_beam_bs])

        self.beam_util_log = np.where(self.beam_util != 0, np.log2(self.beam_util), 0)
        #print(self.beam_util)

        self.sector_util = np.sum(self.beam_util_log, axis=0)  # VERIFICAR AQUI DEPOIS !!!

    def generate_weighted_beam_time(self, t_total, ue_bs, bs_index):
        t_min = 10  # milliseconds
        self.beam_utility(ue_bs=ue_bs, bs_index=bs_index)
        t_beam = np.zeros(shape=self.active_beams.shape)

        for sector_index in np.unique(ue_bs[ue_bs[:, 0] == bs_index][:, 2]).astype(int):
            non_zero = np.where(self.beam_util_log[:, sector_index] != 0)  # to prevent a divide by zero occurence
            t_beam[non_zero, sector_index] = t_min + (self.sector_util[sector_index]/self.beam_util_log[non_zero, sector_index]) \
                                      * (t_total - np.count_nonzero(self.active_beams[:, sector_index])*t_min)  # beam timing according to paper eq.

    def generate_weighted_bw(self, ue_bs, bs_index):
        self.beam_utility(ue_bs=ue_bs, bs_index=bs_index)  # calculating the sector, beam and slice utilities
        warnings.filterwarnings("ignore")
        bw_min = np.where(self.active_beams != 0, self.bw / self.active_beams, 0)  # minimum per beam bw
        warnings.simplefilter('always')

        self.user_bw = np.zeros(shape=ue_bs.shape[0])

        for sector_index in np.unique(ue_bs[ue_bs[:, 0] == bs_index][:, 2]).astype(int):
            for beam_index in np.unique(ue_bs[(ue_bs[:, 0] == bs_index) * (ue_bs[:, 2] == sector_index)][:, 1]).astype(int):
                ue_in_beam_bs = np.where((ue_bs[:, 0] == bs_index) * (ue_bs[:, 1] == beam_index) * (ue_bs[:, 2] == sector_index))
                self.user_bw[ue_in_beam_bs] = bw_min[beam_index, sector_index] + \
                          (self.slice_util[ue_in_beam_bs] /
                           self.beam_util_log[beam_index, sector_index]) * (self.bw - ue_bs[ue_in_beam_bs].shape[0] * bw_min[beam_index, sector_index])

    def generate_beam_bw_new(self, ue_bs=None, bs_index=None):
        # ue_bs -> bs|beam|sector|ch_gain
        if ue_bs is not None:
            if not hasattr(self, 'user_bw'):  # ARRUMAR A INSTÂNCIA DO SELF.USER_BW !!!!!
                self.user_bw = np.zeros(ue_bs.shape[0])
            for sector_index in np.unique(ue_bs[ue_bs[:,0] == bs_index][:, 2]).astype(int):
                for beam_index in np.unique(ue_bs[(ue_bs[:,0] == bs_index) * (ue_bs[:,2] == sector_index)][:, 1]).astype(int):
                    log2_ch_gain = np.log2(1 + 10 ** (ue_bs[(ue_bs[:,0] == bs_index) * (ue_bs[:,1] == beam_index) * (ue_bs[:, 2] == sector_index)][:,3]/10))
                    weighted_ch_gain_map = log2_ch_gain / np.min(log2_ch_gain)
                    sum_weights = np.sum(weighted_ch_gain_map)
                    min_bw = self.bw/sum_weights
                    self.user_bw[np.where((ue_bs[:, 0] == bs_index) * (ue_bs[:, 1] == beam_index) * (ue_bs[:, 2] == sector_index))] = weighted_ch_gain_map * min_bw
        else:
            warnings.filterwarnings("ignore")
            self.beam_bw = np.where(self.active_beams != 0, self.bw / self.active_beams, 0)
            warnings.simplefilter('always')

    def sector_beam_pointing_configuration(self, n_beams):
        # sectors_pointing = np.arange(360/(2*self.n_sectors), 360.1, 360/self.n_sectors)
        self.beams_pointing = np.array([])
        sector_apperture = 360/self.n_sectors
        beams_pointing_0 = np.arange(sector_apperture/(2*n_beams), sector_apperture+0.1, sector_apperture/n_beams)
        self.beams_pointing = beams_pointing_0
        for i in range(1, self.n_sectors):
            self.beams_pointing = np.append(self.beams_pointing, beams_pointing_0 + sector_apperture*i)

    def generate_ant_pattern(self):  # used for sector antennas WITHOUT beamforming
        # horizontal_beamwidth = 360/self.n_sectors
        self.antenna.hor_beamwidth = 360/self.n_sectors
        # self.antenna = self.antenna.ITU1336(gain=gain, frequency=self.frequency, hor_beamwidth=horizontal_beamwidth, ver_beamwidth=10)
        self.antenna.build_diagram()

    def generate_sector_pattern(self, plot=False):  # used for pivot the antennas WITHOUT beamforming
        self.sectors_pointing = np.arange(360 / (2*self.n_sectors), 360, 360 / self.n_sectors)
        for sector_pointing in self.sectors_pointing:
            rotated_azim = self.antenna.sigma + np.deg2rad(sector_pointing)
            rotated_azim = np.where(rotated_azim < 0, rotated_azim + (2 * np.pi), rotated_azim)
            rotated_azim = np.where(rotated_azim > 2 * np.pi, rotated_azim - (2 * np.pi), rotated_azim)
            sector_pattern = np.interp(rotated_azim, self.antenna.sigma, self.antenna.hoz_pattern)
            self.sectors_hor_pattern.append(sector_pattern)

        for downtilt in self.downtilts:
            tilted_pattern = self.antenna.theta - np.deg2rad(downtilt)
            tilted_pattern = np.where(tilted_pattern < 0, tilted_pattern + (2*np.pi), tilted_pattern)
            tilted_pattern = np.where(tilted_pattern > 2 * np.pi, tilted_pattern - (2*np.pi), tilted_pattern)
            tilted_elev_pattern = np.interp(tilted_pattern, self.antenna.theta, self.antenna.ver_pattern)
            self.sectors_ver_pattern.append(tilted_elev_pattern)

        self.sectors_hor_pattern = np.asarray(self.sectors_hor_pattern)
        self.sectors_ver_pattern = np.asarray(self.sectors_ver_pattern)

        if plot:
            plt.polar(self.antenna.sigma, self.sectors_hor_pattern[0])
            for i in range(self.n_sectors - 1):
                plt.plot(self.antenna.sigma, self.sectors_hor_pattern[i+1])
            plt.title('Base station sectors horizontal patterns')
            # plt.show(block=False)

            plt.figure()
            plt.polar(self.antenna.theta, self.sectors_ver_pattern[0])
            for i in range(self.n_sectors - 1):
                plt.plot(self.antenna.theta, self.sectors_ver_pattern[i+1])
            plt.title('Base station sectors vertical patterns')
            plt.show()

    def change_downtilt(self, downtilts, plot=False):
        if np.size(downtilts) == 1:  # if the downtilts variable is not array type
            downtilts = np.zeros(shape=self.n_sectors) + downtilts  # setting all the array values the same
        self.downtilts = downtilts  # this variable must have the shape of the same size of n_sectors
        self.sectors_hor_pattern = []  # rotated antenna patterns for each of the base station sectors
        self.sectors_ver_pattern = []  # tilted elevation pattern
        self.generate_sector_pattern(plot)


