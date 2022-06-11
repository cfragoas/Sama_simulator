import copy
import numpy as np

class Time_Scheduler:
    def __init__(self, simulation_time, time_slot, scheduler_typ, bs_index, t_min=None):
        self.simulation_time = simulation_time
        self.time_slot = time_slot
        self.bs_index = bs_index

        # calculated variables
        self.beam_timing = None
        self.beam_timing_sequence = None
        self.active_beams_index = None
        self.n_sectors = None  # this variable is used to shape the dimensions of some matrices
        self.n_beams = None  # this variable is used to shape the dimensions of some matrices
        if scheduler_typ == 'prop-cmp' or scheduler_typ == 'prop-smp':
            self.weighted_act_beams = None
            if t_min is not None:
                self.t_min = t_min
            else:
                raise ValueError('Need to set t_min when using schedulers prop-cmp or prop-smp.')

    def set_base_dimensions(self, n_sectors, n_beams):
        self.n_sectors = n_sectors  # this variable is used to shape the dimensions of some matrices
        self.n_beams = n_beams  # this variable is used to shape the dimensions of some matrices

    def generate_proportional_beam_timing(self, time_slot, active_beams):
        # this function will generate an equal time distribution for all active beams
        self.beam_timing = [None] * active_beams.shape[1]  # CHECAR SE ESSE CHAPE DÁ A DIMESÃO DE 3 BEAMS !!!!
        for sector_index, _ in enumerate(self.beam_timing):
            sector = np.where(active_beams[:, sector_index] != 0)[0]
            np.random.shuffle(sector)  # randomizing the beam timing sequence
            self.beam_timing[sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

        self.beam_timing_sequence = \
            np.zeros(shape=(active_beams.shape[1], np.round(self.simulation_time / time_slot).astype(int))) \
            + self.n_beams # filling the initial beam_timing_sequence with beam_index that non exist
        for time in np.arange(0, self.simulation_time, time_slot):
            self.next_active_beam()
            for sector_index, sector in enumerate(self.beam_timing):
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
                        self.active_beams_index[sector_index].astype(int)]

    def generate_ue_qtd_proportional_beam_timing(self, time_slot, active_beams, t_index):
        #  this function will allocate the beams times based on the quantities of active UEs in each beam
        users_sector = np.sum(active_beams, axis=0)
        users_sector[users_sector == 0] = 0.1
        # beam_weights = np.round(active_beams / users_sector)
        self.weighted_act_beams = np.round(active_beams / users_sector)
        self.generate_weighted_time_matrix(simulation_time=self.simulation_time - t_index)

        # self.beam_timing = [None] * self.n_sectors
        #
        # for sector_index, _ in enumerate(self.beam_timing):
        #     sector = np.where(self.weighted_act_beams[:, sector_index] != 0)[0]
        #     # np.random.shuffle(sector)  # randomizing the beam timing sequence
        #     sector = sector[np.random.permutation(sector.shape[0])]  # randomizing the beam timing sequence
        #
        #     ordened_weighted_beams = copy.copy(self.weighted_act_beams[:, sector_index])
        #     self.weighted_act_beams[:, sector_index] = 0
        #     self.weighted_act_beams[np.arange(len(sector)), sector_index] = ordened_weighted_beams[sector]  # putting the weights in the same shuffle order of the beams
        #     self.beam_timing[sector_index] = sector  # I really dont know why this line is needed to this code to work!!!
        #
        # self.beam_timing_sequence = np.zeros(
        #     shape=(self.n_sectors,
        #            np.round(self.simulation_time / self.time_slot).astype(int))) + self.n_beams  # filling the initial beam_timing_sequence with beam_index that non exist
        # wighted_act_beams_bkp = copy.copy(self.weighted_act_beams)
        # for time in np.arange(0, self.simulation_time, self.time_slot):
        #     wighted_act_beams = self.next_weighted_active_beam(
        #         self.weighted_act_beams)  # passing the beam list with how many times each beam need to be active
        #     for sector_index, _ in enumerate(self.beam_timing):
        #         if self.active_beams_index[sector_index].astype(int) == -1:
        #             self.active_beams_index[sector_index] = 0
        #             wighted_act_beams[:, sector_index] = wighted_act_beams_bkp[:, sector_index]
        #         if self.beam_timing[sector_index].size != 0:
        #             self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
        #                 self.active_beams_index[sector_index].astype(int)]
        #
        # self.beam_timing_sequence = self.beam_timing_sequence.astype(int)


    def generate_utility_based_beam_timing(self, t_index, ue_bs, active_beams, beam_util, beam_util_log, sector_util):
        # trying to better adjust the beam timing to serve the users uniformly
        # beams with more users with more utilities will get more time slots

        # updating t_min to shrink proportionally to the elapsed time
        t_min = self.t_min*((self.simulation_time - t_index)/self.simulation_time)
        self.generate_utility_weighted_beam_time(t_total=self.simulation_time - t_index, ue_bs=ue_bs, active_beams=active_beams,
                                                 t_min=t_min, beam_util=beam_util, beam_util_log=beam_util_log, sector_util=sector_util)  # T_MIN HERE IS THE MINIMUM RESERVED BEAM TIME !!!!! FIXFIXFIXFIX

        self.generate_weighted_time_matrix(simulation_time=self.simulation_time - t_index)
        # self.beam_timing = [None] * self.weighted_act_beams.shape[1]  # VER SE A DIMENSÃO ESTÁ CERTA AQUI
        # for sector_index, _ in enumerate(self.beam_timing):
        #     sector = np.where(self.weighted_act_beams[:, sector_index] != 0)[0]
        #     # np.random.shuffle(sector)  # randomizing the beam timing sequence
        #     sector = sector[np.random.permutation(sector.shape[0])]  # randomizing the beam timing sequence
        #
        #     ordened_weighted_beams = copy.copy(self.weighted_act_beams[:, sector_index])
        #     self.weighted_act_beams[:, sector_index] = 0
        #     self.weighted_act_beams[np.arange(len(sector)), sector_index] = ordened_weighted_beams[
        #         sector]  # putting the weights in the same shuffle order of the beams
        #     self.beam_timing[
        #         sector_index] = sector  # I really dont know why this line is needed to this code to work!!!
        #
        # self.beam_timing_sequence = np.zeros(
        #     shape=(self.weighted_act_beams.shape[1],
        #            np.round(self.simulation_time / self.time_slot).astype(int))) + self.n_beams  # filling the initial beam_timing_sequence with beam_index that non exist
        # wighted_act_beams_bkp = copy.copy(self.weighted_act_beams)
        # for time in np.arange(0, self.simulation_time, self.time_slot):
        #     wighted_act_beams = self.next_weighted_active_beam(
        #         self.weighted_act_beams)  # passing the beam list with how many times each beam need to be active
        #     for sector_index, _ in enumerate(self.beam_timing):
        #         if self.active_beams_index[sector_index].astype(int) == -1:
        #             self.active_beams_index[sector_index] = 0
        #             wighted_act_beams[:, sector_index] = wighted_act_beams_bkp[:, sector_index]
        #         if self.beam_timing[sector_index].size != 0:
        #             self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
        #                 self.active_beams_index[sector_index].astype(int)]
        #
        # self.beam_timing_sequence = self.beam_timing_sequence.astype(int)

    def generate_weighted_time_matrix(self, simulation_time):
        self.beam_timing = [None] * self.n_sectors

        for sector_index, _ in enumerate(self.beam_timing):
            sector = np.where(self.weighted_act_beams[:, sector_index] != 0)[0]
            # np.random.shuffle(sector)  # randomizing the beam timing sequence
            sector = sector[np.random.permutation(sector.shape[0])]  # randomizing the beam timing sequence

            ordened_weighted_beams = copy.copy(self.weighted_act_beams[:, sector_index])
            self.weighted_act_beams[:, sector_index] = 0
            self.weighted_act_beams[np.arange(len(sector)), sector_index] = ordened_weighted_beams[
                sector]  # putting the weights in the same shuffle order of the beams
            self.beam_timing[
                sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

        self.beam_timing_sequence = np.zeros(
            shape=(self.n_sectors,
                   np.round(simulation_time / self.time_slot).astype(
                       int))) + self.n_beams  # filling the initial beam_timing_sequence with beam_index that non exist
        wighted_act_beams_bkp = copy.copy(self.weighted_act_beams)
        for time in np.arange(0, simulation_time, self.time_slot):
            wighted_act_beams = self.next_weighted_active_beam(
                self.weighted_act_beams)  # passing the beam list with how many times each beam need to be active
            for sector_index, _ in enumerate(self.beam_timing):
                if self.active_beams_index[sector_index].astype(int) == -1:
                    self.active_beams_index[sector_index] = 0
                    wighted_act_beams[:, sector_index] = wighted_act_beams_bkp[:, sector_index]
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
                        self.active_beams_index[sector_index].astype(int)]

        self.beam_timing_sequence = self.beam_timing_sequence.astype(int)

    def generate_utility_weighted_beam_time(self, t_total, ue_bs, t_min, active_beams, beam_util, beam_util_log, sector_util):
        # t_min = 10  # milliseconds
        # self.beam_utility(ue_bs=ue_bs, bs_index=bs_index, c_target=c_target)
        t_beam = np.zeros(shape=beam_util.shape)

        sector_index = np.unique(ue_bs[ue_bs[:, 0] == self.bs_index][:, 2]).astype(int)
        non_zero = (beam_util[:, sector_index] != 0)  # to prevent a divide by zero occurence

        t_beam[beam_util != 0] = (t_min + (beam_util_log[:, sector_index] / sector_util[sector_index])
                                  * (t_total - np.count_nonzero(active_beams[:, sector_index], axis=0) * t_min))[non_zero]

        self.weighted_act_beams = np.round(t_beam).astype(int)

    def next_active_beam(self):
        if self.active_beams_index is None:
            self.active_beams_index = np.zeros(shape=self.n_sectors)
        else:
            self.active_beams_index += 1
            for sector_index, beam in enumerate(self.active_beams_index):
                if beam > len(self.beam_timing[sector_index]) - 1:
                    self.active_beams_index[sector_index] = 0

    def next_weighted_active_beam(self, beam_list=None):
        if beam_list is not None:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)

            for sector_index, beam_index in enumerate(self.active_beams_index):
                try:
                    if np.sum(beam_list[:, sector_index]) != 0:
                        beam_index += 1
                        if beam_index > len(beam_list[:, sector_index]) - 1:
                            beam_index = 0
                        # this block code is slower than the following one
                        # while beam_list[beam_index, sector_index] == 0:
                        #     try:
                        #         non_zero = np.nonzero(beam_list[:, sector_index])[0]
                        #         beam_index = (non_zero[non_zero > beam_index]).min()
                        #     except:
                        #         beam_index = 0
                        while beam_list[beam_index, sector_index] == 0:  # picks the 1st nonzero occurence after beam_index
                            beam_index += 1
                            if beam_index > len(beam_list[:, sector_index]) - 1:  # loops into the first position otherwise
                                beam_index = 0
                        beam_list[beam_index, sector_index] -= 1
                        self.active_beams_index[sector_index] = beam_index
                    else:
                        self.active_beams_index[sector_index] = -1  # informing that the beam list need to be restarted
                except:
                    print('ui')

            return beam_list

        else:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)
            else:
                self.active_beams_index += 1
                for sector_index, beam in enumerate(self.active_beams_index):
                    if beam > len(self.beam_timing[sector_index]) - 1:
                        self.active_beams_index[sector_index] = 0
