from utility_based_fn import Util_fn
import copy
import numpy as np

class Time_Scheduler():
    def __init__(self, simulation_time, scheduler_typ):
        self.simulation_time = simulation_time

        if scheduler_typ == 'prop-cmp' or scheduler_typ == 'prop-smp':
            self.weighted_act_beams = None

    def generate_proportional_beam_timing(self, time_slot, active_beams):
        # same time distribution for all bea
        self.beam_timing = [None] * active_beams.shape[1] # CHEAR SE ESSE CHAPE DÁ A DIMESÃO DE 3 BEAMS !!!!
        for sector_index, _ in enumerate(self.beam_timing):
            sector = np.where(active_beams[:, sector_index] != 0)[0]
            np.random.shuffle(sector)  # randomizing the beam timing sequence
            self.beam_timing[
                sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

        self.beam_timing_sequence = np.zeros(
            shape=(active_beams.shape[1], np.round(self.simulation_time / time_slot).astype(int))) + self.antenna.beams
        for time in np.arange(0, self.simulation_time, time_slot):
            self.next_active_beam()
            for sector_index, sector in enumerate(self.beam_timing):
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
                        self.active_beams_index[sector_index].astype(int)]


    def generate_utility_based_beam_timing(self, time_slot, weighted_act_beams):
        # trying to better adjust the beam timing to serve the users uniformly
        # beams with more users with more utilities will get more time slots
        self.beam_timing = [None] * weighted_act_beams.shape[1]  # VER SE A DIMENSÃO ESTÁ CERTA AQUI
        for sector_index, _ in enumerate(self.beam_timing):
            sector = np.where(weighted_act_beams[:, sector_index] != 0)[0]
            # np.random.shuffle(sector)  # randomizing the beam timing sequence
            sector = sector[np.random.permutation(sector.shape[0])]  # randomizing the beam timing sequence

            ordened_weighted_beams = copy.copy(weighted_act_beams[:, sector_index])
            weighted_act_beams[:, sector_index] = 0
            weighted_act_beams[np.arange(len(sector)), sector_index] = ordened_weighted_beams[
                sector]  # putting the weights in the same shuffle order of the beams
            self.beam_timing[
                sector_index] = sector  # I really dont know why this line is needed to this code to work!!!

        self.beam_timing_sequence = np.zeros(
            shape=(weighted_act_beams.shape[1], np.round(self.simulation_time / time_slot).astype(int))) + self.antenna.beams  # REESCREVER ISSO AQUI
        wighted_act_beams_bkp = copy.copy(weighted_act_beams)
        for time in np.arange(0, self.simulation_time, time_slot):
            wighted_act_beams = self.next_active_beam_new(
                weighted_act_beams)  # passing the beam list with how many times each beam need to be active
            for sector_index, _ in enumerate(self.beam_timing):
                if self.active_beams_index[sector_index].astype(int) == -1:
                    self.active_beams_index[sector_index] = 0
                    wighted_act_beams[:, sector_index] = wighted_act_beams_bkp[:, sector_index]
                if self.beam_timing[sector_index].size != 0:
                    self.beam_timing_sequence[sector_index, time] = self.beam_timing[sector_index][
                        self.active_beams_index[sector_index].astype(int)]

    def generate_weighted_beam_time(self, t_total, ue_bs, bs_index, c_target, t_min):
        # t_min = 10  # milliseconds
        self.beam_utility(ue_bs=ue_bs, bs_index=bs_index, c_target=c_target)
        t_beam = np.zeros(shape=self.active_beams.shape)

        sector_index = np.unique(ue_bs[ue_bs[:, 0] == bs_index][:, 2]).astype(int)
        non_zero = (self.beam_util[:, sector_index] != 0)  # to prevent a divide by zero occurence

        t_beam[self.beam_util != 0] = (t_min + (self.beam_util_log[:, sector_index]/ self.sector_util[sector_index])
                                       * (t_total - np.count_nonzero(self.active_beams[:, sector_index], axis=0) * t_min))[non_zero]

        self.weighted_act_beams = np.round(t_beam).astype(int)


    def next_active_beam(self):
        if self.active_beams_index is None:
            self.active_beams_index = np.zeros(shape=self.n_sectors)
        else:
            self.active_beams_index += 1
            for sector_index, beam in enumerate(self.active_beams_index):
                if beam > len(self.beam_timing[sector_index]) - 1:
                    self.active_beams_index[sector_index] = 0


    def next_active_beam_new(self, beam_list=None):
        if beam_list is not None:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)

            for sector_index, beam_index in enumerate(self.active_beams_index):
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

            return beam_list

        else:
            if self.active_beams_index is None:
                self.active_beams_index = np.zeros(shape=self.n_sectors).astype(int)
            else:
                self.active_beams_index += 1
                for sector_index, beam in enumerate(self.active_beams_index):
                    if beam > len(self.beam_timing[sector_index]) - 1:
                        self.active_beams_index[sector_index] = 0


