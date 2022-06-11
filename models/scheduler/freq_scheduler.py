import numpy as np
import copy

class Freq_Scheduler:
    def __init__(self, bw, bs_index, scheduler_typ, slot_bw=None):
        self.bw = bw
        self.bs_index = bs_index

        # calculated variables
        # if scheduler_typ == 'prop-cmp' or scheduler_typ == 'prop-smp':
        self.user_bw = None
        self.beam_bw = None

        if scheduler_typ == 'RR':
            self.in_queue_ue = None
            self.slot_bw = slot_bw

            if self.slot_bw is None:
                raise ValueError('Need to set slot_bw(MHz) to use the Round Robin Scheduler')

        # TODO - ALTERAR AQUI PARA SEMPRE USAR O USER_BW PARA TODOS OS CASOS

    def generate_proportional_beam_bw(self, active_beams):
        self.beam_bw = np.zeros(shape=active_beams.shape)
        self.beam_bw[active_beams != 0] = (self.bw / active_beams[active_beams != 0])

    def generate_weighted_bw(self, ue_bs, active_beams, slice_util, beam_util):
        # import timeit
        # self.beam_utility(ue_bs=ue_bs, bs_index=bs_index,
        #                   c_target=self.c_target)  # calculating the sector, beam and slice utilities

        self.generate_proportional_beam_bw(active_beams)
        bw_min = self.beam_bw
        # bw_min = np.zeros(shape=self.active_beams.shape)
        # active_beams = self.active_beams != 0
        # bw_min[active_beams] = (self.bw / self.active_beams[active_beams]) / 10

        self.user_bw = np.zeros(shape=ue_bs.shape[0])

        # start = timeit.default_timer()
        for sector_index in np.unique(ue_bs[ue_bs[:, 0] == self.bs_index][:, 2]).astype(int):
            for beam_index in np.unique(ue_bs[(ue_bs[:, 0] == self.bs_index) & (ue_bs[:, 2] == sector_index)][:, 1]).astype(int):
                ue_in_beam_bs = np.where(
                    (ue_bs[:, 0] == self.bs_index) & (ue_bs[:, 1] == beam_index) & (ue_bs[:, 2] == sector_index))
                self.user_bw[ue_in_beam_bs] = bw_min[beam_index, sector_index] + \
                                              (slice_util[ue_in_beam_bs] /
                                               beam_util[beam_index, sector_index]) * (
                                                          self.bw - ue_bs[ue_in_beam_bs].shape[0] * bw_min[
                                                      beam_index, sector_index])

    def generate_RR_bw(self, ue_bs, active_beams):
        # slot_bw is the bandwidth of the fixe frequency slot to be distributed
        if self.in_queue_ue is None:
            self.in_queue_ue = np.zeros(shape=ue_bs.shape[0])
        ue_in_bs = ue_bs[:, 0] == self.bs_index
        # slot_bw = 20
        self.user_bw = np.zeros(shape=ue_bs.shape[0])
        # in_queue_ue = np.zeros(shape=active_beams.shape)
        n_bw_slots = np.ones(shape=active_beams) * self.bw/self.slot_bw
        dummy_queue = copy.copy(active_beams)

        # =================  Dealing with the queue ================
        if self.in_queue_ue is not None and np.sum(self.in_queue_ue) != 0:
            queue_controller = active_beams - self.in_queue_ue - n_bw_slots
            meq_zero = queue_controller >= 0
            less_zero = queue_controller < 0

            # this case is when the queue has LESS ues to be allocated than the number of slots

            [beam_index_list, sector_index_list] = np.where(less_zero)
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_to_receibe_bw_min = np.where(ue_bs[ue_in_bs, 1] == beam_index &
                                                ue_bs[ue_in_bs, 2] == sector_index)
                self.user_bw[ue_to_receibe_bw_min[range(self.in_queue_ue[beam_index, sector_index],
                                                        active_beams[beam_index, sector_index])]] += self.slot_bw

            n_bw_slots[less_zero] = np.abs(queue_controller[less_zero])

            # this case is when the queue has MORE UEs to be allocated than the number of slots
            [beam_index_list, sector_index_list] = np.where(meq_zero)
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_to_receibe_bw_min = np.where(ue_bs[ue_in_bs, 1] == beam_index &
                                                ue_bs[ue_in_bs, 2] == sector_index)

                self.user_bw[ue_to_receibe_bw_min[range(self.in_queue_ue[beam_index, sector_index],
                                                        self.in_queue_ue[beam_index, sector_index] +
                                                        n_bw_slots[beam_index, sector_index])]] += self.slot_bw

            n_bw_slots[meq_zero] = 0
            self.in_queue_ue[meq_zero] = queue_controller[meq_zero]

        # ====================== Dealing with the rest of bw_slots without queue ==============
        while np.sum(dummy_queue) != 0:
            dummy_queue -= n_bw_slots
            less_zero = dummy_queue < 0
            meq_zero = dummy_queue >= 0

            # first, calculate and allocate bw_slots when can do it for all UEs
            [beam_index_list, sector_index_list] = np.where(less_zero)
            while np.sum(dummy_queue[less_zero]) != 0:
                for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                    ue_to_receibe_bw_min = ue_bs[ue_in_bs, 1] == beam_index & ue_bs[ue_in_bs, 2] == sector_index
                    self.user_bw[ue_to_receibe_bw_min] += self.slot_bw

                dummy_queue[meq_zero] -= n_bw_slots
                meq_zero = dummy_queue >= 0

            # second, allocate bw_slots when not all UEs can have it
            [beam_index_list, sector_index_list] = np.where(meq_zero)
            while np.sum(dummy_queue[meq_zero]) != 0:
                for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                    ue_to_receibe_bw_min = np.where(ue_bs[ue_in_bs, 1] == beam_index & ue_bs[ue_in_bs, 2] == sector_index)[range(n_bw_slots)]
                    self.user_bw[ue_to_receibe_bw_min] += self.slot_bw
                    dummy_queue[beam_index, sector_index] = 0
                    self.in_queue_ue[beam_index, sector_index] = range(abs(dummy_queue[beam_index, sector_index]))


        # self.in_queue_ue = in_queue_ue




    def generate_best_CSI_bw(self):
        pass