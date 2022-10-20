import numpy as np
import copy

# The Freq_Scheduler class is responsable for all the different frequency schedullers. It will update the allocated
# bandwidt for all UEs that are allocated in his BS in the variable self.user_bw.

class Freq_Scheduler:
    def __init__(self, bw, bs_index, scheduler_typ, bw_slot=None, tx_power=None, time_slot=None, simulation_time=None):
        self.bw = bw  # the maximum bs in a BS sector
        self.bs_index = bs_index
        self.user_bw = None  # the bw for all the UEs but it will only be updated for the bs_index ones for each obj
        self.beam_bw = None  # the available bw for a beam to be splitted by the UEs

        if scheduler_typ == 'RR':
            self.in_queue_ue = None  # queue of UEs to be served, if necessary
            self.bw_slot = bw_slot  # if necessary, the size of a bandwidth slot

            if self.bw_slot is None:
                raise ValueError('Need to set bw_slot(MHz) to use the Round Robin scheduler')
        if scheduler_typ == 'BCQI':
            self.tx_power = tx_power  # tx_power in dBW
            self.time_ratio = time_slot/simulation_time  # ratio in (ms/ms)

            if self.tx_power is None:
                raise ValueError('Need to set Tx power(dBW) to use the Best-CQI scheduler')

        if scheduler_typ == 'PF':  # need to create both BCQI and RR variables
            self.fake_user_bw = None  # to preserve the RR scheduller after execute BCQI function
            self.last_updated_beams = None  # to preserve the updated beams on the last update for the RR function
            # RR variables
            self.in_queue_ue = None  # queue of UEs to be served, if necessary
            self.bw_slot = bw_slot  # if necessary, the size of a bandwidth slot
            if self.bw_slot is None:
                raise ValueError('Need to set bw_slot(MHz) to use the Proportional Fair scheduler')
            # BCQI variables
            self.tx_power = tx_power  # tx_power in dBW
            if simulation_time == 0:
                print('ui')
            self.time_ratio = time_slot / simulation_time  # ratio in (ms/ms)


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

    def generate_RR_bw(self, ue_bs, active_beams, updated_beams=None):
        # This function will generate the bandwidth allocation for all users (self.user_bw) for all users for the BS
        # object. It will user the Round-Robin algorithm and will make a queue (self.in_queue_ue) that will iterate
        # within the simulations and store the next UE that will receive a bandwidth slot (bw_slot).
        # First, it resolves the queue (if possible) and next will allocate the bw_slots for the fist UE of a beam,
        # updating the queue in both cases when the available bw is not enough.
        # The user_bw is generated in the first time index and is just updated for the last active users.

        ue_in_bs = ue_bs[:, 0] == self.bs_index

        # checking the beams used in the last time index to be update (move the queue) and will erase the allocated
        # bandwidth for the last active users in the last active beams for the last time index
        if updated_beams is None:
            beams_2b_updtd = np.ones(shape=active_beams.shape, dtype=bool)
            self.user_bw = np.zeros(shape=ue_bs.shape[0], dtype=int)
        else:
            beams_2b_updtd = np.zeros(shape=active_beams.shape, dtype=bool)
            non_empty_sectors = updated_beams <= (active_beams.shape[0] - 1)

            beams_2b_updtd[updated_beams[non_empty_sectors], np.array(range(beams_2b_updtd.shape[1]))[non_empty_sectors]] = True

            for sector_index in range(beams_2b_updtd.shape[1]):
                beam_index = updated_beams[sector_index]
                ue_to_erase_bw = ue_in_bs & (ue_bs[:, 1] == beam_index) & (ue_bs[:, 2] == sector_index)
                self.user_bw[ue_to_erase_bw] = 0

        active_beams = active_beams.astype(int)
        # slot_bw is the bandwidth of the fixe frequency slot to be distributed
        if self.in_queue_ue is None:
            self.in_queue_ue = np.zeros(shape=active_beams.shape, dtype=int)
        try:
            n_bw_slots = (np.ones(shape=active_beams.shape) * self.bw/self.bw_slot).astype(int)
        except:
            print('ui')
        non_zero_beams = (active_beams != 0) & beams_2b_updtd

        # =================  Dealing with the queue first ================
        if self.in_queue_ue is not None and np.sum(self.in_queue_ue) != 0:
            in_queue_beams = self.in_queue_ue != 0
            queue_controller = np.zeros(shape=active_beams.shape, dtype=int)
            queue_controller[in_queue_beams] = active_beams[in_queue_beams] - self.in_queue_ue[in_queue_beams] - n_bw_slots[in_queue_beams]
            meq_zero = (queue_controller >= 0) & in_queue_beams & non_zero_beams
            less_zero = (queue_controller < 0) & non_zero_beams

            # this case is when the queue has LESS ues to be allocated than the number of slots (less_zero beams)
            [beam_index_list, sector_index_list] = np.where(less_zero)
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_in_bs_beam_sec = np.where((ue_bs[:, 1] == beam_index) &
                                            (ue_bs[:, 2] == sector_index) & ue_in_bs)[0]
                ue_to_receive_bw_min = ue_in_bs_beam_sec[range(self.in_queue_ue[beam_index, sector_index],
                                                        active_beams[beam_index, sector_index])]
                self.user_bw[ue_to_receive_bw_min] += self.bw_slot
                n_bw_slots[beam_index, sector_index] -= ue_to_receive_bw_min.size

            # this case is when the queue has MORE UEs to be allocated than the number of slots (meq_zero berams)
            [beam_index_list, sector_index_list] = np.where(meq_zero)
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_in_bs_beam_sec = np.where((ue_bs[:, 1] == beam_index) &
                                                (ue_bs[:, 2] == sector_index) & ue_in_bs)[0]

                ue_to_receive_bw_min = ue_in_bs_beam_sec[range(self.in_queue_ue[beam_index, sector_index],
                                                               self.in_queue_ue[beam_index, sector_index] +
                                                               n_bw_slots[beam_index, sector_index])]

                self.user_bw[ue_to_receive_bw_min] += self.bw_slot
                self.in_queue_ue[beam_index, sector_index] = self.in_queue_ue[beam_index, sector_index] + \
                                                             n_bw_slots[beam_index, sector_index]

            n_bw_slots[meq_zero] = 0

        # ====================== Dealing with the rest of bw_slots without queue ==============
        beams_w_bw_slots = (n_bw_slots != 0) & non_zero_beams
        dummy_queue = np.zeros(shape=active_beams.shape)
        dummy_queue[beams_w_bw_slots] = active_beams[beams_w_bw_slots]
        dummy_queue[beams_w_bw_slots] -= n_bw_slots[beams_w_bw_slots]
        less_zero = (dummy_queue < 0) & beams_w_bw_slots

        # first, calculate and allocate bw_slots when can do it for all UEs (less_zero beams)
        while np.sum(less_zero) != 0:  # doing it until it falls in meq_zero
            [beam_index_list, sector_index_list] = np.where(less_zero)
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_to_receive_bw_min = (ue_bs[:, 1] == beam_index) & (ue_bs[:, 2] == sector_index) & ue_in_bs
                self.user_bw[ue_to_receive_bw_min] += self.bw_slot

            n_bw_slots[less_zero] -= active_beams[less_zero]
            dummy_queue = active_beams - n_bw_slots
            less_zero = (dummy_queue < 0) & beams_w_bw_slots

        meq_zero = (dummy_queue >= 0) & beams_w_bw_slots

        # second, allocate bw_slots when not all UEs can have it (meq_zero beams)
        [beam_index_list, sector_index_list] = np.where(meq_zero)
        while np.sum(meq_zero) != 0:
            for [beam_index, sector_index] in zip(beam_index_list, sector_index_list):
                ue_to_receive_bw_min = np.where((ue_bs[:, 1] == beam_index) & (ue_bs[:, 2] == sector_index) & ue_in_bs)[0]
                self.user_bw[ue_to_receive_bw_min[range(n_bw_slots[beam_index, sector_index].astype(int))]] += self.bw_slot
                if active_beams[beam_index, sector_index] - n_bw_slots[beam_index, sector_index] > 0:
                    self.in_queue_ue[beam_index, sector_index] = n_bw_slots[beam_index, sector_index]
                dummy_queue[beam_index, sector_index] = -1
                n_bw_slots[beam_index, sector_index] = 0
            meq_zero = (dummy_queue >= 0) & beams_w_bw_slots


    def generate_best_CQI_bw(self, ue_bs, best_cqi_beams, active_beams=None, c_target=None):
        # this function execute the best channel (Best CQI) channel allocation for all sectors of a BS (self.bs_index)
        from util.util_funcs import shannon_bw

        self.user_bw = np.zeros(shape=ue_bs.shape[0])
        self.sector_bw = np.zeros(shape=best_cqi_beams.shape) + self.bw

        ue_in_bs = ue_bs[:, 0] == self.bs_index  # UEs in self.bs_index filter
        sector_ues_by_cqi = []  # UEs for each sector of self.bs_index (appended because of the different sizes)
        index_controller = np.zeros(shape=best_cqi_beams.shape, dtype='int')   # index to control the BCQI queue allocation order
        non_empty_sectors = np.ones(shape=best_cqi_beams.shape, dtype='bool')  # to filter the sectors without UE
        # best_cqi_ue = np.zeros(shape=best_cqi_beams.shape)

        all_ues_by_cqi = np.argsort(ue_bs[:, 3])[::-1]  # bcqi ues ordered from the highest to lowest
        for sector_index in range(best_cqi_beams.shape[0]):
            ues_in_best_cqi_beam = ue_in_bs & (ue_bs[:, 1] == best_cqi_beams[sector_index]) & (ue_bs[:, 2] == sector_index)
            # best_cqi_ue[sector_index] = np.where(ues_in_best_cqi_beam & (ue_bs[:, 3] == ue_bs[ues_in_best_cqi_beam, 3].max()))  # REVER
            sector_ues_by_cqi.append(all_ues_by_cqi[ues_in_best_cqi_beam[all_ues_by_cqi]])
            if np.sum(ues_in_best_cqi_beam) == 0:
                non_empty_sectors[sector_index] = False  # to discard empty sectors in the following operations

        beam_queue_size = [len(x) - 1 for x in sector_ues_by_cqi]  # to avoid the index_controller to index over the bcqi queue
        non_ended_list = np.ones(shape=best_cqi_beams.shape, dtype='bool')  # indicates the lists that still have UEs to allocate

        while self.sector_bw.sum() != 0:
            best_cqi_ue = np.array([x[index_controller[i]] if x.size != 0 else 0 for i, x in enumerate(sector_ues_by_cqi)])  # select the one UE for each sector using the index_controller
            channels = ue_bs[best_cqi_ue, 3]
            bw_need = np.ceil(shannon_bw(bw=self.bw, tx_power=self.tx_power, channel_state=channels,
                                 c_target=c_target[best_cqi_ue]) / self.time_ratio)

            non_zero_bw = (self.sector_bw >= 0) & non_ended_list & non_empty_sectors  # controls the beams that still have available bw to allocate

            over_bw = (bw_need < self.sector_bw) & non_zero_bw  # more bw than best cqi UE needs
            lack_bw = (bw_need >= self.sector_bw) & non_zero_bw  # less bw than best cqi UE needs

            self.user_bw[best_cqi_ue[lack_bw]] = self.sector_bw[lack_bw]
            self.sector_bw[lack_bw] = 0

            self.user_bw[best_cqi_ue[over_bw]] = bw_need[over_bw]
            self.sector_bw[over_bw] -= bw_need[over_bw]

            index_controller += 1
            non_ended_list = np.array([index_controller[i] < beam_queue_size[i] for i, x in enumerate(sector_ues_by_cqi)])
            index_controller[~non_ended_list] = 0
            self.sector_bw[~non_ended_list] = 0

        # print('BCQI BW')

    def backup_scheduler(self):
        self.fake_user_bw = copy.deepcopy(self.user_bw)

    def restore_scheduler(self):
        self.user_bw = copy.deepcopy(self.fake_user_bw)
        return self.last_updated_beams

    def backup_updated_beams(self, updated_beams):
        self.last_updated_beams = updated_beams
