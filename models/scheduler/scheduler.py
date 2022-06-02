from freq_scheduler import Freq_Scheduler
from time_scheduler import Time_Scheduler

# TODO - LEMBRAR QUE NO MODELO PROPOSTO COMPLETO O C_TARGET PRECISA SER ATUALIZADOR POR FORA

class Scheduler:
    def __init__(self, scheduler_typ, bs_index, bw, simulation_time, c_target=None, tx_power=None):
        self.scheduler_typ = scheduler_typ
        self.bw = bw
        self.t_index = None
        if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
            from utility_based_fn import Util_fn
            self.util_fn = Util_fn(bs_index=bs_index, c_target=c_target, tx_power=tx_power, bw=bw)
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ)
        elif self.scheduler_typ == 'RR':
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ)

    def update(self):
        pass

    def generate_beam_bw(self, active_beams, t_index=None, ue_bs=None, c_target=None):
        if t_index is None:
            self.t_index = 1
        if self.scheduler_typ == 'RR':
            if self.t_index == 1:
                self.freq_scheduler.generate_proportional_beam_bw(active_beams=active_beams)
        elif self.scheduler_typ == 'prop-smp' or self.scheduler_typ == 'prop-cmp':
            if ue_bs is not None or c_target is not None:
                self.util_bsd_bw(active_beams=active_beams, t_index=t_index, ue_bs=ue_bs, c_target=c_target)
                # self.generate_weighted_bw(ue_bs=ue_bs, bs_index=self.bs_index, active_beams=active_beams, t_index=t_index)
            else:
                pass
                # todo fazer a exceção aqui
                # raise Exception

    def util_bsd_bw(self, active_beams, t_index, ue_bs, c_target=None):
        if self.scheduler_typ == 'prop-smp':
            if t_index == 1:
                self.util_fn.update_c_target(shape=ue_bs.shape[0])
                self.util_fn.slice_util(ue_bs=ue_bs, active_beams=active_beams)
                self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.freq_scheduler.generate_weighted_bw(ue_bs=ue_bs, active_beams=active_beams,
                                                         slice_util=self.util_fn.slice_util, beam_util=self.util_fn.beam_util)
        elif self.scheduler_typ == 'prop-cmp':
            self.util_fn.update_c_target(c_target=c_target, shape=ue_bs.shape[0])
            self.util_fn.slice_util(ue_bs=ue_bs, active_beams=active_beams)
            self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
            self.freq_scheduler.generate_weighted_bw(ue_bs=ue_bs, active_beams=active_beams,
                                                     slice_util=self.util_fn.slice_util, beam_util=self.util_fn.beam_util)


    def generate_beam_timing(self, time_slot, t_index=None, weighted_act_beams=None, uniform_time_dist=True):
        # this function point to other functions based on the choosen scheduler
        if t_index is None:
            self.t_index = 1

        if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
            if not uniform_time_dist and weighted_act_beams is not None:  # to check if the beam weights are to be used
                self.time_scheduler.generate_utility_based_beam_timing(time_slot=time_slot, t_index=t_index,
                                                        weighted_act_beams=weighted_act_beams)
        elif self.scheduler_typ == 'RR':
            if t_index == 1:
                self.time_scheduler.generate_proportional_beam_timing(time_slot=1)

        self.beam_timing_sequence = self.beam_timing_sequence.astype(int)