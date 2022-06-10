from models.scheduler.freq_scheduler import Freq_Scheduler
from models.scheduler.time_scheduler import Time_Scheduler

# TODO - LEMBRAR QUE NO MODELO PROPOSTO COMPLETO O C_TARGET PRECISA SER ATUALIZADOR POR FORA

class Scheduler:
    def __init__(self, scheduler_typ, bs_index, bw, simulation_time, time_slot, t_min=None, c_target=None, tx_power=None):
        self.scheduler_typ = scheduler_typ
        self.bw = bw
        self.t_index = None
        if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
            from models.scheduler.utility_based_fn import Util_fn
            self.util_fn = Util_fn(bs_index=bs_index, c_target=c_target, tx_power=tx_power, bw=bw)
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ,
                                                 bs_index=bs_index, time_slot=time_slot, t_min=t_min)
        elif self.scheduler_typ == 'RR':
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ,
                                                 bs_index=bs_index, time_slot=time_slot)

    def update_scheduler(self, active_beams, ue_bs, t_index=0, c_target=None, ue_updt=False):
        self.generate_beam_bw(active_beams=active_beams, t_index=t_index, ue_bs=ue_bs, c_target=c_target, ue_updt=ue_updt)
        self.generate_beam_timing(ue_bs=ue_bs, active_beams=active_beams, t_index=t_index,
                                  c_target=c_target, ue_updt=ue_updt)

    def generate_beam_bw(self, active_beams, t_index, ue_bs=None, c_target=None, ue_updt=False):
        if self.scheduler_typ == 'RR':
            if t_index == 0:
                self.freq_scheduler.generate_proportional_beam_bw(active_beams=active_beams)
        elif self.scheduler_typ == 'prop-smp' or self.scheduler_typ == 'prop-cmp':
            if ue_bs is not None or c_target is not None:
                self.util_bsd_bw(active_beams=active_beams, t_index=t_index, ue_bs=ue_bs, c_target=c_target)
                # self.generate_weighted_bw(ue_bs=ue_bs, bs_index=self.bs_index, active_beams=active_beams, t_index=t_index)
            else:
                raise ValueError('The scheduler type is typed wring or its not supported! Please check the param.yml file.')


    def util_bsd_bw(self, active_beams, t_index, ue_bs, c_target=None, ue_updt=False):
        if self.scheduler_typ == 'prop-smp':
            if t_index == 0:
                self.util_fn.update_c_target(shape=ue_bs.shape[0])
                self.util_fn.slice_utility(ue_bs=ue_bs, active_beams=active_beams)
                if t_index != self.t_index:
                    self.t_index = t_index
                    self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.freq_scheduler.generate_weighted_bw(ue_bs=ue_bs, active_beams=active_beams,
                                                         slice_util=self.util_fn.slice_util, beam_util=self.util_fn.beam_util)
        elif self.scheduler_typ == 'prop-cmp':
            if t_index != self.t_index:
                self.t_index = t_index
                self.util_fn.update_c_target(c_target=c_target, shape=ue_bs.shape[0])
                self.util_fn.slice_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
            self.freq_scheduler.generate_weighted_bw(ue_bs=ue_bs, active_beams=active_beams,
                                                     slice_util=self.util_fn.slice_util, beam_util=self.util_fn.beam_util)


    def generate_beam_timing(self, ue_bs, active_beams, t_index=0, c_target=None, ue_updt=False):
        # reminder: t_min is the minimum reserved per beam time
        # this function point to other functions based on the choosen scheduler
        if t_index == 0:
            # this function call is to pass the base dimensions to simplfy some interal expressions (beams and sectors)
            self.time_scheduler.set_base_dimensions(n_beams=active_beams.shape[0], n_sectors=active_beams.shape[1])

        if self.scheduler_typ == 'prop-cmp':
            if ue_updt:
                if t_index != self.t_index:
                    self.t_index = t_index
                    self.util_fn.update_c_target(c_target=c_target, shape=ue_bs.shape[0])
                    self.util_fn.slice_utility(ue_bs=ue_bs, active_beams=active_beams)
                    self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.util_fn.sector_utility()
                self.time_scheduler.generate_utility_based_beam_timing(t_index=t_index, ue_bs=ue_bs,
                                                                       active_beams=active_beams,
                                                                       beam_util=self.util_fn.beam_util,
                                                                       beam_util_log=self.util_fn.beam_util_log,
                                                                       sector_util=self.util_fn.sector_util)
        elif self.scheduler_typ == 'prop-smp':
            if ue_updt:
                if t_index != self.t_index:
                    self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.util_fn.sector_utility()
                self.time_scheduler.generate_utility_based_beam_timing(t_index=t_index, ue_bs=ue_bs,
                                                                       active_beams=active_beams,
                                                                       beam_util=self.util_fn.beam_util,
                                                                       beam_util_log=self.util_fn.beam_util_log,
                                                                       sector_util=self.util_fn.sector_util)
        elif self.scheduler_typ == 'RR':
            if t_index == 0:
                self.time_scheduler.generate_proportional_beam_timing(time_slot=1, active_beams=active_beams)

        # self.beam_timing_sequence = self.beam_timing_sequence.astype(int)