import copy

from models.scheduler.freq_scheduler import Freq_Scheduler
from models.scheduler.time_scheduler import Time_Scheduler
from numpy import ceil

# Scheduler class is responsable to call the correct time and frequency schedulers depending of the state of the
# simulation on the last time index and the choosen scheduler (scheduler_typ). It has one Freq_Scheduler and one
# Time_Scheduler object.
# A Schedule class object is unique for each base station.

class Scheduler:
    def __init__(self, scheduler_typ, bs_index, bw, simulation_time, time_slot, t_min=None, bw_slot=None,
                 c_target=None, tx_power=None):
        self.scheduler_typ = scheduler_typ  # its a string representing the choosen scheduller
        self.bw = bw  # the bandwidth available for each BS sector (MHz)
        self.t_index = None  # indicates the last t_index when the scheduler is called
        # instatiating the Freq_Scheduler and TIme_Scheduler based on the scheduler_type options
        if self.scheduler_typ == 'prop-cmp' or self.scheduler_typ == 'prop-smp':
            from models.scheduler.utility_based_fn import Util_fn
            self.util_fn = Util_fn(bs_index=bs_index, c_target=c_target, tx_power=tx_power, bw=bw)
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ,
                                                 bs_index=bs_index, time_slot=time_slot, t_min=t_min)
        elif self.scheduler_typ == 'RR':
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ, bw_slot=bw_slot)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ,
                                                 bs_index=bs_index, time_slot=time_slot)
        elif self.scheduler_typ == 'BCQI':
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ,
                                                 tx_power=tx_power, simulation_time=simulation_time,
                                                 time_slot=time_slot)
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time, scheduler_typ=scheduler_typ,
                                                 bs_index=bs_index, time_slot=time_slot)
        elif self.scheduler_typ == 'PF':
            self.freq_scheduler = Freq_Scheduler(bw=bw, bs_index=bs_index, scheduler_typ=scheduler_typ, bw_slot=bw_slot,
                                                 tx_power=tx_power, simulation_time=simulation_time, time_slot=time_slot)
            # it uses half of the time because it splits the scheduling between BCQI and RR
            self.time_scheduler = Time_Scheduler(simulation_time=simulation_time,
                                                 scheduler_typ=scheduler_typ, bs_index=bs_index, time_slot=time_slot)
            # 0 if is best cqi time or 1 if its next UE (like RR) -> [time, frequency]
            # it is a list because of the time/frequency scheduler separation -> [time, frequency]
            self.scheduler_status = [0, 0]  # it starts with the BCQI state
        else:
            raise ValueError('Invalid scheduler type! Please check the param.yml file.')

    def update_scheduler(self, active_beams, ue_bs, t_index=0, c_target=None, ue_updt=False, updated_beams=None):
        # this is the function responsable to call the general time and frequency functions to update the schedullers,
        # if necessary
        print(self.scheduler_status)
        self.generate_beam_bw(active_beams=active_beams, t_index=t_index, ue_bs=ue_bs,
                              c_target=c_target, ue_updt=ue_updt, updated_beams=updated_beams)
        self.generate_beam_timing(ue_bs=ue_bs, active_beams=active_beams, t_index=t_index,
                                  c_target=c_target, ue_updt=ue_updt)

    def generate_beam_bw(self, active_beams, t_index, ue_bs=None, c_target=None, ue_updt=False, updated_beams=None):
        # this function is responsable to call the frequency schedulers for the chosen scheduler_typ
        if self.scheduler_typ == 'RR':
            if self.t_index != t_index:
                self.t_index = t_index
                self.freq_scheduler.generate_RR_bw(ue_bs=ue_bs, active_beams=active_beams, updated_beams=updated_beams)
        elif self.scheduler_typ == 'prop-smp' or self.scheduler_typ == 'prop-cmp':
            if ue_bs is not None or c_target is not None:
                self.util_bsd_bw(active_beams=active_beams, t_index=t_index, ue_bs=ue_bs, c_target=c_target)
                # self.generate_weighted_bw(ue_bs=ue_bs, bs_index=self.bs_index, active_beams=active_beams, t_index=t_index)
            else:
                raise ValueError('The scheduler type is typed wrong or its not supported! Please check the param.yml file.')
        elif self.scheduler_typ == 'BCQI':
            if self.time_scheduler.best_cqi_beams is None or self.t_index != t_index:
                self.generate_beam_timing(ue_bs=ue_bs, active_beams=active_beams, t_index=t_index)
            self.freq_scheduler.generate_best_CQI_bw(ue_bs=ue_bs, best_cqi_beams=self.time_scheduler.best_cqi_beams,
                                                     c_target=c_target)
        elif self.scheduler_typ == 'PF':
            if self.scheduler_status[1] == 0:  # if its BCQI step
                if self.time_scheduler.best_cqi_beams is None or self.t_index != t_index:
                    self.generate_beam_timing(ue_bs=ue_bs, active_beams=active_beams, t_index=t_index)
                self.freq_scheduler.generate_best_CQI_bw(ue_bs=ue_bs, best_cqi_beams=self.time_scheduler.best_cqi_beams,
                                                         c_target=c_target)
                self.scheduler_status[1] = 1  # changing the status for the next step
                self.freq_scheduler.backup_updated_beams(updated_beams)
            else:  # it is next UE step (RR algorithm)
                self.scheduler_status[1] = 0  # changing the status for the next step
                # if self.t_index != t_index:
                #     self.t_index = t_index
                updated_beams = self.freq_scheduler.restore_scheduler()  # because the bcqi scheduller will write over the RR one
                self.freq_scheduler.generate_RR_bw(ue_bs=ue_bs, active_beams=active_beams,
                                                   updated_beams=updated_beams)
                self.freq_scheduler.backup_scheduler()  # making the backup for the next time RR is used

    def generate_beam_timing(self, ue_bs, active_beams, t_index=0, c_target=None, ue_updt=False):
        # reminder: t_min is the minimum reserved per beam time
        # this function is responsible to call the time schedulers and auxiliary functions for the chosen scheduler_typ
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
                    self.t_index = t_index
                    self.util_fn.beam_utility(ue_bs=ue_bs, active_beams=active_beams)
                self.util_fn.sector_utility()
                self.time_scheduler.generate_utility_based_beam_timing(t_index=t_index, ue_bs=ue_bs,
                                                                       active_beams=active_beams,
                                                                       beam_util=self.util_fn.beam_util,
                                                                       beam_util_log=self.util_fn.beam_util_log,
                                                                       sector_util=self.util_fn.sector_util)
        elif self.scheduler_typ == 'RR':
            if ue_updt:
                self.time_scheduler.generate_ue_qtd_proportional_beam_timing(active_beams=active_beams,
                                                                             t_index=t_index)

        elif self.scheduler_typ == 'BCQI':
            if t_index != self.t_index:
                self.t_index = t_index
                self.time_scheduler.generate_best_cqi_beam_timing(ue_bs=ue_bs)

        elif self.scheduler_typ == 'PF':
            if t_index != self.t_index:
                self.t_index = t_index
                if self.scheduler_status[0] == -1:
                    status = 0
                else:
                    status = self.scheduler_status[0]

                self.time_scheduler.generate_proportional_fair_timing(ue_bs=ue_bs, active_beams=active_beams,
                                                                      t_index=t_index, ue_updt=ue_updt,
                                                                      status=status)
                if self.scheduler_status[0] == 0:  # if its BCQI step
                    self.scheduler_status[0] = 1  # changing the status for the next step
                        # self.time_scheduler.generate_best_cqi_beam_timing(ue_bs=ue_bs)
                    # todo - colocar aqui para marcar o UE como atendido nesse step para o próximo
                else:  # it is next UE step
                    self.scheduler_status[0] = 0  # changing the status for the next step
                    # if ue_updt:  # todo - pensar se precisa desse check nesse caso especial
                    #     self.time_scheduler.generate_ue_qtd_proportional_beam_timing(active_beams=active_beams,
                    #                                                                  t_index=t_index)
                    # todo - ver o que fazer aqui (chamar a função de update de time scheduler para PF)

    def util_bsd_bw(self, active_beams, t_index, ue_bs, c_target=None, ue_updt=False):
        # this function is exclusive for the proposed utility-based scheduler call and auxiliary functions (util_fn object)
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

