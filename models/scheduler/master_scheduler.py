import numpy as np
from models.scheduler.scheduler import Scheduler

class Master_scheduler:
    # 1 IN TDD SCHEDULER => UPLINK
    # 0 IN TDD SCHEDULER => DOWNLINK
    def __init__(self):
        self.up_scheduler = False
        self.dwn_scheduler = False
        self.up_tdd_time = None
        self.dwn_tdd_time = None
        self.tdd_scheduler = None


    def create_uplink(self, scheduler_typ, bs_index, bw, simulation_time, time_slot, t_min=None, bw_slot=None,
                 c_target=None, tx_power=None):
        if self.tdd_scheduler is not None:
            simulation_time = np.sum(self.tdd_scheduler == 1)
        # else:
        #     print('No tdd/fdd selected using only the uplink scheduler')

        self.up_scheduler = Scheduler(scheduler_typ=scheduler_typ, bs_index=bs_index, bw=bw,
                                      simulation_time=simulation_time, time_slot=time_slot, t_min=t_min,
                                      bw_slot=bw_slot, c_target=c_target, tx_power=tx_power)

    def create_downlink(self, scheduler_typ, bs_index, bw, time_slot, simulation_time, t_min=None, bw_slot=None,
                 c_target=None, tx_power=None):
        if self.tdd_scheduler is not None:
            # self.dwn_time = np.sum(self.tdd_scheduler == 0)
            simulation_time = np.sum(self.tdd_scheduler == 0)
        # else:
        #     print('No tdd/fdd selected using only the downlink scheduler')
        self.dwn_scheduler = Scheduler(scheduler_typ=scheduler_typ, bs_index=bs_index, bw=bw,
                                       simulation_time=simulation_time, time_slot=time_slot, t_min=t_min,
                                       bw_slot=bw_slot, c_target=c_target, tx_power=tx_power)

    def create_tdd_scheduler(self, simulation_time, t_index=0, up_tdd_time=0.3):
        self.up_tdd_time = up_tdd_time
        self.dwm_tdd_time = 1 - up_tdd_time
        if self.tdd_scheduler is None:
            self.tdd_scheduler = np.zeros(shape=simulation_time, dtype=int)

        micro_tdd_schl_size = simulation_time//100
        self.micro_tdd_scheduler = np.zeros(shape=micro_tdd_schl_size, dtype=int)
        tdd_up_time = np.round(micro_tdd_schl_size * self.up_tdd_time).astype(int)
        # tdd_dwn_time = simulation_time//100 - tdd_up_time
        self.micro_tdd_scheduler[range(0, tdd_up_time)] = 1
        for i in range(100):
            self.tdd_scheduler[range(i*micro_tdd_schl_size, i*micro_tdd_schl_size+micro_tdd_schl_size)] = self.micro_tdd_scheduler


    def create_fdd_scheduler(self):
        pass

    def update_scheduler(self, t_index, downlink_metrics=None, uplink_metrics=None):
        # todo - ver a porra da conta pra saber quantos slots de tempo tem que fazer pra up e down

        if self.up_scheduler is not None:
            self.up_scheduler.update_scheduler(self, active_beams=downlink_metrics['active_beams'],
                                               ue_bs=downlink_metrics['ue_bs'],
                                               t_index=t_index, c_target=downlink_metrics['c_target'],
                                               ue_updt=downlink_metrics['ue_updt'],
                                               updated_beams=downlink_metrics['updated_beams'])
        if self.dwn_scheduler is not None:
            self.dwn_scheduler.update_scheduler(self, active_beams=uplink_metrics['active_beams'],
                                               ue_bs=uplink_metrics['ue_bs'],
                                               t_index=t_index, c_target=uplink_metrics['c_target'],
                                               ue_updt=uplink_metrics['ue_updt'],
                                               updated_beams=uplink_metrics['updated_beams'])

