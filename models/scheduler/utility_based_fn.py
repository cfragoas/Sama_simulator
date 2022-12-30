import numpy as np

class Util_fn:
    def __init__(self, bs_index, bw, c_target, tx_power):
        # parameters for the utility-based scheduling
        self.bs_index = bs_index
        self.c_target = c_target
        self.bw = bw
        self.tx_power = tx_power
        self.slice_util = None
        self.beam_util = None
        self.beam_util_log = None
        self.sector_util = None


    ################################### UTILITY-BASED SCHEDULING BASE FUNCTIONS ########################################

    def update_c_target(self, shape, c_target=None):  # todo - arrumar o shape aqui e da função chamada
        # self.c_target = c_target + np.zeros(shape=ue_bs.shape[0])  # because c_target can be unique for each UE
        if c_target is None:
            if self.c_target is not None:
                c_target = self.c_target
            else:
                raise ValueError('Need to define the target capacity criteria to use the utility-based scheduler !!!!!')
        self.c_target = c_target + np.zeros(shape=shape)  # because c_target can be unique for each UE


    def slice_utility(self, ue_bs, active_beams):  # utility per user bw/snr
        self.slice_util = np.zeros(shape=ue_bs.shape[0])
        bw_need = np.zeros(shape=ue_bs.shape[0])
        snr = np.zeros(shape=ue_bs.shape[0]) - 10000

        c_target = self.c_target * 10E6

        beam_bw = np.zeros(shape=active_beams.shape)

        # segundo teste
        active_beam_index = active_beams != 0
        beam_bw[active_beam_index] = (self.bw / active_beams[active_beam_index]) / 10  # minimum per beam bw
        active_ue = ue_bs[:, 1] != -1

        bw_min = np.zeros(shape=ue_bs.shape[0])
        for ue_index, ue in enumerate(ue_bs):
            bw_min[ue_index] = beam_bw[ue[1], ue[2]] * 10E6  # minimum per user bw

        bw = self.bw * 10E6  # making SNR for a bandwidth of 5MHz
        k = 1.380649E-23  # Boltzmann's constant (J/K)
        t = 290  # absolute temperature
        pw_noise_bw = k * t * bw  # noise power
        # it is important here that tx_pw been in dBW (not dBm!!!)
        tx_pw = 10 ** (self.tx_power / 10)  # converting from dBW to watt
        snr[active_ue] = (tx_pw * 10 ** (ue_bs[active_ue, 3] / 10)) / pw_noise_bw  # signal to noise ratio (linear)
        bw_need[active_ue] = c_target[active_ue]/(np.log2(1 + snr[active_ue]))
        # bw_need[active_ue] = 2 ** (c_target[active_ue] / snr[active_ue]) - 1  # needed bw to achieve the capacity target
        # snr[active_ue][snr[active_ue] < 0] = 1.01  # to prevent a negative utility value in log2
        self.slice_util[active_ue] = (bw_min[active_ue] / bw_need[active_ue]) * np.log2(1 + snr[active_ue])
        # self.slice_util[active_ue & (bw_need < bw_min)] = 10E-12  # TESTANDO ISSO AQUI

    def beam_utility(self, ue_bs, active_beams):
        # ue_bs -> bs|beam|sector|ch_gain

        # self.slice_utility(ue_bs=ue_bs, active_beams=active_beams)
        self.beam_util = np.zeros(shape=active_beams.shape)

        for sector_index in np.unique(ue_bs[ue_bs[:, 0] == self.bs_index][:, 2]).astype(int):
            for beam_index in np.unique(ue_bs[(ue_bs[:, 0] == self.bs_index) & (ue_bs[:, 2] == sector_index)][:, 1]).astype(int):
                ue_in_beam_bs = np.where(
                    (ue_bs[:, 0] == self.bs_index) & (ue_bs[:, 1] == beam_index) & (ue_bs[:, 2] == sector_index))
                self.beam_util[beam_index, sector_index] = np.sum(self.slice_util[ue_in_beam_bs])

        # ================= CHECAR ALTERAÇÃO !!! ====================
        self.beam_util[self.beam_util < 0] = 10E-12  # to prevent a negative utility value in log2
        self.beam_util_log = np.zeros(shape=self.beam_util.shape)
        beams_2calc = self.beam_util != 0  # active beams and beam_util not between 0~1

        # self.beam_util_log[beams_2calc] = np.log2(self.beam_util[beams_2calc])  # with log
        self.beam_util_log[beams_2calc] = self.beam_util[beams_2calc]  # without log

        self.beam_util_log[self.beam_util_log < 0] = 0.0001  # to avoid having allocated time < 0 beeing a detected as active beam

    def sector_utility(self):
        self.sector_util = np.sum(self.beam_util_log, axis=0)  # sector util. is the sum of the beam util.

    ####################################################################################################################