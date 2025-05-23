import copy

import numpy as np

# The User_eq class represents the set of UEs and its functions in a network
# The mais function of the class is to store the UE set and his associations with BS, sector and beam in the network
# UEs can be turned off when its necessary if the flag -1 is used

class User_eq:
    def __init__(self, height=None, tx_power=None, positions=None): 
        self.positions = positions  # x, y of each UE located on the ROI
        self.height = height  # UE height (m)
        self.user_condition = None # Tem que mexer
        self.bs_candidates = None
        self.gain_matrix = None
        self.tx_power = tx_power  # UE tx power in dBW
        self.prop_scenario = None # Propagation scenario of certain user (LOS or NLOS)
        
        self.sector_map = None

        # calculated variables
        self._ue_bs = None  # bs|beam|sector|ch_gain - linked UE and BS indexes - TO BE DELETED

        self.dw_ue_bs = None  # bs|beam|sector|ch_gain - linked UE and BS indexes for the downlink
        self.up_ue_bs = None  # bs|beam|sector|ch_gain - linked UE and BS indexes for the uplink

        self.active_ue = None  # list of UEs that are sensed in the network
        self.ue_bs_total = None  # bs|beam|sector|ch_gain - all UE and BS indexes + non linked

    def acquire_bs_and_beam(self, ch_gain_map, sector_map, pw_5mhz):
        self.sector_map = sector_map.astype(int)
        self.dw_ue_bs = np.ndarray(shape=(ch_gain_map.shape[1], 4))  # bs|beam|sector|ch_gain

        for ue_index in range(ch_gain_map.shape[1]):
            self.dw_ue_bs[ue_index] = np.concatenate((np.array(
                np.unravel_index(np.argmax(ch_gain_map[:, ue_index]), ch_gain_map[:, ue_index].shape)),
                   np.array(self.sector_map[np.unravel_index(np.argmax(ch_gain_map[:, ue_index]),
                     ch_gain_map[:, ue_index].shape)[0], ue_index]),
                       np.array(np.max(ch_gain_map[:, ue_index]))), axis=None)

        # the '+30' here is because of the conversion from dBW to dBm
        inactive_ue = np.where(self.dw_ue_bs[:, 3] + pw_5mhz + 30 < -100)  # ref: ETSI TS 138 101-1 (in 5 MHz) (simplifying for all bands here)
        self.dw_ue_bs[inactive_ue, 0:3] = -1

        # the '+30' here is because of the convertion from dBW to dBm
        self.active_ue = np.where(self.dw_ue_bs[:, 3] + pw_5mhz + 30 > -100) # ref: ETSI TS 138 101-1 (in 5 MHz) (simplifying for all bands here)

        # self.sector_map = self.sector_map[:, self.active_ue][0]  # adjusting the sector map to be the same size as the
        # as the update ue_bs with the active UEs

        # self.ue_bs_total = self.ue_bs
        # self.ue_bs = self.ue_bs[self.active_ue]

        self.dw_ue_bs = self.dw_ue_bs.astype(int)
        self.up_ue_bs = copy.copy(self.dw_ue_bs)  # replicating the relationship table to the uplink


        # self.ue_bs = self.ue_bs[~np.isnan(self.ue_bs[:,3])].astype(int)


        # self.ue_bs[~np.isnan(self.ue_bs)] = self.ue_bs[~np.isnan(self.ue_bs)].astype(int)

    def remove_ue(self, ue_index, downlink=False, uplink=False):  # put a -1 flag to stop communicating with UE that achieve the target capacity
        if downlink:
            self.dw_ue_bs[ue_index, 0:3] = -1
        if uplink:
            self.up_ue_bs[ue_index, 0:3] = -1

    # Get info if the user is indoor or outdoor based on the raster 
    def obtain_user_condition(self,centers,samples,raster_grid):
        if samples is not None:

            ucond = np.ndarray(shape=(centers.shape[0], samples.shape[0]))
            in_out_user = np.ndarray(shape=(centers.shape[0], 2))

            for i, user in enumerate(samples):
                in_out_user = raster_grid[user[0].astype(int),user[1].astype(int)]
                ucond[:,i] = in_out_user
                #print('teste')

            self.user_condition = ucond
        
        else:
            raise Exception('For obtaining user conditions samples must be different than None!')
         