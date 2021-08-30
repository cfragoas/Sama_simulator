import numpy as np

class User_eq:
    def __init__(self, positions, height):
        self.positions = positions
        self.height = height
        self.bs_candidates = None
        self.gain_matrix = None

        self.sector_map = None

        # calculated variables
        self.ue_bs = None  # bs|beam

    def acquire_bs_and_beam(self, ch_gain_map, sector_map):
        self.sector_map = sector_map
        self.ue_bs = np.ndarray(shape=(ch_gain_map.shape[1], 2))  # bs|beam
        # if ch_gain_map is None or self.sector_map is None:
        #     print('Need channel gain map to aquire BS and beam !!!')
        #     return
        # else:
        for ue_index in range(ch_gain_map.shape[1]):
            self.ue_bs[ue_index] = np.unravel_index(np.argmax(ch_gain_map[:, ue_index]), ch_gain_map[:, ue_index].shape)
